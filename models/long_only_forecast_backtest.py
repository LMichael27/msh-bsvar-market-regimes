"""
Long-only Forecast Portfolio Backtest (monthly rebalancing)

This module exposes:
- BacktestConfig: configuration dataclass
- BacktestResult: result dataclass
- prepare_signals_from_oos(): make binary month-end signals from OOS forecasts
- long_only_forecast_backtest(): core backtest using prices + signals
- backtest_from_oos(): convenience wrapper (prices + OOS → result)
- plot_equity_curve(): quick plot helper (saves PNG if path given)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """Configuration for the long-only forecast backtest."""
    tc_bps_roundtrip: float = 10.0   # round-trip transaction cost (bps) on traded notional
    cash_rate_annual: float = 0.00   # annual cash yield (e.g., 0.03 for 3%)
    min_assets_long: int = 1         # minimum number of long signals; else go to cash


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    weights: pd.DataFrame            # monthly portfolio weights per asset
    returns_net: pd.Series           # monthly portfolio net returns (after TC + cash yield)
    equity: pd.Series                # equity curve (start value applied outside or implicit)
    turnover: pd.Series              # monthly L1 turnover (sum abs weight changes)
    metrics: Dict[str, float]        # summary statistics


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _to_month_end_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure month-end frequency for prices (last valid observation per month).
    Accepts daily or already-monthly price levels.
    """
    px = prices.copy()
    if px.index.inferred_type not in ("datetime64", "datetime64tz"):
        px.index = pd.to_datetime(px.index)
    if pd.infer_freq(px.index) not in ("ME", "BM"):
        px = px.resample("ME").last()
    return px


def _prices_to_simple_returns(prices_m: pd.DataFrame) -> pd.DataFrame:
    """Convert month-end prices to simple monthly returns."""
    return prices_m.pct_change()


def _align_signals_month_end(signals: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Align (possibly daily) signals to month-end and reindex to target monthly index.
    Uses last signal in month and forward-fills across missing months.
    """
    sig = signals.copy()
    if sig.index.inferred_type not in ("datetime64", "datetime64tz"):
        sig.index = pd.to_datetime(sig.index)
    if pd.infer_freq(sig.index) not in ("ME", "BM"):
        sig = sig.resample("ME").last()
    sig = sig.reindex(target_index).ffill()
    return sig


# ---------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------
def prepare_signals_from_oos(
    oos_forecast_mean: np.ndarray,
    oos_index: pd.DatetimeIndex,
    asset_names: list[str],
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Turn OOS point forecasts into binary month-end signals:
        signal_{i,t} = 1 if forecast_{i,t} > threshold, else 0
    The signals are stamped at OOS month-ends (oos_index).
    They predict the NEXT month (the backtest will shift them internally).

    Parameters
    ----------
    oos_forecast_mean : ndarray (T_test x N)
        Forecasts for each asset per OOS month (e.g., returns).
    oos_index : DatetimeIndex
        Month-end timestamps corresponding to rows of oos_forecast_mean.
    asset_names : list[str]
        Names of assets (len = N), must match the forecast column order.
    threshold : float
        Threshold for turning forecasts into 0/1 signals (default: 0.0).

    Returns
    -------
    pd.DataFrame
        Binary signals (0/1), index = oos_index (month-end), columns = asset_names.
        These are *publication* dates; the backtest will apply shift(+1).
    """
    if oos_forecast_mean.ndim != 2:
        raise ValueError("oos_forecast_mean must be a 2D array (T_test x N).")
    if oos_forecast_mean.shape[1] != len(asset_names):
        raise ValueError("oos_forecast_mean columns must match len(asset_names).")
    if len(oos_index) != oos_forecast_mean.shape[0]:
        raise ValueError("oos_index length must match oos_forecast_mean rows.")

    sig = (oos_forecast_mean > threshold).astype(int)
    signals = pd.DataFrame(sig, index=pd.to_datetime(oos_index), columns=asset_names)
    return signals


def prepare_signed_signals_from_oos(
    oos_forecast_mean: np.ndarray,
    oos_index: pd.DatetimeIndex,
    asset_names: list[str],
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Turn OOS point forecasts into ternary month-end signals {-1, 0, +1}:
        +1 if forecast > threshold,
        -1 if forecast < -threshold,
        0 otherwise.
    The signals are stamped at OOS month-ends (oos_index) and the backtest will shift them.
    """
    if oos_forecast_mean.ndim != 2:
        raise ValueError("oos_forecast_mean must be a 2D array (T_test x N).")
    if oos_forecast_mean.shape[1] != len(asset_names):
        raise ValueError("oos_forecast_mean columns must match len(asset_names).")
    if len(oos_index) != oos_forecast_mean.shape[0]:
        raise ValueError("oos_index length must match oos_forecast_mean rows.")

    sig = np.zeros_like(oos_forecast_mean, dtype=int)
    sig[oos_forecast_mean > threshold] = 1
    sig[oos_forecast_mean < -threshold] = -1
    signals = pd.DataFrame(sig, index=pd.to_datetime(oos_index), columns=asset_names)
    return signals


# ---------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------
def long_only_forecast_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: Optional[BacktestConfig] = None,
    n_months: Optional[int] = 36,
    start_equity: float = 1.0
) -> BacktestResult:
    """
    Long-only equal-weight portfolio driven by binary signals.
    Signals are assumed to be *published* at month-end t and *applied* in month t+1.
    If fewer than cfg.min_assets_long longs are available, the strategy stays in cash.

    Parameters
    ----------
    prices : DataFrame
        Price levels (daily or monthly), columns = assets (must match signals columns).
    signals : DataFrame
        Binary (0/1) signals at month-ends (publication time). Index must be datetime-like.
        These are shifted +1 inside to avoid look-ahead.
    cfg : BacktestConfig
        Transaction costs, cash yield, min assets constraint.
    n_months : Optional[int]
        If set (default 36), the backtest is restricted to the last n_months available.
        If None, uses the full overlap between prices and signals.
    start_equity : float
        Initial portfolio value for the equity curve.

    Returns
    -------
    BacktestResult
        Weights, net returns, equity curve, turnover, and summary metrics.
    """
    if cfg is None:
        cfg = BacktestConfig()

    # 1) Normalize inputs to monthly month-end indices
    prices_m = _to_month_end_prices(prices)
    rets = _prices_to_simple_returns(prices_m)

    # Align signals to monthly and the same index as returns; then shift by +1 month
    sig_aligned = _align_signals_month_end(signals, rets.index)
    sig_next = sig_aligned.shift(1).fillna(0.0)  # apply *next* month

    # Optional: restrict to last n_months
    if n_months is not None:
        rets = rets.iloc[-n_months:]
        sig_next = sig_next.reindex(rets.index).fillna(0.0)

    # 2) Build equal-weight long-only weights
    W_list = []
    for t, row in sig_next.iterrows():
        long_mask = (row > 0.5)
        n_long = int(long_mask.sum())
        if n_long >= cfg.min_assets_long:
            w = np.where(long_mask, 1.0 / n_long, 0.0)
        else:
            w = np.zeros_like(row.values)
        W_list.append(pd.Series(w, index=sig_next.columns, name=t))
    W = pd.DataFrame(W_list).astype(float)

    # 3) Turnover & transaction costs
    W_prev = W.shift(1).fillna(0.0)
    turnover = (W - W_prev).abs().sum(axis=1)
    tc_rate = cfg.tc_bps_roundtrip / 10000.0
    tc = turnover * tc_rate

    # 4) Net portfolio returns (add cash yield on uninvested weight)
    gross = (W * rets).sum(axis=1)
    cash_weight = 1.0 - W.sum(axis=1).clip(0.0, 1.0)
    monthly_cash_yield = (1.0 + cfg.cash_rate_annual) ** (1.0 / 12.0) - 1.0
    gross_with_cash = gross + cash_weight * monthly_cash_yield
    net = (gross_with_cash - tc).dropna()

    # 5) Equity curve from start_equity
    equity = start_equity * (1.0 + net).cumprod()

    # 6) Metrics
    ann = 12.0
    if len(net) > 0:
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (ann / len(net)) - 1.0
    else:
        cagr = np.nan
    vol_ann = net.std(ddof=1) * np.sqrt(ann) if len(net) > 1 else np.nan
    sharpe = (cagr - cfg.cash_rate_annual) / vol_ann if (vol_ann and vol_ann > 0) else np.nan
    mdd = (equity / equity.cummax() - 1.0).min() if len(equity) else np.nan

    metrics = {
        "CAGR": float(cagr) if np.isfinite(cagr) else np.nan,
        "Vol_Ann": float(vol_ann) if np.isfinite(vol_ann) else np.nan,
        "Sharpe_ex_cash": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "MaxDD": float(mdd) if np.isfinite(mdd) else np.nan,
        "TotalGain": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else np.nan,
    }

    # Keep index-consistent slices
    W = W.loc[net.index]
    turnover = turnover.loc[net.index]

    return BacktestResult(
        weights=W,
        returns_net=net.rename("net_return"),
        equity=equity.rename("equity"),
        turnover=turnover.rename("turnover"),
        metrics=metrics,
    )


def long_short_forecast_backtest(
    prices: pd.DataFrame,
    signed_signals: pd.DataFrame,
    cfg: Optional[BacktestConfig] = None,
    n_months: Optional[int] = 36,
    start_equity: float = 1.0,
) -> BacktestResult:
    """
    Long-short equal-weight portfolio driven by ternary signals {-1,0,1}.
    Each non-zero signal gets equal magnitude and sign. We scale weights so that
    the sum of absolute weights (gross exposure) is 1. Cash is used for any
    remaining weight (should be zero if at least one non-zero signal exists).

    Signals are published at month-end t and applied in t+1 (shifted inside).
    """
    if cfg is None:
        cfg = BacktestConfig()

    prices_m = _to_month_end_prices(prices)
    rets = _prices_to_simple_returns(prices_m)

    sig_aligned = _align_signals_month_end(signed_signals, rets.index)
    sig_next = sig_aligned.shift(1).fillna(0)

    if n_months is not None:
        rets = rets.iloc[-n_months:]
        sig_next = sig_next.reindex(rets.index).fillna(0)

    W_list = []
    for t, row in sig_next.iterrows():
        longs = (row > 0)
        shorts = (row < 0)
        n_long = int(longs.sum())
        n_short = int(shorts.sum())
        n_active = n_long + n_short
        if n_active == 0 or (n_active < cfg.min_assets_long):
            w = np.zeros_like(row.values, dtype=float)
        else:
            # assign equal magnitude to every active signal such that sum abs = 1
            mag = 1.0 / float(n_active)
            w = np.zeros_like(row.values, dtype=float)
            w[longs.values] = mag
            w[shorts.values] = -mag
        W_list.append(pd.Series(w, index=sig_next.columns, name=t))
    W = pd.DataFrame(W_list).astype(float)

    W_prev = W.shift(1).fillna(0.0)
    turnover = (W - W_prev).abs().sum(axis=1)
    tc_rate = cfg.tc_bps_roundtrip / 10000.0
    tc = turnover * tc_rate

    gross = (W * rets).sum(axis=1)
    abs_sum = W.abs().sum(axis=1)
    cash_weight = 1.0 - abs_sum.clip(0.0, 1.0)
    monthly_cash_yield = (1.0 + cfg.cash_rate_annual) ** (1.0 / 12.0) - 1.0
    gross_with_cash = gross + cash_weight * monthly_cash_yield
    net = (gross_with_cash - tc).dropna()

    equity = start_equity * (1.0 + net).cumprod()

    ann = 12.0
    if len(net) > 0:
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (ann / len(net)) - 1.0
    else:
        cagr = np.nan
    vol_ann = net.std(ddof=1) * np.sqrt(ann) if len(net) > 1 else np.nan
    sharpe = (cagr - cfg.cash_rate_annual) / vol_ann if (vol_ann and vol_ann > 0) else np.nan
    mdd = (equity / equity.cummax() - 1.0).min() if len(equity) else np.nan

    metrics = {
        "CAGR": float(cagr) if np.isfinite(cagr) else np.nan,
        "Vol_Ann": float(vol_ann) if np.isfinite(vol_ann) else np.nan,
        "Sharpe_ex_cash": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "MaxDD": float(mdd) if np.isfinite(mdd) else np.nan,
        "TotalGain": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else np.nan,
    }

    W = W.loc[net.index]
    turnover = turnover.loc[net.index]

    return BacktestResult(
        weights=W,
        returns_net=net.rename("net_return"),
        equity=equity.rename("equity"),
        turnover=turnover.rename("turnover"),
        metrics=metrics,
    )


# ---------------------------------------------------------------------
# Convenience: end-to-end from OOS forecasts
# ---------------------------------------------------------------------
def backtest_from_oos(
    prices: pd.DataFrame,
    oos_forecast_mean: np.ndarray,
    oos_index: pd.DatetimeIndex,
    asset_names: list[str],
    cfg: Optional[BacktestConfig] = None,
    n_months: Optional[int] = 36,
    threshold: float = 0.0,
    start_equity: float = 1.0
) -> BacktestResult:
    """
    Convenience wrapper:
    (prices + OOS point forecasts) → (signals) → long-only backtest.

    Parameters
    ----------
    prices : DataFrame
        Price levels (daily or monthly), columns = assets.
    oos_forecast_mean : ndarray (T_test x N)
        OOS point forecasts per month for each asset.
    oos_index : DatetimeIndex
        Month-end timestamps for OOS rows.
    asset_names : list[str]
        Asset column names matching the forecast order.
    cfg : BacktestConfig
        Backtest configuration.
    n_months : Optional[int]
        If set, restrict to last n_months in the backtest.
    threshold : float
        Threshold for signals (forecast > threshold → long).
    start_equity : float
        Initial equity value for the curve.

    Returns
    -------
    BacktestResult
    """
    signals = prepare_signals_from_oos(
        oos_forecast_mean=oos_forecast_mean,
        oos_index=oos_index,
        asset_names=asset_names,
        threshold=threshold
    )
    return long_only_forecast_backtest(
        prices=prices,
        signals=signals,
        cfg=cfg,
        n_months=n_months,
        start_equity=start_equity
    )


# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------
def _compute_baseline_equity_from_prices(prices: pd.DataFrame,
                                         baseline_type: str = "equal",
                                         columns: Optional[list] = None,
                                         start_equity: float = 1.0) -> pd.Series:
    """Compute an equity series for a baseline.

    baseline_type: 'equal' -> equal-weight across columns in prices
    Returns monthly-indexed equity series aligned to prices' month-ends.
    """
    px = prices.copy()
    if px.index.inferred_type not in ("datetime64", "datetime64tz"):
        px.index = pd.to_datetime(px.index)
    # month-end
    if pd.infer_freq(px.index) not in ("ME", "BM"):
        px = px.resample("ME").last()
    rets = px.pct_change().dropna()
    if baseline_type == "equal":
        if columns is not None:
            cols = [c for c in columns if c in rets.columns]
            if len(cols) == 0:
                raise ValueError("No matching columns for equal-weight baseline.")
            rets_sel = rets[cols]
        else:
            rets_sel = rets
        w = np.ones(rets_sel.shape[1]) / float(rets_sel.shape[1])
        port_ret = (rets_sel * w).sum(axis=1)
    else:
        raise ValueError("Unknown baseline_type")
    equity = start_equity * (1.0 + port_ret).cumprod()
    equity.name = f"baseline_{baseline_type}"
    return equity


def plot_equity_curve(
    result: BacktestResult,
    long_short_result: Optional[BacktestResult] = None,
    benchmark_results: Optional[Dict[str, BacktestResult]] = None,
    title: str = "Long-only Forecast Portfolio – Equity",
    ylabel: str = "Value (EUR)",
    xlabel: str = "Time",
    save_path: Optional[str] = None,
    show: bool = True,
    prices_baseline: Optional[pd.DataFrame] = None,
    plot_equal_weight: bool = True,
    start_equity: float = 1.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot equity curve from BacktestResult. Saves PNG if save_path provided.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    result.equity.reindex(result.equity.index).plot(ax=ax, linewidth=2, label="Long-Only")

    if long_short_result is not None:
        try:
            long_short_result.equity.reindex(result.equity.index).plot(ax=ax, linewidth=2, label="Long-Short")
        except Exception:
            # best-effort; don't fail plotting if indices differ
            long_short_result.equity.plot(ax=ax, linewidth=2, label="Long-Short")

    if benchmark_results is not None:
        for name, benchmark in benchmark_results.items():
            try:
                benchmark.equity.reindex(result.equity.index).plot(ax=ax, linewidth=1.8, linestyle="--", label=name)
            except Exception:
                benchmark.equity.plot(ax=ax, linewidth=1.8, linestyle="--", label=name)

    # Optional: plot equal-weight baseline
    if prices_baseline is not None and plot_equal_weight:
        try:
            e_eq = _compute_baseline_equity_from_prices(prices_baseline, baseline_type="equal", start_equity=start_equity, columns=result.weights.columns.tolist())
            # align to strategy index and rebase to same starting value at first common date
            e_eq = e_eq.reindex(result.equity.index)
            common_idx = result.equity.index.intersection(e_eq.index)
            if not common_idx.empty:
                start = common_idx[0]
                # rebase both series to 1.0 at start
                # strategy already starts at 1.0 in result.equity; rescale baseline to same starting point
                if e_eq.loc[start] > 0:
                    e_eq = e_eq / float(e_eq.loc[start]) * float(result.equity.loc[start])
                e_eq = e_eq.dropna()
                if not e_eq.empty:
                    e_eq.plot(ax=ax, linestyle="--", linewidth=1.6, label="Equal-Weight Baseline")
        except Exception:
            pass

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    ax.legend()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax

# ---------------------------------------------------------------------
# Random-Buy&Hold-Vergleich (Histogramm der Endwerte)
# ---------------------------------------------------------------------
def compare_with_random_portfolios(
    prices: pd.DataFrame,
    result: BacktestResult,
    long_short_result: Optional[BacktestResult] = None,
    n_portfolios: int = 1000,
    start_equity: float = 100.0,
    bins: int = 40,
    seed: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Simuliert n_portfolios zufällige Long-Only Buy&Hold-Portfolios (Dirichlet-Gewichte),
    berechnet deren Endwert und plottet ein Histogramm. Zeichnet die Endwerte der
    Strategien (long-only und optional long-short) als vertikale Linien ein.

    Rückgabe: (fig, ax, final_values_array)
    """
    rng = np.random.default_rng(seed)

    # 1) Monats-Returns der Preise
    px_m = _to_month_end_prices(prices)
    rets_m = _prices_to_simple_returns(px_m).dropna()

    # 2) Auf den Zeitraum des Backtests einschränken
    idx = result.equity.index
    rets_m = rets_m.reindex(idx).dropna()
    idx = rets_m.index  # evtl. erste Zeile fällt weg

    # 3) Random Long-Only Gewichte (Dirichlet)
    cols = [c for c in result.weights.columns if c in rets_m.columns]
    if len(cols) == 0:
        raise ValueError("Keine gemeinsame Asset-Schnittmenge zwischen returns und result.weights.")
    R = rets_m[cols].to_numpy(dtype=float)
    N = R.shape[1]

    final_vals = np.empty(n_portfolios, dtype=float)
    for i in range(n_portfolios):
        w = rng.dirichlet(alpha=np.ones(N))          # Long-only, sum=1
        port = (R * w).sum(axis=1)
        eq = start_equity * (1.0 + pd.Series(port, index=idx)).cumprod()
        final_vals[i] = float(eq.iloc[-1])

    # 4) Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(final_vals, bins=bins, alpha=0.8)
    ax.set_xlabel("Final Portfolio Value ($)")
    ax.set_ylabel("Count")
    if title is None:
        title = f"Final Value Distribution: {n_portfolios:,} Random Buy-and-Hold Portfolios"
    ax.set_title(title)

    # 5) Strategielinien
    # robustes Rebasen: auf gemeinsamen Startwert 'start_equity'
    def _final_from_equity(eq: pd.Series) -> float:
        eq = eq.reindex(idx).dropna()
        if len(eq) == 0:
            return np.nan
        ratio = float(eq.iloc[-1] / eq.iloc[0])
        return start_equity * ratio

    strat_final = _final_from_equity(result.equity)
    ax.axvline(strat_final, color="red", linestyle="--",
               label=f"Strategy final (${strat_final:.2f})")

    if long_short_result is not None:
        ls_final = _final_from_equity(long_short_result.equity)
        if np.isfinite(ls_final):
            ax.axvline(ls_final, color="black", linestyle="-.",
                       label=f"Long-Short final (${ls_final:.2f})")

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax, final_vals
