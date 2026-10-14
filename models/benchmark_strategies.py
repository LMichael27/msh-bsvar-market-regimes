"""Benchmark strategies for portfolio comparison.

This module builds a rolling VAR benchmark backtest that returns the same
BacktestResult object used by the main long-only portfolio backtest.

Available strategies:
- VAR-based long-only strategy: fits a rolling VAR on monthly returns and
    allocates weights proportional to positive one-step-ahead forecasts.
"""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from .long_only_forecast_backtest import (
    BacktestConfig,
    BacktestResult,
    _prices_to_simple_returns,
    _to_month_end_prices,
)


def _monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices_m = _to_month_end_prices(prices)
    return _prices_to_simple_returns(prices_m).dropna()


def make_var_forecasts_from_prices(
    prices: pd.DataFrame,
    lookback: int = 60,
    maxlags: int = 3,
    min_train_obs: int = 24,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Create month-end VAR forecasts.

    The signal at month-end t uses returns observed up to t and predicts t+1.
    Entries are raw one-step-ahead forecasts used to build long-only weights.
    """
    rets = _monthly_returns(prices)
    forecasts = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns, dtype=float)

    if rets.shape[1] == 0:
        return forecasts

    for i in range(len(rets)):
        start = max(0, i + 1 - lookback)
        train = rets.iloc[start : i + 1].replace([np.inf, -np.inf], np.nan).dropna()
        if len(train) < max(min_train_obs, 2):
            continue

        try:
            if train.shape[1] == 1:
                forecast = train.iloc[-1].to_numpy(dtype=float)
            else:
                lag_order = min(maxlags, len(train) - 1)
                if lag_order < 1:
                    continue
                fitted = VAR(train).fit(maxlags=lag_order, trend="c")
                forecast = fitted.forecast(train.to_numpy()[-fitted.k_ar :], steps=1)[0]
            forecasts.iloc[i] = np.asarray(forecast, dtype=float)
        except Exception:
            continue

    return forecasts


def var_strategy_backtest(
    prices: pd.DataFrame,
    cfg: Optional[BacktestConfig] = None,
    n_months: Optional[int] = 36,
    start_equity: float = 1.0,
    lookback: int = 60,
    maxlags: int = 3,
    min_train_obs: int = 24,
    threshold: float = 0.0,
) -> BacktestResult:
    """Run a long-only VAR benchmark backtest with forecast-weighted allocations."""
    forecasts = make_var_forecasts_from_prices(
        prices=prices,
        lookback=lookback,
        maxlags=maxlags,
        min_train_obs=min_train_obs,
        threshold=threshold,
    )

    if cfg is None:
        cfg = BacktestConfig()

    prices_m = _to_month_end_prices(prices)
    rets = _prices_to_simple_returns(prices_m).dropna()
    aligned = forecasts.reindex(rets.index).astype(float)

    if n_months is not None:
        rets = rets.iloc[-n_months:]
        aligned = aligned.reindex(rets.index)

    weights = []
    for t, row in aligned.iterrows():
        positive = row.replace([np.inf, -np.inf], np.nan).fillna(0.0) - threshold
        positive = positive.clip(lower=0.0)
        total = float(positive.sum())
        if total > 0:
            w = positive / total
        else:
            w = pd.Series(0.0, index=aligned.columns)
        weights.append(pd.Series(w, index=aligned.columns, name=t))

    W = pd.DataFrame(weights).astype(float)
    W_prev = W.shift(1).fillna(0.0)
    turnover = (W - W_prev).abs().sum(axis=1)
    tc_rate = cfg.tc_bps_roundtrip / 10000.0
    tc = turnover * tc_rate

    gross = (W * rets).sum(axis=1)
    cash_weight = 1.0 - W.sum(axis=1).clip(0.0, 1.0)
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


def build_benchmark_results(
    prices: pd.DataFrame,
    cfg: Optional[BacktestConfig] = None,
    n_months: Optional[int] = 36,
    start_equity: float = 1.0,
    var_lookback: int = 60,
    var_maxlags: int = 3,
    var_min_train_obs: int = 24,
    var_threshold: float = 0.0,
) -> Dict[str, BacktestResult]:
    """Build the VAR benchmark backtest for plotting convenience."""
    var_result = var_strategy_backtest(
        prices=prices,
        cfg=cfg,
        n_months=n_months,
        start_equity=start_equity,
        lookback=var_lookback,
        maxlags=var_maxlags,
        min_train_obs=var_min_train_obs,
        threshold=var_threshold,
    )
    return {"VAR": var_result}
