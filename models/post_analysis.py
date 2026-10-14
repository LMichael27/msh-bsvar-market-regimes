from __future__ import annotations
import os
from typing import List, Tuple, Dict, Sequence, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models.plot_irf import compute_irf  # erwartet Signatur compute_irf(alpha, A0, p, horizon)

def _zscore(x: np.ndarray, axis: int = 0, eps: float = 1e-9) -> np.ndarray:
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / (sd + eps)

def _ensure_dir(path: Optional[str]) -> None:
    if path is not None:
        os.makedirs(path, exist_ok=True)

def _assert_shapes_for_alpha(alpha_draws: np.ndarray, p: int, asset_names: Sequence[str], Y_last: np.ndarray) -> Tuple[int, int, int, int]:
    assert alpha_draws.ndim == 4, f"alpha_draws must have shape (K,N,M,S), got {alpha_draws.shape}"
    K, N, M, S = alpha_draws.shape
    assert K == p * N + 1, f"alpha has K={K}, expected {p * N + 1} for p={p}, N={N}"
    assert Y_last.shape == (p, N), f"Y_last has shape {Y_last.shape}, expected ({p}, {N})"
    assert N == len(asset_names), "Mismatch between alpha_draws N and asset_names length"
    return K, N, M, S

# ============================================================
# FEVD, Forecasts
# ============================================================

def fevd(alpha: np.ndarray, A0: np.ndarray, p: int, horizon: int) -> np.ndarray:
    K_vars = A0.shape[0]
    irfs = compute_irf(alpha, A0, p, horizon)  # (N, N, H+1)
    out = np.zeros((K_vars, K_vars, horizon + 1))
    for h in range(horizon + 1):
        resp = irfs[:, :, h]
        num = resp ** 2
        denom = np.sum(num, axis=1, keepdims=True)
        denom[denom == 0] = 1e-12
        out[:, :, h] = num / denom
    return out

def forecast(alpha: np.ndarray, Y_last: np.ndarray, p: int, steps: int) -> np.ndarray:
    K_vars = Y_last.shape[1]
    intercept = alpha[0, :]
    A_mats = [alpha[1 + i*K_vars: 1 + (i+1)*K_vars, :].T for i in range(p)]
    y_past = Y_last[::-1].copy()  # jüngste zuerst
    forecasts = []
    for _ in range(steps):
        y_next = intercept.copy()
        for lag in range(p):
            y_next += y_past[lag] @ A_mats[lag].T
        forecasts.append(y_next)
        y_past = np.vstack([y_next.reshape(1, -1), y_past[:-1]])
    return np.array(forecasts)

# ------------------------------------------------------------

def _row_stochastic(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    rs = P.sum(axis=1, keepdims=True)
    bad = rs <= eps
    if np.any(bad):
        P[bad[:, 0], :] = 1.0 / P.shape[1]
        rs = P.sum(axis=1, keepdims=True)
    return P / rs

def _stochastic_vec(pi: np.ndarray, M: int, eps: float = 1e-12) -> np.ndarray:
    pi = np.nan_to_num(np.asarray(pi, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    pi = np.clip(pi, 0.0, None)
    s = pi.sum()
    if s <= eps:
        return np.ones(M) / M
    return pi / s

def _ergodic_prob_power(P: np.ndarray, iters: int = 10_000, tol: float = 1e-12) -> np.ndarray:
    M = P.shape[0]
    x = np.ones(M) / M
    for _ in range(iters):
        x_new = x @ P
        if np.max(np.abs(x_new - x)) < tol:
            break
        x = x_new
    return _stochastic_vec(x, M)

def _var_blocks_from_alpha(alpha: np.ndarray, p: int) -> list:
    N = alpha.shape[1]
    blocks = []
    for L in range(p):
        rows = slice(1 + L * N, 1 + (L + 1) * N)
        blocks.append(alpha[rows, :])
    return blocks

def _one_step(alpha: np.ndarray, y_lags: np.ndarray, p: int) -> np.ndarray:
    c = alpha[0, :]
    Bs = _var_blocks_from_alpha(alpha, p)
    y = c.copy()
    for L in range(p):
        y += y_lags[L] @ Bs[L]
    return y

from typing import Dict, Union, Optional, Tuple
import numpy as np

def forecast_with_ci_mc(posteriors: Dict[str, np.ndarray],
                        Y_last: np.ndarray,           # (p,N)
                        p: int,
                        steps: int = 12,
                        ci_level: float = 0.68,
                        regime: Union[int, str] = "mix",
                        initial_probs: Optional[np.ndarray] = None,
                        n_sims: int = 300,
                        seed: Optional[int] = None,
                        scale: str = "log",           # "log" | "frac" | "pct"
                        annualize: bool = False,
                        clip_frac: Optional[float] = None,
                        # Robustheit:
                        rho_target: float = 0.995,    # Ziel-Spektralradius ( < 1 )
                        max_iter_proj: int = 10,      # max. Iterationen für Stabilitätsprojektion
                        max_abs_log: float = 6.0,     # harte Kappe auf Log-Returns (~±400%/Monat)
                        lam_cap: float = 1e4          # Obergrenze für lambda_omega/EWs
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte-Carlo-Forecast für ein MSH-BSVAR mit:
      - robuste Zeilen-Normalisierung der Übergangsmatrix P (Softmax-Fallback bei Logits)
      - stabilisierte VAR-Dynamik (Companion-ρ < rho_target)
      - SPD-Kovarianzen (Eigenvalue-Clipping)
      - flexibler Initialverteilung (letzte S_t-Prob., ergodisch aus P oder uniform)
      - Pfad-Simulation in LOG; optional Ausgabe in frac/pct (+ Annualisierung)

    Rückgabe: (mean, lower, upper) mit Form (steps, N)
    """

    rng = np.random.default_rng(seed)

    alphas = posteriors["alpha"]          # (K,N,M,S)
    A_s    = posteriors["A"]              # (M,N,N,S)
    lamw   = posteriors["lambda_omega"]   # meist (N,M,S)
    P_s    = posteriors.get("PR_TR", None)
    S_t    = posteriors.get("S_t",  None)

    K, N, M, S = alphas.shape
    assert K == 1 + p * N, "alpha hat nicht die erwartete Form (K != 1 + p*N)."
    y_lags0 = Y_last[::-1].copy()  # (p,N)

    # ---------- Helpers ----------
    def _softmax_rows(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        X = X - np.max(X, axis=1, keepdims=True)
        np.exp(X, out=X)
        s = X.sum(axis=1, keepdims=True)
        s[s <= 0] = 1.0
        return X / s

    def _row_stochastic(P: np.ndarray) -> np.ndarray:
        """Macht aus P eine gültige Übergangsmatrix:
           - NaN/Inf -> 0
           - Wenn Werte außerhalb [0,1] oder Zeilensummen weit von 1: Softmax
           - Sonst: clamp >=0 und renorm
           - Leere/ungültige Zeilen -> uniform
        """
        P = np.asarray(P, float)
        if P.ndim != 2 or P.shape[0] != M or P.shape[1] != M:
            raise ValueError(f"P hat unerwartete Form {P.shape}, erwartet (M,M) mit M={M}.")

        P = np.where(np.isfinite(P), P, 0.0)

        row_sum = P.sum(axis=1, keepdims=True)
        out_of_range = (P < 0).any() or (P > 1).any()
        far_from_one = not np.allclose(row_sum, 1.0, atol=1e-6)

        if out_of_range or far_from_one:
            # Wahrscheinlich Logits oder ungeeicht -> Softmax
            P = _softmax_rows(P)
        else:
            # Kleiner numerischer Drift -> clamp + renorm
            P = np.clip(P, 0.0, None)
            row_sum = P.sum(axis=1, keepdims=True)
            row_sum[row_sum <= 0] = 1.0
            P = P / row_sum

        # Fallback: leere/ungültige Zeilen -> gleichverteilt
        bad = ~np.isfinite(P.sum(axis=1)) | (P.sum(axis=1) <= 0)
        if np.any(bad):
            P[bad, :] = 1.0 / M

        # Endgültige Normierung
        P = np.clip(P, 0.0, None)
        row_sum = P.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0] = 1.0
        P = P / row_sum
        return P

    def _extract_P_for_draw(P_s, s: int) -> np.ndarray:
        """Extrahiert für Draw s eine (M,M)-Matrix aus verschiedenen PR_TR-Layouts."""
        if P_s is None:
            return np.eye(M)
        P_arr = np.asarray(P_s)
        # Häufige Fälle:
        if P_arr.ndim == 3 and P_arr.shape[0] == M and P_arr.shape[1] == M:
            idx = min(s, P_arr.shape[2] - 1)
            return P_arr[:, :, idx]
        if P_arr.ndim == 3 and P_arr.shape[1] == M and P_arr.shape[2] == M:
            idx = min(s, P_arr.shape[0] - 1)
            return P_arr[idx, :, :]
        if P_arr.ndim == 2 and P_arr.shape == (M, M):
            return P_arr
        # Fallback: flachbar?
        size = P_arr.size
        if size % (M * M) == 0:
            total_S = size // (M * M)
            flat = P_arr.reshape(total_S, M, M)
            idx = min(s, total_S - 1)
            return flat[idx]
        # Letzter Fallback
        return np.eye(M)

    def _ergodic_prob(P: np.ndarray, k: int = 512) -> np.ndarray:
        P = _row_stochastic(P)
        v = np.ones(M) / M
        for _ in range(k):
            v = v @ P
        v = np.maximum(v, 0)
        s = v.sum()
        return v / s if s > 0 else np.ones(M) / M

    def _stochastic_vec(pi: Optional[np.ndarray]) -> np.ndarray:
        if pi is None:
            return np.ones(M) / M
        pi = np.asarray(pi, float)
        if pi.ndim != 1 or pi.size != M:
            return np.ones(M) / M
        pi = np.where(np.isfinite(pi), np.clip(pi, 0.0, None), 0.0)
        s = pi.sum()
        return pi / s if s > 0 else np.ones(M) / M

    def _Bs_from_alpha(a: np.ndarray):
        # a: (K,N) -> Liste B_i (N,N)
        return [a[1 + i*N: 1 + (i+1)*N, :].T for i in range(p)]

    def _alpha_from_Bs(a: np.ndarray, Bs):
        a_new = a.copy()
        for i, B in enumerate(Bs):
            a_new[1 + i*N: 1 + (i+1)*N, :] = B.T
        return a_new

    def _companion_from_Bs(Bs):
        if p == 0:
            return np.zeros((N, N))
        if p == 1:
            return Bs[0]
        top = np.hstack(Bs)
        bottom = np.eye(N*(p-1), N*p, k=N)
        return np.vstack([top, bottom])

    def _spectral_radius(Bs) -> float:
        C = _companion_from_Bs(Bs)
        vals = np.linalg.eigvals(C)
        return float(np.max(np.abs(vals)))

    def _stabilize_alpha(a: np.ndarray, rho_star: float, max_iter=10) -> np.ndarray:
        """Skaliert alle Lag-Matrizen B_i -> s * B_i (gleiches s), iterativ,
           bis ρ(companion) < rho_star oder max_iter erreicht.
        """
        a_work = a.copy()
        Bs = _Bs_from_alpha(a_work)
        if p == 0:
            return a_work
        for _ in range(max_iter):
            rho = _spectral_radius(Bs)
            if not np.isfinite(rho):
                s = 0.9
            elif rho < rho_star:
                break
            else:
                s = min(0.98 * rho_star / max(rho, 1e-8), 0.999)
            Bs = [s * B for B in Bs]
        return _alpha_from_Bs(a_work, Bs)

    def _spd_from(Ainv: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """Sigma = A^{-1} diag(lam) A^{-T}, numerisch SPD geclippt."""
        lam = np.asarray(lam, float)
        lam = np.clip(lam, 1e-12, lam_cap)
        Sigma = Ainv @ np.diag(lam) @ Ainv.T
        Sigma = 0.5 * (Sigma + Sigma.T)
        w, V = np.linalg.eigh(Sigma)
        w = np.clip(w, 1e-12, lam_cap)
        return (V * w) @ V.T

    def _lam_for_regime(lamw: np.ndarray, m: int, s: int) -> np.ndarray:
        """Robuste Auswahl der diag-Lambdas je Regime m und Draw s."""
        LW = np.asarray(lamw)
        if LW.ndim == 3 and LW.shape[1] == M:
            return LW[:, m, s]
        if LW.ndim == 3 and LW.shape[2] == M:
            return LW[:, s, m]
        if LW.ndim == 2 and LW.shape[1] == M:
            return LW[:, m]
        # Fallback: ursprüngliches Schema (falls Basis * Regime-Skalen gewünscht)
        base = LW[:, 0, s] if LW.ndim >= 3 else LW[:, 0]
        try:
            scale = LW[:, m, s] if LW.ndim >= 3 else LW[:, m]
            return base if m == 0 else base * scale
        except Exception:
            return base

    # ---------- Startverteilung für Regime ----------
    if regime == "mix":
        if initial_probs is None:
            pi0 = None
            if S_t is not None:
                St = np.asarray(S_t)
                # akzeptiert (M,T) oder (T,M)
                if St.ndim == 2:
                    if St.shape[0] == M:
                        pi0 = St[:, -1]
                    elif St.shape[1] == M:
                        pi0 = St[-1, :]
            if pi0 is not None:
                initial_probs = _stochastic_vec(pi0)
            else:
                # Falls P vorhanden: ergodische Verteilung
                P0 = _extract_P_for_draw(P_s, s=0) if P_s is not None else np.eye(M)
                initial_probs = _ergodic_prob(P0)
        else:
            initial_probs = _stochastic_vec(initial_probs)
    else:
        initial_probs = np.eye(M)[int(regime)]

    # ---------- Simulation ----------
    paths = np.zeros((S * n_sims, steps, N))
    row = 0

    for s in range(S):
        # Übergangsmatrix für Draw s
        P_raw = _extract_P_for_draw(P_s, s) if P_s is not None else np.eye(M)
        P = _row_stochastic(P_raw)

        # reduzierte-Form-Kovarianzen je Regime (robust)
        Sigmas = []
        for m in range(M):
            A_m = A_s[m, :, :, s]
            Ainv = np.linalg.inv(A_m)
            lam = _lam_for_regime(lamw, m, s)
            Sigma = _spd_from(Ainv, lam)
            Sigmas.append(Sigma)

        # Stabilitätsprojektion der alphas (je Regime)
        alphas_stable = np.empty_like(alphas[:, :, :, s])
        for m in range(M):
            a_m = alphas[:, :, m, s]
            a_m = _stabilize_alpha(a_m, rho_target, max_iter=max_iter_proj)
            alphas_stable[:, :, m] = a_m

        for _ in range(n_sims):
            # Regimepfad
            states = np.empty(steps, dtype=int)
            states[0] = rng.choice(M, p=initial_probs)
            for h in range(1, steps):
                # Sicherstellen, dass p gültig ist (numerische Guardrails)
                row_p = P[states[h-1], :]
                row_p = np.where(np.isfinite(row_p), np.clip(row_p, 0.0, None), 0.0)
                ssum = row_p.sum()
                if not np.isfinite(ssum) or ssum <= 0:
                    row_p = np.ones(M) / M
                else:
                    row_p = row_p / ssum
                states[h] = rng.choice(M, p=row_p)

            # Forecastpfad in LOG
            y_lags = y_lags0.copy()
            for h in range(steps):
                m = states[h]
                a = alphas_stable[:, :, m]
                # Erwartungswert bedingt auf Lags
                const = a[0, :].copy()
                Bs = [a[1 + i*N:1 + (i+1)*N, :].T for i in range(p)]
                mean_next = const
                for i in range(p):
                    mean_next = mean_next + Bs[i] @ y_lags[i, :].T
                # Schock
                shock = rng.multivariate_normal(np.zeros(N), Sigmas[m])
                y_next_log = mean_next + shock
                y_next_log = np.clip(y_next_log, -max_abs_log, max_abs_log)
                paths[row, h, :] = y_next_log
                # Lags updaten
                if p > 0:
                    y_lags = np.vstack([y_next_log.reshape(1, -1), y_lags[:-1]])
            row += 1

    # ---------- Einheiten-Umrechnung ----------
    if scale not in {"log", "frac", "pct"}:
        raise ValueError("scale must be 'log', 'frac', or 'pct'.")

    if scale in {"frac", "pct"}:
        if clip_frac is not None and clip_frac > 0:
            L = float(np.log1p(abs(clip_frac)))  # z.B. 0.3 -> log(1.3)
            paths = np.expm1(np.clip(paths, -L, L))
            paths = np.clip(paths, -abs(clip_frac), abs(clip_frac))
        else:
            paths = np.expm1(np.clip(paths, -10.0, 10.0))
        if annualize:
            paths = (1.0 + paths) ** 12 - 1.0
        if scale == "pct":
            paths = paths * 100.0

    # nur finite Pfade
    finite = np.isfinite(paths).all(axis=(1, 2))
    if not np.any(finite):
        # Letzter Fallback: falls alles unendlich/NaN, nimm Nullen (konservativ)
        paths = np.zeros_like(paths)
        finite = np.ones(paths.shape[0], dtype=bool)
    paths = paths[finite]

    q_lo = (1 - ci_level) / 2 * 100.0
    q_hi = (1 + ci_level) / 2 * 100.0
    mean  = paths.mean(axis=0)
    lower = np.percentile(paths, q_lo, axis=0, method="linear")
    upper = np.percentile(paths, q_hi, axis=0, method="linear")
    return mean, lower, upper
# ============================================================
# Strukturmatrix-Plot
# ============================================================

def structural_matrix(results: Dict,
                      asset_names: Optional[Sequence[str]] = None,
                      save_path: Optional[str] = None,
                      show: bool = False,
                      regime_names: Optional[Sequence[str]] = None,
                      normalize: bool = False,
                      norm_method: str = "maxabs",
                      per_draw: bool = False,
                      vmin: Optional[float] = None,
                      vmax: Optional[float] = None) -> None:
    """Plot (posterior mean) structural matrices A per regime.

    Added parameters allow optional scaling of entries into [-1, 1]. See
    inline docstring added in patch for detailed description.
    """
    A_samples = results["posteriors"]["A"]  # (M, N, N, S)
    _ensure_dir(save_path)

    M = A_samples.shape[0]
    N = A_samples.shape[1]
    assets = list(asset_names) if asset_names is not None else [f"Var {i+1}" for i in range(N)]
    regimes = list(regime_names) if regime_names is not None else [f"Regime {m+1}" for m in range(M)]

    for m in range(M):
        A_draws = A_samples[m]  # (N, N, S)

        # Normalisation helper
        def _do_norm(mat: np.ndarray) -> np.ndarray:
            if norm_method == "maxabs":
                denom = np.max(np.abs(mat))
                return mat / denom if denom > 0 else mat
            elif norm_method == "row":
                denom = np.max(np.abs(mat), axis=1, keepdims=True)
                denom[denom == 0] = 1.0
                return mat / denom
            elif norm_method == "col":
                denom = np.max(np.abs(mat), axis=0, keepdims=True)
                denom[denom == 0] = 1.0
                return mat / denom
            else:
                raise ValueError(f"Unknown norm_method '{norm_method}'")

        if normalize and per_draw:
            A_proc = np.empty_like(A_draws)
            for s in range(A_draws.shape[2]):
                A_proc[:, :, s] = _do_norm(A_draws[:, :, s])
            A_mean = A_proc.mean(axis=2)
        else:
            A_mean = A_draws.mean(axis=2)
            if normalize:
                A_mean = _do_norm(A_mean)

        annot_fmt = ".2f"
        _vmin = vmin if vmin is not None else (-1 if normalize else None)
        _vmax = vmax if vmax is not None else (1 if normalize else None)

        plt.figure(figsize=(6, 5))
        sns.heatmap(A_mean, annot=True, fmt=annot_fmt, center=0, cmap="coolwarm",
                    xticklabels=assets, yticklabels=assets, vmin=_vmin, vmax=_vmax)
        title_extra = " (normalisiert)" if normalize else ""
        plt.title(f"Posterior Mean of the Structural matrix A – {regimes[m]}")
        plt.xlabel("Shocks (columns)")
        plt.ylabel("Variables (rows)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        # Add note about scale factor if maxabs
        if normalize and norm_method == "maxabs" and not per_draw:
            maxabs_val = np.max(np.abs(A_draws.mean(axis=2)))
            if maxabs_val > 0:
                plt.text(0.0, -0.4, f"", transform=plt.gca().transAxes,
                         fontsize=8)
        plt.tight_layout()
        if save_path is not None:
            fname = os.path.join(save_path, f"structural_matrix_{regimes[m].replace(' ', '_').lower()}.png")
            plt.savefig(fname, dpi=150)
        if show:
            plt.show()
        else:
            plt.close()

# ============================================================
# Bedingte Mittelwerte (modell-basiert)
# ============================================================

def summarize_conditional_means(result: Dict,
                                Y_last: np.ndarray,
                                p: int,
                                asset_names: Sequence[str],
                                save_path: Optional[str] = None,
                                plot: bool = False,
                                regime_names: Optional[Sequence[str]] = None,
                                return_matrix: bool = True) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    ROHE bedingte log-Mittelwerte je Regime: E[r_{t+1} | S_t=m, Y_t..Y_{t-p+1}] mit heutigem Lag-Zustand.
    """
    alpha_draws = result["posteriors"]["alpha"]  # (K, N, M, S)
    K, N, M, S = _assert_shapes_for_alpha(alpha_draws, p, asset_names, Y_last)

    Y_vec = Y_last[::-1].reshape(p * N, 1)
    Y_vec = np.vstack([Y_vec, np.array([[1.0]])])  # +Intercept

    mu_draws = np.empty((M, N, S))
    for s in range(S):
        for m in range(M):
            alpha_s_m = alpha_draws[:, :, m, s]  # (K, N)
            mu_draws[m, :, s] = (alpha_s_m.T @ Y_vec).ravel()

    posterior_mean = mu_draws.mean(axis=2)
    lower_ci = np.percentile(mu_draws, 2.5, axis=2)
    upper_ci = np.percentile(mu_draws, 97.5, axis=2)
    prob_positive = (mu_draws > 0).mean(axis=2)

    regimes = list(regime_names) if regime_names is not None else [f"Regime {m+1}" for m in range(M)]
    summary_df = pd.DataFrame({
        "Posterior Mean": posterior_mean.flatten(),
        "CI Lower (2.5%)": lower_ci.flatten(),
        "CI Upper (97.5%)": upper_ci.flatten(),
        "P(Mean > 0)": prob_positive.flatten(),
    }, index=pd.MultiIndex.from_product([regimes, list(asset_names)], names=["Regime", "Asset"]))

    if save_path is not None:
        summary_df.to_excel(save_path)

    if plot:
        mu_std = _zscore(posterior_mean, axis=0)
        plt.figure(figsize=(10, 6))
        sns.heatmap(mu_std, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                    linewidths=0.5, cbar_kws={"label": "Standardized means (columns)"},
                    xticklabels=asset_names, yticklabels=regimes)
        plt.title("Standardized implicit means per asset (columns) and regime")
        plt.ylabel("Regime")
        plt.xlabel("Asset class")
        plt.tight_layout()
        plt.show()

    return summary_df, (posterior_mean if return_matrix else None)

def conditional_means_heatmap(mu_raw: np.ndarray,
                              asset_names: Sequence[str],
                              regime_names: Optional[Sequence[str]] = None,
                              standardize: bool = True,
                              save_path: Optional[str] = None) -> None:
    M, N = mu_raw.shape
    regimes = list(regime_names) if regime_names is not None else [f"Regime {m+1}" for m in range(M)]
    data = _zscore(mu_raw, axis=0) if standardize else mu_raw

    _ensure_dir(os.path.dirname(save_path) if save_path else None)
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                linewidths=0.5,
                cbar_kws={"label": "Standardised Mean" if standardize else "Log-Means"},
                xticklabels=asset_names, yticklabels=regimes)
    plt.title("Standardized implicit means per asset (columns) and regime" if standardize
              else "Implicit Log-Means per Asset Class and Regime")
    plt.ylabel("Regime")
    plt.xlabel("Asset class")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

# ============================================================
# SAMPLE-basierte Regime-Mittel + Unconditional Mean (NEU)
# ============================================================

def _one_hot(states: np.ndarray, M: int) -> np.ndarray:
    T = states.shape[0]
    W = np.zeros((T, M))
    W[np.arange(T), states.astype(int)] = 1.0
    return W

def _infer_weights(S_info: np.ndarray, M: int) -> np.ndarray:
    """
    Liefert Gewichte W (T,M) je Zeitpunkt t und Regime m.
    Unterstützt:
      - S_info shape (T,M) floats (smoothed probs)
      - S_info shape (T,) ints (harter Pfad)
      - S_info shape (T,S) ints (viele Pfade) -> mittlere One-Hot über S
    """
    S_info = np.asarray(S_info)
    if S_info.ndim == 2 and S_info.shape[1] == M and not np.issubdtype(S_info.dtype, np.integer):
        W = S_info.copy()
        W = np.clip(W, 0.0, None)
        row_sum = W.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return W / row_sum
    if S_info.ndim == 1:
        return _one_hot(S_info, M)
    if S_info.ndim == 2:  # (T,S) harte Pfade
        T, Sdraws = S_info.shape
        W = np.zeros((T, M))
        for m in range(M):
            W[:, m] = (S_info == m).mean(axis=1)
        return W
    raise ValueError("Unrecognized S_info shape for weights.")

def _align_last(R: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Richtet die Längen an den letzten Beobachtungen aus.
    Schneidet bei Bedarf die ältesten Zeilen ab (typischerweise p).
    """
    TR, TW = R.shape[0], W.shape[0]
    if TR == TW:
        return R, W
    if TR > TW:
        return R[TR - TW:, :], W
    else:
        return R, W[TW - TR:, :]

def sample_regime_means_table_and_plot(log_returns: np.ndarray,    # (T,N) log
                                       S_info: np.ndarray,         # (T,), (T,M) oder (T,S)
                                       M: int,
                                       asset_names: Sequence[str],
                                       regime_names: Sequence[str],
                                       annualize: bool = False,
                                       save_path: str = "Figures/plots/sample_regime_means.png",
                                       ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Berechnet sample-basierte Regime-Mittel je Asset (arithmische Monatsrenditen)
    und hängt den unconditional sample mean als letzte Zeile an.
    Längendifferenzen (z.B. T vs. T-p) werden automatisch durch front-trim behoben.
    """
    _ensure_dir(os.path.dirname(save_path) if save_path else None)

    R = np.exp(np.asarray(log_returns)) - 1.0           # (T,N) arithm. Monatsrenditen
    W = _infer_weights(S_info, M)                       # (T,M)
    R, W = _align_last(R, W)                            # Zeitachsen ausrichten
    T, N = R.shape

    pi_hat = W.mean(axis=0)                             # (M,)

    denom = W.sum(axis=0)[:, None]                      # (M,1)
    denom = np.clip(denom, 1e-12, None)
    mu_m = (W[:, :, None] * R[:, None, :]).sum(axis=0) / denom   # (M,N)

    overall = R.mean(axis=0)                            # (N,)
    mix = (pi_hat[:, None] * mu_m).sum(axis=0)          # (N,)

    # Darstellung (%)
    num = np.vstack([mu_m, overall[None, :]]) * 100.0
    labels = np.vectorize(lambda x: f"{x:.2f}%")(num)
    idx = list(regime_names) + ["Uncond. (Sample)"]
    df = pd.DataFrame(labels, index=idx, columns=list(asset_names))

    plt.figure(figsize=(10, 6))
    sns.heatmap(num, annot=labels, fmt="", cmap="coolwarm", center=0,
                xticklabels=asset_names, yticklabels=idx,
                cbar_kws={"label": "Returns (%)"})
    plt.title("Monthly average Returns per Regime (Sample)")
    plt.xlabel("Asset classes")
    plt.ylabel("Regime")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    return df, {"mu_m": mu_m, "pi_hat": pi_hat, "overall": overall, "mix": mix}

# ============================================================
# Conditional Means: log -> % und optional Unconditional-Referenz
# ============================================================

def interpret_mean_returns(posterior_mean_raw: np.ndarray,
                           asset_names: Sequence[str],
                           regime_names: Optional[Sequence[str]] = None,
                           annualize: bool = True,
                           save_path: Optional[str] = "Figures/plots/mean_returns_interpreted.png",
                           uncond_log_returns: Optional[np.ndarray] = None,
                           add_uncond: Optional[str] = None   # None | "row" | "col"
                           ) -> pd.DataFrame:
    M, N = posterior_mean_raw.shape
    regimes = list(regime_names or [f"Regime {m+1}" for m in range(M)])
    cols = list(asset_names)

    numeric = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            r_log = posterior_mean_raw[m, n]
            r_lin = np.exp(r_log) - 1.0
            if annualize:
                r_lin = (1.0 + r_lin) ** 12 - 1.0
            numeric[m, n] = r_lin * 100.0

    if uncond_log_returns is not None and add_uncond in ("row", "col"):
        R = np.asarray(uncond_log_returns)
        R_arith = np.exp(R) - 1.0
        if add_uncond == "row":
            u = R_arith.mean(axis=0)
            if annualize:
                u = (1.0 + u) ** 12 - 1.0
            u_pct = u * 100.0
            numeric = np.vstack([numeric, u_pct.reshape(1, -1)])
            regimes.append("Uncond. (Sample)")
        elif add_uncond == "col":
            col = numeric.mean(axis=1, keepdims=True)
            numeric = np.hstack([numeric, col])
            cols.append("Ø über Assets")

    labels = np.vectorize(lambda x: f"{x:.2f}%")(numeric)
    table = pd.DataFrame(labels, index=regimes, columns=cols)

    _ensure_dir(os.path.dirname(save_path) if save_path else None)
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric, annot=labels, fmt="", cmap="coolwarm", center=0,
                xticklabels=cols, yticklabels=regimes,
                cbar_kws={"label": "Returns (%)"})
    plt.title("Annualised expected Returns per Asset class and Regime" if annualize
              else "Monthly expected Returns per Asset class and Regime")
    plt.xlabel("Asset classes")
    plt.ylabel("Regime")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return table

# ============================================================
# Volatilitäten & Kombiplots & Signifikanz
# ============================================================

def average_volatility_per_regime(result: Dict,
                                  asset_names: Sequence[str],
                                  standardize: bool = True,
                                  plot: bool = True,
                                  regime_names: Optional[Sequence[str]] = None,
                                  save_path: Optional[str] = "Figures/plots/cond_var.png"
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Sigma_s = result["posteriors"]["Sigma"]  # (N, N, M, S)
    N, _, M, S = Sigma_s.shape

    avg_vars = np.zeros((M, N))
    for m in range(M):
        for i in range(N):
            avg_vars[m, i] = np.mean(Sigma_s[i, i, m, :])

    vol_df = pd.DataFrame(avg_vars, columns=asset_names,
                          index=regime_names or [f"Regime {m+1}" for m in range(M)])
    vol_std = vol_df.copy()
    if standardize:
        vol_std.iloc[:, :] = _zscore(vol_df.values, axis=0)

    if plot:
        _ensure_dir(os.path.dirname(save_path) if save_path else None)
        plt.figure(figsize=(10, 6))
        sns.heatmap(vol_std, annot=True, cmap="YlGnBu", center=0, fmt=".2f",
                    linewidths=0.5, cbar_kws={"label": "Standardized Volatility"})
        plt.title("Standardized Volatility per Asset and Regime")
        plt.ylabel("Regime")
        plt.xlabel("Asset classes")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    return vol_std, vol_df

def plot_mean_and_volatility(mu_raw_or_std: pd.DataFrame,
                             vol_raw_or_std: pd.DataFrame,
                             asset_names: Sequence[str],
                             regime_names: Optional[Sequence[str]] = None,
                             save_path: Optional[str] = "Figures/plots/mean_var_non_std.png") -> None:
    if regime_names is not None:
        mu_raw_or_std.index = regime_names
        vol_raw_or_std.index = regime_names

    _ensure_dir(os.path.dirname(save_path) if save_path else None)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    sns.heatmap(mu_raw_or_std, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "Implicit Means"}, ax=axes[0])
    axes[0].set_title("Implicit Means per Asset and Regime")
    axes[0].set_ylabel("Regime")

    sns.heatmap(vol_raw_or_std, annot=True, cmap="YlGnBu", center=0, fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "Volatility"}, ax=axes[1])
    axes[1].set_title("Average Volatility per Asset and Regime")
    axes[1].set_xlabel("Asset class")
    axes[1].set_ylabel("Regime")
    axes[1].set_xticklabels(asset_names, rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def full_mean_vol_visuals(result: Dict,
                          Y_last: np.ndarray,
                          p: int,
                          asset_names: Sequence[str],
                          regime_names: Optional[Sequence[str]] = None,
                          figures_dir: str = "Figures/plots") -> Dict[str, str]:
    _ensure_dir(figures_dir)
    _, mu_raw = summarize_conditional_means(result, Y_last, p, asset_names,
                                            plot=False, regime_names=regime_names, return_matrix=True)
    assert mu_raw is not None

    cond_mean_path = os.path.join(figures_dir, "cond_mean.png")
    conditional_means_heatmap(mu_raw, asset_names, regime_names, standardize=True, save_path=cond_mean_path)

    cond_var_path = os.path.join(figures_dir, "cond_var.png")
    average_volatility_per_regime(result, asset_names, standardize=True, plot=True,
                                  regime_names=regime_names, save_path=cond_var_path)

    mean_ret_path = os.path.join(figures_dir, "mean_returns_interpreted.png")
    interpret_mean_returns(mu_raw, asset_names, regime_names, annualize=False, save_path=mean_ret_path)

    return {"cond_mean": cond_mean_path, "cond_var": cond_var_path, "mean_returns": mean_ret_path}

def plot_mean_significance(prob_or_summary,
                           asset_names,
                           regime_names=None,
                           save_path="Figures/plots/mean_significance.png",
                           show_stars=True,
                           star_levels=(0.90, 0.95, 0.99)):
    if isinstance(prob_or_summary, pd.DataFrame) and "P(Mean > 0)" in prob_or_summary.columns:
        prob = prob_or_summary["P(Mean > 0)"].unstack(level="Asset").values
        if regime_names is None:
            regime_names = list(prob_or_summary.index.levels[0])
    else:
        prob = np.asarray(prob_or_summary)
        if regime_names is None:
            regime_names = [f"Regime {i+1}" for i in range(prob.shape[0])]

    M, N = prob.shape
    labels = np.empty((M, N), dtype=object)

    if show_stars:
        cert = np.maximum(prob, 1.0 - prob)
        star_map = np.where(cert >= star_levels[2], "*",
                    np.where(cert >= star_levels[1], "**",
                    np.where(cert >= star_levels[0], "*", "")))
        for i in range(M):
            for j in range(N):
                labels[i, j] = f"{prob[i, j]*100:.0f}%{star_map[i, j]}"
    else:
        for i in range(M):
            for j in range(N):
                labels[i, j] = f"{prob[i, j]*100:.0f}%"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(prob, annot=labels, fmt="", cmap="coolwarm", center=0.5,
                xticklabels=asset_names, yticklabels=regime_names,
                cbar_kws={"label": "P(Mean > 0)"})
    plt.title("Belief of implicit Means")
    plt.xlabel("Asset class")
    plt.ylabel("Regime")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

ZONE_NAMES = {
    ( 1,  1): "Infl. Growth",
    ( 1, -1): "Infl. Stagnation",
    (-1,  1): "Defl. Growth",
    (-1, -1): "Defl. Stagnation",
}

def plot_cpi_unrate_quadrants(
    df: pd.DataFrame,
    cpi_col: str = "CPI Index",
    unrate_col: str = "UNRATE",
    standardize: bool = False,
    drop_outliers: bool = True,
    outlier_quantiles: tuple[float, float] = (0.01, 0.99),
    robust_limits_quantile: float = 0.95,
    title: str | None = "Inflation vs Unemployment (m/m)",
    save_path: str | None = None,
    show: bool = True,
    point_kwargs: dict | None = None,
):
    """
    X: Δ Unemployment (p.p., m/m)
    Y: CPI Inflation (log %, m/m)

    - drop_outliers: entfernt Punkte außerhalb der gegebenen Quantile je Achse.
    - robust_limits_quantile: setzt symmetrische Achsengrenzen auf das q-Quantil
      des absoluten Werts (max. über X und Y).
    - standardize=True: z-Scores für beide Achsen (optional gleiche Maße).
    """
    point_kwargs = point_kwargs or dict(s=45, color="grey", edgecolor="black", alpha=0.75)

    # --- Transformationen ---
    x_raw = df[unrate_col].astype(float).diff()                           # Δ Unemployment (p.p.)
    y_raw = 100.0 * np.log(df[cpi_col].astype(float).pct_change() + 1.0)  # log-diff in %, stabil

    data = pd.DataFrame({"x": x_raw, "y": y_raw}).dropna()
    x, y = data["x"].to_numpy(), data["y"].to_numpy()

    # --- optional: Standardisierung (gleiche Maße) ---
    if standardize:
        x = (x - np.nanmean(x)) / np.nanstd(x, ddof=0)
        y = (y - np.nanmean(y)) / np.nanstd(y, ddof=0)
        x_label = "Δ Unemployment (z-score)"
        y_label = "CPI (log-diff, z-score)"
    else:
        x_label = "Δ Unemployment (p.p., m/m)"
        y_label = "CPI Inflation (log %, m/m)"

    # --- Ausreißer entfernen (per Quantil-Grenzen je Achse) ---
    removed = 0
    if drop_outliers:
        qx0, qx1 = np.quantile(x, outlier_quantiles)
        qy0, qy1 = np.quantile(y, outlier_quantiles)
        keep = (x >= qx0) & (x <= qx1) & (y >= qy0) & (y <= qy1)
        removed = np.count_nonzero(~keep)
        x, y = x[keep], y[keep]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(x, y, **point_kwargs)

    # Achsenkreuze
    ax.axhline(0, color="silver", lw=1)
    ax.axvline(0, color="silver", lw=1)

    # Robuste, symmetrische, quadratische Skalierung
    Lx = np.quantile(np.abs(x), robust_limits_quantile)
    Ly = np.quantile(np.abs(y), robust_limits_quantile)
    L = max(Lx, Ly)
    if not np.isfinite(L) or L == 0:
        L = np.nanmax(np.abs(np.r_[x, y]))
        if not np.isfinite(L) or L == 0:
            L = 1.0
    L *= 1.10  # kleiner Rand
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal", adjustable="box")

    # Achsentitel / Layout
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    if title:
        ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.25)

    # Quadranten-Beschriftung (Growth ~ -ΔUnemployment)
    pad = 0.04 * L
    labels_positions = {
        ( 1,  1): (-L + pad,  L - pad),  # x<0, y>0
        ( 1, -1): ( L - pad,  L - pad),  # x>0, y>0
        (-1,  1): (-L + pad, -L + pad),  # x<0, y<0
        (-1, -1): ( L - pad, -L + pad),  # x>0, y<0
    }
    for key, (px, py) in labels_positions.items():
        ax.text(px, py, ZONE_NAMES[key],
                ha="left" if px < 0 else "right",
                va="top"  if py > 0 else "bottom",
                fontsize=13, fontstyle="italic", color="black", alpha=0.85)

    # Kleine Info in der Ecke (wie viele entfernt)
    if drop_outliers and removed:
        ax.text(0.99, 0.02, f"{removed} outlier removed",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax