import os
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

# optional, für glatte KDE-Dichten; fällt automatisch auf Histogramm zurück
try:
    from scipy.stats import gaussian_kde as _kde
except Exception:
    _kde = None


# =========================
# Helferfunktionen
# =========================

def _safe_kde(x):
    """KDE – nur wenn genügend Variation vorhanden, sonst None."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if _kde is None or x.size < 5 or np.allclose(x.std(), 0.0):
        return None
    try:
        return _kde(x)
    except Exception:
        return None


def _ess(x, max_lag=None):
    """
    Simple ESS-Schätzung (positive sequence). Nur für Plot-Diagnose.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return np.nan
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return np.nan
    acf = np.correlate(x, x, mode="full")[n - 1:] / denom
    if max_lag is None:
        max_lag = min(n - 1, 1000)
    s = 0.0
    for k in range(1, max_lag + 1):
        rho = float(acf[k])
        if rho <= 0:
            break
        s += 2.0 * rho
    ess = n / (1.0 + s)
    return max(1.0, ess)


def _mcse(x):
    """Monte-Carlo Standard Error auf Basis ESS."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    ess = _ess(x)
    if not np.isfinite(ess) or ess <= 0:
        return np.nan
    sd = np.std(x, ddof=1) if x.size > 1 else 0.0
    return sd / np.sqrt(ess)


def _rolling_mean(x, w=50):
    if w <= 1:
        return x
    x = np.asarray(x, float)
    if x.size < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    rm = (c[w:] - c[:-w]) / float(w)
    pad = np.repeat(rm[0], w // 2)
    pad2 = np.repeat(rm[-1], x.size - rm.size - pad.size)
    return np.r_[pad, rm, pad2]


def _finite_spread(arr, tol=1e-12):
    arr = np.asarray(arr, float)
    if arr.size == 0:
        return False
    return (np.nanmax(arr) - np.nanmin(arr)) > tol


def _auto_bins(x, max_bins=60):
    """Freedman–Diaconis Regel, mit sinnvollen Klammern."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 1:
        return 1
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    data_range = x.max() - x.min()
    if iqr > 0:
        h = 2.0 * iqr * (n ** (-1.0 / 3.0))
        est = int(np.ceil(data_range / h)) if h > 0 else int(np.sqrt(n))
    else:
        est = int(np.sqrt(n))
    return int(np.clip(est, 5, max_bins))


def _safe_logit(p, eps=1e-6):
    """Numerisch stabile Logit-Transformation von Wahrscheinlichkeiten."""
    p = np.asarray(p, float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def find_near_constant(series_dict, tol=1e-12):
    """
    Liste nahezu konstanter Serien – hilfreich zur Diagnose.
    Rückgabe: [(name, mean, range, unique_count), ...]
    """
    out = []
    for k, v in series_dict.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        rng = float(np.nanmax(v) - np.nanmin(v))
        uniq = int(np.unique(np.round(v, 12)).size)
        if rng <= tol or uniq < 2:
            out.append((k, float(np.mean(v)), rng, uniq))
    return out


# =========================
# Extraktion der 1D-Serien
# =========================

def collect_series_from_posteriors(
    posteriors: dict,
    asset_names=None,
    regime_names=None,
    p: int | None = None,
    which=("alpha_intercepts", "lambda_omega", "Sigma_diag", "PR_all", "hyper", "MH_logkernel"),
    max_plots: int | None = 60,
    pr_logit: bool = False,
    logit_eps: float = 1e-6,
):
    """
    Packt wählbare Parameter aus result['posteriors'] in ein geordnetes Dict {name: draws(S,)}.
    'which' steuert, was gesammelt wird.
      - 'alpha_intercepts', 'alpha_ownlag1', 'lambda_omega', 'Sigma_diag',
        'A_diag', 'A_all', 'a', 'PR_all'/'PR_diag', 'hyper', 'w', 'MH_logkernel'
    pr_logit: Wenn True → PR_TR-Wahrscheinlichkeiten per logit transformieren.
    logit_eps: Epsilon für Logit-Clipping (z. B. 1e-9, wenn du Variation nahe 0/1 sehen willst).
    """
    ser = OrderedDict()

    # Hilfsgrößen
    if "alpha" in posteriors:
        K, N, M, _S = posteriors["alpha"].shape
    else:
        N = M = None

    if asset_names is None and N is not None:
        asset_names = [f"Y{j+1}" for j in range(N)]
    if regime_names is None and M is not None:
        regime_names = [f"Regime {m+1}" for m in range(M)]

    def add(name, x):
        if x is None:
            return
        x = np.asarray(x).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        ser[name] = x

    # --- alpha: Intercepts je Gleichung & Regime
    if "alpha" in posteriors and ("alpha_intercepts" in which or "alpha" in which):
        K, N, M, _ = posteriors["alpha"].shape
        for m in range(M):
            for n in range(N):
                draws = posteriors["alpha"][0, n, m, :]
                add(f"Intercept | {asset_names[n]} @ {regime_names[m]}", draws)

    # --- alpha own-lag(1)
    if "alpha" in posteriors and "alpha_ownlag1" in which:
        K, N, M, _ = posteriors["alpha"].shape
        p_guess = (K - 1) // N if p is None else p
        if p_guess >= 1:
            for m in range(M):
                for n in range(N):
                    k_idx = 1 + n  # own-lag(1) für Gleichung n
                    draws = posteriors["alpha"][k_idx, n, m, :]
                    add(f"Lag1 own | {asset_names[n]} @ {regime_names[m]}", draws)

    # --- lambda_omega (N, M, S)
    if "lambda_omega" in posteriors and "lambda_omega" in which:
        lam = posteriors["lambda_omega"]
        N, M, _ = lam.shape
        for m in range(M):
            for n in range(N):
                add(f"lambda_omega[{asset_names[n]}] @ {regime_names[m]}", lam[n, m, :])

    # --- Sigma-Diagonalen (N, N, M, S)
    if "Sigma" in posteriors and "Sigma_diag" in which:
        Sig = posteriors["Sigma"]
        N, _, M, _ = Sig.shape
        for m in range(M):
            for n in range(N):
                add(f"Sigma[{asset_names[n]}]^2 @ {regime_names[m]}", Sig[n, n, m, :])

    # --- Strukturvektor a (M, r, S)
    if "a" in posteriors and "a" in which:
        a = posteriors["a"]
        M, r, _ = a.shape
        for m in range(M):
            for k in range(r):
                add(f"a[{k+1}] @ {regime_names[m]}", a[m, k, :])

    # --- Strukturmatrix A (M, N, N, S) – standardmäßig Diagonale
    if "A" in posteriors and ("A_diag" in which or "A_all" in which):
        A = posteriors["A"]
        M, N, N2, _ = A.shape
        assert N == N2
        for m in range(M):
            for i in range(N):
                add(f"A[{i+1},{i+1}] @ {regime_names[m]}", A[m, i, i, :])
        if "A_all" in which:
            for m in range(M):
                for i in range(N):
                    for j in range(N):
                        if i == j:
                            continue
                        add(f"A[{i+1},{j+1}] @ {regime_names[m]}", A[m, i, j, :])

    # --- Transition PR_TR (M, M, S)
    # WICHTIG: das sind Wahrscheinlichkeiten, NICHT binäre Indikatoren (S_t ist binär).
    if "PR_TR" in posteriors and ("PR_all" in which or "PR_diag" in which):
        P = posteriors["PR_TR"]
        M, _, S = P.shape
        for i in range(M):
            for j in range(M):
                if "PR_diag" in which and i != j:
                    continue
                draws = P[i, j, :]
                label = f"P({regime_names[i]}→{regime_names[j]})"
                if pr_logit:
                    draws = _safe_logit(draws, eps=logit_eps)
                    label += " (logit)"
                add(label, draws)

    # --- Hyperparameter (3, S)
    if "hyper" in posteriors and "hyper" in which:
        H = posteriors["hyper"]
        for k in range(H.shape[0]):
            add(f"hyper[{k+1}]", H[k, :])

    # --- w (d, S) – nur wenn explizit gewünscht
    if "w" in posteriors and "w" in which:
        W = posteriors["w"]
        for k in range(W.shape[0]):
            add(f"w[{k+1}]", W[k, :])

    # --- MH Kernel
    if "MH_logkernel" in posteriors and "MH_logkernel" in which:
        add("MH log-kernel", posteriors["MH_logkernel"])

    # ggf. begrenzen
    if max_plots is not None and len(ser) > max_plots:
        keys = list(ser.keys())
        priority_order = [k for k in keys if k.startswith("Intercept")] + \
                         [k for k in keys if "lambda_omega" in k] + \
                         [k for k in keys if "Sigma[" in k] + \
                         [k for k in keys if k.startswith("P(")] + \
                         [k for k in keys if k.startswith("hyper")] + \
                         [k for k in keys if k.startswith("A[")]
        seen, chosen = set(), []
        for k in priority_order + keys:
            if k not in seen:
                seen.add(k)
                chosen.append(k)
            if len(chosen) >= max_plots:
                break
        ser = OrderedDict((k, ser[k]) for k in chosen)

    return ser


# =========================
# Plotfunktionen
# =========================

def plot_mcmc_panels(
    series_dict: dict[str, np.ndarray],
    bins=60,
    figsize_per_row=1.8,
    save_path: str | None = None,
    show: bool = True,
    tight: bool = True,
    thin: int = 1,
    kde_grid: int = 200,
    verbose: bool = False,
):
    """
    Erzeugt 2-Spalten-Panels (links Dichte, rechts Trace) für alle Serien in 'series_dict'.
    Robust gegen nahezu konstante Serien.
    """
    names = list(series_dict.keys())
    n = len(names)
    if n == 0:
        raise ValueError("Keine Serien zum Plotten geliefert.")

    # ggf. ausdünnen
    def _thin(x):
        return x[::max(1, int(thin))]

    height = max(2, int(np.ceil(n * figsize_per_row)))
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, height))
    if n == 1:
        axes = np.array([axes])  # shape (1,2)

    near_const_list = []

    for i, name in enumerate(names):
        x = np.asarray(series_dict[name], float)
        x = x[np.isfinite(x)]
        x = _thin(x)
        if x.size == 0:
            continue

        # ---- Dichte (links)
        ax_d = axes[i, 0]
        if _finite_spread(x):
            kde = _safe_kde(x)
            if kde is not None:
                xmin, xmax = np.percentile(x, [0.5, 99.5])
                if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
                    xmin, xmax = x.min(), x.max()
                pad = 0.05 * (xmax - xmin + 1e-12)
                xx = np.linspace(xmin - pad, xmax + pad, kde_grid)
                yy = kde(xx)
                ax_d.plot(xx, yy, lw=1.8)
            else:
                ax_d.hist(x, bins=_auto_bins(x, max_bins=bins), density=True, alpha=0.9)
        else:
            # nahezu/komplett konstant → Dirac-artige Darstellung
            mu = float(np.mean(x))
            ax_d.axvline(mu, lw=2)
            ax_d.set_ylim(0, 1)
            near_const_list.append(name)

        mu, sd, ess = np.mean(x), np.std(x, ddof=1), _ess(x)
        mcse = _mcse(x)
        ax_d.set_title(name, fontsize=10)
        ax_d.set_yticks([])
        ax_d.grid(True, alpha=0.3)
        ax_d.text(0.98, 0.92, f"mean={mu:.3g}\nsd={sd:.3g}\nESS≈{ess:.0f}\nMCSE≈{(mcse if np.isfinite(mcse) else np.nan):.2g}",
                  ha="right", va="top", transform=ax_d.transAxes, fontsize=8)

        # ---- Trace (rechts)
        ax_t = axes[i, 1]
        ax_t.plot(np.arange(1, x.size + 1), x, lw=0.7)
        rm = _rolling_mean(x, w=max(10, x.size // 50))
        ax_t.plot(np.arange(1, x.size + 1), rm, lw=1.5, alpha=0.9)
        ax_t.set_title(name, fontsize=10)
        ax_t.grid(True, alpha=0.3)

    if verbose and near_const_list:
        print("Nahezu konstante Serien (als Linie geplottet):")
        for nm in near_const_list[:20]:
            print("  -", nm)
        if len(near_const_list) > 20:
            print(f"  ... und {len(near_const_list)-20} weitere.")

    if tight:
        plt.tight_layout()
    if save_path:
        folder = os.path.dirname(save_path) or "."
        os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================
# Bequemlichkeits-Wrapper
# =========================

def plot_mcmc_posteriors(
    result: dict,
    asset_names=None,
    regime_names=None,
    p: int | None = None,
    which=("alpha_intercepts", "lambda_omega", "Sigma_diag", "PR_all", "hyper", "MH_logkernel"),
    max_plots: int | None = 60,
    save_path: str | None = None,
    show: bool = True,
    thin: int = 1,
    pr_logit: bool = False,
    logit_eps: float = 1e-6,
    verbose: bool = False,
    return_series: bool = False,
):
    """
    Extrahiert Serien aus result['posteriors'] und plottet Panels.
      - pr_logit=True: zeige PR_TR in Logit-Skala (bessere Sicht nahe 0/1)
      - logit_eps: Clipping für Logit (z. B. 1e-9)
      - thin: Ausdünnung der Ketten (z. B. thin=5)
    """
    post = result["posteriors"]
    series = collect_series_from_posteriors(
        posteriors=post,
        asset_names=asset_names,
        regime_names=regime_names,
        p=p,
        which=which,
        max_plots=max_plots,
        pr_logit=pr_logit,
        logit_eps=logit_eps,
    )
    if verbose:
        flats = find_near_constant(series)
        if flats:
            print("Diagnose – nahezu konstante Serien:")
            for name, mean, rng, uniq in flats[:20]:
                print(f"  {name}: mean={mean:.3g}, range={rng:.2e}, unique≈{uniq}")
            if len(flats) > 20:
                print(f"  ... und {len(flats)-20} weitere.")
    plot_mcmc_panels(series, save_path=save_path, show=show, thin=thin, verbose=verbose)
    return series if return_series else None