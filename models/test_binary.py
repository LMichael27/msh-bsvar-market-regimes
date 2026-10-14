
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)
from scipy.stats import norm

from models.restrictions import get_restrictions
from models.initialisation import initialize_aux
from models.priors import create_priors, make_litterman_betaH
from models.msh_bsvar_sampler import run_gibbs_sampler
from models.post_analysis import summarize_conditional_means, forecast_with_ci_mc
from models.permutation2 import get_economic_permutation_from_raw, apply_label_permutation

# ---------- A) Zeit-Split ----------
def time_split_by_index(df_or_arr, split_date=None, test_steps=None):
    if isinstance(df_or_arr, pd.DataFrame):
        X = df_or_arr.sort_index()
        if split_date is not None:
            split_ts = pd.to_datetime(split_date)
            mask_test = X.index >= split_ts
            if not mask_test.any():
                raise ValueError("split_date liegt nach dem letzten Index.")
            split_idx = int(np.argmax(mask_test.values))
        else:
            if test_steps is None:
                raise ValueError("Entweder split_date oder test_steps angeben.")
            split_idx = len(X) - int(test_steps)
        return X.iloc[:split_idx], X.iloc[split_idx:], split_idx
    else:
        X = np.asarray(df_or_arr)
        if split_date is not None:
            raise ValueError("split_date setzt DatetimeIndex voraus (DataFrame).")
        if test_steps is None:
            raise ValueError("Für Arrays bitte test_steps angeben.")
        split_idx = len(X) - int(test_steps)
        return X[:split_idx], X[split_idx:], split_idx

# ---------- B) Regime-Tools ----------
def _row_normalize(P):
    P = np.asarray(P, dtype=float)
    P[P < 0] = 0.0
    rs = P.sum(axis=1, keepdims=True)
    rs = np.clip(rs, 1e-12, None)
    return P / rs

def _stationary_dist(P):
    w, v = np.linalg.eig(P.T)
    j = np.argmin(np.abs(w - 1.0))
    pi = np.real(v[:, j])
    pi = np.maximum(pi, 0.0)
    s = pi.sum()
    return pi / s if s > 0 else np.ones(P.shape[0]) / P.shape[0]

def _posterior_mean_transition(result_train, M):
    Ppost = result_train["posteriors"].get("PR_TR", None)
    if Ppost is None:
        raise KeyError("Posterior enthält keine 'PR_TR'.")
    P = np.asarray(Ppost, dtype=float)
    while P.ndim > 2:
        P = P.mean(axis=-1)
    return _row_normalize(P)

def _last_smoothed_prob(result_train, M):
    post = result_train["posteriors"]
    for key in ("sp", "S_prob", "S_t"):
        arr = post.get(key, None)
        if arr is None:
            continue
        A = np.asarray(arr)
        if A.ndim == 2:
            if A.shape[1] == M:
                v = A[-1, :]
                v = np.maximum(v, 0.0)
                s = v.sum()
                return v / s if s > 0 else np.ones(M) / M
            if A.shape[0] == M:
                v = A[:, -1]
                v = np.maximum(v, 0.0)
                s = v.sum()
                return v / s if s > 0 else np.ones(M) / M
        if A.ndim == 3:
            if A.shape[1] == M:
                v = A[-1, :].mean(axis=-1)
                v = np.maximum(v, 0.0); s = v.sum()
                return v / s if s > 0 else np.ones(M) / M
            if A.shape[0] == M:
                v = A[:, -1].mean(axis=-1)
                v = np.maximum(v, 0.0); s = v.sum()
                return v / s if s > 0 else np.ones(M) / M
    return None

# ---------- C) Analytische Mischung (optional)
def analytic_mixture_forecast(result_train, Y_last_train, p, asset_names, horizon):
    M = result_train["posteriors"]["A"].shape[0]
    _, mu_raw = summarize_conditional_means(
        result=result_train, Y_last=Y_last_train, p=p,
        asset_names=asset_names, plot=False, regime_names=None, return_matrix=True
    )
    mu = np.asarray(mu_raw)
    if mu.shape == (len(asset_names), M):
        mu = mu.T
    Pbar = _posterior_mean_transition(result_train, M)
    pi0 = _last_smoothed_prob(result_train, M)
    if pi0 is None:
        pi0 = _stationary_dist(Pbar)
    pis = np.zeros((horizon, M))
    pi = pi0
    for h in range(horizon):
        pi = pi @ Pbar
        pis[h] = pi
    f_mix = pis @ mu
    scenario = {f"Regime {m+1}": np.repeat(mu[m][None, :], repeats=horizon, axis=0) for m in range(M)}
    return f_mix, scenario

# ---------- D) Fit & Forecast (Train only) ----------
def fit_and_forecast_once(
    returns_full: np.ndarray,
    asset_names: list,
    p: int,
    M: int,
    priors_kwargs: dict,
    sampler_kwargs: dict,
    split_idx: int,
    steps_ahead: int,
    scale: str = "log",
    ci_level: float = 0.68,
):
    Y_train = returns_full[:split_idx, :]
    Y_test  = returns_full[split_idx:, :]

    assert Y_train.shape[0] > 5 * p, "Train-Fenster zu kurz für p Lags."
    assert len(asset_names) == Y_train.shape[1] == Y_test.shape[1], "Asset-Dimensionen inkonsistent."

    N = Y_train.shape[1]
    restrictions = get_restrictions(N=N, M=M)
    aux = initialize_aux(Y=Y_train, p=p, M=M, restrictions=restrictions)

    priors = create_priors(priors_kwargs["hyperparams"], Y=Y_train, p=p, M=M)
    if priors["w"].shape == (M, M):
        priors["w"] = priors["w"].reshape(M * M, 1, order="C")
    priors.update({
        "hyper_a":  priors_kwargs.get("hyper_a", 4),
        "hyper_b":  priors_kwargs.get("hyper_b", 0.2),
        "lambda_a": priors_kwargs.get("lambda_a", 3),
        "lambda_b": priors_kwargs.get("lambda_b", 0.3),
        "omega_a":  priors_kwargs.get("omega_a", 4),
        "omega_b":  priors_kwargs.get("omega_b", 0.2),
        "beta_P":   np.zeros((p * N, N)),
        "beta_H":   make_litterman_betaH(
            N=N, p=p,
            lambda1=priors_kwargs.get("lambda1", 1.5),
            lambda2=priors_kwargs.get("lambda2", 0.5),
            lambda4=priors_kwargs.get("lambda4", 0.7),
            base_variance=priors_kwargs.get("base_variance", 0.5),
        ),
    })

    result_train = run_gibbs_sampler(
        S=sampler_kwargs.get("S", 150),
        burnin=sampler_kwargs.get("burnin", 100),
        priors=priors,
        restrictions=restrictions,
        starting_values=aux,
        nu=sampler_kwargs.get("nu", 2),
        C=sampler_kwargs.get("C", 10),
        debug=sampler_kwargs.get("debug", True),
        filtering_iterations=sampler_kwargs.get("filtering_iterations", 100),
        print_iterations=sampler_kwargs.get("print_iterations", 20),
    )

    # Ökonomische Permutation
    Y_last_train = Y_train[-p:, :]
    _, mu_raw_train = summarize_conditional_means(
        result=result_train, Y_last=Y_last_train, p=p,
        asset_names=asset_names, plot=False, regime_names=None, return_matrix=True
    )
    perm = get_economic_permutation_from_raw(
        mu_raw_train, asset_names=asset_names, M=M,
        stock_name="NDDUUS Index", bond_name="LUATTRUU Index"
    )
    result_train["posteriors"] = apply_label_permutation(result_train["posteriors"], perm)

    # Forecasts
    Pbar = _posterior_mean_transition(result_train, M)
    pi0  = _last_smoothed_prob(result_train, M)
    if pi0 is None:
        pi0 = _stationary_dist(Pbar)

    f_mean_mc, f_lo_mc, f_hi_mc = forecast_with_ci_mc(
        result_train["posteriors"],
        Y_last_train, p=p,
        steps=steps_ahead, ci_level=ci_level,
        regime="mix",
        initial_probs=pi0,
        n_sims=sampler_kwargs.get("n_sims", 400),
        seed=sampler_kwargs.get("seed", 42),
        scale=scale,
        annualize=False,
        clip_frac=None
    )

    # Analytische Mischung (optional für Plot)
    f_mean_an, scenarios = analytic_mixture_forecast(
        result_train, Y_last_train, p=p,
        asset_names=asset_names, horizon=steps_ahead
    )

    return {
        "result_train": result_train,
        "forecast_mean_mc": np.asarray(f_mean_mc),
        "forecast_lower_mc": np.asarray(f_lo_mc),
        "forecast_upper_mc": np.asarray(f_hi_mc),
        "forecast_mean_analytic": np.asarray(f_mean_an),
        "scenario_means": scenarios,
        "Y_test": Y_test,
        "perm": perm,
        # Aliase
        "forecast_mean": np.asarray(f_mean_mc),
        "forecast_lower": np.asarray(f_lo_mc),
        "forecast_upper": np.asarray(f_hi_mc),
    }

# ---------- E) BINÄRE Metriken ----------
def _prob_up_from_ci(mean, lo, hi):
    # 68%-CI ~ ±1 SD → sd ≈ (hi - lo)/2; Guard gegen 0
    sd = np.maximum((hi - lo) / 2.0, 1e-12)
    z = (0.0 - mean) / sd
    return 1.0 - norm.cdf(z)

def evaluate_forecast_binary(f_mean, f_lo, f_hi, Y_test, asset_names, threshold=0.0):
    """
    Erzeugt pro Asset: Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC,
    sowie p(up)-Durchschnitt. Labels: 1 = up (>threshold), 0 = down.
    """
    H, N = Y_test.shape
    rows = []
    y_true = (Y_test > threshold).astype(int)
    y_pred = (f_mean > threshold).astype(int)
    p_up = _prob_up_from_ci(f_mean, f_lo, f_hi)

    for j, name in enumerate(asset_names):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        probs = p_up[:, j]

        acc  = accuracy_score(yt, yp)
        balc = balanced_accuracy_score(yt, yp)
        prec = precision_score(yt, yp, zero_division=0)
        rec  = recall_score(yt, yp, zero_division=0)
        f1   = f1_score(yt, yp, zero_division=0)
        mcc  = matthews_corrcoef(yt, yp) if len(np.unique(yt)) > 1 else np.nan
        cm   = confusion_matrix(yt, yp, labels=[0,1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan, np.nan, np.nan, np.nan)

        rows.append({
            "Asset": name,
            "Accuracy": acc,
            "BalancedAcc": balc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "MCC": mcc,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "Avg_p_up": float(np.mean(probs)),
        })

    return pd.DataFrame(rows)

# ---------- F) Plots ----------
def plot_sign_forecast_vs_actual(oos, asset_names, threshold=0.0, save_prefix=None, dates=None):
    """
    Zeigt pro Asset:
    - tatsächliches Vorzeichen (0/1),
    - vorhergesagtes Vorzeichen (0/1) aus forecast_mean,
    - geschätzte p(up) aus CI.
    
    Parameters
    ----------
    dates : array-like, optional
        Datumsindizes für die X-Achse. Falls vorhanden, wird die X-Achse mit Datumsangaben versehen.
    """
    f_mean = oos["forecast_mean"]
    f_lo = oos["forecast_lower"]
    f_hi = oos["forecast_upper"]
    Y_test = oos["Y_test"]

    y_true = (Y_test > threshold).astype(int)
    y_pred = (f_mean > threshold).astype(int)
    p_up = _prob_up_from_ci(f_mean, f_lo, f_hi)

    H, N = Y_test.shape
    t = np.arange(H)

    for j, name in enumerate(asset_names):
        plt.figure(figsize=(11, 4))
        plt.step(t, y_true[:, j], where="mid", label="Actual sign (1=up)", linewidth=2)
        plt.step(t, y_pred[:, j], where="mid", label="Pred sign (mean>0)", linewidth=2)
        plt.plot(t, p_up[:, j], label="p(up) (from CI)", linewidth=2)
        plt.ylim(-0.1, 1.1)
        plt.title(f"OOS Binary Forecast: {name}")
        
        # Format X-axis with dates if provided
        if dates is not None:
            try:
                dates_arr = np.asarray(dates)
                if len(dates_arr) == H:
                    # Determine number of labels
                    n_labels = min(12, H)
                    step = max(1, H // n_labels)
                    xtick_indices = np.arange(0, H, step)
                    xtick_labels = []
                    for i in xtick_indices:
                        d = dates_arr[int(i)]
                        # Handle numpy datetime64, pandas Timestamp, and datetime objects
                        try:
                            if isinstance(d, (np.datetime64)):
                                # Convert numpy datetime64 to string and extract YYYY-MM
                                d_str = str(d)[:7]  # YYYY-MM format
                                xtick_labels.append(d_str)
                            elif hasattr(d, "strftime"):
                                xtick_labels.append(d.strftime("%Y-%m"))
                            else:
                                d_str = str(d)[:7]
                                xtick_labels.append(d_str)
                        except:
                            xtick_labels.append(str(d)[:7])
                    plt.xticks(xtick_indices, xtick_labels, rotation=45, ha="right", fontsize=9)
                    plt.xlabel("Date")
                else:
                    print(f"[Warning] dates length ({len(dates_arr)}) != H ({H})")
                    plt.xlabel("Horizon (months)")
            except Exception as e:
                print(f"[Warning] Error with dates: {e}")
                plt.xlabel("Horizon (months)")
        else:
            plt.xlabel("Horizon (months)")
        
        plt.ylabel("Probability")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if save_prefix:
            fn = f"{save_prefix}{name.replace(' ', '')}.png"
            plt.savefig(fn, dpi=140)
        plt.show()