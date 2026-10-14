
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Dict, Any

from models.test_binary import (
    time_split_by_index,
    fit_and_forecast_once,
    evaluate_forecast_binary,
)

def grid_search_sign_accuracy(
    returns_df: pd.DataFrame,
    p_grid: Iterable[int] = range(1, 7),
    M_grid: Iterable[int] = range(2, 5),
    test_steps: int = 12,
    base_priors_kwargs: Dict[str, Any] = None,
    base_sampler_kwargs: Dict[str, Any] = None,
    scale: str = "log",
    ci_level: float = 0.68,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Grid-Search über p, M. Bewertung: durchschnittliche Balanced Accuracy (OOS, binär).
    Labels: 1 = return > 0, 0 = sonst.
    """
    assert isinstance(returns_df.index, pd.DatetimeIndex), "returns_df braucht DatetimeIndex."
    returns_df = returns_df.sort_index()
    asset_names = list(returns_df.columns)
    returns = returns_df.to_numpy(dtype=float)

    train_df, test_df, split_idx = time_split_by_index(returns_df, test_steps=test_steps)
    steps_ahead = len(test_df)

    if base_priors_kwargs is None:
        base_priors_kwargs = {
            "hyperparams": [0.5, 0.5, 0.5, -0.7, 0.5],
            "hyper_a": 4, "hyper_b": 0.2,
            "lambda_a": 3, "lambda_b": 0.3,
            "omega_a": 4, "omega_b": 0.2,
            "lambda1": 2.0, "lambda2": 1.0, "lambda4": 1.5,
            "base_variance": 0.5,
        }
    if base_sampler_kwargs is None:
        base_sampler_kwargs = {
            "S": 600, "burnin": 300, "nu": 2, "C": 10,
            "debug": False, "filtering_iterations": 80, "print_iterations": 50,
            "n_sims": 500, "seed": 42,
        }

    rows = []
    for p in p_grid:
        for M in M_grid:
            if verbose:
                print(f"\n[Grid] Starte p={p}, M={M} ...")
            priors_kwargs = dict(base_priors_kwargs)
            sampler_kwargs = dict(base_sampler_kwargs)

            try:
                oos = fit_and_forecast_once(
                    returns_full=returns,
                    asset_names=asset_names,
                    p=p, M=M,
                    priors_kwargs=priors_kwargs,
                    sampler_kwargs=sampler_kwargs,
                    split_idx=split_idx,
                    steps_ahead=steps_ahead,
                    scale=scale,
                    ci_level=ci_level,
                )
                eval_df = evaluate_forecast_binary(
                    oos["forecast_mean"], oos["forecast_lower"], oos["forecast_upper"],
                    oos["Y_test"], asset_names, threshold=0.0
                )
                avg_balacc = float(eval_df["BalancedAcc"].mean())

                row = {
                    "p": p, "M": M,
                    "avg_bal_acc": avg_balacc,
                    "mean_accuracy": float(eval_df["Accuracy"].mean()),
                    "mean_F1": float(eval_df["F1"].mean()),
                }
                for j, a in enumerate(asset_names):
                    row[f"BalAcc_{a}"] = float(eval_df.loc[j, "BalancedAcc"])
                rows.append(row)

                if verbose:
                    print(f"[Grid] p={p}, M={M}  →  avg BalAcc = {avg_balacc:.4f}")

            except Exception as e:
                if verbose:
                    print(f"[Grid] p={p}, M={M}  FEHLER: {e}")
                rows.append({"p": p, "M": M, "avg_bal_acc": np.nan})

    results = pd.DataFrame(rows).sort_values(by="avg_bal_acc", ascending=False).reset_index(drop=True)
    return results

def plot_grid_heatmap_sign(results: pd.DataFrame, title: str = "Grid Search – Avg Balanced Accuracy (binär)"):
    if results.empty or "avg_bal_acc" not in results.columns:
        print("Keine Ergebnisse vorhanden.")
        return
    pivot = results.pivot(index="p", columns="M", values="avg_bal_acc")
    plt.figure(figsize=(6, 4))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(im, label="Avg Balanced Accuracy")
    plt.xticks(ticks=range(pivot.shape[1]), labels=pivot.columns)
    plt.yticks(ticks=range(pivot.shape[0]), labels=pivot.index)
    plt.xlabel("M (Anzahl Regime)")
    plt.ylabel("p (VAR-Lags)")
    plt.title(title)
    plt.tight_layout()
    plt.show()