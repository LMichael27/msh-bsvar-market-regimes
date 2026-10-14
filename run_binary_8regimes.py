import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.priors import create_priors, make_litterman_betaH
from models.mdd_estimation import compute_mdd
from models.sddr import compute_sddr, count_regime_occurrences
from models.initialisation import initialize_aux
from models.msh_bsvar_sampler import run_gibbs_sampler
from models.restrictions import get_restrictions
from models.post_analysis import (
    fevd, structural_matrix, forecast_with_ci_mc,
    average_volatility_per_regime, summarize_conditional_means,
    interpret_mean_returns, conditional_means_heatmap, plot_mean_significance,
    sample_regime_means_table_and_plot
)
from models.permutation2 import get_economic_permutation_from_raw, apply_label_permutation
from models.plot_irf import plot_all_regime_irfs
from models.plot_regimes import (
    plot_regime_probabilities, plot_regime_path,
    empirical_transition_matrix, plot_transition_matrix
)
from models.plot_mcmc import plot_mcmc_posteriors
from models.plot_regimes import plot_current_regime
from models.long_only_forecast_backtest import (
    BacktestConfig, prepare_signals_from_oos,
    backtest_from_oos, plot_equity_curve,
    prepare_signed_signals_from_oos, long_short_forecast_backtest, 
    compare_with_random_portfolios
)
from models.benchmark_strategies import build_benchmark_results

# >>> NEU: Grid-Search auf Balanced Accuracy (binär)
from models.grid_search_binary import grid_search_sign_accuracy, plot_grid_heatmap_sign

# >>> NEU: Binäre Evaluierung/Plots
from models.test_binary import (
    time_split_by_index,
    fit_and_forecast_once,
    evaluate_forecast_binary,
    plot_sign_forecast_vs_actual,
)

# -------------------------------
# 1) Daten einlesen & Returns (DataFrame mit Datumsindex)
# -------------------------------
df = pd.read_csv("data/final.csv", index_col="DATE", parse_dates=["DATE"]).sort_index()
ordering = [
    #"NDDUUS Index",  # US Stocks
    "VIX Index",
    "sp500",
    "LUATTRUU Index", # Bonds
    "DXY Index",
    #"WBCOCBRN Index",
    "XAU Curncy"     # Gold
]

df = df.dropna()
df = df[ordering].astype(float)

# Log-Returns (für das Modell)
returns_df = np.log(df / df.shift(1)).dropna()
asset_names = returns_df.columns.tolist()
returns = returns_df.to_numpy(dtype=float)
T, N = returns.shape

# -------------------------------
# 2) Modellparameter (Startwerte)
# -------------------------------
p = 4
M = 8
S = 30
burnin = 20
horizon = 20

# -------------------------------
# 3) Restriktionen & Startwerte
# -------------------------------
restrictions = get_restrictions(N=N, M=M)
aux = initialize_aux(Y=returns, p=p, M=M, restrictions=restrictions)

# -------------------------------
# 4) Priors
# -------------------------------
hyperparams = [0.5, 0.5, 0.5, -0.7, 0.5]
priors = create_priors(hyperparams, Y=returns, p=p, M=M)
if priors["w"].shape == (M, M):
    priors["w"] = priors["w"].reshape(M * M, 1, order="C")
priors.update({
    "hyper_a": 4,
    "hyper_b": 0.2,
    "lambda_a": 3,
    "lambda_b": 0.3,
    "omega_a": 4,
    "omega_b": 0.2,
    "beta_P": np.zeros((p * N, N)),
    "beta_H": make_litterman_betaH(N=N, p=p, lambda1=1.5, lambda2=0.6, lambda4=0.7, base_variance=0.5)
})
# -------------------------------
# Grid-Search nach (p, M) mit OOS Balanced Accuracy (BINÄR)
# -------------------------------
"""print("\n=== Starte Grid-Search (p=1..6, M=2..4) auf Balanced Accuracy (OOS, binär) ===")
gs_results = grid_search_sign_accuracy(
    returns_df=returns_df,
    p_grid=range(2, 8),
    M_grid=range(8, 9),
    test_steps=12,
    base_priors_kwargs={
        "hyperparams": [0.5, 0.5, 0.5, -0.7, 0.5],
        "hyper_a": 4, "hyper_b": 0.2,
        "lambda_a": 3, "lambda_b": 0.3,
        "omega_a": 4, "omega_b": 0.2,
        "lambda1": 1.5, "lambda2": 0.6, "lambda4": 0.7,
        "base_variance": 0.5,
    },
    base_sampler_kwargs={
        "S": 300, "burnin": 800, "nu": 2, "C": 10,
        "debug": False, "filtering_iterations": 80, "print_iterations": 50,
        "n_sims": 500, "seed": 42,
    },
    scale="log",
    ci_level=0.68,
    verbose=True,
)
os.makedirs("Figures/grid", exist_ok=True)
gs_results.to_csv("Figures/grid/gridsearch_oos_balacc.csv", index=False)
print("\nTop-5 Grid-Search Ergebnisse (Balanced Accuracy):")
print(gs_results.head(5))

# Heatmap
plot_grid_heatmap_sign(gs_results, title="Grid Search – Avg Balanced Accuracy (OOS, binär)")

# Bestes (p, M)
best_row = gs_results.iloc[0]
p = int(best_row["p"])
M = int(best_row["M"])
print(f"\n>>> BESTE WAHL aus Grid-Search: p={p}, M={M}  (avg BalAcc = {best_row['avg_bal_acc']:.4f})")
"""
# -------------------------------
# 5) OOS-Split & BINÄRE Forecast-Evaluierung
# -------------------------------
os.makedirs("Figures/eval", exist_ok=True)
test_steps = 12
train_df, test_df, split_idx = time_split_by_index(returns_df, test_steps=test_steps)
steps_ahead = len(test_df)

priors_kwargs = {
    "hyperparams": hyperparams,
    "hyper_a": 4, "hyper_b": 0.2,
    "lambda_a": 3, "lambda_b": 0.3,
    "omega_a": 4, "omega_b": 0.2,
    "lambda1": 1.0, "lambda2": 0.7, "lambda4": 1.0,
    "base_variance": 0.5,
}
sampler_kwargs = {
    "S": S, "burnin": burnin, "nu": 2, "C": 10,
    "debug": True, "filtering_iterations": 100, "print_iterations": 10,
    "n_sims": 400, "seed": 42,
}

# Wichtig: scale="log" (Returns sind log), für binär: Schwelle 0
oos = fit_and_forecast_once(
    returns_full=returns,
    asset_names=asset_names,
    p=p, M=M,
    priors_kwargs=priors_kwargs,
    sampler_kwargs=sampler_kwargs,
    split_idx=split_idx,
    steps_ahead=steps_ahead,
    scale="log",
    ci_level=0.68,
)

# BINÄRE Gütemaße
eval_cls = evaluate_forecast_binary(
    oos["forecast_mean"], oos["forecast_lower"], oos["forecast_upper"],  # für p(up)
    oos["Y_test"], asset_names, threshold=0.0
)
print("\n=== Out-of-sample Klassifikation (binär: up/down) ===")
print(eval_cls.round(4))
eval_cls.to_csv("Figures/eval/test_metrics_binary.csv", index=False)

# Plots: Actual Sign vs Predicted Sign + p(up)
plot_sign_forecast_vs_actual(
    oos, asset_names, threshold=0.0, save_prefix="Figures/eval/sign_oos_"
)
"""
# -------------------------------
# 6) Gibbs auf Vollsample (für IS-Analysen & Plots)
# -------------------------------
result = run_gibbs_sampler(
    S=S,
    burnin=burnin,
    priors=priors,
    restrictions=restrictions,
    starting_values=aux,
    nu=2,
    C=10,
    debug=True,
    filtering_iterations=100,
    print_iterations=10,
)

# -------------------------------
# 7) Ökonomische Permutation (Vollsample)
# -------------------------------
Y_last = returns[-p:, :]
summary_df, mu_raw = summarize_conditional_means(
    result=result, Y_last=Y_last, p=p,
    asset_names=asset_names, plot=False, regime_names=None, return_matrix=True,
)
perm = get_economic_permutation_from_raw(
    mu_raw, asset_names=asset_names, M=M,
    stock_name="sp500", bond_name="LUATTRUU Index",
)
print(f"\nÖkonomisch gewählte Permutation: {perm}")
result["posteriors"] = apply_label_permutation(result["posteriors"], perm)

# Nach Permutation erneut
summary_df, mu_raw = summarize_conditional_means(
    result=result, Y_last=Y_last, p=p,
    asset_names=asset_names, plot=False, regime_names=None, return_matrix=True,
)

# -------------------------------
# 8) Regime-Namen
# -------------------------------
if M == 8:
    regime_names = ["Infl. Growth High Uncert.", 
                    "Infl. Growth Low Uncert.", 
                    "Infl. Stagnation High Uncert.", 
                    "Infl. Stagnation Low Uncert.", 
                    "Defl. Growth High Uncert.", 
                    "Defl. Growth Low Uncert.", 
                    "Defl. Stagnation High Uncert.", 
                    "Defl. Stagnation Low Uncert."]
else:
    regime_names = [f"Regime {i+1}" for i in range(M)]

# -------------------------------
# 9) Struktur-Diagnostik
# -------------------------------
for m in range(M):
    alpha_m = result["posteriors"]["alpha"][:, :, m, -1]
    cond_num = np.linalg.cond(alpha_m)
    print(f"Regime {m+1}: Condition Number (alpha) = {cond_num:.2e}")

os.makedirs("Figures/plots", exist_ok=True)
structural_matrix(
    result, asset_names=asset_names,
    save_path="Figures/plots/structural_matrices",
    show=True, regime_names=regime_names,
    normalize=True, norm_method="maxabs", per_draw=False,
)

plot_current_regime(
    result,
    regime_names=regime_names,
    save_path="Figures/plots/current_regime.png",
    show=True,
)

# -------------------------------
# 10) (optional) MDD/SDDR
# -------------------------------
mdd_result = compute_mdd(
    result, priors=priors, restrictions=restrictions,
    SP=10000, scale_cov_z=1.0, jitter=1e-6, random_state=42,
)
print(f"\nLog MDD: {mdd_result['log.mdd']}")
sddr_result = compute_sddr(result, priors)
print("\nSDDR (log-scale):\n", np.round(sddr_result["log_sddr"], 2))

# -------------------------------
# 11) Analysen & Plots (weiterhin möglich, unabhängig von binärer OOS-Evaluation)
# -------------------------------
summary_df.index = pd.MultiIndex.from_product([regime_names, asset_names], names=["Regime", "Asset"])
os.makedirs("Figures/plots", exist_ok=True)
summary_df.to_excel("Figures/plots/posterior_mean_summary.xlsx")

S_info = result["posteriors"]["S_t"]
sample_table, decomp = sample_regime_means_table_and_plot(
    log_returns=returns, S_info=S_info, M=M,
    asset_names=asset_names, regime_names=regime_names,
    annualize=False, save_path="Figures/plots/sample_regime_means.png",
)
mix = decomp["mix"] * 100.0
overall = decomp["overall"] * 100.0
print("\nCheck (mix - unconditional) in %-Punkten:\n", np.round(mix - overall, 6))

interpreted = interpret_mean_returns(
    mu_raw, asset_names, regime_names=regime_names, annualize=False,
)

plot_mean_significance(
    summary_df, asset_names, regime_names=regime_names,
    save_path="Figures/plots/mean_significance.png",
    show_stars=True, star_levels=(0.90, 0.95, 0.99),
)

std_flag = M >= 3
conditional_means_heatmap(
    mu_raw, asset_names, regime_names=regime_names,
    standardize=std_flag, save_path="Figures/plots/cond_mean.png",
)

vol_mean_std, vol_df = average_volatility_per_regime(
    result, asset_names, regime_names=regime_names,
    standardize=std_flag, plot=True,
    save_path="Figures/plots/cond_var.png",
)

plot_regime_probabilities(result["posteriors"]["S_t"], regime_names=regime_names, dates=returns_df.index)
plot_regime_path(result, regime_names=regime_names, dates=returns_df.index)

plot_all_regime_irfs(
    posterior=result["posteriors"], p=p, horizon=horizon,
    asset_names=asset_names, save_path="Figures/plots/irfs",
    show=True, regime_names=regime_names,
)

for m in range(M):
    alpha_m = result["posteriors"]["alpha"][:, :, m, -1]
    A0_m = result["posteriors"]["A"][m, :, :, -1]
    fevd_result = fevd(alpha_m, A0_m, p=p, horizon=horizon)
    print(f"\nFEVD für {regime_names[m]} - Variable 0 bei Horizon 20:", np.round(fevd_result[0, :, 20], 2))

# (Optional) Demo-Prognose in % — inhaltlich wie zuvor
Y_last = returns[-p:, :]
forecast_mean, lower, upper = forecast_with_ci_mc(
    result["posteriors"], Y_last, p=p,
    steps=12, ci_level=0.68,
    regime="mix", n_sims=300, seed=42,
    scale="pct", annualize=False,
    clip_frac=0.3, rho_target=0.995, max_iter_proj=10, max_abs_log=6.0, lam_cap=1e4,
)
forecast_df = pd.DataFrame(forecast_mean, columns=asset_names, index=range(1, 13))
plt.figure(figsize=(12, 6))
for col in asset_names:
    plt.plot(forecast_df.index, forecast_df[col], label=col, linewidth=2)
plt.title("Forecast der nächsten 12 Monate")
plt.xlabel("Monate"); plt.ylabel("Rendite (%)")
plt.grid(True); plt.legend(); plt.tight_layout()
os.makedirs("Figures/plots", exist_ok=True)
plt.savefig("Figures/plots/forecast.png", dpi=150)
plt.show()

empirical_matrix = empirical_transition_matrix(result["posteriors"]["S_t"], M)
plot_transition_matrix(empirical_matrix, regime_names=regime_names)

S_t_last = result["posteriors"]["S_t"][:, -1]
regime_counts = count_regime_occurrences(S_t_last)
plt.figure(figsize=(8, 5))
plt.bar(regime_names, regime_counts, edgecolor='black')
plt.title("Anzahl des Vorkommens pro Regime", fontsize=12)
plt.xlabel("Regime", fontsize=10); plt.ylabel("Häufigkeit", fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Figures/plots/regime_counts.png")
plt.show()

plot_mcmc_posteriors(
    result=result,
    asset_names=asset_names,
    regime_names=regime_names,
    p=p,
    which=("alpha_intercepts", "lambda_omega", "Sigma_diag", "PR_all", "hyper", "MH_logkernel"),
    max_plots=60,
    save_path="Figures/plots/mcmc_panels.png",
    show=True,
    thin=1,
    pr_logit=True,
    verbose=True,
)
"""
oos_index = returns_df.index[split_idx : split_idx + oos["forecast_mean"].shape[0]]
cfg = BacktestConfig(tc_bps_roundtrip=10.0, cash_rate_annual=0.00, min_assets_long=1)

bt = backtest_from_oos(
    prices=df,                         # levels (daily or monthly)
    oos_forecast_mean=oos["forecast_mean"],
    oos_index=oos_index,
    asset_names=asset_names,
    cfg=cfg,
    n_months=12                        # last 36 months
)

print(bt.metrics)
# --- Long-short: prepare signed ternary signals and backtest
signed_signals = prepare_signed_signals_from_oos(
    oos_forecast_mean=oos["forecast_mean"],
    oos_index=oos_index,
    asset_names=asset_names,
    threshold=0.0,
)

bt_ls = long_short_forecast_backtest(
    prices=df,
    signed_signals=signed_signals,
    cfg=cfg,
    n_months=12,
)

benchmark_results = build_benchmark_results(
    prices=df,
    cfg=cfg,
    n_months=12,
    start_equity=1.0,
)

print("Long-Only metrics:\n", bt.metrics)
print("Long-Short metrics:\n", bt_ls.metrics)
print("VAR metrics:\n", benchmark_results["VAR"].metrics)

plot_equity_curve(
    bt,
    long_short_result=bt_ls,
    benchmark_results=benchmark_results,
    save_path="Figures/portfolio/equity_long_vs_longshort.png",
    prices_baseline=df,
    plot_equal_weight=True,
    start_equity=1.0,
)

compare_with_random_portfolios(
    prices=df,
    result=bt,
    long_short_result=bt_ls,
    n_portfolios=1000,
    start_equity=100.0,
    bins=40,
    seed=42,
    title="Final Value Distribution: 1,000 Random Buy-and-Hold Portfolios",
    save_path="Figures/portfolio/random_bh_distribution.png",
    show=True,
)