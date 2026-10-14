# MSH-BSVAR: Market Regime Detection

Python implementation of a **Markov-Switching Heteroskedastic Bayesian Structural VAR (MSH-BSVAR)** for detecting latent market regimes and analysing regime-dependent dynamics across asset classes.

This code was written for my Master's thesis at Bocconi University. The full thesis is available as a PDF: [thesis/Master_Thesis.pdf](thesis/Master_Thesis.pdf).

## Idea

The joint dynamics of asset classes like equities, bonds, the US dollar and gold are not stable over time. The model assumes they switch between a small number of latent **regimes**, such as calm or stress periods. A hidden Markov chain drives the switches.

- **Structural VAR** – identifies structural shocks and how they spread across assets.
- **Markov switching** – a latent Markov chain governs transitions between regimes.
- **Heteroskedasticity** – each regime has its own shock variances.
- **Bayesian estimation** – Litterman (Minnesota-type) priors, estimated with a Gibbs sampler that includes a Metropolis–Hastings step for the structural parameters.

## Features

- Smoothed and filtered regime probabilities and regime paths (forward filtering / backward smoothing)
- Posterior and empirical transition matrices
- Regime-specific impulse response functions (IRFs) and forecast error variance decompositions (FEVD)
- Correcting label switching by relabelling regimes on economic criteria
- Model comparison via **Marginal Data Density** (importance sampling)
- Testing parameter restrictions with the **Savage–Dickey density ratio**
- Grid search over lag order `p` and number of regimes `M`
- Out-of-sample forecasts, sign (direction) evaluation, and long-only / long-short backtests against benchmark strategies

## Repository structure

```
├── run_binary.py             # Sign-forecast evaluation + portfolio backtests (M = 4)
├── run_binary_8regimes.py    # Same as run_binary.py with 8 regimes
├── models/
│   ├── msh_bsvar_sampler.py  # Gibbs sampler (calls all sub-steps)
│   ├── initialisation.py     # Initial state of the sampler
│   ├── em_msh.py             # Starting values for regimes / variances / transitions
│   ├── priors.py             # Litterman priors
│   ├── restrictions.py       # Identifying restrictions (Cholesky-type)
│   ├── regression.py         # Regime-specific VAR coefficient draws
│   ├── structrual_mh.py      # Metropolis–Hastings step for structural parameters
│   ├── update_hyper.py       # Hyperparameter updates (inverse gamma)
│   ├── hidden_markov.py      # Transition matrix draws
│   ├── filtering_smoothing.py# Forward filtering / backward smoothing of regimes
│   ├── mdd_estimation.py     # Marginal data density
│   ├── sddr.py               # Savage–Dickey density ratio
│   ├── permutation.py        # Economic relabelling of regimes
│   ├── permutation2.py       # Extended relabelling logic
│   ├── post_analysis.py      # FEVD, forecasts, regime statistics
│   ├── plot_irf.py, plot_regimes.py, plot_mcmc.py
│   ├── test_binary.py                              # Train/test split, OOS forecasting & sign evaluation
│   ├── grid_search_binary.py                       # Grid search over p and M
│   ├── long_only_forecast_backtest.py              # Forecast-driven portfolio backtest
│   └── benchmark_strategies.py                     # VAR benchmark strategy
├── *.ipynb                   # Data download, cleaning, merging, HP filter, ADF tests, EDA
└── thesis/Master_Thesis.pdf  # Full Master's thesis
```

## Getting started

```bash
git clone https://github.com/LMichael27/msh-bsvar-market-regimes.git
cd msh-bsvar-market-regimes
pip install -r requirements.txt
```

Then run the pipeline from the repository root:

```bash
python run_binary.py
```

Figures are written to `Figures/` (`grid/`, `eval/`, `plots/`).

## Data

**The data is not included in this repository.** The thesis used Bloomberg data, which may not be redistributed.

The scripts expect a CSV file in `data/` with a `DATE` column and one price column per asset. For example, `run_binary.py` reads `data/updated.csv` with these columns:

| Column           | Asset                          |
|------------------|--------------------------------|
| `sp500`          | S&P 500                        |
| `LUATTRUU Index` | Bloomberg US Treasury Index    |
| `DXY Index`      | US Dollar Index                |
| `XAU Curncy`     | Gold                           |

To use your own data, change the file path and the `ordering` list at the top of the run script. `yfinance_data_download.ipynb` shows how to get comparable series from public sources.

## Key parameters

Set these at the top of each run script:

| Parameter | Meaning |
|-----------|---------|
| `p` | VAR lag order |
| `M` | Number of regimes |
| `S` | Number of Gibbs draws |

The grid search can replace `p` and `M` with the best-performing combination.

## Author

Michael Ladaa – Master's thesis, Bocconi University
