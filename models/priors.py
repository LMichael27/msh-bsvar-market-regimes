import numpy as np

def create_priors(hyperparams, Y, p, M):
    """
        Erstellt Litterman Priors

        Litterman Prior legt fest, dass eigene Vergangenheitswerte einer Variable wichtiger sind als die von anderen Variablen und das weit zurückliegende Lags weniger relevant sind.
        Somit konzentriert sich der Prior auf autoregressive Effekte.

        Inputs:
        - Hyperparams: Liste von 5 Hyperparameters --> die Hyperparameter steuern die Stärke und die Struktur der Priors
        - Y: Datenmatrix
        - p: Lag-Anzahl im SVAR
        - M: Anzahl der Regime

        Returns:
        - Dictionary mit Prior Matrizen
        """
    if hasattr(Y, "to_numpy"):
        Y = Y.to_numpy(dtype=float)
    else:
        Y = np.asarray(Y, dtype=float)

    N = Y.shape[1]
    T = Y.shape[0]
    TT = T - p - 1

    λ1 = hyperparams[0]
    λ2 = hyperparams[1]
    λ4 = hyperparams[2]
    c  = hyperparams[3]
    A_var = hyperparams[4]

    A_Var = A_var * np.eye(N)

    # --- Std-Abweichungen robust (gegen 0) ---
    eps = 1e-8
    std_devs = []
    for i in range(N):
        series = Y[:, i]
        residuals = series[1:] - series[:-1]   # AR(1)-Ersatz
        std_i = np.std(residuals)
        std_devs.append(max(std_i, eps))       # nie 0 zulassen
    std_devs = np.array(std_devs)

    # Intercept-Prior
    priors_Gamma = std_devs * (λ4 ** 2)

    # Litterman: std^2_i / std^2_j (robust im Nenner)
    litterman_sd = np.ones((N, N))
    for j in range(N):
        denom = max(std_devs[j] ** 2, eps)
        litterman_sd[j, :] = (std_devs ** 2) / denom

    for lag in range(1, p + 1):
        prior_lag = litterman_sd * (λ1 * 2) * (λ2 * 2) * np.exp(c * lag - 1)
        # eigene Lags schwächer schrumpfen
        np.fill_diagonal(prior_lag, np.diag(prior_lag) / (λ2 ** 2))
        priors_Gamma = np.vstack([priors_Gamma, prior_lag])

    prior_mean = np.zeros((1 + N * p, N))
    #for i in range(N):
     #   prior_mean[1 + i * p, i] = 1.0

    prior_variance = np.zeros((1 + N * p, 1 + N * p, N))
    for i in range(N):
        np.fill_diagonal(prior_variance[:, :, i], priors_Gamma[:, i])

    w = 9 * np.eye(M) + np.ones((M, M))
    beta_P = np.zeros((N, N * p))
    beta_H = np.eye(1 + N * p)

    return {
        "Gamma.Var": prior_variance,
        "Gamma.Mean": prior_mean,
        "lambda.alpha": 1,
        "lambda.beta": 1,
        "A.Var": A_Var,
        "w": w,
        "beta_P": beta_P,
        "beta_H": beta_H
    }
def make_litterman_betaH(N, p, lambda1=1.0, lambda2=0.25, lambda4=1.0, base_variance=0.05):
    """
    Diagonale Varianzmatrix (Minnesota-Style):
      Var(b_ii,lag)  ~ base / (lambda1^2) * lag^(2*lambda4)
      Var(b_ij,lag)  ~ base / (lambda1^2 * lambda2^2) * lag^(2*lambda4), i!=j
    Kleinere Varianzen => stärkere Shrinkage (stabilere α).
    """
    K = p * N
    H = np.zeros((K, K))
    for lag in range(1, p + 1):
        for i in range(N):
            for j in range(N):
                idx = (lag - 1) * N + j
                if i == j:
                    var = base_variance / (lambda1 * 2) * (lag * (2 * lambda4))
                else:
                    var = base_variance / (lambda1 * 2 * lambda2 * 2) * (lag ** (2 * lambda4))
                H[idx, idx] = np.clip(var, 1e-8, 1e3)
    return np.diag(np.diag(H))