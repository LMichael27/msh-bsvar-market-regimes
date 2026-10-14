import numpy as np
from numpy.linalg import inv, solve
from scipy.stats import multivariate_normal

def regression_step(aux, priors):
    """
    Ziehe regime-spezifische VAR-Koeffizienten alpha[:, :, m] für jedes Regime separat.
    """

    Y_full = aux["Y"]
    T, N = Y_full.shape
    TT = aux["U"].shape[0]
    p = T - TT
    M = aux["PR_TR"].shape[0]

    # 1) Regressionsdaten X und Y aufbauen
    X = np.ones((TT, 1))
    for lag in range(1, p + 1):
        Y_lag = Y_full[(p - lag):(T - lag), :]
        X = np.hstack([X, Y_lag])
    Y = Y_full[p:, :]  # Zielmatrix

    K = X.shape[1]
    alpha = np.zeros((K, N, M))  # Regime-spezifische alpha

    # 2) Priorstruktur vorbereiten
    alpha_var = float(aux["hyper"][1])
    beta_var = float(aux["hyper"][2])
    beta_diag = np.diag(priors["beta_H"])
    prior_diag = np.concatenate(([1.0 / alpha_var], 1.0 / (beta_var * beta_diag)))
    prior_cov_inv = np.diag(prior_diag)
    prior_mean = np.vstack([np.zeros((1, N)), priors["beta_P"]])

    lambda_m = aux["lambda_omega"].copy()
    for m in range(1, M):
        lambda_m[:, m] *= lambda_m[:, 0]

    # 3) Schleife über Regime
    for m in range(M):
        indices = np.where(aux["xi"][m, :] == 1)[0]
        if len(indices) == 0:
            continue

        X_m = X[indices, :]
        Y_m = Y[indices, :]
        XX = np.zeros((K, K))
        XY = np.zeros((K, N))

        for eq in range(N):
            weight = 1.0 / lambda_m[eq, m]
            XX += weight * (X_m.T @ X_m)
            XY[:, eq] += weight * (X_m.T @ Y_m[:, eq])

        H_m = inv(XX + prior_cov_inv)
        H_m = 0.5 * (H_m + H_m.T)  # symmetrisieren

        alpha_m = np.zeros((K, N))
        for eq in range(N):
            prior_term = prior_cov_inv @ prior_mean[:, eq]
            P_m = H_m @ (XY[:, eq] + prior_term)
            alpha_m[:, eq] = multivariate_normal.rvs(mean=P_m, cov=H_m)

        alpha[:, :, m] = alpha_m

    aux["alpha"] = alpha

    # 4) Residuen berechnen
    U_SF = np.zeros((TT, N))
    U = np.zeros((TT, N))

    for m in range(M):
        indices = np.where(aux["xi"][m, :] == 1)[0]
        if len(indices) == 0:
            continue

        A_m = aux["A"][m]
        alpha_m = alpha[:, :, m]
        X_full = X @ alpha_m
        U_SF[indices] = Y[indices] @ A_m.T - X_full[indices]
        U[indices] = solve(A_m.T, U_SF[indices].T).T

    aux["U_SF"] = U_SF
    aux["U"] = U

    return aux