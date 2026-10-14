import numpy as np
from numpy.linalg import inv, det, slogdet
from scipy.stats import gamma, multivariate_normal, dirichlet, chi2, multivariate_t
from scipy.linalg import eigh, null_space, solve
from scipy.special import gammaln, betaln
from itertools import permutations
from statsmodels.tsa.api import VAR


def update_hyperparameters(aux, priors, restrictions):
    """
    Zieht neue Hyperparameter (Varianzen) aus Inverse-Gamma-Verteilungen.
    - Für Strukturparameter a (Strukturmatrix A)
    - Für Intercepts mu
    - Für VAR-Koeffizienten beta
    Rückgabe: aux mit aktualisiertem "hyper" Eintrag (3x1 Matrix)
    """

    N = aux["Y"].shape[1]
    T = aux["Y"].shape[0]
    TT = aux["U_SF"].shape[0]
    p = T - TT
    r = restrictions["Q"].shape[1]

    alpha = aux["alpha"]
    K = 1 + p * N  # Regressoren inkl. Intercept

    # Check, ob alpha 3D ist (K, N, M) oder 2D (K, N)
    is_regime_specific = alpha.ndim == 3
    M = alpha.shape[2] if is_regime_specific else 1

    # 1) Strukturparameter a
    ga_a = priors["hyper_a"] + r
    ga_b = priors["hyper_b"] + np.sum(aux["a"][:, :, 0] ** 2)
    draw_chisq_a = np.random.chisquare(df=ga_a)
    draw_a = np.clip(ga_b / draw_chisq_a, 1e-2, 1e3)

    # 2) Intercepts mu
    gm_a = priors["hyper_a"] + N * M
    if is_regime_specific:
        alpha_0 = alpha[0, :, :]  # (N, M)
        gm_b = priors["hyper_b"] + float(np.sum(alpha_0 ** 2))
    else:
        alpha_0 = alpha[0, :]  # (N,)
        gm_b = priors["hyper_b"] + float(np.sum(alpha_0 ** 2))
    draw_chisq_m = np.random.chisquare(df=gm_a)
    draw_m = draw_m = np.clip(gm_b / draw_chisq_m, 1e-2, 1e3)

    # 3) Slope-Koeffizienten beta
    gb_a = priors["hyper_a"] + p * N ** 2 * M
    gb_tmp = 0.0
    beta_H = priors["beta_H"]

    if is_regime_specific:
        for m in range(M):
            for n in range(N):
                alpha_n = alpha[1:, n, m].reshape(-1, 1)
                gb_tmp += float(alpha_n.T @ beta_H @ alpha_n)
    else:
        for n in range(N):
            alpha_n = alpha[1:, n].reshape(-1, 1)
            gb_tmp += float(alpha_n.T @ beta_H @ alpha_n)

    gb_b = priors["hyper_b"] + gb_tmp
    draw_chisq_b = np.random.chisquare(df=gb_a)
    draw_b = np.clip(gb_b / draw_chisq_b, 1e-2, 1e3)

    aux["hyper"] = np.array([[draw_a], [draw_m], [draw_b]])

    return aux


def update_inverted_gamma(aux, priors):
    """
    Aktualisiert die regime spezifische Innovationsstruktur:
    - zieht neue 'lambda'-Werte für Basisregime
    - zieht neue 'omega'- Werte für alle weiteren Regime
    - aktualisiert 'lambda.omega' und daraus abgeleitete 'Sigma' je Regime

    --> lambda und omega bestimmende regimeabhängige Volatilität je Asset Klasse
    --> Sigma ist die vollständige Kovarianzmatrix je Regime
    """

    N = aux["Y"].shape[1]
    T = aux["Y"].shape[0]
    TT = aux["U_SF"].shape[0]
    p = T - TT
    M = aux["PR_TR"].shape[0]

    # 1) Ziehung der lambda-Komponente für die Basisregime
    lambda_a = priors["lambda_a"] + TT

    lo_tmp = aux["lambda_omega"].copy()
    lo_tmp[:, 0] = 1.0  # Normierung: Wir nehmen die erste Regime als Referenz

    l_tmp = np.zeros(N)
    for n in range(N):
        for m in range(M):
            indices = np.where(aux["xi"][m, :] == 1)[0]
            if len(indices) > 0:
                resid = aux["U_SF"][indices, n]
                l_tmp[n] += np.sum(resid ** 2) / lo_tmp[n, m]

    lambda_b = l_tmp + priors["lambda_b"]
    draw_chisq_l = chi2.rvs(df=lambda_a, size=N)
    draw_lambda = np.clip(lambda_b / draw_chisq_l, 1e-4, 1e4)  # neue lambda für die Basisregime

    # 2) Ziehung der omega-Komponenetn für alle anderen Regime
    draw = np.zeros((N, M))
    for n in range(N):
        for m in range(1, M):
            T_m = np.sum(aux["xi"][m, :])  # zählt wie oft Regime m aktiv war
            omega_a = priors["omega_a"] + T_m
            indices = np.where(aux["xi"][m, :] == 1)[0]
            if len(indices) > 0:
                resid = aux["U_SF"][indices, n]
                dl_safe = np.maximum(draw_lambda[n], 1e-6)
                omega_b = np.sum(resid ** 2) / dl_safe + priors["omega_b"]
            else:
                omega_b = priors["omega_b"]
            draw_omega = chi2.rvs(df=omega_a)
            draw[n, m] = omega_b / draw_omega

    # 3) Setze Regime 0  als Basisregime
    draw[:, 0] = draw_lambda

    # 4) Aktualisiere lambda und Sigma
    aux["lambda_omega"] = draw

    for m in range(M):
        A_inv = inv(aux["A"][m])
        if m == 0:
            lambda_m = draw[:, m]
        else:
            lambda_m = draw[:, m] * draw[:, 0]  # skaliert nur die Basisvarianz
        aux["Sigma"][:, :, m] = A_inv @ np.diag(lambda_m) @ A_inv.T

    return aux