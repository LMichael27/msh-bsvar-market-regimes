import numpy as np
from numpy.linalg import inv, slogdet
from scipy.stats import multivariate_normal, multivariate_t
from scipy.linalg import solve


def structural_mh(aux, priors, restrictions,
                  nu=5,
                  C=0.05,  
                  min_obs_per_regime=5,
                  jitter=1e-8,
                  max_cond=1e10):
    """
    Verbesserter MH-Step für Strukturparameter a (MSH-BSVAR)
    """

    Y = np.asarray(aux["Y"])
    N = Y.shape[1]
    T = Y.shape[0]

    U = np.asarray(aux["U"])
    TT = U.shape[0]
    p = T - TT

    PR = np.asarray(aux["PR_TR"])
    M = PR.shape[0]

    Q = np.asarray(restrictions["Q"])
    q = np.asarray(restrictions["q"]).reshape(-1, 1)
    r = Q.shape[1]

    # ---------------------------
    # X bauen
    # ---------------------------
    X = np.ones((TT, 1))
    for lag in range(1, p + 1):
        Y_lag = Y[(p - lag):(T - lag), :]
        X = np.hstack([X, Y_lag])

    Y_eff = Y[p:, :]

    # ---------------------------
    # Output Container
    # ---------------------------
    A_s = np.zeros((M, N, N))
    a_s = np.zeros((M, r, 1))

    MH_logkernel_sum = 0.0
    accepts = 0
    proposals = 0

    # ---------------------------
    # Prior
    # ---------------------------
    hyper_var_a = float(aux.get("hyper", [1.0])[0])
    prior_cov_a = hyper_var_a * np.eye(r)

    # ---------------------------
    # xi vorbereiten (weich!)
    # ---------------------------
    xi = np.asarray(aux["xi"])
    if xi.shape == (TT, M):
        xi = xi.T

    if xi.shape != (M, TT):
        raise ValueError("xi must be (M, TT)")

    # ---------------------------
    # Loop über Regime
    # ---------------------------
    for m in range(M):

        weights = xi[m, :] 
        effective_n = np.sum(weights)

        if effective_n < min_obs_per_regime:
            A_s[m] = aux["A"][m]
            a_s[m] = aux["a"][m]
            continue

        alpha_m = aux["alpha"][:, :, m]
        X_alpha = X @ alpha_m

        # ------------------------------------------
        # Designmatrix
        # ------------------------------------------
        I_N = np.eye(N)
        X_tilde = -np.kron(Y_eff, I_N) @ Q
        Y_tilde = (np.kron(Y_eff, I_N) @ q) - X_alpha.reshape(-1, 1)

        # ------------------------------------------
        # Gewichtung (weich!)
        # ------------------------------------------
        lam_m = np.asarray(aux["lambda_omega"])[:, m]
        if m > 0:
            lam_m = lam_m * np.asarray(aux["lambda_omega"])[:, 0]

        lam_m = np.clip(lam_m, 1e-12, None)

        # expandiere Gewichte auf vec-Struktur
        weight_vec = np.repeat(weights, N)
        lambda_vec = np.tile(lam_m, TT)

        inv_lambda = 1.0 / lambda_vec
        W = weight_vec * inv_lambda  

        # ------------------------------------------
        # Stabiler P
        # ------------------------------------------
        WX = X_tilde * W[:, None]
        P_tmp = X_tilde.T @ WX

        P_tmp = 0.5 * (P_tmp + P_tmp.T) + jitter * np.eye(r)

        try:
            P_inv = solve(P_tmp, np.eye(r), assume_a='pos')
            P_inv = 0.5 * (P_inv + P_inv.T)
        except Exception:
            A_s[m] = aux["A"][m]
            a_s[m] = aux["a"][m]
            continue

        # ------------------------------------------
        # Posterior Mean 
        # ------------------------------------------
        WY = Y_tilde * W[:, None]
        a_hat = P_inv @ (X_tilde.T @ WY)

        # ------------------------------------------
        # aktueller Zustand
        # ------------------------------------------
        a_curr = aux["a"][m].reshape(-1, 1)

        U_curr = Y_tilde - X_tilde @ a_curr
        sum_u_curr = float((U_curr.flatten()**2 * W).sum())

        A0_curr = (Q @ a_curr + q).reshape(N, N)

        signA, logdetA = slogdet(A0_curr)
        if (signA <= 0) or (not np.isfinite(logdetA)) or (np.linalg.cond(A0_curr) > max_cond):
            A_s[m] = aux["A"][m]
            a_s[m] = aux["a"][m]
            continue

        log_lik_curr = (
            -0.5 * effective_n * N * np.log(2.0 * np.pi)
            + effective_n * logdetA
            - 0.5 * sum_u_curr
        )

        log_prior_curr = multivariate_normal.logpdf(
            a_curr.flatten(), mean=np.zeros(r), cov=prior_cov_a
        )

        log_kernel_curr = log_lik_curr + log_prior_curr

        # ------------------------------------------
        # Proposal zentriert
        # ------------------------------------------
        proposal_cov = C * P_inv + jitter * np.eye(r)

        try:
            a_cand = multivariate_t.rvs(
                loc=a_hat.flatten(),   
                shape=proposal_cov,
                df=nu
            ).reshape(-1, 1)
        except Exception:
            a_cand = a_curr + 0.01 * np.random.randn(r, 1)

        # ------------------------------------------
        # Kandidat evaluieren
        # ------------------------------------------
        U_cand = Y_tilde - X_tilde @ a_cand
        sum_u_cand = float((U_cand.flatten()**2 * W).sum())

        A0_cand = (Q @ a_cand + q).reshape(N, N)

        signA_c, logdetA_c = slogdet(A0_cand)

        if (signA_c <= 0) or (not np.isfinite(logdetA_c)) or (np.linalg.cond(A0_cand) > max_cond):
            delta = -np.inf
        else:
            log_lik_cand = (
                -0.5 * effective_n * N * np.log(2.0 * np.pi)
                + effective_n * logdetA_c
                - 0.5 * sum_u_cand
            )

            log_prior_cand = multivariate_normal.logpdf(
                a_cand.flatten(), mean=np.zeros(r), cov=prior_cov_a
            )

            log_kernel_cand = log_lik_cand + log_prior_cand
            delta = float(log_kernel_cand - log_kernel_curr)

        # ------------------------------------------
        # MH Schritt (log stabil)
        # ------------------------------------------
        log_u = np.log(np.random.rand())
        proposals += 1

        if log_u < min(0.0, delta):
            a_s[m] = a_cand
            A_s[m] = A0_cand
            MH_logkernel_sum += (log_kernel_cand if np.isfinite(delta) else log_kernel_curr)
            accepts += 1
        else:
            a_s[m] = a_curr
            A_s[m] = A0_curr
            MH_logkernel_sum += log_kernel_curr

    # ---------------------------
    # speichern
    # ---------------------------
    aux["a"] = a_s
    aux["A"] = A_s
    aux["MH_logkernel"] = MH_logkernel_sum / M
    aux["MH_accept_rate"] = accepts / max(1, proposals)

    # ---------------------------
    # Sigma Update
    # ---------------------------
    Sigma = aux.get("Sigma", np.zeros((N, N, M)))

    for m in range(M):
        lam = np.asarray(aux["lambda_omega"])[:, m]
        if m > 0:
            lam = lam * np.asarray(aux["lambda_omega"])[:, 0]

        lam = np.clip(lam, 1e-12, None)

        try:
            Ainv = inv(A_s[m])
        except Exception:
            Ainv = inv(aux["A"][m])

        Sigma[:, :, m] = Ainv @ np.diag(lam) @ Ainv.T

    aux["Sigma"] = Sigma
    return aux