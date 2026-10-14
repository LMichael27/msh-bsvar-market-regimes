import numpy as np
from numpy.linalg import inv, det, slogdet
from scipy.stats import gamma, multivariate_normal, dirichlet, chi2, multivariate_t
from scipy.linalg import eigh, null_space, solve
from scipy.special import gammaln, betaln
from itertools import permutations
from statsmodels.tsa.api import VAR
from models.em_msh import em_msh

def initialize_aux(Y, p, M, restrictions, which_sigma=0):
    """
    Initialisiert alle Startwerte für den MSH-BSVAR Gibbs-Sampler.

    Parameter:
    - Y: Datenmatrix (T x N)
    - p: Anzahl Lags im VAR
    - M: Anzahl der Regime
    - restrictions: Dictionary mit Restrioktionsmatrix ("Q") und freie Parameteranzahl ("q", "dj")
    - which_sigma: Index der führenden Serie für die Regimesortierung

    Rückgabe:
    - aux: Dictionary mit allen Startparametern für den MCMC
    """
    import warnings
    data = np.asarray(Y)
    T, N = data.shape
    TT = T - p
    r = restrictions["Q"].shape[1]
    d_tr = restrictions["PR_M"].shape[1]
    w = np.ones((d_tr, 1)) * 9

    # 1) Robust VAR-Schätzung
    model = VAR(data)
    try:
        var_result = model.fit(p)
    except Exception as e:
        raise RuntimeError(f"⚠️ VAR-Schätzung fehlgeschlagen: {e}\n"
                           f"T={T}, N={N}, p={p}. Versuche z.B. p=1 oder reduziere N.")

    U_temp = var_result.resid
    if np.isnan(U_temp).any():
        raise ValueError("⚠️ VAR-Residuen enthalten NaNs. Prüfe Daten oder Lags.")

    # 2) EM zur Initialisierung
    em_output = em_msh(U_temp, M=M)
    Sigma_temp = em_output["Sigma"]
    order = np.argsort(Sigma_temp[which_sigma, which_sigma, :])
    Sigma_temp = Sigma_temp[:, :, order]

    # 3) Lambda-Omega
    lambda_omega = np.full((N, M), np.nan)
    lambda_omega[:, 0] = np.diag(Sigma_temp[:, :, 0])
    for m in range(1, M):
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.diag(Sigma_temp[:, :, m]) / np.diag(Sigma_temp[:, :, 0])
            lambda_omega[:, m] = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)

    # 4) Strukturmatrizen A & a
    a_s = np.zeros((M, r, 1))
    A_s = np.zeros((M, N, N))
    for m in range(M):
        a_m = np.random.normal(scale=0.2, size=(r, 1))
        A = (restrictions["Q"] @ a_m + restrictions["q"]).reshape(N, N)
        if np.linalg.matrix_rank(A) < N:
            warnings.warn(f"Strukturmatrix A in Regime {m} ist nicht invertierbar.")
            A += np.eye(N) * 1e-4  # leichte Regularisierung
        A_s[m] = A
        a_s[m] = a_m

    # 5) Strukturierte Kovarianzen
    Sigma = np.zeros_like(Sigma_temp)
    for m in range(M):
        D_m = np.diag(np.diag(Sigma_temp[:, :, m]))
        try:
            A_inv = np.linalg.inv(A_s[m])
        except np.linalg.LinAlgError:
            warnings.warn(f"Inversion von A in Regime {m} fehlgeschlagen – Regularisierung verwendet.")
            A_inv = np.linalg.pinv(A_s[m])
        Sigma[:, :, m] = A_inv @ D_m @ A_inv.T

    # 6) VAR-Koeffizienten
    alpha = var_result.coefs.reshape(-1, N)
    intercept = var_result.intercept.reshape(1, -1)
    alpha = np.vstack([intercept, alpha])

    # 7) Designmatrix X
    X = np.ones((TT, 1))
    for lag in range(1, p + 1):
        X = np.hstack([X, data[(p - lag):(T - lag), :]])
    Y_cut = data[p:, :]

    # 8) Regime-Pfad und Fehler
    PR_TR = em_output["PR_TR"]
    Xi_vector = em_output["Xi"]
    xi_matrix = np.eye(M)[:, Xi_vector]
    xi = xi_matrix[order, :]

    U_SF = np.zeros((TT, N))
    U = np.zeros((TT, N))
    X_full = X @ alpha
    for m in range(M):
        indices = np.where(xi[m, :] == 1)[0]
        A_m = A_s[m]
        if len(indices) > 0:
            U_SF[indices] = Y_cut[indices] @ A_m.T - X_full[indices]
            try:
                U[indices] = solve(A_m.T, U_SF[indices].T).T
            except np.linalg.LinAlgError:
                warnings.warn(f"Solve in Regime {m} instabil – Pseudoinverse verwendet.")
                U[indices] = np.linalg.pinv(A_m.T) @ U_SF[indices].T
                U[indices] = U[indices].T

    # 9) Hyperparameter
    hyper = np.ones((3, 1))
    hyper[1, 0] = np.var(alpha[0, :])
    hyper[2, 0] = np.var(alpha[1:, :])

    aux = {
        "Y": data,
        "U": U,
        "U_SF": U_SF,
        "Sigma": Sigma,
        "lambda_omega": lambda_omega,
        "A": A_s,
        "a": a_s,
        "alpha": alpha,
        "PR_TR": PR_TR,
        "w": w,
        "hyper": hyper,
        "xi": xi,
        "MH_logkernel": -np.inf
    }
    return aux