import numpy as np
from IPython.core.pylabtools import figsize
from numpy.linalg import inv, det, slogdet
from scipy.stats import gamma, multivariate_normal, dirichlet, chi2, multivariate_t
from scipy.linalg import eigh, null_space, solve
from scipy.special import gammaln, betaln
from itertools import permutations
from statsmodels.tsa.api import VAR
from models.general_functions import dratioig2
import matplotlib.pyplot as plt

def compute_sddr(qqq, priors):
    """
    Berechnet die Savage-Dickey Density Ratio (SDDR) für jedes Element A[i, j], i != j.
    Das Ziel ist die Bewertung, ob ein Struktureller Effekt null sein könnte (Modellvergleich).
    
    Parameter:
    - qqq: Dictionary mit "last_draws" uns "posteriors" aus Gibbs-Samples
    - priors: Dictionary mit Priorparametern
    
    Rückgabe:
    - Dictionary mit log(SDDR), log(Prior-Dichte), log(Posterior-Dichte)
    """

    aux = qqq["last_draws"]
    post = qqq["posteriors"]
    M = aux["PR_TR"].shape[0]
    p = aux["Y"].shape[0] - aux["U"].shape[0]
    T = aux["Y"].shape[0]
    TT = T - p
    N = aux["Y"].shape[1]
    S = post["alpha"].shape[2] # Anzahl der MCMC Ziehungen

    # 1) Log Prior Werte für jeden A[i, j]
    log_prior = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(N):
            if i != j:
                # M-facher Prior (für Vergleich über Regime hinweg)
                log_prior[i, j] = M * dratioig2(
                    x=1,
                    a1=priors["omega_a"],
                    a2=priors["omega_a"],
                    b1=priors["omega_b"],
                    b2=priors["omega_b"],
                    log=True
                )

    # 2) Posterior-Dichte vorbereiten
    posterior_omega_a = priors["omega_a"] + np.apply_along_axis(count_regime_occurrences, 0, post["S_t"])
    posterior_omega_b = np.zeros((N, M - 1, S)) # b-Werte für Omega im Posterior
    log_post_tmp = np.zeros((N, N, M - 1, S)) # Zwischenspeicher für log Posterior

    for s in range(S):
        for m in range(1, M): # m=1 bis M-1 
            # Beobachtungen im Regime m
            mask = post["S_t"][:, s] == m # Wenn 1-basiert, sonst: == m
            if np.sum(mask) < 2:
                continue # Zu wenig Beobachtungen
            
            U_m = post["U_SF"][mask, :, s] # Fehler im Regime m, Sample s
            for i in range(N):
                for j in range(N):
                    if i != j:
                        # Fehlervarianz für i und j in Regime m
                        var_i = np.var(U_m[:, i])
                        var_j = np.var(U_m[:, j])
                        lambda_i = post["lambda_omega"][i, 0, s]
                        lambda_j = post["lambda_omega"][j, 0, s]
                        
                        # Posterior-b-Parameter
                        b1 = priors["omega_b"] + (np.sum(mask) - 1) * var_i / lambda_i
                        b2 = priors["omega_b"] + (np.sum(mask) - 1) * var_j / lambda_j
                        posterior_omega_b[i, m - 1, s] = b1
                        
                        # Posterior-Dichteverhältnis berechnen
                        log_post_tmp[i, j, m - 1, s] = dratioig2(
                            x=1,
                            a1=posterior_omega_a[m, s],
                            a2=posterior_omega_a[m, s],
                            b1=b1,
                            b2=b2,
                            log=True
                        )

    # 3) Posterior Werte mitteln (logarithmisch)
    log_post = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(N):
            if i != j:
                lps = log_post_tmp[i, j, :, :].flatten()
                lps = lps[np.isfinite(lps)] # Nur gültige Werte
                if len(lps) > 0:
                    max_lps = np.max(lps)
                    # Stabilisiertes log-mean
                    log_post[i, j] = max_lps + np.log(np.sum(np.exp(lps - max_lps))) - np.log(len(lps))

    # 4) SDDR
    log_sddr = log_post - log_prior

    return {
        "log_sddr": log_sddr,
        "log_posterior": log_post,
        "log_prior": log_prior
    }
    
def count_regime_occurrences(S_t):
    """
    Zählt, wie oft jedes Regime im Vektor S_t vorkommt.
    
    Parameter:
    - S_t: Vektor mit Regime-Zuweisungen
    
    Rückgabe: 
    - Vektor [M], wobei jeder Eintrag die Anzahl der jeweiligen Regime enthält.
    """
    M = np.max(S_t) + 1
    return np.array([np.sum(S_t == m) for m in range(M)])

