import numpy as np
from numpy.linalg import inv, det, slogdet
from scipy.stats import gamma, multivariate_normal, dirichlet, chi2, multivariate_t
from scipy.linalg import eigh, null_space, solve
from scipy.special import gammaln, betaln
from itertools import permutations
from statsmodels.tsa.api import VAR

from models.regression import regression_step
from models.structrual_mh import structural_mh
from models.update_hyper import update_inverted_gamma, update_hyperparameters
from models.hidden_markov import update_hidden_markov_chain
from models.filtering_smoothing import filtering_smoothing


def run_gibbs_sampler(S, priors, restrictions, starting_values,
                      nu=5, C=1, debug=False,
                      filtering_iterations=50,
                      print_iterations=100,
                      burnin=0,
                      temper_exponent: float = 1.0,
                      mix_uniform: float = 0.0,
                      min_self_transitions: int = 2,
                      min_run_length: int = 2,
                      clip_transition_diag: float = 0.95,
                      clip_transition_diag_iter: float = None,
                      blend_uniform_tr: float = 0.0,
                      min_offdiag: float = 0.0,
                      enforce_diag_dominance: bool = True,
                      diag_margin: float = 0.01):
    """
    Führt den Gibbs-Sampler für das MSH-BSVAR Modell mit regimespezifischen VAR-Koeffizienten aus.

    Parameter:
    - S: Anzahl an Ziehungen (nach Burn-in)
    - burnin: Anzahl der Burn-in-Iterationen
    - nu, C: Parameter für die t-Vorschlagsverteilung im MH-Schritt
    """
    total_iterations = S + burnin

    # Initialisierung
    aux = starting_values
    Y = aux["Y"]
    N = Y.shape[1]
    T = Y.shape[0]
    TT = aux["U"].shape[0]
    p = T - TT
    M = aux["PR_TR"].shape[0]
    d = restrictions["PR_M"].shape[1]
    r = restrictions["Q"].shape[1]
    K = 1 + p * N

    # Speicher für Posterior-Ziehungen (nur nach Burn-in!)
    posteriors = {
        "w": np.zeros((d, S)),
        "Sigma": np.zeros((N, N, M, S)),
        "lambda_omega": np.zeros((N, M, S)),
        "A": np.zeros((M, N, N, S)),
        "a": np.zeros((M, r, S)),
        "alpha": np.zeros((K, N, M, S)),  # NEU: regimespezifisch!
        "PR_TR": np.zeros((M, M, S)),
        "hyper": np.zeros((3, S)),
        "S_t": np.zeros((TT, S), dtype=int),
        "U_SF": np.zeros((*aux["U_SF"].shape, S)),
        "MH_logkernel": np.zeros(S)
    }

    # Laufende Summe der geglätteten (smoothing) Wahrscheinlichkeiten für jeden Zeitpunkt
    S_prob_sum = np.zeros((TT, M))  # wir akkumulieren sp.T

    # Optionale Abmilderung zu persistenter Start-Übergangsmatrix
    if clip_transition_diag is not None:
        PR = aux["PR_TR"].copy()
        diag = np.diag(PR).copy()
        if np.any(diag > clip_transition_diag):
            for i in range(PR.shape[0]):
                if PR[i, i] > clip_transition_diag:
                    excess = PR[i, i] - clip_transition_diag
                    PR[i, i] = clip_transition_diag
                    off_mask = np.ones(PR.shape[1], dtype=bool); off_mask[i] = False
                    PR[i, off_mask] += excess / off_mask.sum()
            # Renormieren
            PR = np.clip(PR, 1e-12, None)
            PR /= PR.sum(axis=1, keepdims=True)
            aux["PR_TR"] = PR

    def _enforce_diag_dom(PR: np.ndarray, margin: float = 0.0) -> np.ndarray:
        if not enforce_diag_dominance:
            return PR
        Mloc = PR.shape[0]
        out = PR.copy().astype(float)
        out = np.clip(out, 1e-12, None)
        for i in range(Mloc):
            row = out[i, :]
            off_mask = np.ones(Mloc, dtype=bool); off_mask[i] = False
            off_max = float(row[off_mask].max()) if off_mask.any() else 0.0
            desired_diag = max(float(row[i]), off_max + margin)
            if desired_diag >= 1.0:
                desired_diag = 1.0 - 1e-9
            sum_off = float(row[off_mask].sum())
            if sum_off <= 0.0:
                # set uniform small mass to off-diagonals
                row[i] = desired_diag
                rem = max(1.0 - row[i], 0.0)
                row[off_mask] = rem / max(Mloc - 1, 1)
            else:
                row[i] = desired_diag
                rem = max(1.0 - row[i], 1e-12)
                scale = rem / sum_off
                row[off_mask] *= scale
            # normalize for safety
            row = np.clip(row, 1e-12, None)
            row /= row.sum()
            out[i, :] = row
        return out

    # Erzwinge ggf. Diagonaldominanz vor dem ersten Filtering
    aux["PR_TR"] = _enforce_diag_dom(aux["PR_TR"], diag_margin)

    for s in range(total_iterations):
        if debug:
            print("Filtering and Smoothing Step")
        aux = filtering_smoothing(
            aux,
            iterations=filtering_iterations,
            temper_exponent=temper_exponent,
            mix_uniform=mix_uniform,
            min_self_transitions=min_self_transitions,
            min_run_length=min_run_length,
        )

        if debug:
            print("Hidden Markov Chain Step")
        aux = update_hidden_markov_chain(aux, priors, restrictions)
        # Iteratives Begrenzen der Diagonale (Persistenz kappen)
        if clip_transition_diag_iter is not None:
            PRi = aux["PR_TR"]
            diag = np.diag(PRi)
            if np.any(diag > clip_transition_diag_iter):
                for i in range(PRi.shape[0]):
                    if PRi[i, i] > clip_transition_diag_iter:
                        excess = PRi[i, i] - clip_transition_diag_iter
                        PRi[i, i] = clip_transition_diag_iter
                        off_mask = np.ones(PRi.shape[1], dtype=bool); off_mask[i] = False
                        PRi[i, off_mask] += excess / off_mask.sum()
                PRi = np.clip(PRi, 1e-12, None)
                PRi /= PRi.sum(axis=1, keepdims=True)
                aux["PR_TR"] = PRi
        # Uniform-Blending zur Entschärfung extremer Persistenz
        if blend_uniform_tr > 0.0:
            Mloc = aux["PR_TR"].shape[0]
            aux["PR_TR"] = (1 - blend_uniform_tr) * aux["PR_TR"] + blend_uniform_tr * (np.ones((Mloc, Mloc)) / Mloc)
            aux["PR_TR"] /= aux["PR_TR"].sum(axis=1, keepdims=True)
        # Mindest-Off-Diagonale erzwingen
        if min_offdiag > 0.0:
            PRm = aux["PR_TR"]
            Mloc = PRm.shape[0]
            for i in range(Mloc):
                for j in range(Mloc):
                    if i != j and PRm[i, j] < min_offdiag:
                        diff = min_offdiag - PRm[i, j]
                        PRm[i, j] += diff
                        PRm[i, i] -= diff  # von Diagonale abziehen
                # Falls Diagonale negativ geworden: renormieren mild
                if PRm[i, i] < 1e-6:
                    PRm[i, :] = np.clip(PRm[i, :], 1e-12, None)
                    PRm[i, :] /= PRm[i, :].sum()
            aux["PR_TR"] = PRm

        # Nach allen Anpassungen: Diagonaldominanz sicherstellen
        aux["PR_TR"] = _enforce_diag_dom(aux["PR_TR"], diag_margin)

        if debug:
            print("Hyper-Parameters Step")
        aux = update_hyperparameters(aux, priors, restrictions)

        if debug:
            print("Inverted-Gamma Step")
        aux = update_inverted_gamma(aux, priors)

        if debug:
            print("Regression Step")
        aux = regression_step(aux, priors)

        if debug:
            print("Structural Step")
        aux = structural_mh(aux, priors, restrictions, nu=nu, C=C)

        # Nur speichern, wenn Burn-in vorbei ist
        if s >= burnin:
            index = s - burnin  # Index in Posterior-Speicher
            posteriors["lambda_omega"][:, :, index] = aux["lambda_omega"]
            posteriors["PR_TR"][:, :, index] = aux["PR_TR"]
            posteriors["hyper"][:, index] = aux["hyper"].flatten()
            posteriors["w"][:, index] = aux["w"].flatten()
            posteriors["S_t"][:, index] = np.argmax(aux["xi"], axis=0)
            # Smoothing-Probs akkumulieren (falls vorhanden)
            if "sp" in aux:
                # aux["sp"]: (M, TT) -> transpose zu (TT, M)
                S_prob_sum += aux["sp"].T
            posteriors["U_SF"][:, :, index] = aux["U_SF"]
            posteriors["MH_logkernel"][index] = aux["MH_logkernel"]

            for m in range(M):
                posteriors["Sigma"][:, :, m, index] = aux["Sigma"][:, :, m]
                posteriors["A"][m, :, :, index] = aux["A"][m]
                posteriors["a"][m, :, index] = aux["a"][m].flatten()
                posteriors["alpha"][:, :, m, index] = aux["alpha"][:, :, m]

        if s % print_iterations == 0:
            iter_num = s + 1
            if debug:
                print(f"Iteration {iter_num}/{total_iterations}")
            else:
                print(" ", iter_num, end="")

    # Gemittelte smoothing-Wahrscheinlichkeiten (TT, M)
    if np.any(S_prob_sum):
        S_prob = S_prob_sum / float(S)  # (TT, M) rows sum ~1
        # Degeneracy-Check: zu deterministisch?
        row_max = S_prob.max(axis=1)
        degeneracy_ratio = np.mean(row_max > 0.999)
        # Empirische Häufigkeiten aus den gespeicherten harten Pfaden
        S_emp_freq = np.zeros_like(S_prob)
        for s_idx in range(S):
            path = posteriors["S_t"][:, s_idx]
            for t_idx, m_state in enumerate(path):
                S_emp_freq[t_idx, m_state] += 1
        S_emp_freq /= float(S)
        # Glätten (numerische Stabilität)
        eps = 1e-6
        S_emp_freq = S_emp_freq + eps
        S_emp_freq /= S_emp_freq.sum(axis=1, keepdims=True)
        if degeneracy_ratio > 0.8:  # >80% der Zeitpunkte fast deterministisch
            # Nutze empirische Häufigkeiten statt geglätteter sp-Wahrscheinlichkeiten
            S_prob = S_emp_freq
        else:
            # Mische beide Ansätze leicht zur Robustheit
            S_prob = 0.5 * S_prob + 0.5 * S_emp_freq
            S_prob /= S_prob.sum(axis=1, keepdims=True)
        posteriors["S_prob"] = S_prob
    else:
        posteriors["S_prob"] = None

    return {
        "last_draws": aux,
        "posteriors": posteriors
    }