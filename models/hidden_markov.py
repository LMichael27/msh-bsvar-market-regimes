import numpy as np
from numpy.linalg import inv, det, slogdet
from scipy.stats import gamma, multivariate_normal, dirichlet, chi2, multivariate_t
from scipy.linalg import eigh, null_space, solve
from scipy.special import gammaln, betaln
from itertools import permutations
from statsmodels.tsa.api import VAR
from models.filtering_smoothing import ergodic_prob


def update_hidden_markov_chain(aux, priors, restrictions):
    """
    Aktualisiert die Übergangsmatrix PR_TR einer Markov-Kette:
    - nutzt gezählte Regimewechsel aus xi,
    - zieht neue Wahrscheinlichkeiten aus einer Dirichlet-Verteilung (Gibbs),
    - akzeptiert oder verwirft mit Metropolis-Hastings.
    """
    Y = aux["Y"]  # beobachtete Zeitreihe
    U = aux["U"]  # Residuen
    M = aux["PR_TR"].shape[0]  # Anzahl der Regime
    T, N = Y.shape  # T: Beobachtungen, N: Variablen
    TT = U.shape[0]  # Anzahl der Zeitpunkte ohne Lags
    p = T - TT  # Anzahl der Lags im VAR Modell
    r = aux["alpha"].shape[0]  # Länge des Alpha Vektors für die Strukturelle Identifikation
    d = restrictions["PR_M"].shape[1]  # Gesamtanzahl freier Übergangsgewichte
    Q = len(restrictions["dj"])  # Anzahl von Blöcken in Restriktionsmatrix

    PR_TR_old = aux["PR_TR"]

    # 1) Übergänge zählen basierend auf aktuellem xi
    transitions = count_regime_transitions(aux["xi"])  # [M x M]
    transitions_vector = transitions.flatten().reshape(-1, 1)  # [M^2 x 1]

    # 2) Neue Dirichlet-Gewichte ziehen (blockweise)
    w_new = np.zeros((d, 1))
    wj_transitions = np.zeros((d, 1))

    for j in range(Q):
        dj = restrictions["dj"][j]
        if dj == 0:
            continue
        idx_start = sum(restrictions["dj"][:j])
        idx_end = idx_start + dj

        # Übergangssummen für aktuellen Block
        for i in range(d):
            wj_transitions[i] = restrictions["PR_M"][:, i].reshape(1,
                                                                   -1) @ transitions_vector  # Gewichtete Übergangssumme für Position i

            # Ziehe neue Gewichte aus Dirichlet-Verteilung für Block j
            alpha = wj_transitions[idx_start:idx_end] + priors["w"][idx_start:idx_end]

            if alpha.size == 0 or np.any(alpha <= 0):
                continue
            w_new[idx_start:idx_end] = dirichlet.rvs(alpha.flatten())[0].reshape(-1, 1)

    # 3) Neue Übergangsmatrix zusammensetzen aus Restriktionsstruktur
    PR_TR_new = (restrictions["PR_M"] @ w_new).reshape((M, M))

    # 4) Stationäre Verteilung berechnen
    ergodic_old = ergodic_prob(PR_TR_old)
    ergodic_old = np.clip(ergodic_old, 0, 1)
    ergodic_old = ergodic_old / np.sum(ergodic_old)
    ergodic_new = ergodic_prob(PR_TR_new)
    ergodic_new = np.clip(ergodic_new, 0, 1)
    ergodic_new = ergodic_new / np.sum(ergodic_new)

    print("ergodic_new:", ergodic_new)
    print("sum:", np.sum(ergodic_new))
    # 5) Metropolis Hastings Schritt
    # Zufälliger Startstatus, aus alter und neuer Verteilung
    s0_old = np.random.choice(M, p=ergodic_old)
    s0_new = np.random.choice(M, p=ergodic_new)

    # Berechne Akzeptanzwahrscheinlichkeit
    AA = ergodic_new[s0_old] / ergodic_old[s0_old]  # Vergleichen hier die Wahrscheinlichkeit für denselben Startzustand

    # 6) Akzeptiere oder verwerfe neue Matrix
    if np.random.rand() <= AA:
        aux["PR_TR"] = PR_TR_new
        expected_d = restrictions["PR_M"].shape[1]
        if w_new.shape[0] != expected_d:
            w_new = w_new[:expected_d, :]  # nur die ersten d ELemente behalten
            aux["w"] = w_new
    else:
        aux["PR_TR"] = PR_TR_old
    return aux


def count_regime_transitions(xi):
    """
    Zählt Übergänge zwischen Regimen auf Basis der xi-Matrix.
    Gibt eine [M x M] Matrix zurück mit Übergangshäufigkeiten.
    """
    M, TT = xi.shape
    path = np.argmax(xi, axis=0)  # Hole Pfad aus One-Hot Encoding
    count = np.zeros((M, M), dtype=int)

    for t in range(1, TT):
        prev_regime = path[t - 1]
        curr_regime = path[t]
        count[prev_regime, curr_regime] += 1

    return count