import numpy as np
from numpy.linalg import lstsq
from scipy.stats import multivariate_normal


def to_prob(x):
    """Normiert einen Vektor zu Wahrscheinlichkeiten."""
    x = np.maximum(x, 1e-12)
    return x / np.sum(x)


def ergodic_prob(transition_matrix):
    """Berechnet stationäre Verteilung einer Markov-Kette."""
    M = transition_matrix.shape[0]
    A = np.vstack([np.eye(M) - transition_matrix.T, np.ones((1, M))])
    b = np.append(np.zeros(M), 1)
    pi = lstsq(A, b, rcond=None)[0]
    return to_prob(pi)


def filtering(aux):
    """Forward Filtering Schritt."""
    M = aux["PR_TR"].shape[0]
    Y = aux["Y"]
    U = aux["U"]
    T = Y.shape[0]
    TT = U.shape[0]
    K = Y.shape[1]

    eta = np.zeros((M, TT))
    xi = np.zeros((M, TT))

    # Likelihoods für jeden Zeitpunkt und jedes Regime
    for m in range(M):
        sigma_m = aux["Sigma"][:, :, m]
        mvn = multivariate_normal(mean=np.zeros(K), cov=sigma_m, allow_singular=True)
        for t in range(TT):
            eta[m, t] = max(mvn.pdf(U[t]), 1e-300)

    xi_tm1 = ergodic_prob(aux["PR_TR"])

    for t in range(TT):
        num = eta[:, t] * (aux["PR_TR"].T @ xi_tm1)
        den = np.sum(num)

        if not np.isfinite(den) or den <= 0:
            xi[:, t] = np.ones(M) / M
        else:
            xi[:, t] = num / den

        xi_tm1 = xi[:, t]

    return xi


def smoothing(aux, fp):
    """Backward Smoothing Schritt."""
    M, TT = fp.shape
    sp = np.zeros((M, TT))
    sp[:, -1] = to_prob(fp[:, -1])

    for t in range(TT - 2, -1, -1):
        denom = np.dot(aux["PR_TR"].T, fp[:, t])
        denom = np.maximum(denom, 1e-12)
        weight = sp[:, t + 1] / denom
        sp[:, t] = fp[:, t] * (aux["PR_TR"] @ weight)
        sp[:, t] = to_prob(sp[:, t])

    return sp


def count_regime_transitions(xi_sample):
    """Zählt Übergänge zwischen Regimen."""
    M, TT = xi_sample.shape
    path = np.argmax(xi_sample, axis=0)
    counts = np.zeros((M, M), dtype=int)
    for t in range(1, TT):
        counts[path[t - 1], path[t]] += 1
    return counts


def filtering_smoothing(aux, iterations=100, temper_exponent: float = 1.0, mix_uniform: float = 0.0,
                        min_self_transitions: int = 2, min_run_length: int = 1):
    """Sampling des Regimepfads mit Forward-Backward-Verfahren.

    Parameters
    ----------
    iterations : int
        Max. Anzahl Versuche für akzeptablen Pfad.
    temper_exponent : float
        Falls < 1.0: Weicht die geglätteten Wahrscheinlichkeiten auf
        (p_i <- p_i**temper_exponent, danach renormiert). Reduziert künstliche
        Deterministik.
    mix_uniform : float
        Optionaler Anteil (0..1) einer Gleichverteilung, der beigemischt wird:
        p <- (1-mix)p + mix(1/M). Hilft gegen komplette Kollapsierung.
    """
    M = aux["PR_TR"].shape[0]
    T = aux["Y"].shape[0]
    TT = aux["U"].shape[0]

    # Filtering und Smoothing
    fp = filtering(aux)
    sp = smoothing(aux, fp)
    # Tempering / Glättung
    if temper_exponent <= 0:
        temper_exponent = 1.0
    if temper_exponent != 1.0 or mix_uniform > 0.0:
        M = sp.shape[0]
        for t in range(sp.shape[1]):
            p = sp[:, t]
            if temper_exponent != 1.0:
                p = np.power(np.clip(p, 1e-15, 1.0), temper_exponent)
            if mix_uniform > 0.0:
                p = (1 - mix_uniform) * p + mix_uniform * (1.0 / M)
            sp[:, t] = to_prob(p)

    # Ziehe Startpunkt
    xi_TT = to_prob(sp[:, -1])
    xi_sample = np.zeros_like(sp)
    draw = np.random.choice(M, p=xi_TT)
    xi_sample[:, -1] = np.eye(M)[draw]

    # Rückwärtssampling
    # Mindestanzahl Selbstübergänge über alle Regime (min(diag(counts)))
    mnas = max(1, int(min_self_transitions))
    nas = 0
    iteration = 0

    while nas < mnas and iteration < iterations:
        for j in range(1, TT):
            t = TT - j
            denom = np.dot(aux["PR_TR"].T, fp[:, t - 1])
            denom = np.maximum(denom, 1e-12)
            weight = xi_sample[:, t] / denom
            xi_tmj = fp[:, t - 1] * (aux["PR_TR"] @ weight)
            xi_tmj = to_prob(xi_tmj)
            draw = np.random.choice(M, p=xi_tmj)
            xi_sample[:, t - 1] = np.eye(M)[draw]

        transitions = count_regime_transitions(xi_sample)
        nas = np.min(np.diag(transitions))
        iteration += 1

    if iteration < iterations:
        # Enforce Mindest-Lauflänge (z.B. keine Singletons)
        if min_run_length is None or min_run_length <= 1:
            aux["xi"] = xi_sample
        else:
            # Hilfsfunktion zum Glätten kurzer Läufe
            def _enforce_min_run(onehot: np.ndarray, probs: np.ndarray, min_run: int) -> np.ndarray:
                path = np.argmax(onehot, axis=0)  # (TT,)
                Tloc = path.shape[0]
                changed = True
                passes = 0
                while changed and passes < 3:  # wenige Durchläufe reichen
                    changed = False
                    i = 0
                    while i < Tloc:
                        j = i
                        while j + 1 < Tloc and path[j + 1] == path[i]:
                            j += 1
                        run_len = j - i + 1
                        if run_len < min_run:
                            # Kandidaten: vorheriges oder folgendes Regime
                            prev_state = path[i - 1] if i - 1 >= 0 else None
                            next_state = path[j + 1] if j + 1 < Tloc else None
                            choose = None
                            if prev_state is not None and next_state is not None and probs is not None:
                                seg = slice(i, j + 1)
                                # wähle den Nachbarn mit höherer durchschnittlicher Wahrscheinlichkeit in diesem Segment
                                prev_score = probs[seg, prev_state].mean()
                                next_score = probs[seg, next_state].mean()
                                choose = prev_state if prev_score >= next_score else next_state
                            elif prev_state is not None:
                                choose = prev_state
                            elif next_state is not None:
                                choose = next_state
                            else:
                                choose = path[i]
                            path[i:j + 1] = choose
                            changed = True
                            i = j + 1
                        else:
                            i = j + 1
                    passes += 1
                # zurück zu One-Hot
                return np.eye(M, dtype=float)[:, path]

            # Verwende smoothing-Probs sp^T, dimension (M,TT) -> (TT,M)
            sp_T = sp.T if 'sp' in locals() else None
            aux["xi"] = _enforce_min_run(xi_sample, sp_T, int(min_run_length))

    # Zusätzlich: speichere die smoothing-Wahrscheinlichkeiten (weiche probs)
    aux["sp"] = sp  # shape (M, TT)

    return aux