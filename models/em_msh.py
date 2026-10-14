import numpy as np
from scipy.stats import dirichlet


def em_msh(U, M):
    """
    Einfache Initialisierung für ein Markov Switching Heteroskedasticity Modell.

    Inputs:
    - U: Residuenmatrix mit T Beobachtungen und N Variablen (T x N)
    - M: Anzahl der Regime

    Outputs:
    - Dictionary mit:
        - Sigma: Kovarianzmatrizen pro Regime (N x N x M)
        - PR_TR: Übergangsmatrix (M x M) mit gleicher Wahrscheinlichkeit
        - xi: Liste der anfänglichen Regime-Zuweisungen (Länge T)
    """
    T, N = U.shape  # T: Anzahl der Zeitpunkte, N: Anzahl der Variablen
    TT = T

    # 1) Starte mit einer festen Abfolge von Regime-Zuweisungen:
    # Wiederhole die Werte von 0 bis M-1 bis zur Länge T
    xi_vec = np.random.choice(M, size=TT)

    # Erstelle eine one-hot-codierte Matrix der Regimezugehörigkeit (M x T)
    xi = np.random.dirichlet(np.ones(M), size=TT).T
    for t, m in enumerate(xi_vec):
        xi[m, t] = 1

    # 2) Schätze erste Varianzen gleich über alle Regime
    U_var = np.var(U, axis=0)  # Varianz jeder Variable (N)
    Sigma = np.zeros((N, N, M))  # Matrix mit Platz für M Varianz-Kovarianzmatrizen
    for m in range(M):
        Sigma[:, :, m] = np.diag(U_var)  # Diagonalmatrix

    # 3) Initialisiere Übergangsmatrix mit gleicher Wahrscheinlichkeit
    PR_TR = np.full((M, M), 0.05 / (M - 1))
    np.fill_diagonal(PR_TR, 0.95)

    # Rückgabe als Dictionary
    return {
        "Sigma": Sigma,
        "PR_TR": PR_TR,
        "Xi": xi_vec,
        "xi": xi  # One Hot Matrix
    }


def em_msh_stable(U, M, dirichlet_concentration=1.0, diagonal_value=0.95, seed=None, verbose=False):
    """
    Stabile Initialisierung für ein Markov Switching Heteroskedasticity Modell (MSH-BSVAR).

    Inputs:
    - U: Residuenmatrix mit T Beobachtungen und N Variablen (T x N)
    - M: Anzahl der Regime
    - dirichlet_concentration: Konzentrationsparameter für weichere xi-Verteilung (default: 1.0)
    - diagonal_value: Selbstübergangswahrscheinlichkeit in PR_TR (default: 0.95)
    - seed: Optionaler Zufallsseed
    - verbose: Ob Initialwerte geloggt werden sollen (default: False)

    Outputs:
    - Dictionary mit:
        - Sigma: Initiale Kovarianzmatrizen pro Regime (N x N x M)
        - PR_TR: Übergangsmatrix (M x M)
        - xi: Float-basierte, weiche Regime-Zuweisungsmatrix (T x M)
        - xi_vec: Indexvektor mit Max-Zuordnung (T,)
    """
    if seed is not None:
        np.random.seed(seed)

    T, N = U.shape
    TT = T

    # 1) Weiche Initialisierung der Regime-Wahrscheinlichkeiten (Dirichlet)
    xi = np.random.dirichlet([dirichlet_concentration] * M, size=TT)  # Shape: (TT, M)
    xi_vec = np.argmax(xi, axis=1)  # Für Referenz: harteste Komponente

    # 2) Kovarianzmatrizen initialisieren: Diagonal aus Residualvarianzen
    U_var = np.var(U, axis=0)  # N-dim Vektor
    Sigma = np.zeros((N, N, M))
    for m in range(M):
        Sigma[:, :, m] = np.diag(U_var)

    # 3) Übergangsmatrix PR_TR: stark diagonal dominiert
    if M == 1:
        PR_TR = np.array([[1.0]])
    else:
        off_diag_value = (1.0 - diagonal_value) / (M - 1)
        PR_TR = np.full((M, M), off_diag_value)
        np.fill_diagonal(PR_TR, diagonal_value)

    # 4) Optionales Logging
    if verbose:
        print("[em_msh_stable] Initialisierte xi-Proben (erste 5):\n", xi[:5])
        print("[em_msh_stable] Initialisierte PR_TR:\n", PR_TR)
        print("[em_msh_stable] Varianz der Residuen:", U_var)

    return {
        "Sigma": Sigma,
        "PR_TR": PR_TR,
        "xi": xi,  # weich
        "xi_vec": xi_vec  # optional für Tests/Visualisierung
    }