import numpy as np

def get_restrictions(N, M):
    """
    M = Anzahl der Regime
    N = Anzahl der Assetklassen

    Erstellt die Strukturrestriktionen für eine untere Dreicksmatrix A0.
    
    Rückgabe:
    - Q: Matrix mit 1en an den Stellen, die frei sind.
    - q: Vektor mit N^2 Einträgen
    - dj: Liste mit Anzahl freier Parameter je Zeile
    - M: Strukturmatrix für Übergangsmatrix-Parametrisierung (Dirichlet Zeichnung)
    """

    q = np.zeros((N * N, 1)) # Rechte Seite der Restriktionen
    Q_elements = [] # Liste zur Konstruktion von Q
    dj_A0 = [] # Anzahl freier Elemente pro Zeile in A0

    count = 0 # Zählindex für A0 Vektorisierung
    for i in range(N):
        for j in range(N):
            if j < i: # Untere Dreiecksmatrix: j <= i -> frei
                q[count] = 0
                q_row = np.zeros(N * N)
                q_row[count] = 1 # Markiere die Position als frei
                Q_elements.append(q_row)
            # ansonsten bleibt das Element fix auf 0
            elif j == i:
                q[count] = 1
            count += 1
        dj_A0.append(i) # dj = Anzahl der freien Parameter in Zeile i

    # Q hat eine Spalte pro freiem Parameter
    Q = np.array(Q_elements).T # [N^2 x r]
    
    dj_transitions =[M] * M # Jede Zeile hat M freie Übergänge
    
    PR_M = build_M_matrix(dj_transitions)
    return {
        "Q": Q,
        "q": q,
        "dj": dj_A0,
        "PR_M": PR_M # Matrixstruktur für PR_TR Zeichnung
    }

def build_M_matrix(dj):
    """
    Erstellt die M-Matrix für die Ziehung der Übergangsmatrix PR_TR aus der Dirichlet Verteilung.
    
    - dj: Liste mit Anzahl freier Übergangsgewichte je Zeile
    
    Rückgabe:
    - M: Strukturmatrix [d x d], wobei d = sum(dj)
    """
    M = len(dj)
    d = sum(dj) 
    PR_M = np.zeros((M * M, d))
    col_idx = 0
    
    for row in range(M):
        for i in range(dj[row]):
            pos = row * M + i
            PR_M[pos, col_idx] = 1
            col_idx += 1
    return PR_M