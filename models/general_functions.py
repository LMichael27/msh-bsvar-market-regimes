import numpy as np
from numpy.linalg import inv, det, slogdet
from scipy.stats import gamma, multivariate_normal, dirichlet, chi2, multivariate_t
from scipy.linalg import eigh, null_space, solve
from scipy.special import gammaln, betaln
from itertools import permutations
from statsmodels.tsa.api import VAR

# Inverse Gamma Dichtefunktion (genutzt für Varianzschätzungen)
def dig2(x, a, b, log=False):
    """
    Dichte der Inversen-Gamma-Verteilung mit Parametern a, b an Punkt x.
    """
    x = np.maximum(x, 1e-12)
    log_c = gamma.logpdf(a / 2, a / 2, scale=2 / b) # log-Normalisierungskonstante
    log_k = (-(a + 2) / 2) * np.log(x) - 0.5 * (b / x) # log-Kern der Dichte (abhängig von x)
    result = log_k - log_c
    return result if log else np.exp(result)

# Dichte der multivariaten t-Verteilung für MCMC
def dt2(x, mu, precision, a, b, log=False):
    """
    Multivariate t-Dichte für gegebenen Punkt x, Mittelwert mu, Präzision und Hyperparametern a, b.
    """
    N = len(x) # Dimension des Vektors x
    term1 = gamma.logpdf(a/2, a/2) - gamma.logpdf((a + N) / 2, (a + N) / 2) # Teil des Normierungsfaktor der t-Dichte
    term2 = (N / 2) * np.log(np.pi) # Bestandteil des T-Normalisierungsfaktor
    #term3 = -0.5 * np.linalg.slogdet(precision)[1] # Log Determinante der Präzisionsmatrix
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        logdet = np.log(np.linalg.det(precision + 1e-6 * np.eye(len(x))))
    term3 = -0.5 * logdet
    term4 = - (a / 2) * np.log(b) # Skalierung des Inverse Gamma Anteils
    
    # Quadratische Abweichung vom Mittelwert
    residual = x - mu 
    quad = residual.T @ precision @ residual 
    
    # t-Verteilungskern
    term5 = - (a + N) / 2 * np.log(b + quad)
    
    # Gesamtlogdichte
    result = term5 - (term1 + term2 + term3 + term4) 
    return result if log else np.exp(result)

# Erzeugt alle Permutationen von 1 bis n
def all_permutations(n):
    return np.array(list(permutations(range(1, n + 1))))

# Orthogonalkomplement einer Matrix
def orthogonal_complement(X):
    Q, _ = np.linalg.qr(X, mode='complete') # QR Zerlegung für Basis
    return Q[:, X.shape[1]:] # Nicht durch X erzeugte Basisvektoren

# Matrix Wurzel
def matrix_sqrt(A, inverse=False):
    """
    Gibt die Matrixwurzel von A zurück.
    Wenn inverse=True, wird die Inverse der Matrixwurzel berechnet.
    """
    vals, vecs = eigh(A) # Eigenwerte und Eigenvektoren (symmetrisch)
    sqrt_vals = np.sqrt(vals) # Quadratwurzeln der Eigenwerte
    if inverse:
        sqrt_vals = 1 / sqrt_vals # Inverse der Wurzeln (nur falls nötig)
    Λ = np.diag(sqrt_vals)
    return vecs @ Λ @ vecs.T # A^(1/2) = V * D^(1/2) * V'

# Nullraum bereinigen (nützlich für Identifikationsbedingungen)
def null_tw(X):
    """
    Berechnet und sortiert den Nullraum von X nach Spaltenstruktur.
    """
    nullspace = null_space(X)
    abs_null = np.abs(nullspace)
    
    # Sortierung nach der Position der ersten 1 pro Spalte
    index = np.argsort([np.argmax(col == 1) for col in abs_null.T])
    return abs_null[:, index]

# Verhältnis zweier inverse-Gamma-Dichten 
def dratioig2(x, a1, a2, b1, b2, log=False):
    """
    Verhältnis (Dichtequotient) zweier inverse-Gamma-Dichten an Punkt x.
    """
    x = np.maximum(x, 1e-12) # Schutz 
    lcr = (a1 / 2) * np.log(b1) + (a2 / 2) * np.log(b2) - betaln(a1 / 2, a2 / 2) # log - Koeffizientenanteil
    lkr = ((a2 - 2) / 2) * np.log(x) - ((a1 + a2) / 2) * np.log(b1 + b2 * x) # log - Kernanteil der Verteilung
    result = lcr + lkr
    return result if log else np.exp(result)

def in_range(draw, lower, upper):
    """
    Überprüft, ob alle Werte im Vektor "draw" innerhalb der definierten Schranken (lower, upper) liegen.
    """
    return np.all((draw > lower) & (draw < upper))