import numpy as np
from numpy.linalg import LinAlgError
from scipy.stats import multivariate_normal, dirichlet
from scipy.special import logsumexp
from models.general_functions import dt2, dig2  # in_range wird nicht mehr benötigt
from models.filtering_smoothing import ergodic_prob


# ---------------------------------------------------------------------------
# Hilfsfunktionen: Slicing, Transformationen, Jacobian
# ---------------------------------------------------------------------------

def _slices(M: int, N: int, r: int, p: int):
    """
    Liefert nützliche Slices für den (flachen) Parametervektor θ im X-Raum.
    Layout:
      [ lambda_omega | a | alpha | PR_TR(first M-1 per row) | hyper ]
    """
    K = 1 + N * p
    n_lambda = M * N
    n_a      = M * r
    n_alpha  = M * K * N
    n_pr     = M * (M - 1)
    n_hyper  = 3

    i0 = 0
    i1 = i0 + n_lambda
    i2 = i1 + n_a
    i3 = i2 + n_alpha
    i4 = i3 + n_pr
    i5 = i4 + n_hyper

    return (
        K,
        slice(i0, i1),  # s_lam
        slice(i1, i2),  # s_a
        slice(i2, i3),  # s_alpha
        slice(i3, i4),  # s_pr (nur M-1 pro Zeile)
        slice(i4, i5),  # s_hyper
    )


def _x_to_z_row(x: np.ndarray, M: int, N: int, r: int, p: int) -> np.ndarray:
    """
    Mappt einen validen θ-Vektor (X-Raum) -> z (unbeschränkter Z-Raum).
    - positive Blöcke: log
    - Simplex-Reihen (PR_TR): Logits vs. letzte Kategorie
    - Unbeschränktes bleibt unverändert
    """
    K, s_lam, s_a, s_alpha, s_pr, s_h = _slices(M, N, r, p)
    z = x.astype(np.float64).copy()

    # positive -> log
    z[s_lam] = np.log(np.clip(x[s_lam], 1e-12, None))
    z[s_h]   = np.log(np.clip(x[s_h],   1e-12, None))

    # simplex -> logits (gegen letzte Kategorie)
    z_pr = []
    off = 0
    x_pr = x[s_pr]
    for _ in range(M):
        prv = x_pr[off:off+(M-1)]
        # letzte Kategorie:
        pM = max(1e-12, 1.0 - float(np.sum(prv)))
        prv = np.clip(prv, 1e-12, 1.0 - 1e-12)
        z_row = np.log(prv / pM)
        z_pr.append(z_row)
        off += (M - 1)
    z[s_pr] = np.concatenate(z_pr)
    return z


def _z_to_x_row(z: np.ndarray, M: int, N: int, r: int, p: int) -> np.ndarray:
    """
    Mappt z (unbeschränkt) -> θ (X-Raum, gültig).
    - positive Blöcke: exp
    - Simplex-Reihen: Softmax vs. letzte Kategorie (nur erste M-1 werden gespeichert)
    """
    K, s_lam, s_a, s_alpha, s_pr, s_h = _slices(M, N, r, p)
    x = z.astype(np.float64).copy()

    # positive <- exp
    x[s_lam] = np.exp(z[s_lam])
    x[s_h]   = np.exp(z[s_h])

    # simplex <- softmax vs. letzte Kategorie (nur erste M-1 speichern)
    x_pr = []
    off = 0
    z_pr = z[s_pr]
    for _ in range(M):
        z_row = z_pr[off:off+(M-1)]
        # Clip für numerische Stabilität vor exp
        ez = np.exp(np.clip(z_row, -50, 50))
        denom = 1.0 + float(np.sum(ez))
        pr = ez / denom
        x_pr.append(pr)
        off += (M - 1)
    x[s_pr] = np.concatenate(x_pr)
    return x


def _log_jacobian_T(z: np.ndarray, x: np.ndarray, M: int, N: int, r: int, p: int) -> float:
    """
    log|J_T(z)| für die Abbildung z -> θ.
    - exp: Summe der z's
    - Softmax vs. letzte Kategorie: Summe log p_i über ALLE M Kategorien je Zeile
      (letzte Kategorie ist 1 - sum(pr_first_M_minus_1))
    """
    K, s_lam, s_a, s_alpha, s_pr, s_h = _slices(M, N, r, p)
    lj = 0.0
    lj += float(np.sum(z[s_lam])) + float(np.sum(z[s_h]))

    off = 0
    x_pr = x[s_pr]
    for _ in range(M):
        pr = x_pr[off:off+(M-1)]
        pM = max(1e-300, 1.0 - float(np.sum(pr)))
        pr_full = np.clip(np.append(pr, pM), 1e-300, 1.0)
        lj += float(np.sum(np.log(pr_full)))
        off += (M - 1)
    return lj


# ---------------------------------------------------------------------------
# Numerisch stabile Log-Dichte für N-dim. Normal mit diagonaler Varianz
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * np.pi

def _logpdf_diag_gaussian(eps: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """
    eps: (TT, N) Residuen
    lam: (N,) positive Diagonalvarianzen
    Rückgabe: (TT,) Logdichten
    """
    lam = np.clip(lam, 1e-12, 1e12)
    const = -0.5 * (eps.shape[1] * np.log(_TWO_PI) + np.sum(np.log(lam)))
    quad = -0.5 * np.sum((eps * eps) / lam, axis=1)
    return const + quad


# ---------------------------------------------------------------------------
# Log-Likelihood und Log-Prior
# ---------------------------------------------------------------------------

def loglikelihood_draw(draws: np.ndarray, restrictions, N: int, M: int, r: int, p: int,
                       Y: np.ndarray, X: np.ndarray) -> float:
    """
    draws: flacher θ-Vektor im X-Raum (gleiche Struktur wie Posterior-Matrix-Zeilen)
    """
    idx = 0
    lambda_omega = draws[idx: idx + M * N].reshape((N, M)); idx += M * N
    a_matrix     = draws[idx: idx + M * r].reshape((M, r));  idx += M * r
    K = 1 + N * p
    alpha        = draws[idx: idx + M * K * N].reshape((M, K, N)); idx += M * K * N

    pr_flat = draws[idx: idx + M * (M - 1)].reshape(M, M - 1)
    PR_TR = np.hstack([pr_flat, 1 - np.sum(pr_flat, axis=1, keepdims=True)])
    PR_TR = np.clip(PR_TR, 1e-12, 1.0)
    PR_TR = PR_TR / PR_TR.sum(axis=1, keepdims=True)

    TT = Y.shape[0]

    # Log-Beobachtungsdichten für jedes Regime
    log_eta = np.empty((M, TT), dtype=np.float64)
    Q = restrictions["Q"]
    q = restrictions["q"]
    for m in range(M):
        # Strukturmatrix A_m = vec^-1( Q a_m + q )
        A_m = (Q @ a_matrix[m].reshape(-1, 1) + q).reshape(N, N)

        # reduzierte Seite: X @ alpha[m] -> (TT, N)
        X_alpha_m = X @ alpha[m]
        eps = Y @ A_m.T - X_alpha_m

        # Varianz-Parametrisierung
        if m == 0:
            lam = lambda_omega[:, 0]
        else:
            lam = lambda_omega[:, 0] * lambda_omega[:, m]  # multiplicative scales

        log_eta[m, :] = _logpdf_diag_gaussian(eps, lam)

        # Fallback bei NaNs/Infs
        bad = ~np.isfinite(log_eta[m, :])
        if np.any(bad):
            log_eta[m, bad] = -1e300

    # Forward-Filter mit Stabilisierung (Rescaling)
    xi = ergodic_prob(PR_TR)  # stationärer Start
    loglik = 0.0
    for t in range(TT):
        eta_t = np.exp(log_eta[:, t])             # zurück auf lin. Skala
        pred = PR_TR.T @ xi                       # Vorhersage p(s_t)
        alpha_t = eta_t * pred                    # ungeeichte Masse
        c_t = float(np.sum(alpha_t))
        if not np.isfinite(c_t) or c_t <= 0.0:
            c_t = 1e-300
        loglik += np.log(c_t)
        xi = alpha_t / c_t

    return float(loglik)


def logprior_draw(draws: np.ndarray, priors, restrictions, N: int, M: int, r: int, p: int) -> float:
    idx = 0
    lambda_omega = draws[idx: idx + M * N].reshape((N, M)); idx += M * N
    a_matrix     = draws[idx: idx + M * r].reshape((M, r));  idx += M * r
    K = 1 + N * p
    alpha        = draws[idx: idx + M * K * N].reshape((M, K, N)); idx += M * K * N

    pr_flat = draws[idx: idx + M * (M - 1)].reshape(M, M - 1)
    PR_TR = np.hstack([pr_flat, 1 - np.sum(pr_flat, axis=1, keepdims=True)])
    PR_TR = np.clip(PR_TR, 1e-12, 1.0)
    PR_TR = PR_TR / PR_TR.sum(axis=1, keepdims=True)

    idx += M * (M - 1)
    hyper = draws[idx: idx + 3]

    # Priors
    lp_mu = 0.0
    for m in range(M):
        lp_mu += dt2(
            alpha[m, 0, :],
            mu=np.zeros(N),
            precision=np.eye(N),
            a=priors["hyper_a"], b=priors["hyper_b"],
            log=True
        )

    lp_beta = 0.0
    Q = restrictions["Q"]
    q = restrictions["q"]
    for m in range(M):
        A_m = (Q @ a_matrix[m].reshape(-1, 1) + q).reshape(N, N)
        for n in range(N):
            mu_beta = A_m[n, :] @ priors["beta_P"].T  # (pN,)
            lp_beta += dt2(
                alpha[m, 1:, n],
                mu=mu_beta,
                precision=priors["beta_H"],
                a=priors["hyper_a"], b=priors["hyper_b"],
                log=True
            )

    lp_a = 0.0
    for m in range(M):
        lp_a += dt2(
            a_matrix[m],
            mu=np.zeros(r),
            precision=np.eye(r),
            a=priors["hyper_a"], b=priors["hyper_b"],
            log=True
        )

    lp_lambda = float(np.sum(dig2(lambda_omega[:, 0],  a=priors["lambda_a"], b=priors["lambda_b"], log=True)))
    lp_omega  = float(np.sum(dig2(lambda_omega[:, 1:], a=priors["omega_a"],  b=priors["omega_b"],  log=True)))
    lp_hyper  = float(np.sum(dig2(hyper, a=priors["hyper_a"], b=priors["hyper_b"], log=True)))

    # Dirichlet-Prior pro Zeile von PR_TR
    lp_pr_tr = 0.0
    W = priors["w"].reshape(M, M)
    for m in range(M):
        alpha_pr = np.clip(W[m, :], 1e-8, None)
        try:
            lp_pr_tr += float(np.log(dirichlet.pdf(PR_TR[m, :], alpha=alpha_pr)))
        except Exception:
            lp_pr_tr += -1e6  # robuste Sicherung

    return float(lp_mu + lp_beta + lp_a + lp_lambda + lp_omega + lp_hyper + lp_pr_tr)


# ---------------------------------------------------------------------------
# Hauptfunktion: Importance-Sampling MDD mit Z-Transformation
# ---------------------------------------------------------------------------

def compute_mdd(Gibbs_output,
                priors,
                restrictions,
                SP: int = 10_000,
                scale_cov_z: float = 1.0,
                jitter: float = 1e-6,
                random_state=None):
    """
    Berechnet log-MDD per Importance Sampling.
    Proposal: MVN im unbeschränkten Z-Raum; θ = T(z) mit Jacobian-Korrektur.

    Parameter
    ---------
    Gibbs_output : Dict mit "posteriors" und "last_draws"
    priors, restrictions : wie in deinem Projekt
    SP : Anzahl Importance-Draws
    scale_cov_z : optionaler Skalierungsfaktor für die Z-Kovarianz
    jitter : Diagonaljitter für die Z-Kovarianz (PD-Garantie)
    random_state : int | np.random.Generator | None

    Rückgabe
    --------
    Dict mit:
      - "log.mdd" (float)
      - "log.kernel" (np.ndarray)  [log w_i]
      - "log.likelihood", "log.prior", "log.importance" (np.ndarray)
      - "ess" (float), "n_sp" (int), "n_dim" (int)
    """
    rng = np.random.default_rng(random_state)

    aux = Gibbs_output["last_draws"]
    post = Gibbs_output["posteriors"]

    M = int(aux["PR_TR"].shape[0])
    N = int(aux["Y"].shape[1])
    T = int(aux["Y"].shape[0])
    TT = int(aux["U"].shape[0])
    p = T - TT
    r = int(restrictions["Q"].shape[1])
    S = int(post["alpha"].shape[3])  # alpha: (K, N, M, S)

    # Erwartete Länge eines Draws (θ im X-Raum)
    K = 1 + N * p
    expected_len = M * N + M * r + M * N * K + M * (M - 1) + 3

    # Regressoren und Zielmatrix
    X = np.ones((TT, 1), dtype=np.float64)
    for lag in range(1, p + 1):
        Y_lag = aux["Y"][p - lag:T - lag, :]
        X = np.hstack([X, Y_lag])
    Y = aux["Y"][p:, :].astype(np.float64)

    # Posterior-Matrix [S x expected_len] zusammenbauen
    rows = []

    # lambda_omega: (N, M, S) -> (S, N) je m und dann konkat
    for m in range(M):
        rows.append(post["lambda_omega"][:, m, :].T)  # (S, N)
    # a: (M, r, S) -> (S, M*r)
    rows.append(post["a"].transpose(2, 0, 1).reshape(S, -1))
    # alpha: (K, N, M, S) -> (S, M*K*N)
    alpha_post = post["alpha"].transpose(3, 2, 0, 1).reshape(S, -1)
    rows.append(alpha_post)
    # PR_TR: (M, M, S), aber nur die ersten M-1 pro Zeile
    for m in range(M):
        rows.append(post["PR_TR"][m, :-1, :].T)  # (S, M-1)
    # hyper: (3, S) -> (S, 3)
    rows.append(post["hyper"].T)

    posterior_matrix = np.hstack(rows).astype(np.float64, copy=False)
    if posterior_matrix.shape[1] != expected_len:
        raise ValueError(
            f"[FEHLER] Posterior-Dimension inkonsistent: {posterior_matrix.shape[1]} != {expected_len}"
        )

    # --- 1) Transformiere Posterior-Samples θ -> z (unbeschränkt)
    z_samples = np.apply_along_axis(_x_to_z_row, 1, posterior_matrix, M, N, r, p)

    # --- 2) Moments im Z-Raum (mean/cov) + Regularisierung
    mu_z = np.mean(z_samples, axis=0)
    Sig_z = np.cov(z_samples, rowvar=False)

    # PD-Garantie + ggf. Scaling
    d = Sig_z.shape[0]
    Sig_z = Sig_z + jitter * np.eye(d)
    if scale_cov_z != 1.0:
        Sig_z = float(scale_cov_z) * Sig_z

    # --- 3) Ziehe z ~ N(mu_z, Sig_z); mappe deterministisch zurück θ=T(z)
    try:
        z_draws = multivariate_normal.rvs(mean=mu_z, cov=Sig_z, size=SP, random_state=rng)
    except LinAlgError:
        # Fallback: stärkere Diagonalregularisierung
        bump = max(1e-4, 10.0 * jitter)
        z_draws = multivariate_normal.rvs(
            mean=mu_z, cov=(Sig_z + bump * np.eye(d)), size=SP, random_state=rng
        )
    z_draws = z_draws.reshape(-1, mu_z.shape[0])

    theta_draws = np.apply_along_axis(_z_to_x_row, 1, z_draws, M, N, r, p)

    # --- 4) Importance-Dichte q_X(θ) via Change of Variables
    log_qZ = multivariate_normal.logpdf(z_draws, mean=mu_z, cov=Sig_z, allow_singular=True)
    logJ = np.array([_log_jacobian_T(z, x, M, N, r, p) for z, x in zip(z_draws, theta_draws)])
    logis = log_qZ - logJ  # log q_X(θ)

    # --- 5) Log-Likelihood & Log-Prior
    loglik = np.array([loglikelihood_draw(draw, restrictions, N, M, r, p, Y, X)
                       for draw in theta_draws], dtype=np.float64)
    logpri = np.array([logprior_draw(draw, priors, restrictions, N, M, r, p)
                       for draw in theta_draws], dtype=np.float64)

    # --- 6) Importance-Kernel und log-MDD
    lis = loglik + logpri - logis
    finite_mask = np.isfinite(lis)
    if not np.any(finite_mask):
        raise FloatingPointError("Alle Importance-Kerne sind nicht endlich (NaN/Inf).")

    lis = lis[finite_mask]
    log_mdd = float(logsumexp(lis) - np.log(lis.size))

    # Diagnostik: ESS
    # w_i ∝ exp(lis - max)
    w = np.exp(lis - np.max(lis))
    ess = (w.sum() ** 2) / np.sum(w * w)

    return {
        "log.mdd": log_mdd,
        "log.kernel": lis,
        "log.likelihood": loglik[finite_mask],
        "log.prior": logpri[finite_mask],
        "log.importance": logis[finite_mask],
        "ess": float(ess),
        "n_sp": int(SP),
        "n_dim": int(expected_len),
    }