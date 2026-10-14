from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================================
# Helpers
# ======================================================================

def _ensure_dir_for_file(path: Optional[str]) -> None:
    """Erzeugt das Zielverzeichnis für eine Datei, wenn nötig."""
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _is_int_like(a: np.ndarray) -> bool:
    """True, wenn Array ganzzahl-ähnlich ist (auch float mit .0)."""
    return np.allclose(a, np.round(a))


def _to_soft_probs(S: Union[np.ndarray, Sequence], M: Optional[int] = None) -> np.ndarray:
    """
    Bringt verschiedene Eingabe-Formate auf eine weiche Wahrscheinlichkeitsmatrix P (T, M).

    Akzeptierte Eingaben:
      - (T, M) soft probs (Zeilensumme ≈ 1)  -> Rückgabe unverändert
      - (M, T) soft probs (Spaltensumme ≈ 1) -> wird transponiert
      - (T, M) One-Hot                       -> Rückgabe unverändert
      - (T,)    harter Pfad ints             -> zu One-Hot
      - (T, S)  mehrere harte Pfad-Draws     -> One-Hot je Draw, dann über S gemittelt

    Rückgabe:
      - (T, M) mit Zeilensumme exakt 1
    """
    A = np.asarray(S)

    if A.ndim == 1:
        # harter Pfad
        m = int(A.max()) + 1 if M is None else int(M)
        probs = np.eye(m, dtype=float)[A.astype(int)]
        return probs

    if A.ndim != 2:
        raise ValueError(f"S must be 1D or 2D, got shape {A.shape}")

    T, K = A.shape

    # (M, T) soft -> transponieren
    if np.all(A >= 0) and np.all(A <= 1) and np.allclose(A.sum(axis=0), 1.0, atol=1e-3) and (M is None or T == M):
        A = A.T
        T, K = A.shape

    # Bereits (T, M) soft/one-hot?
    if np.all(A >= 0) and np.all(A <= 1) and np.allclose(A.sum(axis=1), 1.0, atol=1e-6):
        return A

    # (T, S) viele harte Draws?
    if _is_int_like(A):
        m = int(A.max()) + 1 if M is None else int(M)
        probs = np.zeros((T, m), dtype=float)
        for s in range(K):
            path = A[:, s].astype(int)
            probs += np.eye(m, dtype=float)[path]
        probs /= float(K)
        # Normierung absichern
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        return probs / row_sum

    # Fallback: Zeilenweise normalisieren (clipping)
    A = np.clip(A, 0.0, None)
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    P = A / row_sum
    return P


def plot_current_regime(result_or_S: Union[dict, np.ndarray, Sequence],
                        regime_names: Optional[Sequence[str]] = None,
                        save_path: Optional[str] = "Figures/plots/current_regime.png",
                        show: bool = True,
                        annotate: bool = True,
                        soften: bool = True,
                        temperature: float = 1.4,
                        uniform_mix: float = 0.05) -> None:
    """Bar-Plot der aktuellen (letzten) Regimewahrscheinlichkeiten und Klassifikation.

    Parameters
    ----------
    result_or_S : dict | array
        Entweder komplettes Gibbs-result (mit posteriors['S_prob'] oder 'S_t'])
        oder direkt eine Darstellung von S (siehe _to_soft_probs).
    regime_names : list[str], optional
        Falls vorhanden werden diese Labels genutzt.
    save_path : str, optional
        Zielpfad für PNG (Ordner wird erstellt). None = nicht speichern.
    show : bool
        Plot anzeigen (True) oder schließen (False).
    annotate : bool
        Werte oberhalb der Balken annotieren.
    """
    # Soft-Probs extrahieren / rekonstruieren
    if isinstance(result_or_S, dict):
        post = result_or_S.get("posteriors", {})
        if "S_prob" in post:
            S_in = post["S_prob"]            # (T, M)
        else:
            S_in = post.get("S_t", None)     # (T, S) oder (T,) oder one-hot
            if S_in is None:
                raise KeyError("Weder 'S_prob' noch 'S_t' in result['posteriors'] gefunden.")
    else:
        S_in = result_or_S

    P = _to_soft_probs(S_in)  # (T, M)
    last_probs = P[-1].astype(float)

    # Softening bei degenerierten Verteilungen (z.B. [1,0,0,0])
    if soften and (last_probs.max() > 0.995 or np.isclose(last_probs.max(), 1.0)):
        # Temperatur-Glättung (größere temperature => flacher)
        if temperature <= 0:
            temperature = 1.0
        powered = np.power(np.clip(last_probs, 1e-12, 1.0), 1.0 / temperature)
        powered /= powered.sum()
        # Mischung mit Uniform
        Mtmp = len(powered)
        last_probs = (1 - uniform_mix) * powered + uniform_mix * (1.0 / Mtmp)
        last_probs /= last_probs.sum()
    M = last_probs.shape[0]

    labels = list(regime_names)[:M] if regime_names else [f"Regime {i+1}" for i in range(M)]
    current_idx = int(np.argmax(last_probs))
    current_label = labels[current_idx]
    current_prob = last_probs[current_idx]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(range(M), last_probs, color=["#1f77b4" if i != current_idx else "#d62728" for i in range(M)], edgecolor="black")
    if annotate:
        for i, (b, v) in enumerate(zip(bars, last_probs)):
            plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Probability")
    plt.ylim(0, 1.05)
    plt.xticks(range(M), labels, rotation=30, ha="right")
    plt.title(f"Current Regime: {current_label} (p = {current_prob*100:.1f}%)")
    plt.tight_layout()
    if save_path:
        _ensure_dir_for_file(save_path)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def regime_probability_diagnostics(result: dict,
                                    save_path: Optional[str] = "Figures/plots/regime_prob_diagnostics.csv",
                                    print_summary: bool = True) -> Optional[dict]:
    """Erstellt Kennzahlen zur Qualität / Nicht-Degeneriertheit der Regimewahrscheinlichkeiten.

    Liefert ein Dict zurück und speichert optional eine CSV.
    Kennzahlen:
      - T, M
      - Anteil Zeitpunkte mit maxProb > 0.9/0.95/0.99
      - Durchschnitt maxProb
      - Mittelwerte der (gemittelten) Übergangsmatrix-Diagonale
    """
    post = result.get("posteriors", {})
    S_prob = post.get("S_prob", None)
    S_t = post.get("S_t", None)
    PR_TR = post.get("PR_TR", None)  # (M,M,S)

    if S_prob is None and S_t is not None:
        # Rekonstruiere weiche probs aus harten Pfaden
        S_prob = _to_soft_probs(S_t)
    if S_prob is None:
        if print_summary:
            print("[Diagnostics] Keine S_prob oder S_t verfügbar.")
        return None

    T = S_prob.shape[0]
    M = S_prob.shape[1]
    row_max = S_prob.max(axis=1)
    diag_mean = None
    if PR_TR is not None and PR_TR.ndim == 3:
        diag_mean = PR_TR.diagonal(axis1=0, axis2=1).mean(axis=1).mean()

    stats = {
        "T": T,
        "M": M,
        "avg_max_prob": float(row_max.mean()),
        "share_gt_0.90": float((row_max > 0.90).mean()),
        "share_gt_0.95": float((row_max > 0.95).mean()),
        "share_gt_0.99": float((row_max > 0.99).mean()),
        "mean_diag_PR_TR": float(diag_mean) if diag_mean is not None else None,
    }

    if print_summary:
        print("[Regime Prob Diagnostics]")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    if save_path:
        import pandas as _pd
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _pd.DataFrame([stats]).to_csv(save_path, index=False)
    return stats


def _most_likely_path(S: Union[np.ndarray, Sequence], M: Optional[int] = None) -> np.ndarray:
    """Liefert einen harten Pfad (T,) als argmax über weiche probs oder Modus über Draws."""
    P = _to_soft_probs(S, M=M)
    return P.argmax(axis=1).astype(int)


def _despike_path(path: np.ndarray, probs: np.ndarray,
                  min_run: int = 2, prob_floor: float = 0.55) -> np.ndarray:
    """
    Glättet ABA-Singletons und sehr kurze Läufe, wenn die Evidenz schwach ist.
    - path: (T,) int
    - probs: (T, M) soft probs
    """
    p = path.copy()
    T = len(p)
    # Einzel-Spikes ersetzen (ABA)
    for t in range(1, T - 1):
        if p[t - 1] == p[t + 1] != p[t] and probs[t, p[t]] < prob_floor:
            p[t] = p[t - 1]

    # Mindestlänge erzwingen
    run_len = 1
    for t in range(1, T):
        if p[t] == p[t - 1]:
            run_len += 1
        else:
            # kurzer Lauf rückwirkend glätten, wenn Evidenz schwach
            if run_len < min_run and probs[t - 1, p[t - 1]] < prob_floor:
                p[t - run_len:t] = p[t]  # an Folge-Regime angleichen
            run_len = 1
    return p


# ======================================================================
# Plots & Matrizen
# ======================================================================

def plot_regime_probabilities(S_t: Union[np.ndarray, Sequence],
                              regime_names: Optional[Sequence[str]] = None,
                              save_path: Optional[str] = "Figures/plots/regime_prob.png",
                              show: bool = True,
                              dates: Optional[Sequence] = None) -> None:
    """
    Heatmap der (geglätteten) Regimewahrscheinlichkeiten.

    S_t kann sein:
      - (T, M) soft probs
      - (T, S) harte Pfad-Draws (ints)
      - (T,)    ein harter Pfad
      - (T, M) One-Hot
    """
    P = _to_soft_probs(S_t)             # (T, M)
    T, M = P.shape

    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Use numeric indices for plotting with pcolormesh
    X, Y = np.arange(T + 1), np.arange(M + 1)
    mesh = ax.pcolormesh(X, Y, P.T, shading='auto', cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(mesh, ax=ax, label="Probability")
    
    # Set X-axis with dates if provided
    if dates is not None:
        try:
            dates_arr = np.asarray(dates)
            if len(dates_arr) == T:
                # Determine number of labels
                n_labels = min(12, T)
                step = max(1, T // n_labels)
                xtick_indices = np.arange(0, T, step)
                xtick_labels = []
                for i in xtick_indices:
                    d = dates_arr[int(i)]
                    # Handle numpy datetime64, pandas Timestamp, and datetime objects
                    try:
                        if isinstance(d, (np.datetime64)):
                            # Convert numpy datetime64 to string and extract YYYY-MM
                            d_str = str(d)[:7]  # YYYY-MM format
                            xtick_labels.append(d_str)
                        elif hasattr(d, "strftime"):
                            xtick_labels.append(d.strftime("%Y-%m"))
                        else:
                            d_str = str(d)[:7]
                            xtick_labels.append(d_str)
                    except:
                        xtick_labels.append(str(d)[:7])
                ax.set_xticks(xtick_indices)
                ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=9)
                ax.set_xlabel("Date")
            else:
                print(f"[Warning] dates length ({len(dates_arr)}) != T ({T})")
                ax.set_xlabel("Time")
        except Exception as e:
            print(f"[Warning] Error setting dates: {e}")
            ax.set_xlabel("Time")
    else:
        ax.set_xlabel("Time")
    
    ax.set_ylabel("Regime")
    ax.set_title("Smoothed Regime Probabilities")
    ax.set_ylim(0, M)

    if regime_names:
        ax.set_yticks(np.arange(0.5, M + 0.5))
        ax.set_yticklabels(list(regime_names)[:M])
    else:
        ax.set_yticks(np.arange(0.5, M + 0.5))
        ax.set_yticklabels([f"Regime {i + 1}" for i in range(M)])

    plt.tight_layout()
    if save_path:
        _ensure_dir_for_file(save_path)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_regime_path(result_or_S: Union[dict, np.ndarray, Sequence],
                     regime_names: Optional[Sequence[str]] = None,
                     use_soft_if_available: bool = True,
                     denoise: bool = False,
                     save_path: Optional[str] = "Figures/plots/regime_path.png",
                     show: bool = True,
                     dates: Optional[Sequence] = None) -> None:
    """
    Zeichnet den (ggf. geglätteten) harten Pfad über die Zeit.
    - result_or_S: entweder das komplette result-Objekt ODER direkt S (siehe _to_soft_probs)
    - use_soft_if_available: wenn dict übergeben und "S_prob" existiert, wird dieses genutzt
    - denoise: optionales Entzacken des argmax-Pfads (nur Visualisierung)
    - dates: optional Datumsindizes für die X-Achse
    """
    # Quelle bestimmen
    if isinstance(result_or_S, dict):
        post = result_or_S.get("posteriors", {})
        if use_soft_if_available and "S_prob" in post:
            S_in = post["S_prob"]          # (T, M) soft
        else:
            S_in = post.get("S_t", None)   # (T, S) harte Draws ODER (T,) Pfad
            if S_in is None:
                raise KeyError("Weder 'S_prob' noch 'S_t' in result['posteriors'] gefunden.")
    else:
        S_in = result_or_S

    P = _to_soft_probs(S_in)               # (T, M)
    path = P.argmax(axis=1).astype(int)    # argmax-Pfad
    if denoise:
        path = _despike_path(path, P, min_run=2, prob_floor=0.55)

    T = len(path)
    # Always plot with numeric indices
    t = np.arange(T)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(t, path + 1, linewidth=1.8, marker='o', markersize=3)
    ax.set_title("Most Likely Regime Path")
    
    # Format X-axis with dates if provided
    if dates is not None:
        try:
            dates_arr = np.asarray(dates)
            if len(dates_arr) == T:
                # Determine number of labels
                n_labels = min(12, T)
                step = max(1, T // n_labels)
                xtick_indices = np.arange(0, T, step)
                xtick_labels = []
                for i in xtick_indices:
                    d = dates_arr[int(i)]
                    # Handle numpy datetime64, pandas Timestamp, and datetime objects
                    try:
                        if isinstance(d, (np.datetime64)):
                            # Convert numpy datetime64 to string and extract YYYY-MM
                            d_str = str(d)[:7]  # YYYY-MM format
                            xtick_labels.append(d_str)
                        elif hasattr(d, "strftime"):
                            xtick_labels.append(d.strftime("%Y-%m"))
                        else:
                            d_str = str(d)[:7]
                            xtick_labels.append(d_str)
                    except:
                        xtick_labels.append(str(d)[:7])
                ax.set_xticks(xtick_indices)
                ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=9)
                ax.set_xlabel("Date")
            else:
                print(f"[Warning] dates length ({len(dates_arr)}) != T ({T})")
                ax.set_xlabel("Time index")
        except Exception as e:
            print(f"[Warning] Error with dates: {e}")
            ax.set_xlabel("Time index")
    else:
        ax.set_xlabel("Time index")
    
    ax.set_ylabel("Regime")
    if regime_names:
        ax.set_yticks(ticks=np.arange(1, P.shape[1] + 1), labels=list(regime_names)[:P.shape[1]])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        _ensure_dir_for_file(save_path)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def compute_mean_transition_matrix(posteriors: dict) -> np.ndarray:
    """
    Erwartungswert der Übergangsmatrix über Draws, zeilenweise normalisiert.
    Erwartet posteriors['PR_TR'] mit Shape (M, M, S).
    """
    PR_TR_all = posteriors["PR_TR"]  # (M, M, S)
    M, _, S = PR_TR_all.shape
    PR_TR_norm = np.zeros_like(PR_TR_all)

    for s in range(S):
        mat = np.asarray(PR_TR_all[:, :, s], dtype=float)
        mat = np.clip(mat, 0.0, None)
        row_sum = mat.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        PR_TR_norm[:, :, s] = mat / row_sum

    return PR_TR_norm.mean(axis=2)


def plot_transition_matrix(transition_matrix: np.ndarray,
                           regime_names: Optional[Sequence[str]] = None,
                           save_path: Optional[str] = "Figures/plots/transition_matrix.png",
                           show: bool = True) -> None:
    """Heatmap der Übergangswahrscheinlichkeiten (M, M)."""
    M = transition_matrix.shape[0]
    if regime_names is None:
        regime_names = [f"Regime {i + 1}" for i in range(M)]

    plt.figure(figsize=(6, 5))
    sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=regime_names, yticklabels=regime_names, cbar=True,
                vmin=0, vmax=1)
    plt.title("Regime Transition Probabilities")
    plt.xlabel("To Regime")
    plt.ylabel("From Regime")
    plt.tight_layout()
    if save_path:
        _ensure_dir_for_file(save_path)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def empirical_transition_matrix(S_t: Union[np.ndarray, Sequence],
                                M: Optional[int] = None) -> np.ndarray:
    """
    Empirische Übergangsmatrix aus weichen Wahrscheinlichkeiten (bevorzugt) oder Pfad-Draws.

    - Wenn S_t (T, M): nutzt erwartete Übergänge E[I(s_t=i) I(s_{t+1}=j)] = Σ_t P_t(i) P_{t+1}(j)
    - Wenn S_t (T, S) ints: wird intern zu (T, M) gemittelt und dann wie oben behandelt.
    - Rückgabe ist zeilenstochastisch (M, M).
    """
    P = _to_soft_probs(S_t, M=M)    # (T, M)
    T, M_eff = P.shape
    num = np.einsum("ti,tj->ij", P[:-1, :], P[1:, :])  # Summe über t
    row_sum = num.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    trans = num / row_sum
    # numerische Stabilisierung / Clipping
    trans = np.clip(trans, 0.0, 1.0)
    return trans