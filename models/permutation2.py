from __future__ import annotations
import numpy as np
from itertools import permutations
from typing import List, Tuple, Dict, Sequence, Optional
from models.post_analysis import _zscore

def all_permutations(n):
    return np.array(list(permutations(range(n))))



def apply_label_permutation(posterior: Dict[str, np.ndarray],
                            perm: Sequence[int]) -> Dict[str, np.ndarray]:
    """
    Permutiert konsistent alle Posterior-Tensors mit einer Regime-Permutation.
    Erwartete Keys (falls vorhanden): 'Sigma' [N,N,M,S], 'lambda_omega' [N,M,S],
    'PR_TR' [M,M,S], 'A' [M,N,N,S], 'a' [M,r,S], 'alpha' [K,N,M,S], 'S_t' [T,S].
    """
    perm = np.asarray(perm, dtype=int)
    out: Dict[str, np.ndarray] = {}
    for k, v in posterior.items():
        out[k] = v

    if "Sigma" in posterior:
        out["Sigma"] = posterior["Sigma"][:, :, perm, :]
    if "lambda_omega" in posterior:
        out["lambda_omega"] = posterior["lambda_omega"][:, perm, :]
    if "PR_TR" in posterior:
        out["PR_TR"] = posterior["PR_TR"][perm][:, perm, :]
    if "A" in posterior:
        out["A"] = posterior["A"][perm, :, :, :]
    if "a" in posterior:
        out["a"] = posterior["a"][perm, :, :]
    if "alpha" in posterior:
        out["alpha"] = posterior["alpha"][:, :, perm, :]

    # S_t enthält Integerzustände (0..M-1) pro Draw: [T, S]
    if "S_t" in posterior:
        inv_perm = np.argsort(perm)
        S_t = posterior["S_t"]
        # vektorisiert: map alte Labels -> neue mittels inv_perm
        out["S_t"] = inv_perm[S_t]

    return out


def get_economic_permutation_from_raw(mu_raw: np.ndarray,
                                      asset_names: Sequence[str],
                                      M: int,
                                      stock_name: str = "NDDUUS Index",
                                      bond_name: str = "LUATTRUU Index") -> List[int]:
    """
    Bestimmt eine Permutation, die Regime nach (Stocks, Bonds) sortiert.
    - mu_raw: (M, N) ROHE log-Mittel (aus summarize_conditional_means)
    - stock_name/bond_name: Spaltennamen in asset_names
    Zielvektoren für M in {2,3,4} sind vordefiniert. Minimiert euklidische Distanz.
    """
    name_to_idx = {n: i for i, n in enumerate(asset_names)}
    if stock_name not in name_to_idx or bond_name not in name_to_idx:
        # Try fuzzy matching to rescue minor typos
        try:
            from difflib import get_close_matches
        except Exception:
            get_close_matches = None

        resolved = {}
        for role, nm in (("stock_name", stock_name), ("bond_name", bond_name)):
            if nm in name_to_idx:
                resolved[role] = nm
            elif get_close_matches is not None:
                cand = get_close_matches(nm, list(name_to_idx.keys()), n=1, cutoff=0.8)
                if cand:
                    print(f"[permutation2] INFO: Using fuzzy match for {role}: '{nm}' -> '{cand[0]}'")
                    resolved[role] = cand[0]
        if len(resolved) < 2:
            # Fallback: identische Permutation, aber mit klarer Warnung
            missing = []
            if stock_name not in resolved:
                missing.append(f"stock_name '{stock_name}'")
            if bond_name not in resolved:
                missing.append(f"bond_name '{bond_name}'")
            print(
                "[permutation2] WARNING: Could not find the following asset(s) in asset_names: "
                + ", ".join(missing)
                + ". Falling back to identity permutation."
            )
            return list(range(M))
        stock_name, bond_name = resolved.get("stock_name", stock_name), resolved.get("bond_name", bond_name)

    s_idx, b_idx, = name_to_idx[stock_name], name_to_idx[bond_name]
    regime_scores = mu_raw[:, [s_idx, b_idx]]  # (M,2)
    # spaltenweise standardisieren (vergleichbar über Regime)
    regime_scores = _zscore(regime_scores, axis=0)

    if M == 4:
        target = np.array([[+1, -1],  # Infl. Growth  (Stocks+, Bonds-)
                           [-1, -1],  # Infl. Stagn.
                           [+1, +1],  # Defl. Growth
                           [-1, +1]]) # Defl. Stagn.
    elif M == 3:
        target = np.array([[+1, -1],   # Growth
                           [-1, +1],   # Stagnation
                           [-1, -1]])  # Defensive/Neutral
    elif M == 2:
        target = np.array([[+1, -1],   # Risk-On
                           [-1, +1]])  # Risk-Off
    else:
        return list(range(M))

    # ebenfalls standardisieren
    target = _zscore(target, axis=0)

    # Brute-force über alle Permutationen (M <= 6 ok)
    from itertools import permutations
    best_perm, min_cost = list(range(M)), float("inf")
    for perm in permutations(range(M)):
        perm = list(perm)
        cost = 0.0
        for i in range(min(M, target.shape[0])):
            cost += np.linalg.norm(regime_scores[perm[i]] - target[i])
        if cost < min_cost:
            min_cost, best_perm = cost, perm
    return list(best_perm)


