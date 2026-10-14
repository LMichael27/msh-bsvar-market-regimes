import numpy as np
import matplotlib.pyplot as plt
import os


def _map_asset_names(names):
    mapping = {
        "sp500": "Equities",
        "luattruu index": "Bonds",
        "dxy index": "Dollar",
        "xau curncy": "Gold",
    }

    mapped = []
    for n in names:
        if not isinstance(n, str):
            mapped.append(n)
            continue
        key = n.lower().strip()
        if "sp500" in key:
            mapped.append(mapping["sp500"])
        elif "luattruu" in key or "luattruu index" in key:
            mapped.append(mapping["luattruu index"])
        elif "dxy" in key or "dxy index" in key:
            mapped.append(mapping["dxy index"])
        elif "xau" in key or "xau curncy" in key:
            mapped.append(mapping["xau curncy"])
        else:
            mapped.append(n)

    return mapped

def compute_irf(alpha, A0, p, horizon=20):
    K = alpha.shape[1]
    intercept = alpha[0, :]
    A_matrices = [alpha[1 + i * K: 1 + (i + 1) * K, :].T for i in range(p)]

    A_comp = np.zeros((K * p, K * p))
    for i in range(p):
        A_comp[:K, i * K:(i + 1) * K] = A_matrices[i]
    if p > 1:
        A_comp[K:, :-K] = np.eye(K * (p - 1))

    irfs_rf = np.zeros((K, K, horizon + 1))
    irfs_rf[:, :, 0] = np.eye(K)

    state = np.eye(K * p)
    for h in range(1, horizon + 1):
        state = A_comp @ state
        irfs_rf[:, :, h] = state[:K, :K]

    A_inv = np.linalg.inv(A0)
    irfs_struct = np.zeros_like(irfs_rf)
    for h in range(horizon + 1):
        irfs_struct[:, :, h] = -irfs_rf[:, :, h] @ A_inv  # negativer Schock

    return irfs_struct

def plot_irf_regime(alpha, A0, regime=0, asset_names=None, horizon=20, regime_name=None):
    irf = compute_irf(alpha, A0, p=alpha.shape[0] // alpha.shape[1] - 1, horizon=horizon)
    N = A0.shape[0]
    time = np.arange(horizon + 1)

    fig, axs = plt.subplots(N, N, figsize=(3 * N, 2.5 * N), squeeze=False)
    if asset_names is None:
        asset_names = [f"Asset {i + 1}" for i in range(N)]
    else:
        asset_names = _map_asset_names(asset_names)

    for i in range(N):
        for j in range(N):
            axs[i, j].plot(time, irf[i, j, :], label=f"Impulse {asset_names[j]}")
            axs[i, j].axhline(0, color="gray", linestyle="--", linewidth=0.5)
            axs[i, j].set_title(f"{asset_names[i]} response to {asset_names[j]}")
            axs[i, j].grid(True)

    if regime_name is None:
        title = f"Impulse Response Functions - Regime {regime + 1}"
    else:
        title = f"Impulse Response Functions - {regime_name}"

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_all_regime_irfs(posterior, p, horizon=20, asset_names=None, save_path=None, show=True, regime_names=None):
    M = posterior["A"].shape[0]
    K = posterior["A"].shape[1]

    if asset_names is None:
        asset_names = [f"Var {i + 1}" for i in range(K)]
    else:
        asset_names = _map_asset_names(asset_names)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    if regime_names is None:
        regime_names = [f"Regime {m + 1}" for m in range(M)]

    for m in range(M):
        alpha = posterior["alpha"][:, :, m, -1]
        A0 = posterior["A"][m, :, :, -1]
        irf = compute_irf(alpha, A0, p, horizon)
        time = np.arange(horizon + 1)

        fig, axs = plt.subplots(K, K, figsize=(3 * K, 2.5 * K), squeeze=False)
        fig.suptitle(f"IRFs - {regime_names[m]}", fontsize=16)

        for i in range(K):
            for j in range(K):
                axs[i, j].plot(time, irf[i, j, :], label=f"{asset_names[j]}")
                axs[i, j].axhline(0, color="gray", linestyle="--", linewidth=0.5)
                axs[i, j].set_title(f"{asset_names[i]} response to {asset_names[j]}")
                axs[i, j].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path is not None:
            safe_name = regime_names[m].replace(" ", "_").lower()
            filename = os.path.join(save_path, f"irf_{safe_name}.png")
            fig.savefig(filename)

        if show:
            plt.show()
        else:
            plt.close(fig)