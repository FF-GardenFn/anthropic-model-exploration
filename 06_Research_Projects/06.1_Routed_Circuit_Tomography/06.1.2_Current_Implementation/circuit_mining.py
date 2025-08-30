import numpy as np


def top_eig(x, k=1):
    w, v = np.linalg.eigh(x)
    idx = np.argsort(w)[::-1][:k]
    return w[idx], v[:, idx]


def mine_scores(kernels):
    out = []
    for K in kernels:
        w, _ = top_eig(K, 1)
        out.append(float(w[0]))
    return out