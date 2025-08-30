import numpy as np


def hat(K, lam):
    n = K.shape[0]
    return K @ np.linalg.pinv(K + lam * np.eye(n))


def rkhs_score(K):
    w, _ = np.linalg.eigh(K)
    w = np.sort(w)[::-1]
    if len(w) < 2:
        return float(w[0]) if len(w) else 0.0
    return float(w[0] - w[1])