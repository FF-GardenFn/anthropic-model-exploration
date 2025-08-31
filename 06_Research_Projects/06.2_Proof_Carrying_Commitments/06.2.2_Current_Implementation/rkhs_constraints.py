import numpy as np


def kernel_linear(x):
    return x @ x.T


def simple_hat(K, lam):
    n = K.shape[0]
    return K @ np.linalg.pinv(K + lam * np.eye(n))