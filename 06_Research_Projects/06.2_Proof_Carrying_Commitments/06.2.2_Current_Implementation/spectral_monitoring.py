import numpy as np


def spectral_props(M):
    vals = np.linalg.eigvals(M)
    vals = np.real(vals)
    return float(np.max(vals)), float(np.min(vals))


def spectral_alerts(M, max_bound, min_bound):
    vmax, vmin = spectral_props(M)
    alerts = []
    if vmax > max_bound:
        alerts.append("max_exceeded")
    if vmin < min_bound:
        alerts.append("min_below")
    return alerts