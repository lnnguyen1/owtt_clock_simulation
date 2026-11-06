"""
allan.py - Allan deviation tools.

Works on ANY phase record x(t) sampled every tau0 seconds:
simulated clocks, ZED-F9P PPS logs, or CSAC "Phase" telemetry from ClockStudio
(convert ns to seconds first).
"""

import numpy as np


def overlapping_adev(x, tau0, taus=None):
    """
    Overlapping Allan deviation from phase data.

    x     : 1-D array of phase (time error) in seconds, evenly spaced
    tau0  : sample spacing in seconds
    taus  : averaging times to evaluate (seconds); default ~3 per decade

    Returns (taus, adev, n_terms). n_terms = how many differences were averaged;
    fewer terms means a less certain estimate (large tau).
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if taus is None:
        max_m = (N - 1) // 4                      # keep at least ~2N/4 terms
        ms = np.unique(np.round(np.logspace(0, np.log10(max_m), 3 * int(np.log10(max_m) + 1))).astype(int))
    else:
        ms = np.unique(np.round(np.asarray(taus) / tau0).astype(int))
        ms = ms[(ms >= 1) & (2 * ms < N)]

    out_tau, out_adev, out_n = [], [], []
    for m in ms:
        d = x[2 * m:] - 2 * x[m:-m] + x[:-2 * m]   # second difference of phase
        avar = np.mean(d**2) / (2.0 * (m * tau0) ** 2)
        out_tau.append(m * tau0)
        out_adev.append(np.sqrt(avar))
        out_n.append(len(d))
    return np.array(out_tau), np.array(out_adev), np.array(out_n)


def fit_q1_q2(taus, adev):
    """
    Fit the two-state model's Allan variance (Galleani 2008, eq. 47):
        sigma^2(tau) = q1/tau + q2*tau/3
    Least squares in relative error, so small and large tau count equally.
    Returns (q1, q2), each clipped to >= 0.
    """
    taus = np.asarray(taus, float)
    s2 = np.asarray(adev, float) ** 2
    A = np.column_stack([1.0 / taus, taus / 3.0]) / s2[:, None]
    b = np.ones_like(taus)
    (q1, q2), *_ = np.linalg.lstsq(A, b, rcond=None)
    return max(q1, 0.0), max(q2, 0.0)