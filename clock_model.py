"""
clock_model.py - simulate the time error of an atomic clock.

Based on the two-state clock model in:
    L. Galleani, "A tutorial on the two-state model of the atomic clock noise",
    Metrologia 45 (2008) S175-S182.

Quantities (same names as the paper):
    x(t)  phase deviation  = clock reading minus true time, in seconds
    y(t)  frequency deviation = dx/dt, dimensionless (e.g. 3e-10)

Random part (paper eqs. 41-46):
    q1  white frequency noise      -> Allan deviation slopes DOWN  (~ 1/sqrt(tau))
    q2  random-walk frequency noise -> Allan deviation slopes UP    (~ sqrt(tau))
    Allan variance of the model (eq. 47):  sigma_y^2(tau) = q1/tau + q2*tau/3

Deterministic part (paper eq. 50, plus the "third state" the paper mentions
for rubidium-type clocks, which the CSAC behaves like):
    x0     initial phase offset (s)         - e.g. sync error at the surface
    y0     initial frequency offset         - e.g. calibration error
    aging  conant frequency drift (1/s)   - frequency changes linearly in time
"""

import numpy as np

SECONDS_PER_MONTH = 30 * 86400


class ClockParams:
    """All the numbers that describe one clock."""

    def __init__(self, name, q1, q2, aging=0.0, x0=0.0, y0=0.0):
        self.name = name
        self.q1 = q1        # white FM noise level  [s]
        self.q2 = q2        # random-walk FM level  [1/s]
        self.aging = aging  # frequency drift rate  [1/s]
        self.x0 = x0        # initial phase offset  [s]
        self.y0 = y0        # initial freq offset   [-]

    def adev_theory(self, tau):
        """Allan deviation predicted by the model (paper eq. 47), random part only."""
        tau = np.asarray(tau, dtype=float)
        return np.sqrt(self.q1 / tau + self.q2 * tau / 3.0)


# ---------------------------------------------------------------------------
# Microchip CSAC SA65 (Industrial, 090-02789-002), from the datasheet DS00003478F
#   ADEV: 3e-10 @1 s, 1e-10 @10 s, 3e-11 @100 s, 1e-11 @1000 s
#   Those four points lie on a 1/sqrt(tau) line -> pure white FM:
#       q1 = (3e-10)^2 * 1 s = 9e-20 s
#   No upward (random-walk) slope is visible up to 1000 s, so q2 is only bounded:
#       q2 * 1000/3 must stay below 1e-22 - 9e-23  ->  q2 <= 3e-26 /s
#   We use that upper bound (conservative). Replace with fitted values from
#   your own CSAC data once you have it.
#   Aging: < 9e-10 per month (typical, after 30 days of operation).
# ---------------------------------------------------------------------------
CSAC_SA65_SPEC_ADEV = {1: 3e-10, 10: 1e-10, 100: 3e-11, 1000: 1e-11}


def csac_sa65(x0=0.0, y0=0.0, aging=9e-10 / SECONDS_PER_MONTH):
    return ClockParams("CSAC SA65", q1=9e-20, q2=3e-26, aging=aging, x0=x0, y0=y0)


def simulate_phase(params, duration_s, dt_s, n_runs=1, rng=None,
                   include_noise=True, include_deterministic=True):
    """
    Simulate clock phase error x(t).

    Returns (t, x): t has shape (n,), x has shape (n_runs, n), both in seconds.
    Uses the exact discrete-time model (paper eqs. 60-70), so any dt_s is valid.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = int(round(duration_s / dt_s)) + 1
    t = np.arange(n) * dt_s
    x = np.zeros((n_runs, n))

    if include_noise:
        # Covariance of the noise added each step (paper eq. 68)
        T = dt_s
        c11 = params.q1 * T + params.q2 * T**3 / 3.0
        c12 = params.q2 * T**2 / 2.0
        c22 = params.q2 * T
        cov = np.array([[c11, c12], [c12, c22]])
        # Cholesky factor (tiny jitter keeps it valid if q2 == 0)
        L = np.linalg.cholesky(cov + np.eye(2) * 1e-60)
        z = rng.standard_normal((n_runs, n - 1, 2))
        eta = z @ L.T                         # correlated (eta_x, eta_y) per step
        eta_x, eta_y = eta[..., 0], eta[..., 1]

        # y_n = y_{n-1} + eta_y   ;   x_n = x_{n-1} + T*y_{n-1} + eta_x   (eq. 69)
        y_noise = np.concatenate([np.zeros((n_runs, 1)), np.cumsum(eta_y, axis=1)], axis=1)
        dx = T * y_noise[:, :-1] + eta_x
        x[:, 1:] += np.cumsum(dx, axis=1)

    if include_deterministic:
        x += params.x0 + params.y0 * t + 0.5 * params.aging * t**2

    return t, x