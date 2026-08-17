"""
Bias-aware EKF: Day 11's filter with the gyro bias promoted to a state.

State: [x, z, vx, vz, theta, omega, b_omega]

Everything from Day 11 carries over one dimension larger. The process model is
the same six equations plus a seventh, `db/dt = 0`, and the attitude
measurement changes in exactly one place: the rate row now reads *both* omega
and the bias, because the instrument genuinely cannot tell them apart in a
single reading.

    omega_meas = omega + b_omega + noise

That single shared row is what makes the problem non-trivial and also what
makes it solvable. The two states are indistinguishable in any one measurement
and behave quite differently over time -- omega is driven by the commanded
torque and swings through the flip, while the bias barely moves -- so given
enough readings the filter can separate them. It is the same reason a slowly
drifting baseline can be pulled out of any time series.

The bias process noise has to be small and non-zero. Exactly zero freezes the
filter's belief after the first update, because a state with no process noise
and no uncertainty left has a Kalman gain of zero and stops learning. Too large
and the filter treats the bias as fast-moving noise and chases it. Small and
non-zero lets the estimate keep tracking a slow thermal drift, which is what a
real bias does.
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.ekf import EKF, default_process_noise                 # noqa: E402
from src.dynamics_6dof import Vehicle6DoF                      # noqa: E402
from src.aero import AeroConfig                                # noqa: E402

N_STATE_BIAS = 7

# Nav sensor still sees no bias -- one extra zero column.
H_NAV_BIAS = np.zeros((4, N_STATE_BIAS))
H_NAV_BIAS[0, 0] = H_NAV_BIAS[1, 1] = 1.0
H_NAV_BIAS[2, 2] = H_NAV_BIAS[3, 3] = 1.0

# Attitude sensor: theta as before, and the rate row sums omega and the bias.
H_ATT_BIAS = np.zeros((2, N_STATE_BIAS))
H_ATT_BIAS[0, 4] = 1.0
H_ATT_BIAS[1, 5] = 1.0
H_ATT_BIAS[1, 6] = 1.0


def default_process_noise_bias(scale=1.0, bias_walk_deg_s=0.02):
    """
    Day 11's tuned 6x6 block, plus a random-walk variance for the bias.

    The six original entries are left exactly as Day 11 measured them -- that
    tuning was hard-won and is not what changed today. `bias_walk_deg_s` is the
    filter's belief about how fast the bias drifts, in deg/s per sqrt(s); the
    true process defaults to 0.01, so the filter is told the bias may wander
    somewhat faster than it really does. Erring loose keeps the estimate
    tracking rather than latching.
    """
    Q6 = default_process_noise(scale)
    Q = np.zeros((N_STATE_BIAS, N_STATE_BIAS))
    Q[:6, :6] = Q6
    Q[6, 6] = float(np.radians(bias_walk_deg_s)) ** 2
    return Q


def default_initial_covariance_bias(bias_sigma_deg_s=2.0):
    """
    Prior at ignition, with generous doubt about the bias.

    Starting the bias estimate at zero with tight covariance would tell the
    filter it already knows the answer, and it would take most of the flight to
    be argued out of it. A 2 deg/s one-sigma says: could be anything in the
    range an uncalibrated gyro plausibly has.
    """
    from src.ekf import default_initial_covariance
    P = np.zeros((N_STATE_BIAS, N_STATE_BIAS))
    P[:6, :6] = default_initial_covariance()
    P[6, 6] = float(np.radians(bias_sigma_deg_s)) ** 2
    return P


class BiasEKF(EKF):
    """
    Day 11's filter, augmented. Predict/update usage is unchanged.

    The only overrides are the seventh row of the process model (the bias does
    not move), the measurement matrices, and the finite-difference step for the
    new state. Everything else -- RK4 mean propagation, the numerical Jacobian,
    the Joseph-form update -- is inherited untouched, which is the point of
    having refactored the base class to read its dimension off the state.
    """

    def __init__(self, x0, m0, vehicle: Vehicle6DoF = None,
                 aero: AeroConfig = None, P0=None, Q=None, b0=0.0):
        x0 = np.asarray(x0, dtype=float)
        if x0.size == 6:
            x0 = np.append(x0, float(b0))
        super().__init__(x0, m0, vehicle, aero,
                         P0=default_initial_covariance_bias() if P0 is None
                         else P0,
                         Q=default_process_noise_bias() if Q is None else Q)
        self._eps = np.append(self._eps, 1e-6)

    def _deriv(self, x, sigma, delta):
        """The six original equations, plus a bias that does not move."""
        d6 = super()._deriv(np.asarray(x, dtype=float)[:6], sigma, delta)
        return np.append(d6, 0.0)

    def update_nav(self, z_meas, R):
        return self._update(H_NAV_BIAS, R, z_meas)

    def update_attitude(self, z_meas, R):
        return self._update(H_ATT_BIAS, R, z_meas)

    # -- reporting ------------------------------------------------------
    def state(self):
        s = super().state()
        s["b_omega"] = float(self.x[6])
        return s

    @property
    def bias(self) -> float:
        """Current bias estimate [rad/s]."""
        return float(self.x[6])

    @property
    def bias_sigma(self) -> float:
        """One-sigma uncertainty on the bias [rad/s]."""
        return float(np.sqrt(max(self.P[6, 6], 0.0)))

    def corrected_rate(self) -> float:
        """
        The rate the vehicle is actually turning at, bias removed.

        This is the quantity guidance should steer on, and the one the
        unaugmented filter cannot produce.
        """
        return float(self.x[5])


# ======================================================================
# Two simultaneous sensor errors
# ======================================================================
N_STATE_DUAL = 8

# Nav sensor: the downrange-velocity row now reads vx and its bias together,
# exactly as the rate row does for the gyro.
H_NAV_DUAL = np.zeros((4, N_STATE_DUAL))
H_NAV_DUAL[0, 0] = H_NAV_DUAL[1, 1] = 1.0
H_NAV_DUAL[2, 2] = H_NAV_DUAL[3, 3] = 1.0
H_NAV_DUAL[2, 7] = 1.0

H_ATT_DUAL = np.zeros((2, N_STATE_DUAL))
H_ATT_DUAL[0, 4] = 1.0
H_ATT_DUAL[1, 5] = 1.0
H_ATT_DUAL[1, 6] = 1.0


def default_process_noise_dual(scale=1.0, bias_walk_deg_s=0.02,
                               accel_walk_ms=0.05):
    """The 7x7 bias block, plus a random walk for the velocity bias."""
    Q7 = default_process_noise_bias(scale, bias_walk_deg_s)
    Q = np.zeros((N_STATE_DUAL, N_STATE_DUAL))
    Q[:7, :7] = Q7
    Q[7, 7] = float(accel_walk_ms) ** 2
    return Q


def default_initial_covariance_dual(bias_sigma_deg_s=2.0,
                                    accel_sigma_ms=3.0):
    P = np.zeros((N_STATE_DUAL, N_STATE_DUAL))
    P[:7, :7] = default_initial_covariance_bias(bias_sigma_deg_s)
    P[7, 7] = float(accel_sigma_ms) ** 2
    return P


class DualBiasEKF(BiasEKF):
    """
    Eight states: the vehicle, a gyro bias, and a velocity-channel bias.

    This is the generalisation the pattern promises. Nothing structural
    changes: one more row of `f` that stays at zero, one more diagonal entry
    in `Q`, and one more column in the measurement matrices with a 1 wherever
    the instrument cannot separate the error from the signal.

    The two biases are estimable at the same time because they are observed by
    *different instruments*. The gyro bias shows up only in the attitude
    sensor's rate channel and the velocity bias only in the nav sensor's
    downrange channel, so there is no path by which one can be mistaken for
    the other -- which is exactly why adding the second one does not disturb
    the first.
    """

    def __init__(self, x0, m0, vehicle: Vehicle6DoF = None,
                 aero: AeroConfig = None, P0=None, Q=None, b0=0.0, a0=0.0):
        x0 = np.asarray(x0, dtype=float)
        if x0.size == 6:
            x0 = np.append(x0, [float(b0), float(a0)])
        elif x0.size == 7:
            x0 = np.append(x0, float(a0))
        # Skip BiasEKF.__init__, which would size the 7-state defaults.
        EKF.__init__(self, x0, m0, vehicle, aero,
                     P0=default_initial_covariance_dual() if P0 is None
                     else P0,
                     Q=default_process_noise_dual() if Q is None else Q)
        self._eps = np.append(self._eps, [1e-6, 1e-4])

    def _deriv(self, x, sigma, delta):
        d6 = EKF._deriv(self, np.asarray(x, dtype=float)[:6], sigma, delta)
        return np.append(d6, [0.0, 0.0])

    def update_nav(self, z_meas, R):
        return self._update(H_NAV_DUAL, R, z_meas)

    def update_attitude(self, z_meas, R):
        return self._update(H_ATT_DUAL, R, z_meas)

    def state(self):
        s = super().state()
        s["b_vx"] = float(self.x[7])
        return s

    @property
    def accel_bias(self) -> float:
        """Current velocity-channel bias estimate [m/s]."""
        return float(self.x[7])

    @property
    def accel_bias_sigma(self) -> float:
        return float(np.sqrt(max(self.P[7, 7], 0.0)))
