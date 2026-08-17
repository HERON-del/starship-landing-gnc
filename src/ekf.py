"""
Extended Kalman filter for the planar flip-and-land vehicle.

State:  [x, z, vx, vz, theta, omega]
Input:  (sigma, delta) -- the commanded throttle and gimbal, known exactly

Mass is carried alongside rather than estimated. It is not observed by either
instrument, and it is not really uncertain: the vehicle knows what it commanded
and Isp is a constant, so propagating it deterministically is both simpler and
more accurate than asking the filter to infer it from data that says nothing
about it.

Two things make this an *extended* filter rather than a plain one. The process
model is the true coupled dynamics -- thrust points along the body axis plus the
gimbal, drag goes as the square of airspeed through an attitude-dependent area
-- so the mean is propagated by RK4 through the real thing, exactly. Only the
covariance needs a linear model, and its Jacobian is taken numerically by
central differences about the current estimate. That is slower per call than
hand-derived partials and much harder to get wrong, and since it only shapes
uncertainty rather than the estimate itself, a locally accurate Jacobian is
enough. The same argument the trust region rests on.

The filter is told nothing about the wind. That is the whole point: gusts are
what the process noise `Q` exists to represent, and a filter that knew the
disturbance would not be measuring anything interesting.
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.dynamics_6dof import Vehicle6DoF, G0, G_EARTH      # noqa: E402
from src.aero import AeroConfig, aero_acceleration           # noqa: E402
from src.sensors import H_NAV, H_ATT                         # noqa: E402

N_STATE = 6


def default_process_noise(scale: float = 1.0) -> np.ndarray:
    """
    Process noise: how much the filter distrusts its own dynamics per second.

    Weighted toward the velocity rows because that is where the disturbance
    physically enters -- a gust is an unmodelled acceleration, not an
    unmodelled teleport. Position picks up its uncertainty through integration
    of the velocity it did not know about, which the propagation handles on its
    own; adding much position noise on top double-counts it.

    The magnitude was measured rather than guessed, and the first guess was
    wrong by two orders of magnitude. Sweeping the scale over the closed loop:

        scale     est error     miss
        x1          2.22 m     34.45 m
        x10         1.55 m     11.74 m
        x100        1.60 m      3.13 m

    Note what barely moves. A filter tuned a hundred times tighter estimates
    about as accurately on average and lands eleven times further away, because
    an under-confident Q makes the filter trust its own dynamics through gusts
    it cannot see, and the resulting error is a *lag* rather than noise. Mean
    estimation error hides that completely -- a lagging estimate and a noisy
    one look alike by that measure, and behave nothing alike in a control loop,
    since successive replans average noise out and cannot average out a bias.
    """
    q = np.array([0.5, 0.5, 20.0, 20.0, 1e-3, 1e-2]) ** 2
    return np.diag(q) * float(scale)


def default_initial_covariance() -> np.ndarray:
    """Prior at ignition: roughly the nav sensor's own noise, loosened."""
    p = np.array([5.0, 5.0, 1.0, 1.0, np.radians(1.0), np.radians(2.0)]) ** 2
    return np.diag(p)


class EKF:
    """
    Extended Kalman filter over the six translational and rotational states.

    Usage is predict/update: `predict` carries the estimate and its covariance
    forward under the commanded control, `update_nav` and `update_attitude`
    fold in whichever reading has arrived. Both updates are ordinary linear
    Kalman corrections, since both instruments observe the state directly.
    """

    def __init__(self, x0, m0, vehicle: Vehicle6DoF = None,
                 aero: AeroConfig = None, P0=None, Q=None):
        self.vehicle = vehicle or Vehicle6DoF()
        self.aero = aero if aero is not None else AeroConfig()
        self.x = np.asarray(x0, dtype=float).copy()
        self.m = float(m0)
        self.P = default_initial_covariance() if P0 is None else np.array(P0,
                                                                          float)
        self.Q = default_process_noise() if Q is None else np.array(Q, float)
        # Dimension is read off the state rather than fixed, so an augmented
        # filter can subclass this one instead of copying it. Day 12 adds a
        # gyro-bias state and changes nothing else in the machinery.
        self.n = self.x.size
        # Finite-difference steps, per state, since metres and radians are not
        # comparable quantities to perturb by the same amount.
        self._eps = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-5, 1e-5])

    # -- process model --------------------------------------------------
    def _deriv(self, x, sigma, delta):
        """
        The vehicle's own dynamics, in the filter's six states.

        Identical in form to `dynamics_in_wind` with no wind, because the wind
        is exactly what the filter does not know.
        """
        xp, z, vx, vz, th, om = x
        veh = self.vehicle
        m = max(self.m, veh.m_dry)

        T = float(np.clip(sigma, 0.0, veh.T_max))
        d = float(np.clip(delta, -veh.delta_max, veh.delta_max))
        if self.m <= veh.m_dry:
            Tx = Tz = tau = 0.0
        else:
            Tx = T * np.sin(th + d)
            Tz = T * np.cos(th + d)
            tau = T * veh.L_engine * np.sin(d)

        if self.aero is not None and self.aero.enabled:
            ax, az = aero_acceleration(vx, vz, max(z, 0.0), th, m, self.aero)
        else:
            ax = az = 0.0

        return np.array([vx, vz,
                         Tx / m + float(ax),
                         Tz / m + float(az) - G_EARTH,
                         om,
                         tau / veh.I_pitch])

    def _rk4(self, x, sigma, delta, dt):
        k1 = self._deriv(x, sigma, delta)
        k2 = self._deriv(x + 0.5 * dt * k1, sigma, delta)
        k3 = self._deriv(x + 0.5 * dt * k2, sigma, delta)
        k4 = self._deriv(x + dt * k3, sigma, delta)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _jacobian(self, x, sigma, delta, dt):
        """
        d(propagated state)/d(state), by central differences.

        Central rather than forward: the extra evaluations are cheap here and
        the error falls as the square of the step instead of linearly, which
        matters because the step has to be small enough not to smear the
        attitude terms and large enough to survive cancellation.
        """
        n = self.n
        F = np.zeros((n, n))
        # Scaled per state, since metres and radians are not comparable.
        eps = self._eps
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = eps[j]
            hi = self._rk4(x + dx, sigma, delta, dt)
            lo = self._rk4(x - dx, sigma, delta, dt)
            F[:, j] = (hi - lo) / (2.0 * eps[j])
        return F

    # -- filter steps ---------------------------------------------------
    def predict(self, sigma, delta, dt):
        """Carry estimate and covariance forward under the commanded control."""
        if dt <= 0:
            return self.x
        F = self._jacobian(self.x, sigma, delta, dt)
        self.x = self._rk4(self.x, sigma, delta, dt)
        self.P = F @ self.P @ F.T + self.Q * dt

        # Mass is known, not estimated: it follows from what was commanded.
        veh = self.vehicle
        if self.m > veh.m_dry:
            self.m = max(self.m - float(np.clip(sigma, 0.0, veh.T_max))
                         * dt / (veh.isp * G0), veh.m_dry)
        return self.x

    def _update(self, H, R, z_meas):
        """
        One linear correction, in Joseph form.

        `(I - KH) P` is algebraically correct and numerically fragile: it
        subtracts two nearly equal matrices, and over a few hundred updates the
        roundoff can drive the covariance out of symmetry and eventually
        indefinite. The Joseph form costs an extra multiply and stays positive
        semi-definite by construction, which is worth it in a filter that runs
        for the whole descent.
        """
        H = np.asarray(H, float)
        R = np.asarray(R, float)
        y = np.asarray(z_meas, float) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)      # kill any residual asymmetry
        return y

    def update_nav(self, z_meas, R):
        return self._update(H_NAV, R, z_meas)

    def update_attitude(self, z_meas, R):
        return self._update(H_ATT, R, z_meas)

    # -- reporting ------------------------------------------------------
    def state(self) -> dict:
        """Estimate in the shape the guidance loop expects."""
        return {"x": self.x[0], "z": self.x[1], "vx": self.x[2],
                "vz": self.x[3], "theta": self.x[4], "omega": self.x[5],
                "m": self.m}

    def sigma(self) -> np.ndarray:
        """One-sigma uncertainty per state, for plotting against the error."""
        return np.sqrt(np.clip(np.diag(self.P), 0.0, None))

    def position_sigma(self) -> float:
        return float(np.hypot(*self.sigma()[:2]))
