"""
Simulated sensors for closed-loop navigation.

Day 10's guidance loop read the truth state directly, which is the one thing a
real vehicle never gets. These are the two instruments it would actually have,
each with its own noise and its own update rate:

    attitude sensor   fast, and only knows which way the vehicle points
    navigation sensor slower and noisier, and knows where it is

Neither is sufficient on its own. The attitude sensor says nothing about
position, and dead-reckoning position from attitude alone drifts without bound.
The nav sensor knows position but arrives too rarely to steer on directly --
guidance replans every 0.5 s and a 5 Hz sensor has told it something new only
twice in that window, each time with metres of noise on it. Fusing them is what
the EKF is for.

The state vector everywhere in this module is the vehicle's seven-element one,
`[x, z, vx, vz, theta, omega, m]`, and both measurement models are linear
selections from it. That is realistic -- radar and Doppler observe position and
velocity fairly directly, and a horizon sensor observes attitude fairly
directly. What is nonlinear is the dynamics connecting those observations
between updates, which is precisely why the filter has to be extended.
"""

from dataclasses import dataclass

import numpy as np

# State order the filter and the sensors agree on.
STATE_LABELS = ("x", "z", "vx", "vz", "theta", "omega")

# Linear observation models. Constant, because the nonlinearity is in the
# process model rather than in what the instruments see.
H_NAV = np.zeros((4, 6))
H_NAV[0, 0] = H_NAV[1, 1] = H_NAV[2, 2] = H_NAV[3, 3] = 1.0

H_ATT = np.zeros((2, 6))
H_ATT[0, 4] = H_ATT[1, 5] = 1.0


@dataclass
class SensorConfig:
    """Noise and update rate for both instruments."""

    # Navigation sensor: radar altimeter and Doppler velocimeter.
    nav_rate_hz: float = 5.0
    sigma_x: float = 3.0                    # [m]
    sigma_z: float = 2.0                    # [m]
    sigma_vx: float = 0.5                   # [m/s]
    sigma_vz: float = 0.5                   # [m/s]

    # Attitude sensor: rate gyro with a horizon reference.
    att_rate_hz: float = 20.0
    sigma_theta: float = np.radians(0.3)    # [rad]
    sigma_omega: float = np.radians(0.5)    # [rad/s]

    # A slowly-varying gyro bias, off by default. Real rate gyros have one,
    # and a filter that does not estimate it cannot remove it -- the point of
    # the bias experiment.
    omega_bias: float = 0.0                 # [rad/s]

    # Set very large to switch the nav sensor off without special-casing the
    # loop, which is how the dead-reckoning experiment is run.
    nav_enabled: bool = True

    @property
    def dt_nav(self) -> float:
        return 1.0 / self.nav_rate_hz

    @property
    def dt_att(self) -> float:
        return 1.0 / self.att_rate_hz

    def R_nav(self) -> np.ndarray:
        return np.diag([self.sigma_x ** 2, self.sigma_z ** 2,
                        self.sigma_vx ** 2, self.sigma_vz ** 2])

    def R_att(self) -> np.ndarray:
        return np.diag([self.sigma_theta ** 2, self.sigma_omega ** 2])

    def summary(self) -> str:
        return "\n".join([
            "Sensors",
            f"  nav       : {self.nav_rate_hz:>5.1f} Hz  "
            f"sigma [{self.sigma_x:.1f} m, {self.sigma_z:.1f} m, "
            f"{self.sigma_vx:.2f} m/s, {self.sigma_vz:.2f} m/s]"
            + ("" if self.nav_enabled else "   DISABLED"),
            f"  attitude  : {self.att_rate_hz:>5.1f} Hz  "
            f"sigma [{np.degrees(self.sigma_theta):.2f} deg, "
            f"{np.degrees(self.sigma_omega):.2f} deg/s]"
            + (f"   bias {np.degrees(self.omega_bias):+.2f} deg/s"
               if self.omega_bias else ""),
        ])


def measure_nav(truth, cfg: SensorConfig, rng) -> np.ndarray:
    """One noisy nav reading: [x, z, vx, vz]."""
    sig = np.array([cfg.sigma_x, cfg.sigma_z, cfg.sigma_vx, cfg.sigma_vz])
    return np.asarray(truth[:4], dtype=float) + rng.normal(size=4) * sig


def measure_attitude(truth, cfg: SensorConfig, rng) -> np.ndarray:
    """
    One noisy attitude reading: [theta, omega].

    The bias is added to the rate channel only, which is where a real gyro
    carries it, and it is deliberately not visible to the filter.
    """
    sig = np.array([cfg.sigma_theta, cfg.sigma_omega])
    out = np.asarray(truth[4:6], dtype=float) + rng.normal(size=2) * sig
    out[1] += cfg.omega_bias
    return out


class SensorSuite:
    """
    Both instruments on one clock.

    `due(t)` reports which readings have become available since the last call,
    so the loop can run at whatever rate it likes and still see each sensor at
    its own. Measurements are generated only when they are due -- asking for a
    reading that has not arrived yet would be inventing data.
    """

    def __init__(self, cfg: SensorConfig = None, seed: int = 0):
        self.cfg = cfg or SensorConfig()
        self.rng = np.random.default_rng(seed)
        self._next_nav = 0.0
        self._next_att = 0.0

    def due(self, t: float, truth) -> dict:
        """Readings that have come due at or before `t`."""
        out = {}
        if self.cfg.nav_enabled and t + 1e-12 >= self._next_nav:
            out["nav"] = measure_nav(truth, self.cfg, self.rng)
            # Step past t rather than by one interval, so a coarse caller
            # cannot accumulate a backlog of stale readings.
            while self._next_nav <= t + 1e-12:
                self._next_nav += self.cfg.dt_nav
        if t + 1e-12 >= self._next_att:
            out["att"] = measure_attitude(truth, self.cfg, self.rng)
            while self._next_att <= t + 1e-12:
                self._next_att += self.cfg.dt_att
        return out
