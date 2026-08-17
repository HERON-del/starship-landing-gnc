"""
The true gyro bias, and a rate sensor that carries it.

Day 11's instruments had zero-mean noise: on average they told the truth, and a
Kalman filter is built to exploit exactly that, since averaging enough unbiased
readings converges on the answer. A bias is different in kind. A rate gyro
reading `omega + 1 deg/s` reads it *every time*; no amount of averaging removes
it, because it is not random relative to itself.

Day 11 measured what that costs. Sweeping an unestimated bias through the
closed loop, the miss ran 3.13, 3.43, 10.22 and 13.67 m at 0, 0.5, 1 and
2 deg/s -- while the position estimation error stayed flat near 1.5 m. The
filter reported good health the whole way down and the steering degraded
anyway, which is precisely the argument for giving the bias its own state.

This module is the reality side of that: a true bias no filter can see, and the
measurement it corrupts. `src/ekf_bias.py` is the filter that can back it out.
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.sensors import SensorConfig      # noqa: E402


class GyroBiasProcess:
    """
    The true bias: a slow random walk about a nonzero starting offset.

    The walk matters. A bias fixed for all time would be estimable once and
    then forgotten, which is not how thermal drift behaves and would make the
    filter's job artificially easy. Letting it wander slowly is both physically
    honest and the thing that forces the filter to keep tracking rather than
    latching onto its first answer.
    """

    def __init__(self, b0=np.radians(1.0), sigma_walk=np.radians(0.01),
                 seed=0):
        """
        b0 : starting bias [rad/s]. An uncalibrated MEMS gyro might carry
             0.5 to 2 deg/s.
        sigma_walk : diffusion rate [rad/s per sqrt(s)] -- how fast the bias
             itself drifts over a flight.
        """
        self.b0 = float(b0)
        self.b = float(b0)
        self.sigma_walk = float(sigma_walk)
        self.rng = np.random.default_rng(seed)

    def step(self, dt):
        """Advance the true bias by `dt` and return it."""
        if dt > 0:
            self.b += self.rng.normal() * self.sigma_walk * np.sqrt(dt)
        return self.b


def measure_attitude_biased(truth, bias_true, cfg: SensorConfig, rng):
    """
    An attitude reading corrupted by the true bias as well as by noise.

    Only the rate channel carries it, which is where a real gyro carries it,
    and the filter is never told the value -- it has to infer it from the fact
    that a bias and a genuine rotation behave differently over time even though
    they are indistinguishable in any single reading.
    """
    sig = np.array([cfg.sigma_theta, cfg.sigma_omega])
    out = np.asarray(truth[4:6], dtype=float) + rng.normal(size=2) * sig
    out[1] += float(bias_true)
    return out


class BiasedSensorSuite:
    """
    Day 11's suite with the rate channel biased.

    Holds the true bias process so the loop advances it on the same clock the
    vehicle flies on, and records its history so a test can check the filter's
    estimate against what the bias actually did rather than against its
    starting value.
    """

    def __init__(self, cfg: SensorConfig = None, seed: int = 0,
                 bias_process: GyroBiasProcess = None):
        from src.sensors import measure_nav
        self._measure_nav = measure_nav
        self.cfg = cfg or SensorConfig()
        self.rng = np.random.default_rng(seed)
        self.bias = bias_process or GyroBiasProcess(seed=seed + 1000)
        self._next_nav = 0.0
        self._next_att = 0.0
        self.history = {"t": [], "b_true": []}

    def advance(self, t, dt):
        """Step the true bias, and log it."""
        b = self.bias.step(dt)
        self.history["t"].append(t)
        self.history["b_true"].append(b)
        return b

    def due(self, t, truth):
        """Readings that have come due at or before `t`."""
        out = {}
        if self.cfg.nav_enabled and t + 1e-12 >= self._next_nav:
            out["nav"] = self._measure_nav(truth, self.cfg, self.rng)
            while self._next_nav <= t + 1e-12:
                self._next_nav += self.cfg.dt_nav
        if t + 1e-12 >= self._next_att:
            out["att"] = measure_attitude_biased(truth, self.bias.b, self.cfg,
                                                 self.rng)
            while self._next_att <= t + 1e-12:
                self._next_att += self.cfg.dt_att
        return out
