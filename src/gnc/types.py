"""
Shared data contracts between solvers and the 3-D viewer.

Everything the web front end knows about a problem comes through these two
objects, so adding a new problem (Week 2 SCvx, Week 3 6-DoF, Week 5 MPC)
never requires touching JavaScript:

  * `Param`      declares one tunable knob -> the UI auto-builds a control for it
  * `Trajectory` is the universal result format -> the 3-D scene renders any of them

The frame convention is fixed everywhere: right-handed, **+Y is up**, ground
plane is y = 0, and the landing pad sits at the origin. Three.js uses the same
convention, so no axis juggling happens in the browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Sequence

import numpy as np

Vec3 = Sequence[float]


# ----------------------------------------------------------------------
# Parameter declaration
# ----------------------------------------------------------------------
@dataclass
class Param:
    """One tunable input. The UI renders a slider, a toggle, or a dropdown."""

    key: str
    label: str
    default: Any
    kind: Literal["float", "int", "bool", "choice"] = "float"
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str = ""
    group: str = "General"
    help: str = ""
    choices: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------
@dataclass
class Series:
    """A named scalar time series, plotted in the telemetry strip."""

    key: str
    label: str
    unit: str
    values: list[float]
    # "state" series have N+1 samples, "control" series have N (zero-order hold)
    on: Literal["state", "control"] = "state"


@dataclass
class Trajectory:
    """
    Universal solved-trajectory payload.

    A 1-D problem fills only the y component and leaves attitude as identity;
    a 6-DoF problem fills everything. The renderer does not care which.
    """

    t_state: list[float]                 # (N+1,)
    t_control: list[float]               # (N,)
    position: list[list[float]]          # (N+1, 3)  m
    velocity: list[list[float]]          # (N+1, 3)  m/s
    thrust: list[list[float]]            # (N, 3)    m/s^2 (or N applied accel)
    attitude: list[list[float]]          # (N+1, 4)  quaternion [x, y, z, w]
    series: list[Series] = field(default_factory=list)

    status: str = "unknown"
    feasible: bool = False
    cost: float | None = None
    solve_time_ms: float | None = None
    solver: str = ""
    thrust_max: float = 1.0              # for plume / colour normalisation
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["series"] = [asdict(s) for s in self.series]
        return d


# ----------------------------------------------------------------------
# Helpers used by the problem modules
# ----------------------------------------------------------------------
def quat_from_vector(v: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """
    Quaternion [x, y, z, w] rotating the body +Y axis onto `v`.

    Physical vehicles point their long axis along the thrust vector, so this is
    how a 3-DoF (translation-only) solution gets a plausible attitude for
    rendering. A real 6-DoF solver will return its own attitude and skip this.
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0])

    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])

    a = up / np.linalg.norm(up)
    b = v / n
    dot = float(np.dot(a, b))

    if dot > 1.0 - 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -1.0 + 1e-9:
        # 180 deg: any axis perpendicular to `a` works
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 0.0, 1.0]))
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])

    axis = np.cross(a, b)
    s = np.sqrt((1.0 + dot) * 2.0)
    q = np.array([axis[0] / s, axis[1] / s, axis[2] / s, s * 0.5])
    return q / np.linalg.norm(q)


def quats_from_pitch(theta: np.ndarray) -> np.ndarray:
    """
    Quaternions for a planar pitch angle measured from vertical.

    `theta = 0` is upright, `theta = pi/2` is horizontal. The body axis points
    along `(sin theta, cos theta, 0)`, matching the thrust convention in
    dynamics_6dof, and the renderer builds its vehicle along +Y — so this is a
    rotation of `-theta` about +Z.

    Unlike `attitudes_from_thrust`, which infers an attitude the solver never
    computed, this is the optimiser's own state variable. Use it whenever the
    problem actually solves for attitude.

    Parameters
    ----------
    theta : array-like, shape (n,)
        Pitch from vertical [rad].

    Returns
    -------
    np.ndarray, shape (n, 4)
        Quaternions as [x, y, z, w].
    """
    theta = np.asarray(theta, dtype=float)
    half = -0.5 * theta
    out = np.zeros((theta.size, 4))
    out[:, 2] = np.sin(half)
    out[:, 3] = np.cos(half)
    return out


def attitudes_from_thrust(thrust: np.ndarray, rel_deadband: float = 0.02) -> np.ndarray:
    """
    Per-state attitude quaternions from an (N, 3) thrust history.

    During a coast arc the solver returns thrust that is numerically tiny but
    not exactly zero, and its *direction* is pure noise. Pointing the vehicle
    along it makes the render jitter, so anything below `rel_deadband` of the
    peak holds the last commanded attitude instead — which is also what a real
    vehicle does when the engines are off.

    Control has N samples but state has N+1, so the final attitude repeats the
    last control sample.
    """
    if thrust.size == 0:
        return np.array([[0.0, 0.0, 0.0, 1.0]])

    mag = np.linalg.norm(thrust, axis=1)
    floor = float(mag.max()) * rel_deadband

    quats: list[np.ndarray] = []
    last = np.array([0.0, 0.0, 0.0, 1.0])
    for k in range(thrust.shape[0]):
        if mag[k] > floor:
            last = quat_from_vector(thrust[k])
        quats.append(last)

    # A leading coast has no previous command to hold; back-fill it with the
    # first real one so the vehicle does not snap upright on ignition.
    first_real = next((i for i, m in enumerate(mag) if m > floor), None)
    if first_real is not None:
        for i in range(first_real):
            quats[i] = quats[first_real]

    quats.append(quats[-1])
    return np.asarray(quats)
