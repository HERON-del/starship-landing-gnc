"""
3-D aerodynamics: angle of attack, sideslip, a drag/lift/side-force
decomposition, and aerodynamic moments. Laid on top of Day 14's validated
rigid body -- the same one-physics-layer-per-day pattern as Day 6, which put
aero on top of an already-working rotational model.

What is actually new relative to Day 6
--------------------------------------
Day 6's `aero.py` already had drag *and* lift, with the same `Cl_max sin(2a)`
curve and the same coefficients. What it did not have is sideslip, a side
force, or any moment at all -- it pushed the vehicle without ever rotating it,
because in Days 5 to 12 every torque came from the engine gimbal.

And it had one thing wrong. Day 6 blends reference area and drag coefficient by
`theta`, the pitch from *vertical*, when the quantity that decides how much
vehicle the air sees is the angle between the body axis and the *relative
wind*. Those agree only when the wind is vertical. Day 6 is internally
inconsistent about it too: it computes the wind-relative angle correctly for
lift and then uses the vertical-relative angle for area and drag. This file
uses the wind-relative angle throughout, and `tests/test_aero_3d.py` measures
what the difference is worth in the regime this project actually flies.

Frames
------
`u, v, w` are the components of the vehicle's velocity relative to the air, in
the BODY frame, with +x the long axis exactly as in Day 14:

    V              = |(u, v, w)|
    alpha          = atan2(w, u)                 angle of attack, pitch plane
    beta           = asin(v / V)                 sideslip, out of plane
    angle_off_axis = atan2(hypot(v, w), u)       total angle off the long axis

alpha is Day 6's angle generalised; beta and `angle_off_axis` have no planar
equivalent -- for planar motion beta is identically zero and `angle_off_axis`
is |alpha|.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.quaternion import quat_to_rotmatrix                   # noqa: E402
from src.atmosphere import density                             # noqa: E402
from src.dynamics_3d import (                                  # noqa: E402
    Vehicle3D, dynamics_3d_derivative, rk4_step_3d_dynamics,
    propagate_3d_dynamics, IDX_VEL, IDX_QUAT,
)


@dataclass
class AeroConfig3D:
    """
    3-D aerodynamic configuration.

    Geometry and coefficients match Day 6's `AeroConfig` exactly wherever there
    is a counterpart, so the reduction check compares two models of one vehicle.
    """

    diameter: float = 9.0
    length: float = 50.0

    Cd_nose: float = 0.3          # axial flow
    Cd_belly: float = 1.2         # broadside flow
    Cl_max: float = 0.4           # peak lift coefficient, at alpha = 45 deg
    Cy_beta: float = 0.8          # side force per radian of sideslip

    #: Centre of pressure along the body axis, measured from the centre of
    #: mass. POSITIVE puts it aft, toward the engine, which makes the vehicle
    #: weathervane into the wind like an arrow. Negative puts it forward and
    #: the vehicle diverges instead. Which one a real Starship has in the
    #: belly-flop is the reason it carries flaps; see the note in the test
    #: suite, and treat this default as a choice rather than a measurement.
    x_cp: float = 5.0

    #: Below this relative speed the wind direction is numerically meaningless,
    #: so the angles are reported as zero and every force and moment is zero.
    #: Same guard and same value as Day 6.
    v_min: float = 5.0

    enabled: bool = True

    @property
    def A_nose(self) -> float:
        return np.pi * (self.diameter / 2.0) ** 2

    @property
    def A_belly(self) -> float:
        return self.diameter * self.length

    def summary(self) -> str:
        return "\n".join([
            "AeroConfig3D",
            f"  A_nose / A_belly  : {self.A_nose:>10.1f} / "
            f"{self.A_belly:.1f} m^2  ({self.A_belly / self.A_nose:.1f}x)",
            f"  Cd_nose / Cd_belly: {self.Cd_nose:>10.2f} / {self.Cd_belly:.2f}",
            f"  Cl_max            : {self.Cl_max:>10.2f}",
            f"  Cy_beta           : {self.Cy_beta:>10.2f} per rad",
            f"  x_cp              : {self.x_cp:>10.1f} m "
            f"({'aft, stable' if self.x_cp > 0 else 'forward, unstable'})",
            f"  v_min             : {self.v_min:>10.1f} m/s",
        ])


# ======================================================================
# Relative wind and the two angles
# ======================================================================
def relative_wind_body(v_inertial, wind_inertial, q):
    """Vehicle velocity relative to the air, in the body frame."""
    v_rel = np.asarray(v_inertial, dtype=float) - np.asarray(
        wind_inertial, dtype=float)
    return quat_to_rotmatrix(np.asarray(q, dtype=float)).T @ v_rel


def aero_angles(v_rel_body, v_min=5.0):
    """
    (alpha, beta, V, angle_off_axis), angles in radians.

    Below `v_min` the angles are reported as zero because the direction of a
    near-zero vector is noise. V itself is still returned truthfully -- callers
    that want to know how slow it is should read that rather than infer it from
    the angles.
    """
    u, v, w = (float(x) for x in v_rel_body)
    V = float(np.sqrt(u * u + v * v + w * w))
    if V < v_min:
        return 0.0, 0.0, V, 0.0
    return (float(np.arctan2(w, u)),
            float(np.arcsin(np.clip(v / V, -1.0, 1.0))),
            V,
            float(np.arctan2(np.hypot(v, w), u)))


def angles_to_relative_wind(alpha, beta, V):
    """Inverse of `aero_angles`, for round-trip testing."""
    return np.array([V * np.cos(alpha) * np.cos(beta),
                     V * np.sin(beta),
                     V * np.sin(alpha) * np.cos(beta)])


# ======================================================================
# Coefficients
# ======================================================================
def effective_area_and_Cd(angle_off_axis, cfg: AeroConfig3D = None):
    """
    Blend the nose and belly values by how far the wind is off the long axis.

    `sin` is the right shape here rather than a convenience: it is zero at both
    0 and 180 degrees -- nose-on and tail-on present the same circular section
    -- and peaks at exactly 90 degrees, the broadside case. At 90 degrees this
    returns Day 6's belly values unchanged.
    """
    cfg = cfg or AeroConfig3D()
    s = abs(np.sin(angle_off_axis))
    return (cfg.A_nose + (cfg.A_belly - cfg.A_nose) * s,
            cfg.Cd_nose + (cfg.Cd_belly - cfg.Cd_nose) * s)


def lift_coefficient(alpha, cfg: AeroConfig3D = None):
    """Bluff-body lift curve: zero nose-on and broadside, peak at 45 deg."""
    cfg = cfg or AeroConfig3D()
    return cfg.Cl_max * np.sin(2.0 * alpha)


# ======================================================================
# Force and moment
# ======================================================================
def aero_force_body(v_rel_body, altitude, cfg: AeroConfig3D = None):
    """
    Drag, lift and side force in the body frame.

    Drag opposes the full relative wind. Lift is perpendicular to it within the
    pitch plane. The side force acts along body y and opposes sideslip, which
    is the aerodynamic equivalent of a restoring spring.
    """
    cfg = cfg or AeroConfig3D()
    v_rel_body = np.asarray(v_rel_body, dtype=float)
    alpha, beta, V, off_axis = aero_angles(v_rel_body, cfg.v_min)
    if not cfg.enabled or V < cfg.v_min:
        return np.zeros(3)

    q_dyn = 0.5 * density(max(float(altitude), 0.0)) * V * V
    A_eff, Cd_eff = effective_area_and_Cd(off_axis, cfg)
    u, _, w = v_rel_body

    F = -(q_dyn * A_eff * Cd_eff) * (v_rel_body / V)

    # Lift acts perpendicular to the wind, in the pitch plane, and the sign
    # is not a free choice. At positive angle of attack the vehicle is moving
    # in +z relative to the air, so the air pushes it in -z; the perpendicular
    # component of that reaction has to continue pushing the same way, not
    # against it. Taking [-w, 0, u] instead -- as the Day 15 guide does --
    # produces a lift that overwhelms the drag's normal component at small
    # angles and turns the vehicle AWAY from the wind, which would make a
    # centre of pressure aft of the centre of mass destabilising. See the
    # weathervaning test.
    uw = float(np.hypot(u, w))
    if uw > 1e-9:
        F = F + (q_dyn * A_eff * lift_coefficient(alpha, cfg)) * (
            np.array([w, 0.0, -u]) / uw)

    F[1] -= q_dyn * A_eff * cfg.Cy_beta * beta
    return F


def aero_moment_body(F_aero_body, cfg: AeroConfig3D = None):
    """
    Moment from a centre of pressure offset along the body axis.

    Deliberately the same `r x F` pattern as Day 14's engine torque, and it
    inherits both of that pattern's consequences: no roll moment ever, and
    exactly zero moment when the force is collinear with the offset -- which
    for aerodynamics means purely axial flow.
    """
    cfg = cfg or AeroConfig3D()
    return np.cross(np.array([-cfg.x_cp, 0.0, 0.0]),
                    np.asarray(F_aero_body, dtype=float))


def aero_force_and_moment_body(v_inertial, wind_inertial, q, altitude,
                               cfg: AeroConfig3D = None):
    """Inertial state in, body-frame force and moment out."""
    cfg = cfg or AeroConfig3D()
    F = aero_force_body(relative_wind_body(v_inertial, wind_inertial, q),
                        altitude, cfg)
    return F, aero_moment_body(F, cfg)


# ======================================================================
# Combined dynamics: Day 14's rigid body, plus today's aero
# ======================================================================
def aero_wrench_fn(aero_cfg: AeroConfig3D, wind_inertial=(0.0, 0.0, 0.0)):
    """
    A body-frame wrench closure for `dynamics_3d_derivative`.

    Day 15 is a layer, not a fork. Gravity, the body-to-inertial rotation,
    Euler's equations and the mass flow all stay in `dynamics_3d.py` with one
    implementation and one test suite; this supplies the extra force and torque
    and nothing else.
    """
    cfg = aero_cfg or AeroConfig3D()
    wind = np.asarray(wind_inertial, dtype=float)

    def wrench(s):
        return aero_force_and_moment_body(
            s[IDX_VEL], wind, s[IDX_QUAT], s[2], cfg)
    return wrench


def dynamics_3d_with_aero_derivative(s, T_mag, delta_y, delta_z,
                                     vehicle: Vehicle3D,
                                     aero_cfg: AeroConfig3D = None,
                                     wind_inertial=(0.0, 0.0, 0.0),
                                     include_gyro=True):
    """Thrust and gravity from Day 14, plus today's aerodynamic wrench."""
    return dynamics_3d_derivative(
        s, T_mag, delta_y, delta_z, vehicle, include_gyro=include_gyro,
        extra_body_wrench=aero_wrench_fn(aero_cfg, wind_inertial))


def rk4_step_3d_with_aero(s, T_mag, delta_y, delta_z, dt, vehicle,
                          aero_cfg: AeroConfig3D = None,
                          wind_inertial=(0.0, 0.0, 0.0), include_gyro=True):
    """One RK4 step of the combined model."""
    return rk4_step_3d_dynamics(
        s, T_mag, delta_y, delta_z, dt, vehicle, include_gyro=include_gyro,
        extra_body_wrench=aero_wrench_fn(aero_cfg, wind_inertial))


def propagate_3d_with_aero(s0, control_fn, t_span, dt, vehicle,
                           aero_cfg: AeroConfig3D = None,
                           wind_inertial=(0.0, 0.0, 0.0), include_gyro=True):
    """Propagate the combined model. `control_fn(t, s) -> (T, dy, dz)`."""
    return propagate_3d_dynamics(
        s0, control_fn, t_span, dt, vehicle, include_gyro=include_gyro,
        extra_body_wrench=aero_wrench_fn(aero_cfg, wind_inertial))


# ======================================================================
# Readouts
# ======================================================================
def angle_history(s_hist, wind_inertial=(0.0, 0.0, 0.0),
                  cfg: AeroConfig3D = None):
    """(alpha, beta, V, angle_off_axis) at every sample, in radians."""
    cfg = cfg or AeroConfig3D()
    wind = np.asarray(wind_inertial, dtype=float)
    return np.array([
        aero_angles(relative_wind_body(s[IDX_VEL], wind, s[IDX_QUAT]),
                    cfg.v_min)
        for s in np.asarray(s_hist)])


def static_margin_sign(cfg: AeroConfig3D = None) -> str:
    """Whether this configuration weathervanes into the wind or away from it."""
    cfg = cfg or AeroConfig3D()
    if abs(cfg.x_cp) < 1e-12:
        return "neutral"
    return "stable (weathervanes into the wind)" if cfg.x_cp > 0 \
        else "unstable (diverges from the wind)"
