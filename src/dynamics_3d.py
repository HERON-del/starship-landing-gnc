"""
Full 3-D rigid-body dynamics: Euler's rotational equations with a 3-axis
inertia tensor, a two-axis engine gimbal producing genuine 3-D torque, and
thrust and gravity forces. Built directly on Day 13's quaternion machinery.

State: s = [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]   (14,)

    position, velocity   INERTIAL frame, +z up
    quaternion           body to inertial, scalar first (Day 13 convention)
    angular velocity     BODY frame, which is what a rate gyro reports
    mass                 scalar

Body axes: +x is the vehicle's long axis, pointing away from the engine, so an
upright vehicle has body +x along inertial +z. y and z are the two transverse
axes. This is the standard aerospace body frame and it is *not* the convention
Day 13's viewer entry rendered with, which put the long axis on body +z -- that
was a rendering choice in a file with no vehicle in it, and the two never meet.
`attitude_from_pitch` and `pitch_from_attitude` are the bridge to Day 5's single
pitch angle, and `tests/test_dynamics_3d.py` uses them to check that the planar
model and this one are the same model.

Scope: thrust and gravity only. Aerodynamics is Day 15's addition, laid on top
of this once it is validated -- the same one-physics-layer-per-day pattern as
Day 6, which added aero on top of an already-working rotational model.

Known limitation, stated once here: the inertia tensor is constant while the
mass depletes. A vehicle that burns 30 t of its 130 t does change its inertia,
and this model does not track that. Day 5 made the same simplification with a
scalar I_pitch, so the two stay comparable; correcting it is a change to both.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.quaternion import (                                   # noqa: E402
    quat_to_rotmatrix, quat_kinematics, quat_normalize, quat_from_axis_angle,
)

G0 = 9.80665          # Isp reference gravity
G_EARTH = 9.80665     # local gravity

N_STATE_3D = 14
IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)
IDX_QUAT = slice(6, 10)
IDX_OMEGA = slice(10, 13)
IDX_MASS = 13

#: The vehicle's long axis in the body frame.
BODY_LONG = np.array([1.0, 0.0, 0.0])


@dataclass
class Vehicle3D:
    """
    3-D vehicle parameters.

    Every value with a Day 5 counterpart carries the same number, so the
    reduction check in the test suite compares two models of one vehicle
    rather than two different vehicles.
    """

    m_dry: float = 100_000.0
    m_prop_initial: float = 30_000.0

    n_engines: int = 3
    thrust_per_engine: float = 2.3e6
    throttle_min: float = 0.40
    throttle_max: float = 1.00
    isp: float = 327.0

    length: float = 50.0            # vehicle length [m]
    L_engine: float = 25.0          # CoM to gimbal point, along body -x [m]
    delta_max_deg: float = 15.0     # max deflection, EACH axis [deg]

    # Thin cylinder: I_roll = 1/2 m r^2 with m ~ m_wet and r = 4.5 m gives
    # 1.32e6, which is where this number comes from. I_pitch_yaw is Day 5's
    # I_pitch unchanged -- same vehicle, same transverse-axis inertia.
    I_roll: float = 1.3e6
    I_pitch_yaw: float = 2.7e7
    #: Overrides the yaw-axis inertia, breaking axisymmetry. Left as None the
    #: body is axisymmetric, which is what every claim in this file assumes;
    #: set it and the roll axis stops decoupling.
    I_yaw: float = None

    @property
    def m_wet(self) -> float:
        return self.m_dry + self.m_prop_initial

    @property
    def T_max(self) -> float:
        return self.n_engines * self.thrust_per_engine * self.throttle_max

    @property
    def T_min(self) -> float:
        return self.n_engines * self.thrust_per_engine * self.throttle_min

    @property
    def delta_max(self) -> float:
        return np.radians(self.delta_max_deg)

    @property
    def is_axisymmetric(self) -> bool:
        return self.I_yaw is None or self.I_yaw == self.I_pitch_yaw

    def __post_init__(self):
        # Cached once. The derivative is evaluated four times per RK4 step and
        # rebuilding a 3x3 there costs more than the physics does.
        izz = self.I_pitch_yaw if self.I_yaw is None else float(self.I_yaw)
        object.__setattr__(self, "_I_diag",
                           np.array([self.I_roll, self.I_pitch_yaw, izz]))

    @property
    def I_diag(self) -> np.ndarray:
        """
        The three principal moments.

        The tensor is diagonal by construction -- a cylindrical vehicle's
        principal axes are its body axes -- so the rotational equation is three
        divisions rather than a linear solve. That is not a shortcut taken for
        speed; a non-diagonal tensor would mean the body frame was not aligned
        with the principal axes, which this model does not allow.
        """
        return self._I_diag

    @property
    def I_body(self) -> np.ndarray:
        """The same thing as a matrix, for the conservation helpers."""
        return np.diag(self._I_diag)

    @property
    def tau_max(self) -> float:
        """Peak transverse torque, full thrust and full deflection [N m]."""
        return self.T_max * self.L_engine * np.sin(self.delta_max)

    def summary(self) -> str:
        I = self.I_body
        return "\n".join([
            "Vehicle configuration (3-D rigid body)",
            f"  Wet / dry mass    : {self.m_wet:>12,.0f} / "
            f"{self.m_dry:,.0f} kg",
            f"  T_max / T_min     : {self.T_max / 1e6:>12.2f} / "
            f"{self.T_min / 1e6:.2f} MN",
            f"  I_roll            : {I[0, 0]:>12.2e} kg m^2",
            f"  I_pitch / I_yaw   : {I[1, 1]:>12.2e} / {I[2, 2]:.2e} kg m^2",
            f"  transverse / roll : {I[1, 1] / I[0, 0]:>12.1f} x",
            f"  CG-to-engine      : {self.L_engine:>12.0f} m",
            f"  Gimbal max        : {self.delta_max_deg:>12.0f} deg per axis",
            f"  tau_max           : {self.tau_max:>12,.0f} N m",
            f"  axisymmetric      : {str(self.is_axisymmetric):>12s}",
        ])


# ======================================================================
# Forces and torques
# ======================================================================
def gimbal_force_and_torque_body(T_mag, delta_y, delta_z, vehicle):
    """
    Thrust force and the torque it produces, both in the body frame.

    `delta_y` deflects the thrust toward body -z and is exactly Day 5's single
    gimbal axis. `delta_z` deflects it toward body +y and is new. Exact trig,
    not a small-angle approximation -- at the 15 degree limit the two differ by
    about 1%, which is small, but there is no reason to accept it.

    The torque is r x F with r = [-L, 0, 0], which works out to
    [0, L*Fz, -L*Fy]. Two consequences fall straight out of that and are worth
    saying here rather than discovering later:

      * the x component is identically zero, so **this model cannot produce
        roll torque at all**, at any deflection. Real vehicles get roll
        authority by differentially throttling several engines; this project
        has modelled the engines as one effective thruster since Day 2, and
        that choice is what costs the roll axis.
      * at zero deflection the torque is exactly zero, because the force then
        acts along a line through the centre of mass.
    """
    thrust_dir = np.array([
        np.cos(delta_z) * np.cos(delta_y),
        np.sin(delta_z) * np.cos(delta_y),
        -np.sin(delta_y),
    ])
    F_body = T_mag * thrust_dir
    r_engine = np.array([-vehicle.L_engine, 0.0, 0.0])
    return F_body, np.cross(r_engine, F_body)


def gyroscopic_term(omega, I_body):
    """
    The omega x (I omega) term in Euler's equations.

    This is the piece with no planar equivalent. In a single-axis rotation
    omega and I omega are parallel, so the cross product is identically zero
    and Euler's equations collapse to Day 5's tau = I alpha. Kept as its own
    function so that claim can be measured rather than asserted -- see the
    `include_gyro` flag below.
    """
    omega = np.asarray(omega, dtype=float)
    return np.cross(omega, I_body @ omega)


# ======================================================================
# Dynamics
# ======================================================================
def dynamics_3d_derivative(s, T_mag, delta_y, delta_z, vehicle,
                           include_gyro=True):
    """
    The 14-state derivative: thrust and gravity, Euler's rotational equations
    with gyroscopic coupling, and mass depletion.

    `include_gyro=False` drops the coupling term. That is not a physical model
    -- it exists so the term's actual contribution can be measured by
    difference instead of argued about.
    """
    s = np.asarray(s, dtype=float)
    q = s[IDX_QUAT]
    omega = s[IDX_OMEGA]
    m = float(s[IDX_MASS])

    F_body, tau_body = gimbal_force_and_torque_body(
        T_mag, delta_y, delta_z, vehicle)

    R = quat_to_rotmatrix(q)
    F_thrust_inertial = R @ F_body
    F_gravity_inertial = np.array([0.0, 0.0, -G_EARTH * m])
    a_inertial = (F_thrust_inertial + F_gravity_inertial) / m

    I = vehicle.I_diag
    net = tau_body
    if include_gyro:
        net = net - np.cross(omega, I * omega)
    domega = net / I

    return np.concatenate([
        s[IDX_VEL],
        a_inertial,
        quat_kinematics(q, omega),
        domega,
        [-T_mag / (vehicle.isp * G0)],
    ])


def rk4_step_3d_dynamics(s, T_mag, delta_y, delta_z, dt, vehicle,
                         include_gyro=True):
    """One RK4 step, then renormalise the quaternion and floor the mass."""
    def f(state):
        return dynamics_3d_derivative(state, T_mag, delta_y, delta_z, vehicle,
                                      include_gyro=include_gyro)

    k1 = f(s)
    k2 = f(s + 0.5 * dt * k1)
    k3 = f(s + 0.5 * dt * k2)
    k4 = f(s + dt * k3)
    s_new = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    s_new[IDX_QUAT] = quat_normalize(s_new[IDX_QUAT])
    s_new[IDX_MASS] = max(s_new[IDX_MASS], vehicle.m_dry)
    return s_new


def propagate_3d_dynamics(s0, control_fn, t_span, dt, vehicle,
                          include_gyro=True):
    """
    Propagate the 14-state model.

    `control_fn(t, s) -> (T_mag, delta_y, delta_z)` so control can depend on
    time, on state, or on neither.
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    n = max(int(round((tf - t0) / dt)), 1)
    t_hist = np.zeros(n + 1)
    s_hist = np.zeros((n + 1, N_STATE_3D))
    t_hist[0] = t0
    s_hist[0] = np.asarray(s0, dtype=float)

    for k in range(n):
        T_mag, delta_y, delta_z = control_fn(t_hist[k], s_hist[k])
        s_hist[k + 1] = rk4_step_3d_dynamics(
            s_hist[k], T_mag, delta_y, delta_z, dt, vehicle,
            include_gyro=include_gyro)
        t_hist[k + 1] = t0 + (k + 1) * dt
    return t_hist, s_hist


def make_initial_state_3d(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0),
                          quat=(1.0, 0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0),
                          m=None, vehicle=None):
    """Assemble the 14-state vector, quaternion normalised."""
    if m is None:
        m = vehicle.m_wet if vehicle is not None else 130_000.0
    return np.concatenate([
        np.asarray(pos, dtype=float),
        np.asarray(vel, dtype=float),
        quat_normalize(np.asarray(quat, dtype=float)),
        np.asarray(omega, dtype=float),
        [float(m)],
    ])


# ======================================================================
# The bridge to Day 5's planar model
# ======================================================================
def attitude_from_pitch(theta):
    """
    Quaternion for Day 5's pitch-from-vertical.

    theta = 0 is upright (body +x along inertial +z) and theta = pi/2 is the
    belly-flop, body +x along inertial +x. That is a rotation about +y by
    theta - pi/2, which puts the long axis at (sin theta, 0, cos theta) -- the
    direction Day 5's Tx = T sin(theta), Tz = T cos(theta) assumes.
    """
    return quat_from_axis_angle([0.0, 1.0, 0.0], float(theta) - np.pi / 2.0)


def pitch_from_attitude(q):
    """
    Day 5's pitch angle recovered from a quaternion, signed.

    Reads the long axis in the inertial frame and takes atan2 of its downrange
    over its vertical component. Exact for planar motion, a projection for
    anything else.
    """
    v = quat_to_rotmatrix(
        quat_normalize(np.asarray(q, dtype=float))) @ BODY_LONG
    return float(np.arctan2(v[0], v[2]))


def tilt_from_vertical(q):
    """Unsigned angle between the long axis and inertial up [rad]."""
    v = quat_to_rotmatrix(
        quat_normalize(np.asarray(q, dtype=float))) @ BODY_LONG
    return float(np.arccos(np.clip(v[2], -1.0, 1.0)))


# ======================================================================
# Conserved quantities -- the checks that are theorems
# ======================================================================
def angular_momentum_inertial(s, vehicle):
    """L = R(q) (I omega). Torque-free, this is a constant *vector*."""
    s = np.asarray(s, dtype=float)
    return quat_to_rotmatrix(s[IDX_QUAT]) @ (vehicle.I_body @ s[IDX_OMEGA])


def rotational_kinetic_energy(s, vehicle):
    """KE = 1/2 omega . (I omega). Torque-free, this is constant."""
    omega = np.asarray(s, dtype=float)[IDX_OMEGA]
    return 0.5 * float(np.dot(omega, vehicle.I_body @ omega))
