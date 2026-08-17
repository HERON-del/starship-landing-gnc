"""
The 13-state 3-D kinematic model.

Deliberately not rigid-body dynamics. There is no inertia tensor here, no
forces and no torques: translational acceleration and body-frame angular
acceleration are *prescribed* from outside, and this file's entire job is to
get the bookkeeping right -- which frame each quantity lives in, and how the
quaternion evolves. Day 14 replaces the prescriptions with Euler's equations.

Separating them is worth the extra file. Every three-dimensional bug from here
on will be a frame confusion or a sign error, and it is far easier to find one
in a model whose answers can be checked against closed-form rotations than in
one where the forces are also in question.

State: s = [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]   (13,)

    position, velocity   INERTIAL frame
    quaternion           body to inertial
    angular velocity     BODY frame, which is what a rate gyro reports
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.quaternion import (                                   # noqa: E402
    quat_kinematics, quat_normalize, quat_identity, quat_to_euler,
)

N_STATE_3D = 13
IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)
IDX_QUAT = slice(6, 10)
IDX_OMEGA = slice(10, 13)


def make_initial_state(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0),
                       quat=None, omega=(0.0, 0.0, 0.0)):
    """Assemble a 13-state vector, quaternion normalised."""
    q = quat_identity() if quat is None else quat_normalize(quat)
    return np.concatenate([np.asarray(pos, dtype=float),
                           np.asarray(vel, dtype=float),
                           q,
                           np.asarray(omega, dtype=float)])


def kinematics_derivative(s, a_inertial, alpha_body):
    """
    State derivative.

    `a_inertial` is translational acceleration in the inertial frame and
    `alpha_body` is angular acceleration in the body frame. Both are given,
    not computed -- that is what makes this kinematics rather than dynamics.
    """
    s = np.asarray(s, dtype=float)
    return np.concatenate([
        s[IDX_VEL],
        np.asarray(a_inertial, dtype=float),
        quat_kinematics(s[IDX_QUAT], s[IDX_OMEGA]),
        np.asarray(alpha_body, dtype=float),
    ])


def rk4_step_3d(s, a_inertial, alpha_body, dt, renormalize=True):
    """
    One RK4 step, then re-impose the unit-norm constraint.

    RK4 knows nothing about the constraint: each stage nudges the quaternion
    along a direction only approximately tangent to the unit sphere, so the
    norm creeps away from 1 and the quaternion quietly stops representing a
    pure rotation. Renormalising is not a tidiness measure.

    It happens once, on the combined state, rather than inside each stage --
    normalising the intermediate stages would corrupt the RK4 weights.
    `renormalize=False` exists so the test suite can measure how bad the drift
    gets without it.
    """
    k1 = kinematics_derivative(s, a_inertial, alpha_body)
    k2 = kinematics_derivative(s + 0.5 * dt * k1, a_inertial, alpha_body)
    k3 = kinematics_derivative(s + 0.5 * dt * k2, a_inertial, alpha_body)
    k4 = kinematics_derivative(s + dt * k3, a_inertial, alpha_body)
    s_new = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if renormalize:
        s_new[IDX_QUAT] = quat_normalize(s_new[IDX_QUAT])
    return s_new


def propagate_3d(s0, a_inertial_fn, alpha_body_fn, t_span, dt,
                 renormalize=True):
    """
    Integrate over `t_span`, returning the time grid and state history.

    `a_inertial_fn(t, s)` and `alpha_body_fn(t, s)` supply the prescribed
    accelerations, so a caller can specify anything from a free tumble (both
    zero) to a commanded slew without this file knowing what a force is.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    n = max(int(round((t1 - t0) / dt)), 1)
    ts = np.linspace(t0, t0 + n * dt, n + 1)
    out = np.zeros((n + 1, N_STATE_3D))
    out[0] = np.asarray(s0, dtype=float)
    for k in range(n):
        s = out[k]
        out[k + 1] = rk4_step_3d(s, a_inertial_fn(ts[k], s),
                                 alpha_body_fn(ts[k], s), dt,
                                 renormalize=renormalize)
    return ts, out


# ======================================================================
# Convenience readouts
# ======================================================================
def euler_history(states):
    """Roll, pitch and yaw at every sample, for plotting against time."""
    return np.array([quat_to_euler(s[IDX_QUAT]) for s in np.asarray(states)])


def norm_history(states):
    """Quaternion norm at every sample -- 1.0 if the constraint is holding."""
    return np.array([np.linalg.norm(s[IDX_QUAT]) for s in np.asarray(states)])


def zero_accel(t, s):
    """No translational acceleration."""
    return np.zeros(3)


def zero_alpha(t, s):
    """No angular acceleration: a free tumble at constant body rate."""
    return np.zeros(3)


def gravity_accel(g=9.80665, axis=2):
    """Uniform gravity along `-axis`, for a ballistic reference case."""
    a = np.zeros(3)
    a[axis] = -g

    def _f(t, s):
        return a
    return _f
