"""
Quaternion algebra for 3-D attitude.

Days 1-12 got away with a single pitch angle because the vehicle only ever
rotated about one axis. In three dimensions orientation is a point in SO(3),
and three unconstrained numbers cannot cover it without a singularity: Euler
angles lose a degree of freedom at pitch = +/-90 degrees, where the roll and
yaw axes become parallel. That is gimbal lock, and it is not an aesthetic
problem -- arbitrarily small physical motions there demand arbitrarily large
jumps in the angles, which is exactly what a controller reading those angles
cannot survive.

A unit quaternion uses four numbers and one constraint instead, and has no
such singularity anywhere. The price is that the constraint is not preserved
by numerical integration and must be re-imposed by hand.

Convention, stated once and used everywhere, because mixing conventions is the
usual source of three-dimensional sign bugs:

    - scalar-first Hamilton quaternion, q = [w, x, y, z]
    - q rotates BODY to INERTIAL:  v_inertial = R(q) @ v_body
    - omega is in the BODY frame, which is what a rate gyro measures and what
      Day 11's attitude sensor already reports
    - kinematics:  qdot = 0.5 * q (x) [0, omega]

One quirk worth knowing before it looks like a bug: `q` and `-q` are the same
rotation. The unit quaternions double-cover SO(3), so a round trip through a
matrix or through Euler angles may hand back the negated quaternion, and that
is correct. `quats_equal` compares up to sign for this reason.
"""

import numpy as np

# Below this the vector part of a quaternion carries no reliable direction,
# and near |sin(pitch)| = 1 the Euler decomposition is inside gimbal lock.
_EPS = 1e-12
GIMBAL_LOCK_TOL = 1e-6


# ======================================================================
# Core algebra
# ======================================================================
def quat_multiply(q1, q2):
    """
    Hamilton product q1 (x) q2. Not commutative, because rotations are not.

    Composition order is the thing people get wrong: this is the rotation you
    get by applying q1 and then applying q2 expressed in the frame q1 left
    behind.
    """
    w1, x1, y1, z1 = np.asarray(q1, dtype=float)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q):
    """Conjugate, which for a unit quaternion is the inverse rotation."""
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array([w, -x, -y, -z])


def quat_norm(q):
    return float(np.linalg.norm(np.asarray(q, dtype=float)))


def quat_normalize(q):
    """Project back onto the unit sphere. Mandatory after integration."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < _EPS:
        return quat_identity()
    return q / n


def quat_identity():
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_from_axis_angle(axis, angle):
    """
    Rotation of `angle` radians about `axis`, right-handed.

    A positive rotation about +z carries +x toward +y.
    """
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < _EPS:
        return quat_identity()
    axis = axis / n
    h = 0.5 * float(angle)
    return np.concatenate([[np.cos(h)], np.sin(h) * axis])


def quat_to_axis_angle(q):
    """Inverse of `quat_from_axis_angle`, with angle in [0, pi]."""
    q = quat_normalize(q)
    if q[0] < 0.0:                      # pick the short way round
        q = -q
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    s = np.sqrt(max(1.0 - q[0] * q[0], 0.0))
    if s < _EPS:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return q[1:] / s, float(angle)


# ======================================================================
# Matrix and Euler bridges
# ======================================================================
def quat_to_rotmatrix(q):
    """
    Direction cosine matrix, body to inertial.

    Anything in an integration loop uses this rather than the sandwich
    product: one matrix-vector product instead of two quaternion multiplies.
    """
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotmatrix_to_quat(R):
    """
    Inverse of `quat_to_rotmatrix`, by Shepperd's method.

    The naive formula divides by sqrt(1 + trace(R)), which collapses at
    trace(R) = -1 -- a 180 degree rotation, not an exotic case. Shepperd's
    method picks whichever of the four components is largest and solves for
    that one first, so the divisor is never small.
    """
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s,
                      0.25 * s,
                      (R[0, 1] + R[1, 0]) / s,
                      (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s,
                      (R[0, 1] + R[1, 0]) / s,
                      0.25 * s,
                      (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s,
                      (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s,
                      0.25 * s])
    return quat_normalize(q)


def quat_to_euler(q):
    """
    ZYX Tait-Bryan angles (roll, pitch, yaw) in radians.

    Returns the gimbal-locked branch when |sin(pitch)| reaches 1: roll is set
    to zero and the whole rotation is carried in yaw, because at that
    configuration only their sum is observable. That is a property of the
    representation, not a failure of this function -- which is the entire
    reason the rest of the project stores a quaternion instead.
    """
    w, x, y, z = quat_normalize(q)
    sin_p = 2.0 * (w * y - z * x)
    if abs(sin_p) >= 1.0 - GIMBAL_LOCK_TOL:
        pitch = np.copysign(np.pi / 2.0, sin_p)
        roll = 0.0
        yaw = float(np.copysign(2.0, sin_p) * np.arctan2(x, w))
        return np.array([roll, pitch, yaw])
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(sin_p)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([float(roll), float(pitch), float(yaw)])


def euler_to_quat(roll, pitch, yaw):
    """ZYX Tait-Bryan angles to quaternion: yaw, then pitch, then roll."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def is_near_gimbal_lock(q, margin_deg=1.0):
    """Whether this orientation sits close enough to pitch = +/-90 to matter."""
    pitch = quat_to_euler(q)[1]
    return bool(abs(abs(pitch) - np.pi / 2.0) < np.radians(margin_deg))


# ======================================================================
# Acting on vectors, and kinematics
# ======================================================================
def quat_rotate_vector(q, v):
    """
    Rotate `v` from body to inertial by the sandwich product q (x) [0,v] (x) q*.

    Equivalent to `quat_to_rotmatrix(q) @ v`, and kept because the agreement
    between the two is a genuine check on the convention: they are derived
    differently and can only agree if both treat q as body-to-inertial.
    """
    q = quat_normalize(q)
    v = np.asarray(v, dtype=float)
    out = quat_multiply(quat_multiply(q, np.concatenate([[0.0], v])),
                        quat_conjugate(q))
    return out[1:]


def quat_kinematics(q, omega_body):
    """
    qdot = 0.5 * q (x) [0, omega_body].

    The three-dimensional generalisation of Day 5's `dtheta/dt = omega`. The
    angular velocity is packaged as a pure quaternion so it composes through
    the same Hamilton product as everything else.
    """
    q = np.asarray(q, dtype=float)
    w = np.asarray(omega_body, dtype=float)
    return 0.5 * quat_multiply(q, np.concatenate([[0.0], w]))


def quats_equal(q1, q2, tol=1e-6):
    """Equality up to sign, since q and -q are the same rotation."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    return bool(np.allclose(q1, q2, atol=tol)
                or np.allclose(q1, -q2, atol=tol))


def quat_angle_between(q1, q2):
    """Rotation angle [rad] separating two orientations, in [0, pi]."""
    rel = quat_multiply(quat_conjugate(quat_normalize(q1)),
                        quat_normalize(q2))
    return quat_to_axis_angle(rel)[1]
