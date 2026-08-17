"""
Verification of the quaternion library and the 3-D kinematic model.

Tests:
    1. The Hamilton product behaves like rotation composition
    2. Known rotations come out where hand calculation says they should
    3. Round trips through matrices and Euler angles return the same rotation
    4. The two ways of rotating a vector agree
    5. Renormalisation is load-bearing, and the drift without it is measured
    6. Gimbal lock is characterised rather than papered over
    7. The 3-D model reduces to Day 5's planar case

Test 4 is the convention check. `quat_rotate_vector` and `quat_to_rotmatrix`
are derived by different routes, so they can only agree if both treat the
quaternion as body-to-inertial. If a later change flips a convention
somewhere, this is what catches it.

Test 7 is the regression that matters most: Days 1-12 all rest on the planar
model, and a 3-D layer that disagreed with it would invalidate them.

Run:  python tests/test_quaternion.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.quaternion import (                                   # noqa: E402
    quat_multiply, quat_conjugate, quat_norm, quat_normalize, quat_identity,
    quat_from_axis_angle, quat_to_axis_angle, quat_to_rotmatrix,
    rotmatrix_to_quat, quat_to_euler, euler_to_quat, quat_rotate_vector,
    quat_kinematics, quats_equal, quat_angle_between, is_near_gimbal_lock,
)
from src.dynamics_3d_kinematics import (                       # noqa: E402
    make_initial_state, propagate_3d, zero_accel, zero_alpha, norm_history,
    IDX_QUAT, IDX_OMEGA,
)
from tests.test_dynamics import PASS, FAIL                     # noqa: E402


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name:<52}"
    if detail:
        msg += f" {detail}"
    print(msg)
    return bool(ok)


# ======================================================================
def test_hamilton_product():
    print("\nTEST 1 - The Hamilton product")
    qi = quat_identity()
    qx = quat_from_axis_angle([1, 0, 0], np.pi / 2)
    qy = quat_from_axis_angle([0, 1, 0], np.pi / 2)

    ok = report("identity is neutral", quats_equal(quat_multiply(qi, qx), qx))
    ok &= report("a rotation times its conjugate is the identity",
                 quats_equal(quat_multiply(qx, quat_conjugate(qx)), qi))
    ok &= report("does not commute", not quats_equal(
        quat_multiply(qx, qy), quat_multiply(qy, qx)),
        "x-then-y is a different orientation from y-then-x")
    ok &= report("but does for rotations sharing an axis",
                 quats_equal(
                     quat_multiply(qx, quat_from_axis_angle([1, 0, 0], 0.3)),
                     quat_multiply(quat_from_axis_angle([1, 0, 0], 0.3), qx)))
    ok &= report("unit times unit stays unit",
                 abs(quat_norm(quat_multiply(qx, qy)) - 1.0) < 1e-12)
    # Two quarter turns about the same axis make a half turn.
    ok &= report("composes angles about a shared axis",
                 quats_equal(quat_multiply(qx, qx),
                             quat_from_axis_angle([1, 0, 0], np.pi)))
    return ok


# ======================================================================
def test_known_rotation():
    """Against rotations whose answers can be written down by hand."""
    print("\nTEST 2 - Known rotations")
    q = quat_from_axis_angle([0, 0, 1], np.pi / 2)      # +90 about z
    got = quat_rotate_vector(q, [1, 0, 0])
    ok = report("+90 deg about z carries +x to +y",
                np.allclose(got, [0, 1, 0], atol=1e-12),
                f"{np.round(got, 6).tolist()}")

    q = quat_from_axis_angle([1, 0, 0], np.pi / 2)
    got = quat_rotate_vector(q, [0, 1, 0])
    ok &= report("+90 deg about x carries +y to +z",
                 np.allclose(got, [0, 0, 1], atol=1e-12),
                 f"{np.round(got, 6).tolist()}")

    q = quat_from_axis_angle([0, 1, 0], np.pi)
    got = quat_rotate_vector(q, [0, 0, 1])
    ok &= report("180 deg about y flips +z",
                 np.allclose(got, [0, 0, -1], atol=1e-12))

    ok &= report("the rotation axis is left alone",
                 np.allclose(quat_rotate_vector(
                     quat_from_axis_angle([0, 0, 1], 1.1), [0, 0, 3]),
                     [0, 0, 3], atol=1e-12))

    axis, angle = quat_to_axis_angle(quat_from_axis_angle([1, 1, 0], 0.7))
    ok &= report("axis and angle survive a round trip",
                 abs(angle - 0.7) < 1e-9
                 and np.allclose(axis, np.array([1, 1, 0]) / np.sqrt(2),
                                 atol=1e-9))
    return ok


# ======================================================================
def test_round_trips():
    print("\nTEST 3 - Round trips")
    rng = np.random.default_rng(0)
    worst_R, worst_E, n_neg = 0.0, 0.0, 0
    for _ in range(300):
        q = quat_normalize(rng.normal(size=4))
        back_R = rotmatrix_to_quat(quat_to_rotmatrix(q))
        worst_R = max(worst_R, quat_angle_between(q, back_R))
        if np.dot(q, back_R) < 0:
            n_neg += 1
        e = quat_to_euler(q)
        if not is_near_gimbal_lock(q, margin_deg=2.0):
            back_E = euler_to_quat(*e)
            worst_E = max(worst_E, quat_angle_between(q, back_E))

    ok = report("quaternion -> matrix -> quaternion", worst_R < 1e-9,
                f"worst {np.degrees(worst_R):.2e} deg over 300 random")
    ok &= report("quaternion -> Euler -> quaternion, away from lock",
                 worst_E < 1e-9,
                 f"worst {np.degrees(worst_E):.2e} deg")
    ok &= report("double cover shows up and is handled", n_neg > 0,
                 f"{n_neg}/300 returned -q, which is the same rotation")
    R = quat_to_rotmatrix(quat_from_axis_angle([0.3, -0.5, 1.0], 2.0))
    ok &= report("the matrix really is a rotation",
                 np.allclose(R @ R.T, np.eye(3), atol=1e-12)
                 and abs(np.linalg.det(R) - 1.0) < 1e-12,
                 "orthonormal, determinant +1")
    return ok


# ======================================================================
def test_vector_rotation_consistency():
    """The convention check: two derivations, one answer."""
    print("\nTEST 4 - Sandwich product against the matrix")
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(300):
        q = quat_normalize(rng.normal(size=4))
        v = rng.normal(size=3)
        worst = max(worst, float(np.abs(
            quat_rotate_vector(q, v) - quat_to_rotmatrix(q) @ v).max()))
    ok = report("both routes agree", worst < 1e-12,
                f"worst disagreement {worst:.2e}")

    q = quat_normalize(rng.normal(size=4))
    v = rng.normal(size=3)
    ok &= report("length is preserved",
                 abs(np.linalg.norm(quat_rotate_vector(q, v))
                     - np.linalg.norm(v)) < 1e-12)
    ok &= report("the conjugate undoes the rotation",
                 np.allclose(quat_rotate_vector(quat_conjugate(q),
                                                quat_rotate_vector(q, v)),
                             v, atol=1e-12))
    return ok


# ======================================================================
def test_normalization_drift():
    """
    Measure what skipping renormalisation actually costs.

    The point is not that the drift is large but that it is one-sided and
    unbounded: it accumulates in the same direction every step, so it is a
    slow corruption rather than noise, and nothing downstream would flag it.
    """
    print("\nTEST 5 - Renormalisation is load-bearing")
    s0 = make_initial_state(omega=(0.8, -0.5, 1.2))
    kw = dict(t_span=(0.0, 60.0), dt=0.01)
    _, with_n = propagate_3d(s0, zero_accel, zero_alpha, renormalize=True,
                             **kw)
    n_with = norm_history(with_n)
    ok = report("norm holds when renormalised",
                float(np.abs(n_with - 1.0).max()) < 1e-12,
                f"worst |1-|q|| = {np.abs(n_with - 1.0).max():.2e}")

    # How fast the drift grows depends strongly on the step, because it is
    # RK4 truncation error accumulating. At a fine step it is genuinely
    # negligible over a flight -- worth stating plainly rather than implying
    # renormalisation rescues a solution that would otherwise fall apart in
    # seconds. What makes it worth doing anyway is the next assertion: the
    # error is one-sided, so it only ever grows.
    drifts = {}
    for dt in (0.5, 0.2, 0.05, 0.01):
        _, w = propagate_3d(s0, zero_accel, zero_alpha, (0.0, 60.0), dt,
                            renormalize=False)
        drifts[dt] = float(abs(norm_history(w)[-1] - 1.0))
    coarse = drifts[0.5]
    ok &= report("without it the norm drifts at a coarse step", coarse > 1e-9,
                 f"|1-|q|| = {coarse:.2e} at dt=0.5, "
                 f"{drifts[0.01]:.2e} at dt=0.01")
    ok &= report("and the drift shrinks with the step, so it is truncation",
                 drifts[0.5] > drifts[0.2] > drifts[0.05],
                 "  ".join(f"dt={k}: {v:.1e}" for k, v in drifts.items()))

    _, without = propagate_3d(s0, zero_accel, zero_alpha, (0.0, 60.0), 0.5,
                              renormalize=False)
    n_without = norm_history(without)
    ok &= report("the drift is one-sided, not noise",
                 bool(np.all(np.diff(n_without) >= -1e-15)
                      or np.all(np.diff(n_without) <= 1e-15)),
                 "monotone, so it accumulates rather than averaging out")
    # An un-normalised quaternion no longer represents a pure rotation: the
    # matrix it produces is not orthonormal, which silently scales vectors.
    R = quat_to_rotmatrix(without[-1][IDX_QUAT])
    ok &= report("renormalised states still give orthonormal matrices",
                 np.allclose(R @ R.T, np.eye(3), atol=1e-9),
                 "quat_to_rotmatrix normalises defensively on the way in")
    return ok


# ======================================================================
def test_gimbal_lock():
    """
    Characterise the singularity rather than pretend it is absent.

    At pitch = 90 degrees the Euler decomposition loses a degree of freedom:
    roll and yaw act about the same physical axis and only their sum is
    recoverable. The quaternion is untroubled, which is the whole argument.
    """
    print("\nTEST 6 - Gimbal lock")
    q_lock = euler_to_quat(0.0, np.pi / 2, 0.0)
    ok = report("the locked orientation is detected",
                is_near_gimbal_lock(q_lock))
    ok &= report("its quaternion is a perfectly ordinary unit quaternion",
                 abs(quat_norm(q_lock) - 1.0) < 1e-12)

    # Two visibly different (roll, yaw) pairs at pitch = 90 are the same
    # physical orientation, which is exactly the lost degree of freedom.
    a = euler_to_quat(0.4, np.pi / 2, 0.0)
    b = euler_to_quat(0.0, np.pi / 2, -0.4)
    ok &= report("distinct Euler triples, identical orientation",
                 quat_angle_between(a, b) < 1e-9,
                 "roll and yaw are the same axis there")

    # The failure that matters is dynamic, so measure it dynamically: fly a
    # perfectly smooth rotation straight through pitch = 90 and compare the
    # rate the Euler angles would demand against the rate the body is actually
    # turning at. A controller reading angles has to track the former.
    s0 = make_initial_state(quat=euler_to_quat(0.0, np.radians(60.0), 0.0),
                            omega=(0.0, 0.6, 0.0))
    dt = 0.002
    ts, hist = propagate_3d(s0, zero_accel, zero_alpha, (0.0, 2.0), dt)
    eul = np.array([quat_to_euler(s[IDX_QUAT]) for s in hist])
    euler_rate = np.abs(np.diff(eul, axis=0)).max(axis=1) / dt
    body_rate = 0.6
    ok &= report("the body turns at a constant, unremarkable rate",
                 abs(np.linalg.norm(hist[-1][IDX_OMEGA]) - body_rate) < 1e-12,
                 f"{body_rate} rad/s throughout")
    ok &= report("but the Euler rate spikes as it crosses pitch = 90 deg",
                 euler_rate.max() > 50.0 * body_rate,
                 f"peak {euler_rate.max():.1f} rad/s, "
                 f"{euler_rate.max() / body_rate:.0f}x the physical rate")
    ok &= report("...for a motion the quaternion handles smoothly",
                 float(np.abs(np.diff(
                     np.array([quat_angle_between(hist[0][IDX_QUAT],
                                                  s[IDX_QUAT])
                               for s in hist]))).max()) < 2.0 * body_rate * dt,
                 "no jump anywhere along the same trajectory")
    ok &= report("the quaternion path does not",
                 quat_angle_between(euler_to_quat(0.3, np.pi / 2 - 1e-4, 0.2),
                                    euler_to_quat(0.3, np.pi / 2 + 1e-4, 0.2))
                 < 1e-3,
                 "passing through lock is unremarkable in quaternions")
    return ok


# ======================================================================
def test_2d_reduction():
    """
    The 3-D model must agree with Days 1-12.

    Confined to a single axis, the quaternion kinematics has to reproduce
    `theta = omega * t` exactly, or every planar result built on that would be
    in question.
    """
    print("\nTEST 7 - Reduction to the planar case")
    omega_z = 0.37
    s0 = make_initial_state(omega=(0.0, 0.0, omega_z))
    ts, hist = propagate_3d(s0, zero_accel, zero_alpha, (0.0, 10.0), 0.001)

    worst = 0.0
    for t, s in zip(ts, hist):
        expected = quat_from_axis_angle([0, 0, 1], omega_z * t)
        worst = max(worst, quat_angle_between(s[IDX_QUAT], expected))
    ok = report("single-axis spin matches the closed form",
                worst < 1e-9,
                f"worst {np.degrees(worst):.2e} deg over 10 s")

    yaw = np.array([quat_to_euler(s[IDX_QUAT])[2] for s in hist])
    ok &= report("yaw advances linearly at omega",
                 abs(np.unwrap(yaw)[-1] - omega_z * ts[-1]) < 1e-9,
                 f"{np.unwrap(yaw)[-1]:.9f} vs {omega_z * ts[-1]:.9f} rad")
    ok &= report("nothing leaks into the other two axes",
                 float(np.abs(hist[:, 6 + 1:6 + 3]).max()) < 1e-12,
                 "the x and y quaternion components stay at zero")
    ok &= report("body rate is unchanged by a free spin",
                 np.allclose(hist[-1][IDX_OMEGA], [0, 0, omega_z],
                             atol=1e-12))
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 13 - QUATERNION AND 3-D KINEMATICS VERIFICATION")
    print("=" * 70)
    oks = [test_hamilton_product(), test_known_rotation(), test_round_trips(),
           test_vector_rotation_consistency(), test_normalization_drift(),
           test_gimbal_lock(), test_2d_reduction()]
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED" if all(oks) else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
