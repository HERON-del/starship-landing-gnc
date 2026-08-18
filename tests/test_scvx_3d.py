"""
Verification of the 3-D SCvx solver.

Groups:
    1. Hamilton L(q) and R(p) reproduce the quaternion product exactly
    2. dR/dq against central differences of the form actually linearised
    3. The gyroscopic Jacobian against central differences
    4. The quaternion-kinematics linearisation is exact at the reference and
       second-order away from it
    5. force_to_gimbal inverts Day 14's exact trig
    6. The exact-convex control set is respected by the returned solution
    7. Boundary conditions are met
    8. KNOWN FAILURE, guarded: the virtual control does not converge, so the
       plan is not dynamically feasible

Run:  python tests/test_scvx_3d.py
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.scvx_3d import (                                      # noqa: E402
    quat_L_matrix, quat_R_matrix, rotmatrix_unnormalized, dR_dq_matrices,
    gyro_term, linearize_gyro, linearize_rotated_force, force_to_gimbal,
    solve_scvx_3d, OMEGA_MAX_DEFAULT,
)
from src.quaternion import (                                   # noqa: E402
    quat_multiply, quat_normalize, quat_to_rotmatrix, quat_kinematics,
)
from src.dynamics_3d import (                                  # noqa: E402
    Vehicle3D, gimbal_force_and_torque_body, attitude_from_pitch,
    tilt_from_vertical,
)

PASS, FAIL, NOTE = "[PASS]", "[FAIL]", "[NOTE]"
_SOLVED = {}


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  {tag} {name:<54}" + (f" {detail}" if detail else ""))
    return ok


def note(text, detail=""):
    print(f"  {NOTE} {text:<54}" + (f" {detail}" if detail else ""))


def solved():
    """One solve, shared by the groups that need it."""
    if "r" not in _SOLVED:
        _SOLVED["r"] = solve_scvx_3d(
            N=25, t_f=8.0, pos0=(300.0, 0.0, 420.0), vel0=(-30.0, 0.0, -130.0),
            theta0_deg=25.0, gamma_gs_deg=80.0, verbose=False)
    return _SOLVED["r"]


# ======================================================================
def test_hamilton_matrices():
    print("\nTEST 1 - Hamilton left/right multiplication matrices")
    rng = np.random.default_rng(16)
    ok = True
    worst = 0.0
    for _ in range(400):
        a, b = rng.normal(size=4), rng.normal(size=4)
        prod = quat_multiply(a, b)
        worst = max(worst, float(np.abs(quat_L_matrix(a) @ b - prod).max()),
                    float(np.abs(quat_R_matrix(b) @ a - prod).max()))
    ok &= report("L(q) @ p and R(p) @ q both equal q (x) p",
                 worst < 1e-12, f"worst = {worst:.2e} over 400 pairs")
    return ok


def test_dR_dq():
    print("\nTEST 2 - dR/dq Jacobians")
    rng = np.random.default_rng(17)
    ok = True

    agree = max(float(np.abs(rotmatrix_unnormalized(q)
                             - quat_to_rotmatrix(q)).max())
                for q in (quat_normalize(rng.normal(size=4))
                          for _ in range(200)))
    ok &= report("the raw form equals quat_to_rotmatrix on unit q",
                 agree < 1e-14, f"worst = {agree:.2e}")

    worst = 0.0
    for _ in range(200):
        q = quat_normalize(rng.normal(size=4))
        dRs = dR_dq_matrices(q)
        for i in range(4):
            e = np.zeros(4)
            e[i] = 1e-6
            num = (rotmatrix_unnormalized(q + e)
                   - rotmatrix_unnormalized(q - e)) / 2e-6
            worst = max(worst, float(np.abs(num - dRs[i]).max()))
    ok &= report("all four match central differences of the raw form",
                 worst < 1e-8, f"worst = {worst:.2e}")

    # The trap. `quat_to_rotmatrix` normalises its argument, so it is not the
    # function being linearised, and differencing it fails by order one. The
    # guide's Test 1 is described as checking exactly that.
    q = quat_normalize(np.array([0.6, 0.3, -0.5, 0.4]))
    e = np.zeros(4)
    e[0] = 1e-6
    wrong = float(np.abs((quat_to_rotmatrix(q + e) - quat_to_rotmatrix(q - e))
                         / 2e-6 - dR_dq_matrices(q)[0]).max())
    ok &= report("and differencing the NORMALISED form does not match",
                 wrong > 0.1, f"off by {wrong:.2f} -- the obvious wrong test")
    note("normalising projects out the radial direction",
         "same value, different derivative")
    return ok


def test_gyro_jacobian():
    print("\nTEST 3 - Gyroscopic Jacobian")
    rng = np.random.default_rng(18)
    I = Vehicle3D().I_diag
    ok = True
    worst = 0.0
    for _ in range(300):
        w = rng.normal(scale=0.5, size=3)
        _, J = linearize_gyro(w, I)
        for i in range(3):
            e = np.zeros(3)
            e[i] = 1e-7
            num = (gyro_term(w + e, I) - gyro_term(w - e, I)) / 2e-7
            worst = max(worst, float(np.abs(num - J[:, i]).max())
                        / max(float(np.abs(J[:, i]).max()), 1.0))
    ok &= report("matches central differences (relative)",
                 worst < 1e-6, f"worst = {worst:.2e}")

    # Axisymmetric bodies have no roll row, which is Day 14's result arriving
    # as a property of the Jacobian rather than of a sweep.
    _, J = linearize_gyro(np.array([0.3, 0.7, -0.4]), I)
    ok &= report("roll row is identically zero for I_yy = I_zz",
                 float(np.abs(J[0]).max()) < 1e-9)
    return ok


def test_quaternion_kinematics_linearisation():
    print("\nTEST 4 - Quaternion-kinematics linearisation")
    rng = np.random.default_rng(19)
    ok = True

    def true_qdot(q, w):
        return quat_kinematics(q, w)

    def lin_qdot(q, w, q_r, w_r):
        wq = np.concatenate([[0.0], w])
        wq_r = np.concatenate([[0.0], w_r])
        return 0.5 * (quat_L_matrix(q_r) @ wq
                      + quat_R_matrix(wq_r) @ q - quat_L_matrix(q_r) @ wq_r)

    worst = 0.0
    for _ in range(300):
        q_r = quat_normalize(rng.normal(size=4))
        w_r = rng.normal(scale=0.3, size=3)
        worst = max(worst, float(np.abs(
            lin_qdot(q_r, w_r, q_r, w_r) - true_qdot(q_r, w_r)).max()))
    ok &= report("exact at the reference point", worst < 1e-13,
                 f"worst = {worst:.2e}")

    # A product-rule expansion of a bilinear term has error exactly
    # dq (x) dw, so halving the perturbation must quarter the error.
    q_r = quat_normalize(np.array([0.7, 0.1, -0.6, 0.35]))
    w_r = np.array([0.05, -0.2, 0.1])
    errs = []
    for h in (1e-1, 5e-2, 2.5e-2):
        dq, dw = h * np.array([0.3, -0.5, 0.2, 0.1]), h * np.array([1.0, -.5, .3])
        errs.append(float(np.abs(
            lin_qdot(q_r + dq, w_r + dw, q_r, w_r)
            - true_qdot(q_r + dq, w_r + dw)).max()))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    ok &= report("error is second order in the step",
                 all(3.5 < r < 4.5 for r in ratios),
                 f"halving the step divides the error by "
                 f"{', '.join(f'{r:.2f}' for r in ratios)}")
    return ok


def test_force_to_gimbal():
    print("\nTEST 5 - Force vector back to gimbal angles")
    v = Vehicle3D()
    rng = np.random.default_rng(20)
    ok = True
    worst = 0.0
    for _ in range(500):
        T = rng.uniform(v.T_min, v.T_max)
        dy, dz = rng.uniform(-v.delta_max, v.delta_max, 2)
        F, _ = gimbal_force_and_torque_body(T, dy, dz, v)
        T2, dy2, dz2 = force_to_gimbal(F)
        worst = max(worst, abs(T2 - T) / T, abs(dy2 - dy), abs(dz2 - dz))
    ok &= report("round-trips Day 14's exact trig", worst < 1e-12,
                 f"worst = {worst:.2e} over 500 commands")
    ok &= report("zero force does not divide by zero",
                 force_to_gimbal(np.zeros(3)) == (0.0, 0.0, 0.0))
    return ok


def test_exact_convex_set():
    print("\nTEST 6 - The exact-convex control set is respected")
    r = solved()
    v = Vehicle3D()
    ok = True
    F, sig = r["F"], r["sigma"]
    nF = np.linalg.norm(F, axis=1)

    ok &= report("||F|| <= sigma at every node",
                 float((nF - sig).max()) < 1e-6,
                 f"worst violation = {float((nF - sig).max()):.2e} N")
    ok &= report("T_min <= sigma <= T_max",
                 sig.min() >= v.T_min - 1e-6 and sig.max() <= v.T_max + 1e-6,
                 f"sigma in [{sig.min() / 1e6:.2f}, {sig.max() / 1e6:.2f}] MN")
    gim = np.degrees(np.arctan2(np.linalg.norm(F[:, 1:], axis=1), F[:, 0]))
    ok &= report("gimbal cone respected",
                 float(gim.max()) <= v.delta_max_deg + 1e-6,
                 f"peak deflection = {float(gim.max()):.3f} deg of "
                 f"{v.delta_max_deg:.0f}")
    gs = np.degrees(np.arctan2(np.linalg.norm(r["pos"][:-1, :2], axis=1),
                               np.maximum(r["pos"][:-1, 2], 1e-9)))
    ok &= report("glideslope cone respected",
                 float(gs.max()) <= r["gamma_gs_deg"] + 1e-3,
                 f"peak = {float(gs.max()):.2f} deg of {r['gamma_gs_deg']:.0f}")

    # The relaxation ||F|| <= sigma is only lossless if it is tight.
    note("lossless-convexification gap", f"{r['lcvx_gap']:.2e} -- see Test 8")
    return ok


def test_boundary_conditions():
    print("\nTEST 7 - Boundary conditions")
    r = solved()
    ok = True
    ok &= report("lands at the pad",
                 float(np.linalg.norm(r["pos"][-1])) < 1e-3,
                 f"|pos| = {float(np.linalg.norm(r['pos'][-1])):.2e} m")
    ok &= report("lands at rest",
                 float(np.linalg.norm(r["vel"][-1])) < 1e-3,
                 f"|vel| = {float(np.linalg.norm(r['vel'][-1])):.2e} m/s")
    ok &= report("lands upright",
                 abs(r["final_tilt_deg"]) < 1e-2,
                 f"tilt = {r['final_tilt_deg']:.2e} deg")
    ok &= report("body rates zero at touchdown",
                 float(np.linalg.norm(r["omega"][-1])) < 1e-4)
    ok &= report("rate limit respected throughout",
                 float(np.linalg.norm(r["omega"], axis=1).max())
                 <= OMEGA_MAX_DEFAULT + 1e-6)
    ok &= report("propellant not exceeded",
                 r["fuel"] <= Vehicle3D().m_prop_initial + 1e-6,
                 f"{r['fuel']:,.0f} kg of "
                 f"{Vehicle3D().m_prop_initial:,.0f} available")
    note("upright is NOT the identity quaternion here",
         "the guide's q[N] = [1,0,0,0] is the belly-flop")
    return ok


def test_known_non_convergence():
    """
    The solver does not converge, and this pins that down rather than hiding
    it. Everything above verifies that the sub-problem is the right convex
    problem; this records that the outer loop does not drive its own dynamics
    defect to tolerance, so what comes out is not yet a flyable trajectory.
    """
    print("\nTEST 8 - KNOWN FAILURE: the outer loop does not converge")
    r = solved()
    ok = True
    vc = r["vc_norm"]

    ok &= report("virtual control does NOT reach vc_tol",
                 vc > 1e-3,
                 f"|nu| = {vc:.3e}, tolerance is 1e-6")
    rp = r["replay"]
    ok &= report("and the plan therefore does not fly",
                 rp["miss_m"] > 10.0,
                 f"replayed through the true model: miss "
                 f"{rp['miss_m']:.1f} m at {rp['speed_ms']:.1f} m/s")

    note("This test asserts the failure so it cannot regress silently.", "")
    note("  Measured, and NOT the causes: it is not the Euler step -- the", "")
    note("  miss does not fall with N (141 m at N=15, 247 at 25, 418 at 40,", "")
    note("  240 at 60, 417 at 90, no trend). It is not an under-sized trust", "")
    note("  region either -- the defect FALLS as eta grows (0.557 at 0.2,", "")
    note("  0.416 at 0.5, 0.310 at 1.0), which is the opposite signature.", "")
    note("  What it looks like is an over-constrained sub-problem: hard", "")
    note("  terminal conditions on all four blocks, a throttle floor", "")
    note("  that puts minimum deceleration at 21 m/s^2 against 9.8, and", "")
    note("  a fixed horizon. The solver pays slack because it cannot meet", "")
    note("  them. Lengthening the horizon eases the defect (0.42 at 8 s,", "")
    note("  0.26 at 11, 0.18 at 14) without improving the replay.", "")
    note("  Next thing to try: free final time, or terminal conditions as", "")
    note("  penalties rather than hard equalities.", "")
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 16 - 3-D SCvx SOLVER VERIFICATION")
    print("=" * 70)
    results = [
        test_hamilton_matrices(),
        test_dR_dq(),
        test_gyro_jacobian(),
        test_quaternion_kinematics_linearisation(),
        test_force_to_gimbal(),
        test_exact_convex_set(),
        test_boundary_conditions(),
        test_known_non_convergence(),
    ]
    print("\n" + "=" * 70)
    ok = all(results)
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    if ok:
        print("NOTE: Test 8 passing means the known failure is still present.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
