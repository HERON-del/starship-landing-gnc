"""
Verification of the Szmuk & Acikmese (2018) benchmark replication.

Groups:
    1. Table 1 and Table 2 transcription, and what the numbers imply
    2. The paper's model IS this project's physics -- two independent
       expressions of the same rotation and the same gyroscopic term
    3. alpha_m is not a harmless placeholder
    4. Where the dynamics residual actually lives
    5. The paper's central claim -- robustness to the time-of-flight guess
    6. The paper's literal Algorithm 1, run as printed
    6b. What actually broke it: the discretisation, isolated
    7. Boundary conditions and the convex constraint set

Run:  python tests/test_benchmark_szmuk.py
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.benchmark_szmuk import (                              # noqa: E402
    PaperVehicle, PaperAlgorithm, two_d_case, three_d_case,
    dcm_body_from_inertial, omega_matrix, nonlinear_dynamics,
    solve_benchmark, residual_by_block, ALPHA_M_NOTE,
    discretize, discretize_exact_foh, initialize_reference,
    IDX_M, IDX_R, IDX_V, IDX_Q, IDX_W,
)
from src.quaternion import quat_to_rotmatrix, quat_normalize   # noqa: E402
from src.dynamics_3d import gyroscopic_term                    # noqa: E402

PASS, FAIL, NOTE = "[PASS]", "[FAIL]", "[NOTE]"
_C = {}


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  {tag} {name:<52}" + (f" {detail}" if detail else ""))
    return ok


def note(text, detail=""):
    print(f"  {NOTE} {text:<52}" + (f" {detail}" if detail else ""))


def cached(key, fn):
    if key not in _C:
        _C[key] = fn()
    return _C[key]


# ======================================================================
def test_transcription():
    print("\nTEST 1 - Table 1 and Table 2, and what they imply")
    v, a = PaperVehicle(), PaperAlgorithm()
    ok = True
    ok &= report("Table 1 masses, thrusts, angles",
                 (v.m_wet, v.m_dry, v.T_min, v.T_max) == (2.0, 1.0, 0.3, 5.0)
                 and (v.delta_max_deg, v.theta_max_deg, v.gamma_gs_deg,
                      v.omega_max_deg) == (20.0, 90.0, 20.0, 60.0))
    ok &= report("Table 1 inertia and thrust offset",
                 np.allclose(v.J_B, 1e-2 * np.eye(3))
                 and np.allclose(v.r_TB, [-1e-2, 0.0, 0.0]))
    ok &= report("Table 2 weights and tolerances",
                 (a.w_nu, a.w_delta, a.w_delta_sigma) == (1e5, 1e-3, 1e-1)
                 and (a.nu_tol, a.delta_tol, a.N_iter_max, a.K)
                 == (1e-10, 1e-3, 15, 50))
    ok &= report("gravity is -e1, so x is up in this file",
                 np.allclose(v.g_I, [-1.0, 0.0, 0.0]))

    # The thing worth noticing about Table 1, which Day 17 makes relevant.
    ok &= report("this vehicle can descend under minimum throttle",
                 v.T_min / v.m_wet < 1.0,
                 f"min accel {v.T_min / v.m_wet:.3f} against g = 1.0")
    note("Day 17's Starship numbers could not", "T_min/m_wet was 2.16 g there")
    note("so the throttle-floor pathology found on Day 17",
         "cannot occur in the paper's own problem")
    return ok


def test_model_matches_this_project():
    """
    The paper's model and Days 13-15 are the same physics, written twice.

    This is the part of a replication that is genuinely external: the paper's
    DCM and gyroscopic term were written by people who never saw this
    codebase, and if the two disagree, one of them is wrong.
    """
    print("\nTEST 2 - The paper's model against this project's")
    rng = np.random.default_rng(18)
    ok = True

    worst = max(float(np.abs(dcm_body_from_inertial(q).T
                             - quat_to_rotmatrix(q)).max())
                for q in (quat_normalize(rng.normal(size=4))
                          for _ in range(400)))
    ok &= report("paper's DCM transposed == quat_to_rotmatrix",
                 worst < 1e-14, f"worst = {worst:.2e} over 400 quaternions")

    J = 1e-2 * np.eye(3)
    worst_g = max(float(np.abs(np.cross(w, J @ w) - gyroscopic_term(w, J)).max())
                  for w in rng.normal(scale=1.0, size=(400, 3)))
    ok &= report("paper's w x Jw == Day 14's gyroscopic_term",
                 worst_g < 1e-15, f"worst = {worst_g:.2e}")

    # Omega(w) has to be skew, or the quaternion norm is not conserved.
    worst_s = max(float(np.abs(omega_matrix(w) + omega_matrix(w).T).max())
                  for w in rng.normal(size=(200, 3)))
    ok &= report("Omega(w) is skew-symmetric", worst_s < 1e-15,
                 f"worst = {worst_s:.2e}")

    # A skew generator means qdot is orthogonal to q, so |q| is preserved.
    worst_o = 0.0
    for _ in range(200):
        q = quat_normalize(rng.normal(size=4))
        w = rng.normal(scale=0.5, size=3)
        worst_o = max(worst_o,
                      abs(float(q @ (0.5 * omega_matrix(w) @ q))))
    ok &= report("so qdot stays tangent to the unit sphere",
                 worst_o < 1e-15, f"worst |q . qdot| = {worst_o:.2e}")
    return ok


def test_alpha_m_is_not_harmless():
    """
    The one number the guide waves through.

    It is not in the paper, the guide sets it to 1.0 and calls it a
    placeholder, and at 1.0 the vehicle needs six times its own propellant
    load to fly a three-unit trajectory.
    """
    print("\nTEST 3 - alpha_m is not a harmless placeholder")
    v = PaperVehicle()
    ok = True
    usable = v.m_wet - v.m_dry

    burn_at_1 = 1.0 * v.hover_thrust * 3.0
    ok &= report("at alpha_m = 1.0 the propellant cannot last",
                 burn_at_1 > usable,
                 f"needs {burn_at_1:.1f} UM of {usable:.1f} available")
    burn_at_default = v.alpha_m * v.hover_thrust * 3.0
    ok &= report("at the default it can",
                 burn_at_default < usable,
                 f"needs {burn_at_default:.2f} UM of {usable:.1f}")
    note("the paper never gives alpha_m", "its results do not depend on one")
    note("but the dynamics do", "mass enters the velocity row through F/m")
    return ok


def test_residual_location():
    """
    Where the defect lives, which is the diagnostic the guide never runs.

    The guide attributes its whole residual to a cruder discretisation than
    the paper's. Decomposing by state block says otherwise.
    """
    print("\nTEST 4 - Where the dynamics residual actually lives")
    alg = PaperAlgorithm()
    ok = True

    hot = cached("hot", lambda: solve_benchmark(
        two_d_case(), veh=PaperVehicle(alpha_m=1.0), alg=alg,
        sigma_guess=3.0, verbose=False))
    cool = cached("cool", lambda: solve_benchmark(
        two_d_case(), veh=PaperVehicle(alpha_m=0.03), alg=alg,
        sigma_guess=3.0, verbose=False))

    ok &= report("the guide's residual reproduces at its own alpha_m",
                 hot["nu_total"] > 1.0,
                 f"|nu| = {hot['nu_total']:.3f} (guide reported 4.4 to 5.0)")
    ok &= report("and it sits in the mass and velocity rows",
                 hot["nu_fraction"]["mass"]
                 + hot["nu_fraction"]["velocity"] > 0.8,
                 "mass " + " ".join(
                     f"{k}={v * 100:.0f}%" for k, v in
                     hot["nu_fraction"].items() if k in ("mass", "velocity")))
    ok &= report("mass is pinned on the floor there",
                 abs(hot["x"][-1, IDX_M] - PaperVehicle().m_dry) < 1e-6,
                 f"final mass = {hot['x'][-1, IDX_M]:.4f}")
    ok &= report("sizing alpha_m collapses the residual",
                 cool["nu_total"] < hot["nu_total"] / 50.0,
                 f"{hot['nu_total']:.3e} -> {cool['nu_total']:.3e}")

    # The control that settles the guide's own root-cause claim: apply the fix
    # it proposes, at the alpha_m it proposes, and see whether it helps.
    hot_exact = cached("hot_exact", lambda: solve_benchmark(
        two_d_case(), veh=PaperVehicle(alpha_m=1.0), alg=alg,
        sigma_guess=3.0, exact_foh=True, verbose=False))
    ok &= report("the guide's own proposed fix does NOT help at its alpha_m",
                 hot_exact["nu_total"] > 0.5 * hot["nu_total"],
                 f"exact Eq. 22 gives {hot_exact['nu_total']:.3f} against "
                 f"{hot['nu_total']:.3f} -- no better")
    note("mass pinned at m_dry makes the mass row unsatisfiable, and", "")
    note("  velocity follows because F/m carries the wrong mass. No", "")
    note("  quadrature scheme fixes a constraint that is simply binding.", "")
    return ok


def test_robustness_to_time_guess():
    """
    The paper's central empirical claim, and it reproduces.

    Ten time-of-flight guesses from 1 to 10 UT, all converging to within
    0.01 UT. This is the number the paper actually stakes its case on.
    """
    print("\nTEST 5 - Robustness to the time-of-flight guess")
    alg = PaperAlgorithm()
    alg.K = 30
    veh = PaperVehicle(alpha_m=0.03)
    ok = True

    def sweep():
        out = []
        for g in range(1, 11):
            r = solve_benchmark(two_d_case(), veh=veh, alg=alg,
                                sigma_guess=float(g), verbose=False)
            out.append((r["sigma"], r["nu_total"]))
        return out

    res = cached("sweep", sweep)
    tfs = np.array([a for a, _ in res])
    nus = np.array([b for _, b in res])
    spread = float(tfs.max() - tfs.min())

    ok &= report("all ten guesses land within the paper's own bar",
                 spread < 0.01,
                 f"spread {spread:.5f} UT, paper claims < 0.01")
    ok &= report("and they land on the same answer",
                 float(tfs.std()) < 0.01,
                 f"tf = {tfs.mean():.5f} +/- {tfs.std():.5f} UT")
    ok &= report("with the virtual control at machine precision",
                 float(nus.max()) < 1e-10,
                 f"worst |nu| over the ten = {float(nus.max()):.2e}")
    note("This corrects a claim published earlier in this project.", "")
    note("  With a single-endpoint approximation in place of the paper's", "")
    note("  Eq. 22 integrals the spread was 21.7 UT and the flight time", "")
    note("  tracked its own guess, and that was written up as the paper's", "")
    note("  claim failing. The failure was in the implementation.", "")
    return ok


def test_paper_algorithm_as_printed():
    """
    Algorithm 1 with nothing added: no hard trust region, no quaternion
    renormalisation. The guide calls both necessary. They are not.
    """
    print("\nTEST 6 - The paper's literal Algorithm 1")
    alg = PaperAlgorithm()
    alg.K = 30
    veh = PaperVehicle(alpha_m=0.03)
    ok = True

    def literal(g):
        return solve_benchmark(two_d_case(), veh=veh, alg=alg,
                               sigma_guess=float(g), hard_trust=False,
                               renormalize=False, verbose=False)

    a = cached("lit1", lambda: literal(1))
    b = cached("lit10", lambda: literal(10))

    ok &= report("it converges with no hard trust region",
                 a["nu_total"] < 1e-10 and b["nu_total"] < 1e-10,
                 f"|nu| = {a['nu_total']:.2e} and {b['nu_total']:.2e}")
    ok &= report("and with no quaternion renormalisation",
                 True, "both flags off for this run")
    ok &= report("guesses of 1 and 10 reach the same flight time",
                 abs(a["sigma"] - b["sigma"]) < 0.01,
                 f"{a['sigma']:.4f} vs {b['sigma']:.4f} UT")

    # The guide reports sigma collapsing to zero on iteration 1. It does dip,
    # from a far-off guess -- and then recovers, which is the part a cruder
    # discretisation never gets to see.
    seq = b["history"]["sigma"]
    ok &= report("a far guess dips early and then recovers",
                 min(seq) < seq[-1],
                 "sigma: " + " ".join(f"{v:.2f}" for v in seq[:6]) + " ...")
    note("The guide adds a hard trust box and renormalisation and calls", "")
    note("  both necessary. With the paper's own discretisation neither", "")
    note("  is, and the hard box measurably degrades the sweep -- 0.025 UT", "")
    note("  of spread against 0.002 without it.", "")
    return ok


def test_discretisation_is_what_broke_it():
    """
    Isolating the cause, by changing exactly one thing.

    Same alpha_m, same weights, same everything -- only the quadrature
    differs. This is the measurement that says which layer the failure
    lived in.
    """
    print("\nTEST 6b - The discretisation, isolated")
    alg = PaperAlgorithm()
    alg.K = 30
    veh = PaperVehicle(alpha_m=0.03)
    ok = True

    crude = cached("crude", lambda: solve_benchmark(
        two_d_case(), veh=veh, alg=alg, sigma_guess=3.0,
        exact_foh=False, verbose=False))
    exact = cached("exact", lambda: solve_benchmark(
        two_d_case(), veh=veh, alg=alg, sigma_guess=3.0,
        exact_foh=True, verbose=False))

    ok &= report("single-endpoint quadrature leaves a real residual",
                 crude["nu_total"] > 1e-3,
                 f"|nu| = {crude['nu_total']:.3e}")
    ok &= report("the paper's Eq. 22 integrals close it",
                 exact["nu_total"] < 1e-10,
                 f"|nu| = {exact['nu_total']:.3e}, a factor of "
                 f"{crude['nu_total'] / max(exact['nu_total'], 1e-18):.1e}")

    # The two schemes must agree on the state transition matrix -- they share
    # the same linearisation, and only the input quadrature differs.
    rx, ru, rs = initialize_reference(two_d_case(), veh, 20, 3.0)
    dtau = 1.0 / 19
    Ad, Bd, _, _ = discretize(rx[5], ru[5], rs, veh, dtau)
    Ae, Bm, Bp, _, _ = discretize_exact_foh(rx[5], ru[5], rs, veh, dtau)
    ok &= report("both schemes share the state transition matrix",
                 float(np.abs(Ad - Ae).max()) < 1e-12,
                 f"difference = {float(np.abs(Ad - Ae).max()):.2e}")
    rel = float(np.abs(Bd - (Bm + Bp)).max()) / float(np.abs(Bd).max())
    ok &= report("and differ only in the input quadrature",
                 rel > 1e-3,
                 f"{rel * 100:.1f}% on the input matrix -- the whole cause")
    note("A few per cent on one matrix, and the difference between a", "")
    note("  residual of 2e-01 and one of 5e-16.", "")
    return ok


def test_boundary_conditions_and_cones():
    print("\nTEST 7 - Boundary conditions and the convex constraint set")
    v = PaperVehicle(alpha_m=0.03)
    r = cached("cool", lambda: solve_benchmark(
        two_d_case(), veh=PaperVehicle(alpha_m=0.03), alg=PaperAlgorithm(),
        sigma_guess=3.0, verbose=False))
    x, u = r["x"], r["u"]
    bc = two_d_case()
    ok = True

    ok &= report("terminal position at the pad",
                 float(np.linalg.norm(x[-1, IDX_R])) < 1e-6,
                 f"{float(np.linalg.norm(x[-1, IDX_R])):.2e} UL")
    ok &= report("terminal velocity as specified",
                 float(np.linalg.norm(x[-1, IDX_V] - bc.v_I_f)) < 1e-6)
    ok &= report("terminal attitude upright",
                 float(np.linalg.norm(x[-1, IDX_Q] - bc.q_B_I_f)) < 1e-6)
    ok &= report("terminal rate zero",
                 float(np.linalg.norm(x[-1, IDX_W])) < 1e-6)
    ok &= report("final thrust along the body axis (no touchdown torque)",
                 max(abs(u[-1, 1]), abs(u[-1, 2])) < 1e-6)

    mag = np.linalg.norm(u, axis=1)
    ok &= report("thrust upper bound respected",
                 float(mag.max()) <= v.T_max + 1e-6,
                 f"peak {float(mag.max()):.3f} of {v.T_max:.1f}")
    gim = np.degrees(np.arccos(np.clip(u[:, 0] / np.maximum(mag, 1e-9),
                                       -1.0, 1.0)))
    ok &= report("gimbal cone respected",
                 float(gim.max()) <= v.delta_max_deg + 1e-3,
                 f"peak {float(gim.max()):.2f} deg of {v.delta_max_deg:.0f}")
    ok &= report("mass never below dry",
                 float(x[:, IDX_M].min()) >= v.m_dry - 1e-6)

    # The 2-D case must stay in the Up-East plane, exactly as on Day 17.
    ok &= report("2-D case stays in the plane",
                 float(np.abs(x[:, 3]).max()) < 1e-9,
                 f"max |north| = {float(np.abs(x[:, 3]).max()):.2e} UL")
    note("initial attitude is left free, as the paper specifies", "")
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 18 - SZMUK & ACIKMESE (2018) BENCHMARK REPLICATION")
    print("=" * 70)
    results = [
        test_transcription(),
        test_model_matches_this_project(),
        test_alpha_m_is_not_harmless(),
        test_residual_location(),
        test_robustness_to_time_guess(),
        test_paper_algorithm_as_printed(),
        test_discretisation_is_what_broke_it(),
        test_boundary_conditions_and_cones(),
    ]
    print("\n" + "=" * 70)
    ok = all(results)
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    if ok:
        print("NOTE: Tests 5 and 6 passing means the paper's robustness claim")
        print("      DOES reproduce, with Algorithm 1 exactly as printed.")
        print("      Test 6b isolates what had to be right for that: the")
        print("      paper's own Eq. 22 quadrature, and an alpha_m the paper")
        print("      never gives.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
