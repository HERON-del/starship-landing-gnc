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
                 f"{hot['nu_total']:.3f} -> {cool['nu_total']:.4f}, a factor "
                 f"of {hot['nu_total'] / max(cool['nu_total'], 1e-12):.0f}")
    note("so the discretisation is not what the guide says it is", "")
    note("  mass pinned at m_dry makes the mass row unsatisfiable, and", "")
    note("  velocity follows because F/m carries the wrong mass.", "")
    return ok


def test_robustness_to_time_guess():
    """
    The paper's central empirical claim, which the guide declines to test.

    Ten time-of-flight guesses, all converging to within 0.01 UT. It is the
    one number the paper actually stakes its case on, and the one this
    replication is really trying to check.
    """
    print("\nTEST 5 - Robustness to the time-of-flight guess")
    alg = PaperAlgorithm()
    veh = PaperVehicle(alpha_m=0.03)
    ok = True

    def sweep():
        return [solve_benchmark(two_d_case(), veh=veh, alg=alg,
                                sigma_guess=float(g), verbose=False)["sigma"]
                for g in (1, 3, 5, 8, 10)]

    tfs = np.array(cached("sweep", sweep))
    spread = float(tfs.max() - tfs.min())
    ok &= report("the claim does NOT reproduce here",
                 spread > 0.01,
                 f"spread {spread:.2f} UT across five guesses, paper says "
                 f"< 0.01")
    corr = float(np.corrcoef(np.array([1, 3, 5, 8, 10], dtype=float), tfs)[0, 1])
    ok &= report("the answer tracks the guess instead",
                 corr > 0.8,
                 f"correlation between guess and result = {corr:.3f}")
    note("tf found: " + ", ".join(f"{t:.1f}" for t in tfs), "")
    note("A free variable that follows its own initial guess", "")
    note("  is not being solved for. See Test 6 for why.", "")
    return ok


def test_paper_algorithm_as_printed():
    """
    Algorithm 1 with no hard trust region, exactly as the paper prints it.

    The guide says this collapses sigma toward zero and oscillates. It does
    not. It drives the virtual control to machine precision and inflates
    sigma without bound -- the opposite direction, which matters because the
    two failures need opposite fixes.
    """
    print("\nTEST 6 - The paper's literal Algorithm 1")
    alg = PaperAlgorithm()
    veh = PaperVehicle(alpha_m=0.03)
    ok = True

    soft = cached("soft", lambda: solve_benchmark(
        two_d_case(), veh=veh, alg=alg, sigma_guess=1.0,
        hard_trust=False, verbose=False))
    seq = soft["history"]["sigma"]

    ok &= report("it solves, every iteration",
                 soft["ever_solved"] and len(seq) == alg.N_iter_max)
    ok &= report("virtual control reaches machine precision",
                 soft["nu_total"] < 1e-12,
                 f"|nu| = {soft['nu_total']:.2e}, far below the hard-trust "
                 f"version")
    ok &= report("sigma does NOT collapse toward zero",
                 min(seq) > 0.5,
                 f"minimum sigma over the run = {min(seq):.2f}")
    ok &= report("it inflates instead, monotonically",
                 seq[-1] > 3.0 * seq[0],
                 f"{seq[0]:.2f} -> {seq[-1]:.2f} UT")

    note("Time is nearly free in the paper's own cost weights.", "")
    note(f"  w_nu = {alg.w_nu:.0e} multiplies |nu|; sigma's coefficient", "")
    note("  is 1. Feasibility outprices minimum-time 100,000 to 1, so the", "")
    note("  optimiser buys any amount of flight time to shed a little", "")
    note("  virtual control. The guide's hard trust region hides this by", "")
    note("  pinning sigma near its guess -- which is exactly why Test 5", "")
    note("  finds the answer tracking the guess.", "")
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
        test_boundary_conditions_and_cones(),
    ]
    print("\n" + "=" * 70)
    ok = all(results)
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    if ok:
        print("NOTE: Tests 5 and 6 passing means the paper's robustness claim")
        print("      does NOT reproduce here. That is the finding, asserted so")
        print("      it cannot regress into a silent success.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
