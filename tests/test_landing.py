"""
Verification suite for the constrained landing optimization.

Tests check that:

    1. The nominal problem is feasible and produces a reasonable trajectory
    2. All constraints are satisfied (not just claimed by the solver)
    3. The glideslope cone is respected at every time step
    4. Thrust magnitude stays within [T_min, T_max]
    5. Thrust pointing angle stays within theta_max
    6. Mass dynamics are consistent (mass decreases monotonically)
    7. The lossless relaxation is actually tight
    8. An impossible problem correctly returns infeasible

Test 7 is not in the original plan and is the one that matters most. The
convexified problem only bounds ||T|| <= sigma; a solver can satisfy that while
leaving ||T|| well below T_min, which produces a trajectory that burns
minimum-throttle propellant while generating less than minimum-throttle force.
Checking ||T|| <= sigma alone passes trivially and hides it. Checking that the
two are *equal* is what makes the convexification meaningful.

Run:  python tests/test_landing.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.landing_problem import solve_landing          # noqa: E402
from src.dynamics import Vehicle, G_EARTH              # noqa: E402


def _supports_ansi() -> bool:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return False
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
        except Exception:
            return False
    return True


if _supports_ansi():
    PASS, FAIL = "\033[92m[PASS]\033[0m", "\033[91m[FAIL]\033[0m"
else:
    PASS, FAIL = "[PASS]", "[FAIL]"


def report(name: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name:<44}"
    if detail:
        msg += f" {detail}"
    print(msg)
    return ok


# ======================================================================
# TEST 1 - Nominal problem is feasible
# ======================================================================
def test_nominal_solution():
    print("\nTEST 1 - Nominal problem is feasible")
    vehicle = Vehicle()
    result = solve_landing(verbose=False)

    ok = report("solver returns optimal",
                result["status"] in ("optimal", "optimal_inaccurate"),
                f"status = {result['status']}")
    if not ok:
        print("  Cannot continue - problem was infeasible.")
        return False, None

    v_final = np.hypot(result["vx"][-1], result["vz"][-1])
    ok &= report("lands at target (x=0)", abs(result["x"][-1]) < 1.0,
                 f"x_final = {result['x'][-1]:.4f} m")
    ok &= report("lands at target (z=0)", abs(result["z"][-1]) < 1.0,
                 f"z_final = {result['z'][-1]:.4f} m")
    ok &= report("zero final velocity", v_final < 1.0,
                 f"|v_final| = {v_final:.4f} m/s")
    ok &= report("fuel consumed > 0", result["fuel"] > 0,
                 f"fuel = {result['fuel']:,.0f} kg")
    ok &= report("fuel consumed < propellant load",
                 result["fuel"] < vehicle.m_prop_initial,
                 f"{result['fuel']:,.0f} < {vehicle.m_prop_initial:,.0f} kg")
    return ok, result


# ======================================================================
# TEST 2 - Constraints hold in the returned solution
# ======================================================================
def test_constraints_satisfied(result):
    print("\nTEST 2 - All constraints satisfied in solution")
    vehicle = Vehicle()
    ok = True

    # Glideslope: |x| <= z / tan(gamma)
    gamma = result["gamma_gs_deg"]
    tan_gs = np.tan(np.radians(gamma))
    violations = int(np.sum(np.abs(result["x"]) * tan_gs > result["z"] + 0.01))
    ok &= report(f"glideslope ({gamma:.0f} deg) at all nodes", violations == 0,
                 f"violations: {violations}")

    min_alt = float(np.min(result["z"]))
    ok &= report("altitude >= 0 everywhere", min_alt >= -0.01,
                 f"min altitude = {min_alt:.4f} m")

    sigma = result["sigma"]
    T_actual = np.hypot(result["Tx"], result["Tz"])

    ok &= report("sigma >= T_min", sigma.min() >= vehicle.T_min - 1.0,
                 f"min sigma = {sigma.min()/1e6:.3f} MN "
                 f"(T_min = {vehicle.T_min/1e6:.3f})")
    ok &= report("sigma <= T_max", sigma.max() <= vehicle.T_max + 1.0,
                 f"max sigma = {sigma.max()/1e6:.3f} MN "
                 f"(T_max = {vehicle.T_max/1e6:.3f})")
    ok &= report("||T|| <= sigma (SOC)", bool(np.all(T_actual <= sigma + 1.0)),
                 f"max violation = {np.max(T_actual - sigma):.1f} N")

    # Pointing
    theta_max = result["theta_max_deg"]
    cos_theta = np.cos(np.radians(theta_max))
    ok &= report(f"thrust pointing <= {theta_max:.0f} deg",
                 bool(np.all(result["Tz"] >= sigma * cos_theta - 1.0)))

    mass_diff = np.diff(result["m"])
    ok &= report("mass monotonically decreasing",
                 bool(np.all(mass_diff <= 0.01)),
                 f"max increase = {np.max(mass_diff):.4f} kg")
    ok &= report("mass >= m_dry", result["m"].min() >= vehicle.m_dry - 1.0,
                 f"min mass = {result['m'].min():,.0f} kg")
    return ok


# ======================================================================
# TEST 3 - The relaxation is lossless in practice, not just in theory
# ======================================================================
def test_relaxation_is_tight(result):
    print("\nTEST 3 - Lossless convexification is actually tight")
    vehicle = Vehicle()
    sigma = result["sigma"]
    T_actual = np.hypot(result["Tx"], result["Tz"])
    gap = sigma - T_actual
    rel_gap = gap.max() / vehicle.T_min

    ok = report("sigma == ||T|| (relaxation tight)", rel_gap < 0.01,
                f"max gap = {gap.max():,.0f} N ({100*rel_gap:.2f}% of T_min)")
    # The physical constraint the relaxation is standing in for. If this fails
    # the trajectory commands less thrust than the engines can produce.
    ok &= report("||T|| >= T_min (physically flyable)",
                 T_actual.min() >= vehicle.T_min * 0.99,
                 f"min ||T|| = {T_actual.min()/vehicle.T_min:.3f} x T_min")
    return ok


# ======================================================================
# TEST 4 - Impossible problem returns infeasible
# ======================================================================
def test_infeasible():
    print("\nTEST 4 - Impossible problem returns infeasible")
    result = solve_landing(
        x0=10000.0, z0=200.0, vx0=-300.0, vz0=-10.0,
        t_burn=5.0, verbose=False,
    )
    ok = report("status is infeasible",
                result["status"] in ("infeasible", "infeasible_inaccurate"),
                f"status = {result['status']}")

    # 5 km downrange at 500 m altitude is outside the glideslope cone before
    # the dynamics are even considered — geometry alone rules it out.
    geo = solve_landing(x0=5000.0, z0=500.0, vx0=-200.0, vz0=-50.0,
                        t_burn=15.0, verbose=False)
    ok &= report("shallow long-range entry infeasible",
                 geo["status"] in ("infeasible", "infeasible_inaccurate"),
                 f"status = {geo['status']}")
    return ok


def main():
    print("=" * 70)
    print("DAY 3 - CONSTRAINED LANDING OPTIMIZATION VERIFICATION")
    print("=" * 70)

    ok1, result = test_nominal_solution()
    if result is not None:
        ok2 = test_constraints_satisfied(result)
        ok3 = test_relaxation_is_tight(result)
    else:
        ok2 = ok3 = False
    ok4 = test_infeasible()

    all_ok = all([ok1, ok2, ok3, ok4])
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
