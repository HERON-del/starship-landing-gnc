"""
Verification suite for free-final-time landing with trapezoidal discretization.

Tests:
    1. Free-time trapezoidal solves and lands
    2. Free time uses less fuel than a fixed 20 s burn (gravity-loss argument)
    3. Euler and trapezoidal agree on fuel and duration
    4. All path constraints hold in the free-time solution
    5. The optimiser's own control profile, replayed through the verified RK4
       integrator, actually lands the vehicle — and trapezoidal beats Euler

Test 5 is the one that matters. Tests 1-4 ask the optimiser whether it obeyed
its own discretized model, which it always does. Test 5 asks whether that model
resembles reality, by flying the commanded thrust through the independently
verified Day 2 simulator and measuring where the vehicle actually ends up. A
solution can look perfect inside the optimiser and still miss the pad.

Run:  python tests/test_free_time.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.landing_free_time import solve_landing_free_time      # noqa: E402
from src.landing_problem import (                              # noqa: E402
    solve_landing,
    feasible_entry_state,
    max_downrange,
)
from src.dynamics import Vehicle, dynamics_3dof                # noqa: E402
from src.integrators import propagate                          # noqa: E402
from tests.test_dynamics import PASS, FAIL                     # noqa: E402

N_NODES = 50
T_NOMINAL = 20.0


def report(name: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name:<48}"
    if detail:
        msg += f" {detail}"
    print(msg)
    return bool(ok)


def shared_entry(vehicle, gamma=80.0, theta=30.0):
    """One entry state for every variant, so the numbers are commensurable."""
    z0, vz0 = feasible_entry_state(vehicle, T_NOMINAL, theta)
    return dict(x0=0.75 * max_downrange(z0, gamma), z0=z0,
                vx0=-40.0, vz0=vz0)


# ======================================================================
# TEST 1
# ======================================================================
def test_free_time_solves(entry):
    print("\nTEST 1 - Free-time trapezoidal is feasible")
    r = solve_landing_free_time(N=N_NODES, method="trapz", verbose=False,
                                t_nominal=T_NOMINAL, **entry)
    ok = report("solver returns optimal", r["status"].startswith("optimal"),
                f"status = {r['status']}")
    if not ok:
        return False, None

    ok &= report("lands at target",
                 abs(r["x"][-1]) < 1.0 and abs(r["z"][-1]) < 1.0,
                 f"({r['x'][-1]:.4f}, {r['z'][-1]:.4f}) m")
    ok &= report("zero final velocity",
                 np.hypot(r["vx"][-1], r["vz"][-1]) < 1.0,
                 f"|v| = {np.hypot(r['vx'][-1], r['vz'][-1]):.4f} m/s")
    ok &= report("burn time strictly inside the bounds",
                 8.0 < r["t_f"] < 34.0, f"t_f = {r['t_f']:.2f} s")
    ok &= report("fuel > 0", r["fuel"] > 0, f"fuel = {r['fuel']:,.0f} kg")
    return ok, r


# ======================================================================
# TEST 2
# ======================================================================
def test_free_beats_fixed(entry):
    print("\nTEST 2 - Free time uses less fuel than a fixed 20 s burn")
    r_fixed = solve_landing(N=N_NODES, t_burn=T_NOMINAL, verbose=False, **entry)
    r_free = solve_landing_free_time(N=N_NODES, method="trapz", verbose=False,
                                     t_nominal=T_NOMINAL, **entry)

    if not r_fixed["status"].startswith("optimal"):
        print("  Fixed-time solve failed - cannot compare.")
        return False
    if not r_free["status"].startswith("optimal"):
        print("  Free-time solve failed - cannot compare.")
        return False

    saved = r_fixed["fuel"] - r_free["fuel"]
    ok = report("free-time fuel <= fixed-time fuel",
                r_free["fuel"] <= r_fixed["fuel"] * 1.02,
                f"{r_free['fuel']:,.0f} vs {r_fixed['fuel']:,.0f} kg "
                f"(saved {saved:,.0f}, {100 * saved / r_fixed['fuel']:.1f}%)")
    ok &= report("optimiser chose a shorter burn",
                 r_free["t_f"] < T_NOMINAL,
                 f"{r_free['t_f']:.2f} s vs {T_NOMINAL:.1f} s")
    return ok


# ======================================================================
# TEST 3
# ======================================================================
def test_trapz_vs_euler(entry):
    print("\nTEST 3 - Trapezoidal vs Euler agreement")
    r_e = solve_landing_free_time(N=N_NODES, method="euler", verbose=False,
                                  t_nominal=T_NOMINAL, **entry)
    r_t = solve_landing_free_time(N=N_NODES, method="trapz", verbose=False,
                                  t_nominal=T_NOMINAL, **entry)
    if not (r_e["status"].startswith("optimal")
            and r_t["status"].startswith("optimal")):
        print("  One or both solves failed - cannot compare.")
        return False, None, None

    diff_pct = abs(r_e["fuel"] - r_t["fuel"]) / r_e["fuel"] * 100
    ok = report("fuel values agree within 15%", diff_pct < 15.0,
                f"Euler={r_e['fuel']:,.0f}, Trapz={r_t['fuel']:,.0f} "
                f"({diff_pct:.1f}% apart)")
    ok &= report("burn times agree within 5 s",
                 abs(r_e["t_f"] - r_t["t_f"]) < 5.0,
                 f"Euler={r_e['t_f']:.2f}, Trapz={r_t['t_f']:.2f} s")
    return ok, r_e, r_t


# ======================================================================
# TEST 4
# ======================================================================
def test_constraints(result, gamma=80.0, theta=30.0):
    print("\nTEST 4 - Path constraints satisfied")
    veh = Vehicle()
    ok = True

    tan_gs = np.tan(np.radians(gamma))
    violations = int(np.sum(np.abs(result["x"]) * tan_gs > result["z"] + 1.0))
    ok &= report("glideslope satisfied at all nodes", violations == 0,
                 f"violations: {violations}")
    ok &= report("altitude >= 0", np.min(result["z"]) >= -0.01,
                 f"min = {np.min(result['z']):.4f} m")

    sigma = result["sigma"]
    T_mag = np.hypot(result["Tx"], result["Tz"])
    ok &= report("sigma >= T_min", np.min(sigma) >= veh.T_min - 100,
                 f"min = {np.min(sigma) / 1e6:.3f} MN")
    ok &= report("sigma <= T_max", np.max(sigma) <= veh.T_max + 100,
                 f"max = {np.max(sigma) / 1e6:.3f} MN")
    ok &= report("relaxation tight (sigma == ||T||)",
                 np.max(sigma - T_mag) < 0.01 * veh.T_min,
                 f"max gap = {np.max(sigma - T_mag):,.0f} N")
    cos_t = np.cos(np.radians(theta))
    ok &= report("pointing within theta_max",
                 np.all(result["Tz"] >= sigma * cos_t - 100),
                 f"max tilt = "
                 f"{np.degrees(np.arctan2(np.abs(result['Tx']), result['Tz'])).max():.1f} deg")
    ok &= report("mass monotonically decreasing",
                 np.all(np.diff(result["m"]) <= 1.0),
                 f"max increase = {np.max(np.diff(result['m'])):.4f} kg")
    ok &= report("mass >= m_dry", np.min(result["m"]) >= veh.m_dry - 1.0,
                 f"min = {np.min(result['m']):,.0f} kg")
    return ok


# ======================================================================
# TEST 5 - replay through the verified integrator
# ======================================================================
def _replay(result, entry, vehicle):
    """
    Fly the optimiser's commanded thrust through the Day 2 RK4 integrator and
    return the terminal state error.

    The control is reconstructed the way each scheme assumes it: zero-order
    hold for Euler, linear interpolation for trapezoidal.
    """
    t_f = result["t_f"]
    Tx, Tz = result["Tx"], result["Tz"]
    n_ctrl = len(Tx)
    trapz = n_ctrl == len(result["t"])

    if trapz:
        t_ctrl = np.linspace(0.0, t_f, n_ctrl)

        def control(t, state, veh):
            return np.array([np.interp(t, t_ctrl, Tx),
                             np.interp(t, t_ctrl, Tz)])
    else:
        dt_ctrl = t_f / n_ctrl

        def control(t, state, veh):
            k = min(int(t / dt_ctrl), n_ctrl - 1)
            return np.array([Tx[k], Tz[k]])

    y0 = np.array([entry["x0"], entry["z0"],
                   entry["vx0"], entry["vz0"], vehicle.m_wet])
    _, y = propagate(dynamics_3dof, y0, (0.0, t_f), t_f / 4000,
                     control, vehicle, method="rk4")
    return {
        "pos_err": float(np.hypot(y[-1, 0], y[-1, 1])),
        "vel_err": float(np.hypot(y[-1, 2], y[-1, 3])),
        "fuel": float(vehicle.m_wet - y[-1, 4]),
    }


def test_replay(r_euler, r_trapz, entry):
    print("\nTEST 5 - Optimiser control replayed through the verified RK4 integrator")
    veh = Vehicle()
    ok = True
    errs = {}
    for name, r in (("Euler", r_euler), ("Trapz", r_trapz)):
        e = _replay(r, entry, veh)
        errs[name] = e
        print(f"         {name:<6} position error {e['pos_err']:8.1f} m, "
              f"velocity error {e['vel_err']:7.2f} m/s, "
              f"fuel {e['fuel']:,.0f} kg")

    # Position miss is the claim that holds up. Terminal velocity is a much
    # smaller number in both cases and the ordering flips between runs, so
    # asserting a direction on it would be asserting noise. Both are reported.
    ok &= report("trapezoidal lands closer than Euler",
                 errs["Trapz"]["pos_err"] < errs["Euler"]["pos_err"],
                 f"{errs['Trapz']['pos_err']:.1f} m vs "
                 f"{errs['Euler']['pos_err']:.1f} m")

    v_entry = np.hypot(entry["vx0"], entry["vz0"])
    combined = {k: e["pos_err"] / entry["z0"] + e["vel_err"] / v_entry
                for k, e in errs.items()}
    ok &= report("trapezoidal lower combined normalised miss",
                 combined["Trapz"] < combined["Euler"],
                 f"{combined['Trapz']:.2e} vs {combined['Euler']:.2e}")

    # Only trapezoidal is held to a tight bound. Euler's miss is the *result*
    # of this test, not a failure of it — a first-order scheme at N=50 really
    # does put the vehicle tens of metres off. It still gets a loose sanity
    # bound so that an actual bug would surface rather than being read as
    # "expected discretisation error".
    ok &= report("trapezoidal replay lands within 1% of the descent",
                 errs["Trapz"]["pos_err"] < 0.01 * entry["z0"],
                 f"{errs['Trapz']['pos_err']:.1f} m of {entry['z0']:,.0f} m "
                 f"({100 * errs['Trapz']['pos_err'] / entry['z0']:.2f}%)")
    ok &= report("Euler replay within a loose 5% sanity bound",
                 errs["Euler"]["pos_err"] < 0.05 * entry["z0"],
                 f"{errs['Euler']['pos_err']:.1f} m of {entry['z0']:,.0f} m "
                 f"({100 * errs['Euler']['pos_err'] / entry['z0']:.2f}%)")

    ratio = errs["Euler"]["pos_err"] / max(errs["Trapz"]["pos_err"], 1e-9)
    print(f"         -> same node count, {ratio:.1f}x smaller miss from "
          f"second-order collocation")
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 4 - FREE FINAL TIME & TRAPEZOIDAL VERIFICATION")
    print("=" * 70)

    veh = Vehicle()
    entry = shared_entry(veh)
    print(f"Shared entry state: ({entry['x0']:,.0f}, {entry['z0']:,.0f}) m, "
          f"({entry['vx0']:.1f}, {entry['vz0']:.1f}) m/s")

    ok1, r_trapz = test_free_time_solves(entry)
    ok2 = test_free_beats_fixed(entry)
    ok3, r_e, r_t = test_trapz_vs_euler(entry)
    ok4 = test_constraints(r_trapz) if r_trapz is not None else False
    ok5 = test_replay(r_e, r_t, entry) if r_e is not None else False

    all_ok = all([ok1, ok2, ok3, ok4, ok5])
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
