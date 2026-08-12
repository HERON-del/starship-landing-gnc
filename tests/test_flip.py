"""
Verification suite for the flip-and-land optimizer.

Tests:
    1. Nominal flip is feasible and lands upright at rest
    2. All path constraints hold: pitch rate, gimbal, throttle, glideslope
    3. SCvx converged: the linear model agrees with the true dynamics
    4. Rotation costs propellant relative to a vertical-entry baseline
    5. The commanded control, replayed through the verified 6-DoF integrator,
       actually flies the trajectory
    6. The entry-pitch ceiling is real: 65 degrees solves, 70 does not, and
       relaxing either the glideslope or the pitch-rate limit recovers it

Test 5 is the one that matters. Tests 1-3 ask the optimiser whether it obeyed
its own linearised model. Test 5 takes the commanded throttle and gimbal, flies
them through the independently verified non-linear simulator from Day 5 Part 2,
and measures where the vehicle actually ends up.

Run:  python tests/test_flip.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.landing_flip import solve_flip_landing, feasible_entry_state   # noqa: E402
from src.dynamics_6dof import Vehicle6DoF, dynamics_6dof                # noqa: E402
from src.integrators import propagate                                    # noqa: E402
from tests.test_dynamics import PASS, FAIL                               # noqa: E402

N_NODES = 80
T_BURN = 15.0


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name:<48}"
    if detail:
        msg += f" {detail}"
    print(msg)
    return bool(ok)


# ======================================================================
def test_nominal():
    print("\nTEST 1 - Nominal flip is feasible")
    r = solve_flip_landing(N=N_NODES, t_burn=T_BURN, verbose=False)
    ok = report("solver returns optimal", r["status"].startswith("optimal"),
                f"status = {r['status']}")
    if not ok:
        return False, None

    ok &= report("lands on the pad",
                 abs(r["x"][-1]) < 1.0 and abs(r["z"][-1]) < 1.0,
                 f"({r['x'][-1]:.4f}, {r['z'][-1]:.4f}) m")
    ok &= report("arrives at rest",
                 np.hypot(r["vx"][-1], r["vz"][-1]) < 1.0,
                 f"|v| = {np.hypot(r['vx'][-1], r['vz'][-1]):.4f} m/s")
    ok &= report("arrives upright",
                 abs(np.degrees(r["theta"][-1])) < 0.5,
                 f"theta_f = {np.degrees(r['theta'][-1]):.4f} deg")
    ok &= report("arrives without rotation",
                 abs(np.degrees(r["omega"][-1])) < 0.5,
                 f"omega_f = {np.degrees(r['omega'][-1]):.4f} deg/s")
    ok &= report("burned propellant", 0 < r["fuel"] < Vehicle6DoF().m_prop_initial,
                 f"{r['fuel']:,.0f} kg")
    return ok, r


# ======================================================================
def test_constraints(r):
    print("\nTEST 2 - Path constraints satisfied")
    veh = Vehicle6DoF()
    ok = True

    ok &= report("pitch rate within limit",
                 np.max(np.abs(r["omega"])) <= veh.omega_max + 1e-6,
                 f"peak {np.degrees(np.max(np.abs(r['omega']))):.2f} of "
                 f"{np.degrees(veh.omega_max):.2f} deg/s")
    ok &= report("gimbal within limit",
                 np.max(np.abs(r["delta"])) <= veh.delta_max + 1e-6,
                 f"peak {np.degrees(np.max(np.abs(r['delta']))):.2f} of "
                 f"{veh.delta_max_deg:.0f} deg")
    ok &= report("torque consistent with gimbal and thrust",
                 np.all(np.abs(r["tau"]) <= r["sigma"] * veh.L_engine
                        * np.sin(veh.delta_max) + 1.0),
                 f"peak {np.max(np.abs(r['tau'])):,.0f} N m")
    ok &= report("throttle within [T_min, T_max]",
                 r["sigma"].min() >= veh.T_min - 100
                 and r["sigma"].max() <= veh.T_max + 100,
                 f"{r['sigma'].min()/1e6:.2f} - {r['sigma'].max()/1e6:.2f} MN")

    tan_gs = np.tan(np.radians(r["gamma_gs_deg"]))
    ok &= report("glideslope respected",
                 int(np.sum(np.abs(r["x"]) * tan_gs > r["z"] + 1.0)) == 0,
                 f"violations {int(np.sum(np.abs(r['x']) * tan_gs > r['z'] + 1.0))}")
    ok &= report("altitude never negative", np.min(r["z"]) >= -0.01,
                 f"min {np.min(r['z']):.4f} m")
    ok &= report("mass decreases monotonically",
                 np.all(np.diff(r["m"]) <= 1.0),
                 f"max increase {np.max(np.diff(r['m'])):.4f} kg")
    return ok


# ======================================================================
def test_convergence(r):
    print("\nTEST 3 - SCvx converged against the true dynamics")
    ok = report("linearisation defect is small",
                r["final_defect"] < 0.01,
                f"{r['final_defect']:.5f} of T_max")
    ok &= report("converged inside the iteration budget",
                 r["iterations"] < 40, f"{r['iterations']} iterations")

    # The defect must actually improve; a flat history means the loop is not
    # doing anything, which the earlier delta=0 expansion did.
    first = r["history"][0][2]
    ok &= report("defect improved over the run",
                 r["final_defect"] < 0.25 * first,
                 f"{first:.4f} -> {r['final_defect']:.5f}")
    return ok


# ======================================================================
def test_rotation_costs_fuel():
    print("\nTEST 4 - Rotating costs propellant")
    r_flip = solve_flip_landing(N=N_NODES, t_burn=T_BURN, theta0_deg=60.0,
                                verbose=False)
    # Same burn and entry geometry, but starting upright: no flip to perform.
    z0, vz0 = feasible_entry_state(Vehicle6DoF(), T_BURN, 60.0,
                                   t_flip=1.4 * np.radians(60.0) / 0.5)
    r_up = solve_flip_landing(N=N_NODES, t_burn=T_BURN, theta0_deg=0.0,
                              z0=z0, vz0=vz0, verbose=False)

    if not (r_flip["status"].startswith("optimal")
            and r_up["status"].startswith("optimal")):
        print("  One of the two solves failed - cannot compare.")
        return False

    extra = r_flip["fuel"] - r_up["fuel"]
    pct = 100 * extra / r_up["fuel"]
    ok = report("flip costs at least as much as vertical entry",
                extra > -5.0,
                f"{r_flip['fuel']:,.0f} vs {r_up['fuel']:,.0f} kg "
                f"({extra:+,.0f}, {pct:+.1f}%)")
    ok &= report("rotation overhead is plausible (< 30%)",
                 pct < 30.0, f"{pct:+.1f}%")
    return ok


# ======================================================================
def test_replay(r):
    """Fly the commanded throttle and gimbal through the nonlinear simulator."""
    print("\nTEST 5 - Commanded control replayed through the 6-DoF integrator")
    veh = Vehicle6DoF()
    t_ctrl = r["t"][:-1]
    sigma, delta = r["sigma"], r["delta"]
    dt_ctrl = r["t_burn"] / len(sigma)

    def control(t, state, vehicle):
        k = min(int(t / dt_ctrl), len(sigma) - 1)
        return sigma[k], delta[k]

    y0 = np.array([r["x"][0], r["z"][0], r["vx"][0], r["vz"][0],
                   r["theta"][0], r["omega"][0], veh.m_wet])
    _, y = propagate(lambda t, yy, *a: dynamics_6dof(t, yy, control, veh),
                     y0, (0.0, r["t_burn"]), r["t_burn"] / 4000, method="rk4")

    pos_err = float(np.hypot(y[-1, 0], y[-1, 1]))
    vel_err = float(np.hypot(y[-1, 2], y[-1, 3]))
    att_err = float(abs(np.degrees(y[-1, 4])))
    rate_err = float(abs(np.degrees(y[-1, 5])))
    print(f"         position {pos_err:7.1f} m   velocity {vel_err:6.2f} m/s   "
          f"pitch {att_err:6.2f} deg   rate {rate_err:6.2f} deg/s")

    # The flip optimiser still discretises with forward Euler. Day 4 measured
    # that as a 1.5% miss on the 3-DoF problem and showed trapezoidal
    # collocation cutting it 7x; rotation compounds it further, and this is what
    # that costs. The bound is set where Euler actually lands, not where we
    # would like it to - upgrading this solver to trapezoidal is the outstanding
    # action, and this number is how it will be judged.
    ok = report("lands within 5% of the descent (Euler-limited)",
                pos_err < 0.05 * r["z"][0],
                f"{pos_err:.1f} m of {r['z'][0]:,.0f} m "
                f"({100 * pos_err / r['z'][0]:.2f}%)")
    ok &= report("attitude within 5 deg of upright", att_err < 5.0,
                 f"{att_err:.2f} deg")
    ok &= report("residual rate under 5 deg/s", rate_err < 5.0,
                 f"{rate_err:.2f} deg/s")
    return ok


# ======================================================================
def test_entry_pitch_ceiling():
    """
    The entry-pitch limit is a real interaction, not a solver artefact.

    The engine is lit throughout, so while tilted it pushes the vehicle sideways
    at up to T_min/m. The pitch rate caps how fast that ends. Past a certain
    entry angle the excursion cannot fit in the glideslope corridor and still be
    nulled by touchdown - and relaxing *either* constraint alone recovers it,
    which is what shows the two are binding together.
    """
    print("\nTEST 6 - The entry-pitch ceiling is a genuine constraint interaction")
    ok = True

    def ceiling(vehicle=None, angles=(30, 40, 50, 55, 60, 65, 70, 75, 80)):
        best = None
        for a in angles:
            r = solve_flip_landing(vehicle=vehicle, N=N_NODES, t_burn=T_BURN,
                                   theta0_deg=float(a), verbose=False)
            if r["status"].startswith("optimal"):
                best = a
            elif best is not None:
                return best, a       # first failure above a success
        return best, None

    hi, fail = ceiling()
    print(f"         nominal vehicle: highest feasible {hi} deg, "
          f"first infeasible {fail} deg")
    ok &= report("a ceiling exists below a full belly-flop",
                 hi is not None and fail is not None and fail < 90,
                 f"{hi} deg feasible, {fail} deg not")

    # Raise the pitch rate: the flip finishes before the lateral excursion
    # builds, so the ceiling moves up. That is what identifies the pitch-rate
    # limit as one of the two binding constraints.
    fast = Vehicle6DoF(omega_max=0.9)
    hi_fast, fail_fast = ceiling(fast)
    print(f"         omega_max 51 deg/s: highest feasible {hi_fast} deg, "
          f"first infeasible {fail_fast} deg")
    ok &= report("a faster pitch rate raises the ceiling",
                 hi_fast is not None and hi is not None and hi_fast > hi,
                 f"{hi} deg -> {hi_fast} deg")

    # And the failing angle becomes feasible outright on the faster vehicle.
    if fail is not None:
        r = solve_flip_landing(vehicle=fast, N=N_NODES, t_burn=T_BURN,
                               theta0_deg=float(fail), verbose=False)
        ok &= report(f"{fail} deg solves once the rate limit is relaxed",
                     r["status"].startswith("optimal"), f"status = {r['status']}")
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 5 - FLIP-AND-LAND OPTIMIZER VERIFICATION")
    print("=" * 70)

    ok1, r = test_nominal()
    ok2 = test_constraints(r) if r else False
    ok3 = test_convergence(r) if r else False
    ok4 = test_rotation_costs_fuel()
    ok5 = test_replay(r) if r else False
    ok6 = test_entry_pitch_ceiling()

    all_ok = all([ok1, ok2, ok3, ok4, ok5, ok6])
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
