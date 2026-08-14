"""
Verification of the complete SCvx solver.

Tests:
    1. It converges
    2. Terminal constraints are satisfied
    3. Virtual control vanishes
    4. Free final time is real -- it moves, and it moves the dynamics
    5. Log-mass is consistent with the mass it claims to represent
    6. Fuel against the Day 7 solver
    7. Trapezoidal collocation is measurably more accurate than Euler
    8. Perturbed initial conditions

Test 4 is not the formality it looks like. A free-time implementation can
declare `t_f` a variable, bound it, give it a trust region and a penalty, and
still never let it touch a dynamics constraint -- in which case the penalty
drives it to a bound and the "optimisation" reports a constant. So it is not
enough to check that `t_f` differs from the guess: the test forces the bounds
around a value the solver would not otherwise choose and checks that the fuel
*changes*, which it only can if `t_f` is genuinely coupled.

Test 7 is the one the day exists for. Day 5's suite left trapezoidal
collocation as the outstanding action and said "this number is how it will be
judged" of its own Euler replay error. This measures both solvers on the same
problem, with the burn duration pinned so the only difference is the
discretisation, and replays each through the independently verified nonlinear
integrator.

Run:  python tests/test_scvx_complete.py
"""

import os
import sys
import warnings

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.scvx_complete import solve_scvx_complete                # noqa: E402
from src.scvx import solve_scvx                                  # noqa: E402
from src.scvx_params import SCvxParams                           # noqa: E402
from src.dynamics_6dof import Vehicle6DoF                        # noqa: E402
from src.dynamics_aero import dynamics_full                      # noqa: E402
from src.aero import AeroConfig                                  # noqa: E402
from src.integrators import propagate                            # noqa: E402
from tests.test_dynamics import PASS, FAIL                       # noqa: E402

warnings.filterwarnings("ignore")

T_GUESS = 8.0
THETA0_DEG = 30.0


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name:<54}"
    if detail:
        msg += f" {detail}"
    print(msg)
    return bool(ok)


def solve(**kw):
    kw.setdefault("aero", AeroConfig())
    kw.setdefault("t_burn_guess", T_GUESS)
    kw.setdefault("theta0_deg", THETA0_DEG)
    kw.setdefault("verbose", False)
    return solve_scvx_complete(**kw)


def replay(r, vehicle):
    """Fly the commanded throttle and gimbal through the 6-DoF simulator."""
    sigma, delta = r["sigma"], r["delta"]
    t_f = r.get("t_f", r.get("t_burn"))
    dt_ctrl = t_f / len(sigma)

    def control(t, state, veh):
        k = min(int(t / dt_ctrl), len(sigma) - 1)
        return sigma[k], delta[k]

    y0 = np.array([r["x"][0], r["z"][0], r["vx"][0], r["vz"][0],
                   r["theta"][0], r["omega"][0], vehicle.m_wet])
    _, y = propagate(
        lambda t, yy, *a: dynamics_full(t, yy, control, vehicle, AeroConfig()),
        y0, (0.0, t_f), t_f / 4000, method="rk4")
    return (float(np.hypot(y[-1, 0], y[-1, 1])),
            float(abs(np.degrees(y[-1, 4]))),
            float(abs(np.degrees(y[-1, 5]))))


# ======================================================================
def test_convergence():
    print("\nTEST 1 - Convergence")
    r = solve()
    ok = report("solver reports converged", r.get("status") == "converged")
    ok &= report("converges in 35 iterations or fewer",
                 r.get("iterations", 99) <= 35,
                 f"{r.get('iterations', '?')} iterations")
    ok &= report("solves in under 90 s", r.get("elapsed", 999) < 90.0,
                 f"{r.get('elapsed', 0):.1f}s")
    return ok, r


def test_terminal(r):
    print("\nTEST 2 - Terminal constraints")
    ok = report("x_f on the pad", abs(r["x"][-1]) < 5.0, f"{r['x'][-1]:.4f} m")
    ok &= report("z_f on the pad", abs(r["z"][-1]) < 5.0,
                 f"{r['z'][-1]:.4f} m")
    ok &= report("vx_f at rest", abs(r["vx"][-1]) < 3.0,
                 f"{r['vx'][-1]:.4f} m/s")
    ok &= report("vz_f at rest", abs(r["vz"][-1]) < 3.0,
                 f"{r['vz'][-1]:.4f} m/s")
    ok &= report("theta_f upright", abs(np.degrees(r["theta"][-1])) < 2.0,
                 f"{np.degrees(r['theta'][-1]):.4f} deg")
    ok &= report("omega_f not rotating", abs(np.degrees(r["omega"][-1])) < 5.0,
                 f"{np.degrees(r['omega'][-1]):.4f} deg/s")
    return ok


def test_virtual_control(r):
    print("\nTEST 3 - Virtual control")
    ok = report("slack vanishes", r["vc_norm"] < 1e-6,
                f"|nu| = {r['vc_norm']:.2e}")
    dfc = [d for d in r["history"]["defect"] if not np.isnan(d)]
    ok &= report("true nonlinear defect decreases",
                 len(dfc) >= 3 and dfc[-1] < 0.1 * dfc[0],
                 f"{dfc[0]:.2e} -> {dfc[-1]:.2e}")
    return ok


def test_free_time(r):
    """
    The variable must move, stay legal, be robust to the guess -- and,
    critically, actually be coupled to the dynamics.
    """
    print("\nTEST 4 - Free final time")
    lo, hi = r["t_f_bounds"]
    ok = report("t_f present and positive", r.get("t_f", 0) > 0)
    ok &= report("t_f within bounds", lo <= r["t_f"] <= hi,
                 f"{r['t_f']:.3f} s in [{lo:.1f}, {hi:.1f}]")
    ok &= report("t_f moved off the initial guess",
                 abs(r["t_f"] - T_GUESS) > 1e-3,
                 f"{r['t_f']:.3f} s vs guess {T_GUESS:.1f} s")
    ok &= report("t_f did not simply peg at a bound",
                 min(abs(r["t_f"] - lo), abs(r["t_f"] - hi)) > 0.05,
                 f"{r['t_f']:.3f} s, bounds [{lo:.1f}, {hi:.1f}]")

    # Robustness to the guess. The entry state is sized from the guess, so
    # different guesses are genuinely different problems -- what should hold is
    # that each finds an interior optimum rather than falling to a bound.
    #
    # Interiority is the claim under test here, and it is kept separate from
    # convergence on purpose. A 10 s guess does not fully converge, but that is
    # the aerodynamic deficit Day 7 already measured at this burn time and
    # entry pitch, not something free time introduced -- and it still places
    # t_f in the interior. Test 8 reports the convergence side.
    interior = True
    detail = []
    for g in (6.0, 10.0):
        r2 = solve(t_burn_guess=g)
        lo2, hi2 = r2["t_f_bounds"]
        margin = min(abs(r2["t_f"] - lo2), abs(r2["t_f"] - hi2))
        interior &= margin > 0.05
        detail.append(f"guess {g:.0f}s -> {r2['t_f']:.2f}s "
                      f"({r2.get('status')})")
    ok &= report("finds an interior optimum from other guesses", interior,
                 "; ".join(detail))

    # The coupling test. Force the bounds somewhere the solver would not
    # choose; if t_f reaches the dynamics at all, the fuel must change.
    pinned = solve(t_f_min=0.85 * r["t_f"], t_f_max=0.9 * r["t_f"])
    moved = abs(pinned["fuel"] - r["fuel"]) > 1.0
    ok &= report("t_f is coupled to the dynamics (fuel responds)",
                 moved and pinned.get("status") != "failed",
                 f"{r['fuel']:,.0f} kg at {r['t_f']:.2f} s -> "
                 f"{pinned['fuel']:,.0f} kg at {pinned['t_f']:.2f} s")
    return ok


def test_log_mass(r):
    print("\nTEST 5 - Log-mass consistency")
    veh = Vehicle6DoF()
    err = float(np.max(np.abs(veh.m_wet * np.exp(r["zm"]) - r["m"])))
    ok = report("m_wet exp(z_m) matches m", err < 1.0, f"max err {err:.2e} kg")
    ok &= report("z_m starts at zero", abs(r["zm"][0]) < 1e-9,
                 f"{r['zm'][0]:.2e}")
    ok &= report("mass decreases monotonically",
                 bool(np.all(np.diff(r["m"]) <= 1e-6)),
                 f"max increase {float(np.max(np.diff(r['m']))):.2e} kg")
    ok &= report("final mass above dry", r["m"][-1] >= veh.m_dry - 1e-6,
                 f"{r['m'][-1]:,.0f} kg vs dry {veh.m_dry:,.0f} kg")
    ok &= report("z_m stays inside its bounds",
                 bool(np.all(r["zm"] <= 1e-9)
                      and np.all(r["zm"] >= np.log(veh.m_dry / veh.m_wet) - 1e-9)))
    return ok


def test_fuel_vs_day7(r):
    print("\nTEST 6 - Fuel against the Day 7 solver")
    d7 = solve_scvx(aero=AeroConfig(), t_burn=T_GUESS,
                    theta0_deg=THETA0_DEG, verbose=False)
    if d7.get("status") != "converged":
        print("  Day 7 did not converge - checking the budget only.")
        return report("fuel inside the propellant load",
                      0 < r["fuel"] < Vehicle6DoF().m_prop_initial,
                      f"{r['fuel']:,.0f} kg")
    ok = report("fuel no worse than Day 7 (free time should help)",
                r["fuel"] <= d7["fuel"] * 1.01,
                f"Day 8 {r['fuel']:,.0f} kg vs Day 7 {d7['fuel']:,.0f} kg "
                f"({100 * (r['fuel'] - d7['fuel']) / d7['fuel']:+.2f}%)")
    ok &= report("fuel positive and inside the load",
                 0 < r["fuel"] < Vehicle6DoF().m_prop_initial,
                 f"{r['fuel']:,.0f} kg")
    ok &= report("linearisation no worse than Day 7's",
                 r["thrust_defect"] <= d7["thrust_defect"] * 1.5,
                 f"{r['thrust_defect']:.2e} vs {d7['thrust_defect']:.2e}")
    return ok


def test_trapz_accuracy():
    """
    Trapezoidal against Euler on the same problem, judged by the simulator.

    The burn duration is pinned to the Day 7 value so free time cannot
    contribute: the only difference left is the discretisation.
    """
    print("\nTEST 7 - Trapezoidal collocation vs Euler")
    veh = Vehicle6DoF()
    d7 = solve_scvx(aero=AeroConfig(), t_burn=T_GUESS,
                    theta0_deg=THETA0_DEG, verbose=False)
    d8 = solve(t_f_min=T_GUESS, t_f_max=T_GUESS)

    ok = report("pinned-time Day 8 solve converged",
                d8.get("status") == "converged")
    if d7.get("status") != "converged" or d8.get("status") != "converged":
        print("  One solver did not converge - skipping the comparison.")
        return False

    p7, a7, w7 = replay(d7, veh)
    p8, a8, w8 = replay(d8, veh)
    descent = d8["z"][0]
    print(f"         Euler (Day 7): {p7:7.3f} m  "
          f"({100 * p7 / descent:.3f}% of descent), pitch {a7:.3f} deg")
    print(f"         Trapz (Day 8): {p8:7.3f} m  "
          f"({100 * p8 / descent:.3f}% of descent), pitch {a8:.3f} deg")

    ok &= report("trapz replay error is smaller than Euler's", p8 < p7,
                 f"{p8:.3f} m vs {p7:.3f} m ({p7 / max(p8, 1e-9):.1f}x better)")
    ok &= report("trapz replay lands within 0.5% of the descent",
                 p8 < 0.005 * descent,
                 f"{p8:.3f} m of {descent:,.0f} m")
    ok &= report("trapz replay arrives upright", a8 < 5.0, f"{a8:.3f} deg")
    ok &= report("trapz replay residual rate small", w8 < 5.0,
                 f"{w8:.3f} deg/s")

    # Collocation residual on the solved trajectory, the doc's own check.
    dt = d8["t_f"] / d8["N"]
    res = max(
        float(np.max(np.abs(np.diff(d8["x"])
                            - 0.5 * dt * (d8["vx"][:-1] + d8["vx"][1:])))),
        float(np.max(np.abs(np.diff(d8["z"])
                            - 0.5 * dt * (d8["vz"][:-1] + d8["vz"][1:])))),
    )
    ok &= report("trapz position collocation residual under 1 m", res < 1.0,
                 f"{res:.2e} m")

    # And the check that separates a rocket from a drone.
    gim = np.degrees(np.abs(d8["delta"])).max()
    ok &= report("gimbal within its limit at every node",
                 gim <= veh.delta_max_deg + 1e-6,
                 f"peak {gim:.2f} of {veh.delta_max_deg:.0f} deg")
    return ok


def test_robustness():
    print("\nTEST 8 - Robustness to perturbed conditions")
    cases = [
        ("upright entry",    dict(theta0_deg=0.0)),
        ("shallow entry",    dict(theta0_deg=20.0)),
        ("short guess",      dict(t_burn_guess=6.0)),
        ("long guess",       dict(t_burn_guess=10.0)),
        ("loose glideslope", dict(gamma_gs_deg=60.0)),
        ("coarse grid",      dict(N=40)),
        ("fine grid",        dict(N=120)),
    ]
    ok = True
    converged = 0
    for name, kw in cases:
        r = solve(**kw)
        landed = (r.get("status") != "failed"
                  and abs(r["x"][-1]) < 5.0 and abs(r["z"][-1]) < 5.0
                  and np.hypot(r["vx"][-1], r["vz"][-1]) < 3.0)
        if r.get("status") == "converged":
            converged += 1
        ok &= report(f"reaches the pad: {name}", landed,
                     f"{r.get('status'):>11}, t_f={r.get('t_f', 0):.2f}s, "
                     f"{r['fuel']:,.0f} kg")
    ok &= report("most cases close completely",
                 converged >= len(cases) - 2,
                 f"{converged} of {len(cases)} converged")
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 8 - COMPLETE SCvx SOLVER VERIFICATION")
    print("=" * 70)

    ok1, r = test_convergence()
    ok2 = test_terminal(r)
    ok3 = test_virtual_control(r)
    ok4 = test_free_time(r)
    ok5 = test_log_mass(r)
    ok6 = test_fuel_vs_day7(r)
    ok7 = test_trapz_accuracy()
    ok8 = test_robustness()

    all_ok = all([ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8])
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
