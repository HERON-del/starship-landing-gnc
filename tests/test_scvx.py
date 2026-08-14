"""
Verification suite for the SCvx solver.

Tests:
    1. SCvx converges on the nominal problem
    2. Terminal constraints are satisfied
    3. Virtual control does its job -- both halves of it
    4. The trust region adapts rather than drifting
    5. Fuel and accuracy against the Day 5/6 optimiser on the same problem
    6. Perturbed initial conditions are handled or honestly reported
    7. The trajectory is one an actual rocket could fly

Test 3 is where the day's idea is checked, and it has two halves that are easy
to conflate. On a feasible problem the virtual control must vanish -- otherwise
the reported trajectory does not satisfy its own dynamics. On an *infeasible*
one it must instead settle to a positive constant that measures the deficit,
without the solver crashing and without it pretending to have landed. The
second half is the property the Day 5/6 loop did not have: it returned
`infeasible` and left you to guess by how much and in which row.

Test 7 is the one that matters most, for the same reason test 5 mattered on Day
5. Tests 1-4 ask the optimiser whether it satisfied its own model. Test 7 asks
whether that model was a rocket: whether the thrust vector it commands is
reachable with a 15 degree gimbal, and whether replaying the commanded throttle
and gimbal through the independently verified nonlinear simulator actually flies
the trajectory. The Day 7 guide's formulation passes tests 1-4 and fails test 7
at every node.

Run:  python tests/test_scvx.py
"""

import os
import sys
import warnings

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.scvx import solve_scvx                                  # noqa: E402
from src.scvx_params import SCvxParams                           # noqa: E402
from src.dynamics_6dof import Vehicle6DoF                        # noqa: E402
from src.dynamics_aero import dynamics_full                      # noqa: E402
from src.aero import AeroConfig                                  # noqa: E402
from src.integrators import propagate                            # noqa: E402
from src.landing_flip import solve_flip_landing                  # noqa: E402
from tests.test_dynamics import PASS, FAIL                       # noqa: E402

warnings.filterwarnings("ignore")

# The nominal case is the regime Day 6 concluded the vehicle actually flies:
# coast on the belly with the engines off, ignite near-upright, burn briefly.
T_BURN = 8.0
THETA0_DEG = 30.0

# An entry the vehicle cannot recover from: 100 m off-axis with 15 m/s of
# lateral velocity and eight seconds to null both, while the attitude is
# already committed to the flip. Used to check that infeasibility is *reported*
# rather than crashed on.
INFEASIBLE_CASE = dict(x0=100.0, vx0=-15.0)


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name:<52}"
    if detail:
        msg += f" {detail}"
    print(msg)
    return bool(ok)


def solve(**kw):
    kw.setdefault("aero", AeroConfig())
    kw.setdefault("t_burn", T_BURN)
    kw.setdefault("theta0_deg", THETA0_DEG)
    kw.setdefault("verbose", False)
    return solve_scvx(**kw)


# ======================================================================
def test_convergence():
    """SCvx converges on the nominal problem, and does it quickly."""
    print("\nTEST 1 - Convergence on the nominal problem")
    r = solve()
    ok = report("solver reports converged", r.get("status") == "converged")
    ok &= report("converges in 25 iterations or fewer",
                 r.get("iterations", 99) <= 25,
                 f"{r.get('iterations', '?')} iterations")
    ok &= report("solves in under 60 s", r.get("elapsed", 999) < 60.0,
                 f"{r.get('elapsed', 0):.1f}s")
    ok &= report("trust region and step history recorded",
                 len(r["history"]["eta"]) == r["iterations"])
    return ok, r


# ======================================================================
def test_terminal_constraints(r):
    """The vehicle arrives on the pad, at rest, upright and not rotating."""
    print("\nTEST 2 - Terminal constraints")
    ok = report("x_f on the pad", abs(r["x"][-1]) < 5.0,
                f"{r['x'][-1]:.3f} m")
    ok &= report("z_f on the pad", abs(r["z"][-1]) < 5.0,
                 f"{r['z'][-1]:.3f} m")
    ok &= report("vx_f at rest", abs(r["vx"][-1]) < 3.0,
                 f"{r['vx'][-1]:.3f} m/s")
    ok &= report("vz_f at rest", abs(r["vz"][-1]) < 3.0,
                 f"{r['vz'][-1]:.3f} m/s")
    ok &= report("theta_f upright", abs(np.degrees(r["theta"][-1])) < 2.0,
                 f"{np.degrees(r['theta'][-1]):.3f} deg")
    ok &= report("omega_f not rotating",
                 abs(np.degrees(r["omega"][-1])) < 5.0,
                 f"{np.degrees(r['omega'][-1]):.3f} deg/s")
    return ok


# ======================================================================
def test_virtual_control(r):
    """
    Both halves of what virtual control is for.

    On a feasible problem the slack must vanish, or the trajectory does not
    obey the dynamics it claims to. On an infeasible one it must converge to a
    constant that measures the shortfall -- and that constant must be a
    property of the *problem*, not of the trust region, which is what
    distinguishes a genuine deficit from a linearisation artefact.
    """
    print("\nTEST 3 - Virtual control")
    ok = report("slack vanishes on the feasible problem",
                r["vc_norm"] < 1e-6, f"|nu| = {r['vc_norm']:.2e}")

    # The true nonlinear residual must fall too. ||nu|| going to zero only says
    # the linear model satisfied itself; this says it was describing reality.
    dfc = [d for d in r["history"]["defect"] if not np.isnan(d)]
    ok &= report("true nonlinear defect decreases over iterations",
                 len(dfc) >= 3 and dfc[-1] < 0.1 * dfc[0],
                 f"{dfc[0]:.2e} -> {dfc[-1]:.2e}")

    # --- the infeasible case -----------------------------------------
    bad = solve(**INFEASIBLE_CASE, params=SCvxParams(max_iter=40))
    ok &= report("infeasible problem still returns a trajectory",
                 bad.get("status") != "failed" and "x" in bad)
    ok &= report("...and is not reported as converged",
                 bad.get("status") == "unconverged")
    ok &= report("...with slack that is positive, not zero",
                 bad["vc_norm"] > 1e-3, f"|nu| = {bad['vc_norm']:.2e}")

    tail = [v for v in bad["history"]["vc_norm"][-6:] if not np.isnan(v)]
    spread = (max(tail) - min(tail)) / max(tail) if tail else 1.0
    ok &= report("...that has settled to a constant",
                 spread < 0.01, f"spread {100 * spread:.3f}% over last "
                                f"{len(tail)} iterations")

    # A linearisation artefact scales with the step the linearisation had to
    # cover, so tightening the trust region an order of magnitude would drive
    # it down an order of magnitude. A genuine deficit barely moves. Note the
    # bar is "does not trend to zero", not "is identical" -- the least
    # infeasible point is not unique, so the settled value wanders a little.
    sweep = [(e, solve(**INFEASIBLE_CASE,
                       params=SCvxParams(max_iter=40, eta_0=e))["vc_norm"])
             for e in (0.5, 0.1, 0.05)]
    vals = [v for _, v in sweep]
    ratio = max(vals) / min(vals)
    ok &= report("...and does not shrink as the trust region tightens",
                 ratio < 1.5,
                 "  ".join(f"eta0={e}: {v:.3e}" for e, v in sweep)
                 + f"  (spread {ratio:.2f}x for a 10x tighter region)")

    # The deficit is a real infeasibility, so the no-slack Day 5 optimiser
    # should not be able to solve the same problem.
    d5 = solve_flip_landing(aero=AeroConfig(), N=80, t_burn=T_BURN,
                            theta0_deg=THETA0_DEG, x0=100.0, vx0=-15.0,
                            verbose=False)
    ok &= report("...and the no-slack optimiser cannot solve it",
                 not str(d5.get("status", "")).startswith("optimal"),
                 f"Day 5 status: {d5.get('status')}")
    return ok


# ======================================================================
def test_trust_region(r):
    """The radius adapts to measured model agreement rather than drifting."""
    print("\nTEST 4 - Trust-region behaviour")
    h = r["history"]
    eta = np.array(h["eta"], dtype=float)
    rho = np.array([x for x in h["rho"] if not np.isnan(x)], dtype=float)

    ok = report("radius recorded every iteration", len(eta) >= 3)
    ok &= report("radius changed during the solve", eta.max() != eta.min(),
                 f"[{eta.min():.4f}, {eta.max():.4f}]")
    ok &= report("radius ends smaller than it started", eta[-1] < eta[0],
                 f"{eta[0]:.4f} -> {eta[-1]:.4f}")
    ok &= report("step quality computed every iteration", len(rho) == len(eta))
    ok &= report("some steps were rejected", not all(h["accepted"]),
                 f"{sum(1 for a in h['accepted'] if not a)} of "
                 f"{len(h['accepted'])} rejected")
    ok &= report("every rejected step shrank the radius",
                 all(eta[i + 1] <= eta[i] + 1e-12
                     for i in range(len(eta) - 1)
                     if not h["accepted"][i] and h["rho"][i] == h["rho"][i]))
    return ok


# ======================================================================
def test_against_day6(r):
    """
    The same problem, solved by the Day 5/6 optimiser.

    Fuel should land in the same place -- it is the same physics. The
    interesting comparison is accuracy: SCvx bounds its linearisation error by
    construction, where the ad-hoc loop only hoped to.
    """
    print("\nTEST 5 - Against the Day 5/6 optimiser on the same problem")
    d5 = solve_flip_landing(aero=AeroConfig(), N=80, t_burn=T_BURN,
                            theta0_deg=THETA0_DEG, verbose=False)
    if not str(d5.get("status", "")).startswith("optimal"):
        print("  Day 5 optimiser did not solve - comparing against budget only.")
        return report("SCvx fuel is within the propellant load",
                      0 < r["fuel"] < Vehicle6DoF().m_prop_initial,
                      f"{r['fuel']:,.0f} kg")

    ratio = r["fuel"] / d5["fuel"]
    ok = report("fuel within 30% of the Day 5/6 answer", 0.7 < ratio < 1.3,
                f"SCvx {r['fuel']:,.0f} kg vs Day 5 {d5['fuel']:,.0f} kg "
                f"(ratio {ratio:.3f})")
    ok &= report("fuel is positive and inside the load",
                 0 < r["fuel"] < Vehicle6DoF().m_prop_initial,
                 f"{r['fuel']:,.0f} / {Vehicle6DoF().m_prop_initial:,.0f} kg")
    ok &= report("linearisation is at least as accurate as Day 5's",
                 r["thrust_defect"] <= d5["final_defect"],
                 f"{r['thrust_defect']:.2e} vs {d5['final_defect']:.2e}")
    return ok


# ======================================================================
def test_robustness():
    """
    Perturbed initial conditions.

    The bar is deliberately not "everything converges". Some of these problems
    have no solution -- entry pitch is what decides that once drag is active,
    because `Cd A` belly-on is 28x its upright value. What is required is that
    the solver never crashes, always returns a trajectory that reaches the pad,
    and says plainly which case it could not close.
    """
    print("\nTEST 6 - Robustness to perturbed initial conditions")
    cases = [
        ("upright entry",     dict(theta0_deg=0.0)),
        ("shallow entry",     dict(theta0_deg=20.0)),
        ("steep entry",       dict(theta0_deg=40.0)),
        ("short burn",        dict(t_burn=6.0)),
        ("long burn",         dict(t_burn=10.0)),
        ("loose glideslope",  dict(gamma_gs_deg=60.0)),
        ("coarse grid",       dict(N=40)),
        ("fine grid",         dict(N=120)),
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
                     f"{r.get('status'):>11}, |nu| = {r['vc_norm']:.1e}, "
                     f"{r['fuel']:,.0f} kg")
    ok &= report("at least half the cases close completely",
                 converged >= len(cases) // 2,
                 f"{converged} of {len(cases)} converged")
    return ok


# ======================================================================
def test_flyable(r):
    """
    Is this a trajectory an actual rocket could fly?

    Three checks the Day 7 guide's formulation fails. Its control is a free
    thrust vector plus an independent torque, so the thrust it commands sits a
    mean of 43 degrees and a maximum of 115 degrees off the body axis at every
    one of 80 nodes, against a 15 degree gimbal. It still lands exactly on the
    pad at exactly zero speed, which is precisely why this test exists.
    """
    print("\nTEST 7 - The trajectory is one a rocket could fly")
    veh = Vehicle6DoF()
    N = r["N"]

    gimbal = np.degrees(np.abs(r["delta"]))
    ok = report("gimbal within its limit at every node",
                gimbal.max() <= veh.delta_max_deg + 1e-6,
                f"peak {gimbal.max():.2f} of {veh.delta_max_deg:.0f} deg")

    # The engine is bolted on: the thrust vector must lie within delta_max of
    # the body axis. This is the check that separates a rocket from a drone.
    off = np.degrees(np.abs(np.arctan2(
        np.sin(np.arctan2(r["Tx"], r["Tz"]) - r["theta"][:N]),
        np.cos(np.arctan2(r["Tx"], r["Tz"]) - r["theta"][:N]))))
    ok &= report("thrust vector reachable from the body axis",
                 off.max() <= veh.delta_max_deg + 1e-6,
                 f"peak {off.max():.2f} deg, "
                 f"{int((off > veh.delta_max_deg + 1e-6).sum())}/{N} nodes over")

    ok &= report("throttle inside its band at every node",
                 r["sigma"].min() >= veh.T_min - 1.0
                 and r["sigma"].max() <= veh.T_max + 1.0,
                 f"[{r['sigma'].min() / 1e6:.2f}, "
                 f"{r['sigma'].max() / 1e6:.2f}] MN")
    ok &= report("pitch rate inside its limit",
                 np.abs(r["omega"]).max() <= veh.omega_max + 1e-6,
                 f"peak {np.degrees(np.abs(r['omega']).max()):.1f} deg/s")

    # --- replay through the independently verified simulator ----------
    sigma, delta = r["sigma"], r["delta"]
    dt_ctrl = r["t_burn"] / len(sigma)

    def control(t, state, vehicle):
        k = min(int(t / dt_ctrl), len(sigma) - 1)
        return sigma[k], delta[k]

    y0 = np.array([r["x"][0], r["z"][0], r["vx"][0], r["vz"][0],
                   r["theta"][0], r["omega"][0], veh.m_wet])
    _, y = propagate(
        lambda t, yy, *a: dynamics_full(t, yy, control, veh, AeroConfig()),
        y0, (0.0, r["t_burn"]), r["t_burn"] / 4000, method="rk4")

    pos_err = float(np.hypot(y[-1, 0], y[-1, 1]))
    att_err = float(abs(np.degrees(y[-1, 4])))
    rate_err = float(abs(np.degrees(y[-1, 5])))
    print(f"         replay: position {pos_err:7.2f} m   "
          f"pitch {att_err:6.2f} deg   rate {rate_err:6.2f} deg/s")

    # The subproblem still discretises with forward Euler. Day 4 measured that
    # as a 1.5% miss on the 3-DoF problem; rotation and drag compound it. The
    # bound is set where Euler actually lands. Trapezoidal collocation inside
    # the SCvx loop is Day 8's job, and this number is how it will be judged.
    ok &= report("replay lands within 5% of the descent (Euler-limited)",
                 pos_err < 0.05 * r["z"][0],
                 f"{pos_err:.2f} m of {r['z'][0]:,.0f} m "
                 f"({100 * pos_err / r['z'][0]:.2f}%)")
    ok &= report("replay arrives within 5 deg of upright", att_err < 5.0,
                 f"{att_err:.2f} deg")
    ok &= report("replay residual rate under 5 deg/s", rate_err < 5.0,
                 f"{rate_err:.2f} deg/s")
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 7 - SCvx SOLVER VERIFICATION")
    print("=" * 70)

    ok1, r = test_convergence()
    ok2 = test_terminal_constraints(r)
    ok3 = test_virtual_control(r)
    ok4 = test_trust_region(r)
    ok5 = test_against_day6(r)
    ok6 = test_robustness()
    ok7 = test_flyable(r)

    all_ok = all([ok1, ok2, ok3, ok4, ok5, ok6, ok7])
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
