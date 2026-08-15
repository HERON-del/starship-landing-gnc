"""
Verification of warm starting and closed-loop guidance.

Tests:
    1. The shifted reference is well formed
    2. Warm against cold, judged honestly
    3. The guidance loop runs to touchdown
    4. Replans fit inside the guidance cycle
    5. The loop tracks its own plan
    6. Closed against open loop over many seeds
    7. Replanning does not run the tank dry

Test 2 is where it is easiest to fool yourself. The guide measures the
warm-start speedup by capping warm solves at four iterations and comparing
against an uncapped cold solve, which guarantees a speedup of at least the cap
whether or not warm starting does anything at all. Run to the same tolerance,
the effect here is absent. What warm starting genuinely buys is command
accuracy inside a fixed budget, and that is what this asserts.

Test 6 is the one that matters, for the reason Day 9 established: a single
seed is an anecdote. Both strategies fly identical gust sequences, paired.

Run:  python tests/test_closed_loop.py
"""

import os
import sys
import warnings

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.warm_start import shift_reference, MIN_HORIZON_S      # noqa: E402
from src.closed_loop import (                                  # noqa: E402
    run_closed_loop, run_open_loop, _plan, _as_state,
)
from src.scvx_complete import solve_scvx_complete              # noqa: E402
from src.scvx_params import SCvxParams                         # noqa: E402
from src.dynamics_6dof import Vehicle6DoF                      # noqa: E402
from src.aero import AeroConfig                                # noqa: E402
from tests.test_dynamics import PASS, FAIL                     # noqa: E402

warnings.filterwarnings("ignore")

N_NODES = 40
Z0, VZ0, THETA0 = 420.0, -130.0, 25.0


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name:<52}"
    if detail:
        msg += f" {detail}"
    print(msg)
    return bool(ok)


def _base_plan(veh, aero):
    truth = np.array([0.0, Z0, 0.0, VZ0, np.radians(THETA0), 0.0, veh.m_wet])
    return _plan(veh, aero, N_NODES, 2.0 * Z0 / abs(VZ0), _as_state(truth), 30)


# ======================================================================
def test_shift_reference():
    print("\nTEST 1 - Shifted reference is well formed")
    veh, aero = Vehicle6DoF(), AeroConfig()
    plan = _base_plan(veh, aero)
    ok = report("base plan converged", plan.get("status") == "converged")
    if not ok:
        return False, None, None

    elapsed = 1.0
    t = plan["t"]
    state = {k: float(np.interp(elapsed, t, plan[k]))
             for k in ("x", "z", "vx", "vz", "theta", "omega", "m")}
    state["x"] += 4.0                       # a deliberate 4 m of drift
    ref, remaining, gap = shift_reference(plan, elapsed, state, N_NODES, veh)

    ok &= report("reference returned", ref is not None)
    if ref is None:
        return False, None, None
    ok &= report("N+1 state nodes", len(ref["x"]) == N_NODES + 1)
    ok &= report("N control intervals", len(ref["sigma"]) == N_NODES)
    ok &= report("node 0 is the measurement, not the prediction",
                 abs(ref["x"][0] - state["x"]) < 1e-9)
    ok &= report("horizon shortened by the time flown",
                 abs(remaining - (plan["t_f"] - elapsed)) < 1e-9,
                 f"{remaining:.3f}s of {plan['t_f']:.3f}s")
    ok &= report("gap reports the drift it was given",
                 abs(gap - 4.0) < 1e-6, f"{gap:.3f} m")
    ok &= report("time-scale factor reset to 1", ref["kt"] == 1.0)
    ok &= report("log-mass is finite and non-positive",
                 np.all(np.isfinite(ref["zm"])) and np.all(ref["zm"] <= 1e-9))
    ok &= report("declines when the horizon is gone",
                 shift_reference(plan, plan["t_f"] - 0.5 * MIN_HORIZON_S,
                                 state, N_NODES, veh)[0] is None)
    return ok, plan, veh


# ======================================================================
def test_warm_vs_cold(plan, veh):
    """
    Both halves, judged the way each should be.

    To convergence, warm starting does not help this solver -- its iteration
    count is set by the trust-region schedule, not by where the reference
    starts. Inside a fixed budget it helps a great deal, which is the property
    a guidance loop actually needs.
    """
    print("\nTEST 2 - Warm start, measured honestly")
    aero = AeroConfig()
    t = plan["t"]
    elapsed = 1.5
    state = {k: float(np.interp(elapsed, t, plan[k]))
             for k in ("x", "z", "vx", "vz", "theta", "omega", "m")}
    state["x"] += 3.0
    state["vx"] += 2.0
    ref, remaining, _ = shift_reference(plan, elapsed, state, N_NODES, veh)

    common = dict(vehicle=veh, aero=aero, N=N_NODES, t_burn_guess=remaining,
                  x0=state["x"], z0=state["z"], vx0=state["vx"],
                  vz0=state["vz"],
                  theta0_deg=np.degrees(state["theta"]),
                  omega0=state["omega"], m0=state["m"], verbose=False)

    truth = solve_scvx_complete(params=SCvxParams(max_iter=40, min_iter=1),
                                **common)
    ok = report("a converged answer exists to compare against",
                truth.get("status") == "converged")
    if not ok:
        return False

    # Same tolerance, both starts. No cap on either.
    warm_full = solve_scvx_complete(params=SCvxParams(max_iter=40, min_iter=1),
                                    initial_ref=ref, **common)
    cold_full = solve_scvx_complete(params=SCvxParams(max_iter=40, min_iter=1),
                                    **common)
    ok &= report("both reach the same answer to convergence",
                 abs(warm_full["t_f"] - cold_full["t_f"]) < 0.5,
                 f"warm {warm_full['iterations']} iters vs cold "
                 f"{cold_full['iterations']} -- no speedup, and none claimed")

    # Fixed budget: the guidance case.
    def cmd_error(r):
        return abs(np.degrees(r["delta"][0] - truth["delta"][0]))

    warm3 = solve_scvx_complete(params=SCvxParams(max_iter=3, min_iter=1),
                                initial_ref=ref, **common)
    cold3 = solve_scvx_complete(params=SCvxParams(max_iter=3, min_iter=1),
                                **common)
    ew, ec = cmd_error(warm3), cmd_error(cold3)
    ok &= report("in a 3-iteration budget, warm steers closer", ew < ec,
                 f"warm {ew:.2f} deg vs cold {ec:.2f} deg off the "
                 f"converged gimbal")
    ok &= report("...and is close enough to fly", ew < 2.0,
                 f"{ew:.2f} deg")
    ok &= report("commanded thrust is the same either way",
                 abs(warm3["sigma"][0] - cold3["sigma"][0]) < 1e3,
                 f"{abs(warm3['sigma'][0] - cold3['sigma'][0]) / 1e3:.1f} kN "
                 f"apart")
    return ok


# ======================================================================
def test_loop_runs():
    print("\nTEST 3 - The guidance loop runs to touchdown")
    r = run_closed_loop(wind_seed=7, verbose=False)
    ok = report("loop completed", r.get("status") == "flown")
    if not ok:
        return False, None
    ok &= report("it replanned repeatedly", r["n_replans"] >= 5,
                 f"{r['n_replans']} cycles")
    ok &= report("no replan was abandoned", r["n_failed_replans"] == 0,
                 f"{r['n_failed_replans']} failed")
    ok &= report("it reached the ground", r["truth"]["z"][-1] < 25.0,
                 f"last logged z = {r['truth']['z'][-1]:.1f} m")
    ok &= report("propellant left at touchdown", r["margin"] > 0,
                 f"{r['margin']:,.0f} kg")
    return ok, r


# ======================================================================
def test_replan_fits_the_cycle(r):
    print("\nTEST 4 - Replans fit inside the guidance cycle")
    dt = r["guidance_dt"]
    ok = report("mean replan faster than the cycle",
                r["mean_solve_time"] < dt,
                f"{r['mean_solve_time']:.3f}s of {dt:.2f}s")
    ok &= report("worst replan faster than the cycle",
                 r["max_solve_time"] < dt,
                 f"{r['max_solve_time']:.3f}s")
    ok &= report("the budget was actually used",
                 max(r["replan"]["iterations"]) <= r["budget"],
                 f"budget {r['budget']}")
    return ok


# ======================================================================
def test_tracking(r):
    """The loop should stay close to the plan it is flying."""
    print("\nTEST 5 - The loop tracks its own plan")
    gaps = np.array([g for g in r["replan"]["gap"] if np.isfinite(g)])
    ok = report("tracking gap stays small", np.nanmax(gaps) < 5.0,
                f"worst {np.nanmax(gaps):.2f} m, median "
                f"{np.median(gaps):.2f} m")
    ok &= report("the gap does not run away",
                 gaps[-1] < max(5.0, 3.0 * np.median(gaps)),
                 f"last {gaps[-1]:.2f} m")
    return ok


# ======================================================================
def test_against_open_loop():
    """
    Paired over seeds, both flying identical gusts.

    Day 9 is the reason this is not a single run: one seed is an anecdote, and
    the interesting quantity here has a wide spread.
    """
    print("\nTEST 6 - Closed against open loop, paired over seeds")
    seeds = range(8)
    cls, ols = [], []
    for s in seeds:
        cls.append(run_closed_loop(wind_seed=s, verbose=False))
        ols.append(run_open_loop(wind_seed=s, verbose=False))
    pairs = [(c, o) for c, o in zip(cls, ols)
             if c.get("status") == "flown" and o.get("status") == "flown"]
    ok = report("both strategies flew every seed", len(pairs) == len(cls),
                f"{len(pairs)}/{len(cls)}")
    if not pairs:
        return False

    nearer = sum(1 for c, o in pairs if c["miss"] < o["miss"])
    cl_miss = np.median([c["miss"] for c, _ in pairs])
    ol_miss = np.median([o["miss"] for _, o in pairs])
    ok &= report("closed loop lands nearer more often than not",
                 nearer > len(pairs) / 2,
                 f"{nearer}/{len(pairs)} seeds; median miss "
                 f"{cl_miss:.2f} m vs {ol_miss:.2f} m")
    ok &= report("closed-loop miss is inside tolerance", cl_miss < 5.0,
                 f"{cl_miss:.2f} m")
    return ok


# ======================================================================
def test_fuel_bounded():
    print("\nTEST 7 - Replanning does not run the tank dry")
    seeds = range(6)
    diffs, margins = [], []
    for s in seeds:
        c = run_closed_loop(wind_seed=s, verbose=False)
        o = run_open_loop(wind_seed=s, verbose=False)
        if c.get("status") == "flown" and o.get("status") == "flown":
            diffs.append(c["fuel"] - o["fuel"])
            margins.append(c["margin"])
    ok = report("comparisons available", bool(diffs))
    if not diffs:
        return False
    ok &= report("closed-loop propellant within 2 t of open-loop",
                 abs(np.median(diffs)) < 2000.0,
                 f"median {np.median(diffs):+,.0f} kg")
    ok &= report("every run kept propellant in hand", min(margins) > 0,
                 f"worst margin {min(margins):,.0f} kg")
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 10 - CLOSED-LOOP GUIDANCE VERIFICATION")
    print("=" * 70)

    ok1, plan, veh = test_shift_reference()
    ok2 = test_warm_vs_cold(plan, veh) if plan is not None else False
    ok3, r = test_loop_runs()
    ok4 = test_replan_fits_the_cycle(r) if r else False
    ok5 = test_tracking(r) if r else False
    ok6 = test_against_open_loop()
    ok7 = test_fuel_bounded()

    all_ok = all([ok1, ok2, ok3, ok4, ok5, ok6, ok7])
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
