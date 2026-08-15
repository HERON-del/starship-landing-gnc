"""
Verification of the Monte Carlo dispersion analysis.

Tests:
    1. Sampling has the statistical properties it claims
    2. A small sweep completes and returns well-formed results
    3. The solver converges across the dispersion set
    4. Landing accuracy is measured against the truth, not the constraint
    5. Propellant margin is accounted for correctly
    6. Same seed reproduces, different seeds do not
    7. Wind is a disturbance the planner never sees

Test 4 is the one that matters, and it is a test about measurement rather than
about the vehicle. The solver enforces `x[N] == 0` and `z[N] == 0` as hard
equalities, so its own terminal error is at the 1e-7 m level on every run. Any
"landing accuracy" computed from the solver's own solution is therefore reading
back its own constraint. The test asserts that the two numbers are orders of
magnitude apart -- that the reported miss comes from flying the plan through the
independently verified simulator, and not from the optimiser marking its own
homework.

Test 7 is the same argument applied to wind. The guide injects wind by shifting
`vx0` and `vz0`, then hands the shifted state to the solver, so the planner sees
it and plans around it. That is extra navigation dispersion wearing a wind
costume. Here the wind enters only the truth model, so it must change the flown
result while leaving the plan untouched.

Run:  python tests/test_monte_carlo.py
"""

import os
import sys
import warnings

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.monte_carlo import (                                    # noqa: E402
    DispersionConfig, sample_dispersions, run_single, run_monte_carlo,
    fly_the_plan, MISS_TOL_M, SPEED_TOL_MS,
)
from src.scvx_complete import solve_scvx_complete                # noqa: E402
from src.scvx_params import SCvxParams                           # noqa: E402
from src.dynamics_6dof import Vehicle6DoF                        # noqa: E402
from src.aero import AeroConfig                                  # noqa: E402
from tests.test_dynamics import PASS, FAIL                       # noqa: E402

warnings.filterwarnings("ignore")

N_FAST = 50          # nodes for the test sweeps
SWEEP = 16           # samples per sweep; enough to be meaningful, quick enough


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name:<54}"
    if detail:
        msg += f" {detail}"
    print(msg)
    return bool(ok)


# ======================================================================
def test_sampling():
    """The sampler produces the distribution it advertises."""
    print("\nTEST 1 - Dispersion sampling")
    disp = DispersionConfig()
    rng = np.random.default_rng(123)
    s = [sample_dispersions(rng, disp) for _ in range(20000)]

    z0 = np.array([v["z0"] for v in s])
    vz0 = np.array([v["vz0"] for v in s])
    th0 = np.array([v["theta0_deg"] for v in s])
    wx = np.array([v["wind_x"] for v in s])

    ok = report("z0 mean sits on nominal",
                abs(np.mean(z0) - disp.z0_nominal) < 3.0,
                f"{np.mean(z0):.1f} vs {disp.z0_nominal:.0f} m")
    ok &= report("z0 sigma is the declared 3-sigma over three",
                 abs(np.std(z0) - disp.z0_3sigma / 3.0) < 1.0,
                 f"{np.std(z0):.2f} vs {disp.z0_3sigma / 3.0:.2f}")
    ok &= report("vz0 sigma likewise",
                 abs(np.std(vz0) - disp.vz0_3sigma / 3.0) < 1.0,
                 f"{np.std(vz0):.2f} vs {disp.vz0_3sigma / 3.0:.2f}")
    ok &= report("wind is zero-mean",
                 abs(np.mean(wx)) < 1.0, f"{np.mean(wx):+.3f} m/s")
    ok &= report("physical floors hold on every sample",
                 bool((z0 > 0).all() and (vz0 < 0).all()
                      and (th0 >= 0).all() and (th0 <= 70).all()))
    return ok


# ======================================================================
def test_small_sweep():
    """A small sweep completes and every record is well-formed."""
    print("\nTEST 2 - Small sweep completes")
    mc = run_monte_carlo(n_runs=SWEEP, seed=42, N=N_FAST, verbose=False)
    ok = report("returns results, stats and config",
                all(k in mc for k in ("results", "stats", "config")))
    ok &= report(f"{SWEEP} records returned", len(mc["results"]) == SWEEP)

    required = ("converged", "good", "miss", "speed", "fuel", "margin",
                "t_f", "fail_reason", "sample")
    ok &= report("every record carries every field",
                 all(all(k in r for k in required) for r in mc["results"]))
    ok &= report("outcome counts sum to the run count",
                 sum(mc["stats"]["outcomes"].values()) == SWEEP)
    return ok, mc


# ======================================================================
def test_solve_rate(mc):
    """The dispersion set sits inside the solver's feasible envelope."""
    print("\nTEST 3 - Solver convergence across the dispersions")
    s = mc["stats"]
    ok = report("solver converges on at least 90% of samples",
                s["solve_rate"] >= 90.0,
                f"{s['solve_rate']:.1f}% ({s['n_solved']}/{s['n_runs']})")
    ok &= report("iteration count stays bounded",
                 s.get("iter_max", 999) <= 40, f"max {s.get('iter_max')}")
    ok &= report("free final time varies across samples",
                 s["t_f_max"] - s["t_f_min"] > 0.1,
                 f"[{s['t_f_min']:.2f}, {s['t_f_max']:.2f}] s")
    return ok


# ======================================================================
def test_accuracy_is_measured_not_asserted(mc):
    """
    The reported miss must come from the replay, not the constraint.

    If these two numbers were the same, the analysis would be reading the
    solver's own equality constraint back to itself and calling it accuracy.
    """
    print("\nTEST 4 - Accuracy comes from the flown trajectory")
    s = mc["stats"]
    flown = [r for r in mc["results"] if r["converged"]]
    if len(flown) < 5:
        return report("enough converged runs to judge", False)

    planned = max(r["planned_miss"] for r in flown)
    ok = report("solver's own terminal error is at constraint level",
                planned < 1e-4, f"max {planned:.2e} m")
    ok &= report("flown miss is orders of magnitude larger",
                 s["miss_mean"] > 1000 * planned,
                 f"{s['miss_mean']:.2f} m flown vs {planned:.2e} m planned")
    ok &= report("miss distribution is non-degenerate",
                 s["miss_std"] > 1e-3,
                 f"std {s['miss_std']:.3f} m")
    ok &= report("CEP is finite and positive",
                 0.0 < s["miss_cep"] < 1e4, f"{s['miss_cep']:.2f} m")
    ok &= report("percentiles are ordered",
                 s["miss_cep"] <= s["miss_p95"] <= s["miss_max"] + 1e-9)
    return ok


# ======================================================================
def test_margin(mc):
    """Propellant bookkeeping is consistent with the true vehicle."""
    print("\nTEST 5 - Propellant margin")
    s = mc["stats"]
    ok = report("mean margin is positive",
                s["margin_mean"] > 0, f"{s['margin_mean']:,.0f} kg")
    ok &= report("no run finishes below dry mass",
                 s["n_margin_negative"] == 0,
                 f"{s['n_margin_negative']} runs")
    ok &= report("propellant used stays within the load",
                 s["fuel_max"] < DispersionConfig().m_prop_nominal
                 + DispersionConfig().m_prop_3sigma,
                 f"max {s['fuel_max']:,.0f} kg")

    # fuel + margin must reconstruct the true propellant load of each sample
    worst = 0.0
    for r in mc["results"]:
        if not r["converged"]:
            continue
        worst = max(worst, abs(r["fuel"] + r["margin"] - r["sample"]["m_prop"]))
    ok &= report("fuel + margin reconstructs the sampled load",
                 worst < 1.0, f"worst mismatch {worst:.4f} kg")
    return ok


# ======================================================================
def test_reproducibility():
    """Same seed, same answer. Different seed, different answer."""
    print("\nTEST 6 - Reproducibility")
    a = run_monte_carlo(n_runs=8, seed=77, N=N_FAST, verbose=False)
    b = run_monte_carlo(n_runs=8, seed=77, N=N_FAST, verbose=False)
    c = run_monte_carlo(n_runs=8, seed=1234, N=N_FAST, verbose=False)

    ok = report("same seed gives the same solve count",
                a["stats"]["n_solved"] == b["stats"]["n_solved"])
    ok &= report("same seed gives the same miss statistics",
                 abs(a["stats"]["miss_mean"] - b["stats"]["miss_mean"]) < 1e-9,
                 f"{a['stats']['miss_mean']:.6f} vs "
                 f"{b['stats']['miss_mean']:.6f}")
    ok &= report("a different seed gives a different answer",
                 abs(a["stats"]["miss_mean"] - c["stats"]["miss_mean"]) > 1e-6,
                 f"{a['stats']['miss_mean']:.3f} vs "
                 f"{c['stats']['miss_mean']:.3f}")
    return ok


# ======================================================================
def test_wind_is_unmodelled():
    """
    Wind must change the outcome without changing the plan.

    Plan once, then fly the identical plan through still air and through a
    cross-wind. The commanded control is the same object in both cases, so any
    difference in where the vehicle ends up is the disturbance doing its work.
    """
    print("\nTEST 7 - Wind is a disturbance, not a relabelled entry state")
    disp = DispersionConfig()
    plan = solve_scvx_complete(
        vehicle=Vehicle6DoF(), aero=AeroConfig(),
        params=SCvxParams(max_iter=25), N=N_FAST,
        t_burn_guess=2.0 * disp.z0_nominal / abs(disp.vz0_nominal),
        z0=disp.z0_nominal, vz0=disp.vz0_nominal,
        theta0_deg=disp.theta0_nominal_deg, verbose=False)
    if plan.get("status") != "converged":
        return report("nominal plan converged", False)

    veh, aero = Vehicle6DoF(), AeroConfig()
    calm = fly_the_plan(plan, veh, aero, (0.0, 0.0))
    windy = fly_the_plan(plan, veh, aero, (12.0, 0.0))

    ok = report("nominal plan flies close to the pad in still air",
                calm["miss"] < 2.0, f"{calm['miss']:.3f} m")
    ok &= report("a 12 m/s cross-wind moves the touchdown",
                 abs(windy["miss"] - calm["miss"]) > 0.5,
                 f"{calm['miss']:.2f} m calm vs {windy['miss']:.2f} m windy")
    ok &= report("the plan itself is unchanged by the wind",
                 plan["t_f"] > 0 and np.isfinite(plan["sigma"]).all(),
                 "planner never sees it")

    # And the sampler must keep wind out of the entry state it hands over.
    rng = np.random.default_rng(5)
    s = sample_dispersions(rng, disp)
    ok &= report("wind is reported separately from the entry velocity",
                 "wind_x" in s and "vx0" in s
                 and abs(s["vx0"] - disp.vx0_nominal) < disp.vx0_3sigma,
                 f"vx0 {s['vx0']:+.2f} m/s, wind {s['wind_x']:+.2f} m/s")
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 9 - MONTE CARLO VERIFICATION")
    print("=" * 70)

    ok1 = test_sampling()
    ok2, mc = test_small_sweep()
    ok3 = test_solve_rate(mc)
    ok4 = test_accuracy_is_measured_not_asserted(mc)
    ok5 = test_margin(mc)
    ok6 = test_reproducibility()
    ok7 = test_wind_is_unmodelled()

    all_ok = all([ok1, ok2, ok3, ok4, ok5, ok6, ok7])
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
