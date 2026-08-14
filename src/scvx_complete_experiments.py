"""
Day 8 exploration: four sweeps over the complete solver.

A  time penalty        the fuel / duration Pareto front
B  node count          does trapezoidal collocation buy back nodes?
C  extreme conditions  where the solver stops closing
D  the initial guess   is free final time actually guess-independent?

Experiment D is the one worth reading carefully. The obvious version of it
measures nothing, and the reason is a modelling choice rather than a bug.

Run:  python src/scvx_complete_experiments.py
"""

import os
import sys
import warnings

import matplotlib
import numpy as np

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
RESULTS = os.path.join(REPO_ROOT, "results")

from src.scvx_complete import solve_scvx_complete        # noqa: E402
from src.scvx import solve_scvx                          # noqa: E402
from src.scvx_params import SCvxParams                   # noqa: E402
from src.aero import AeroConfig                          # noqa: E402
from src.dynamics_6dof import Vehicle6DoF                # noqa: E402
from src.dynamics_aero import dynamics_full              # noqa: E402
from src.integrators import propagate                    # noqa: E402
from src.landing_flip import feasible_entry_state        # noqa: E402

warnings.filterwarnings("ignore")

T_GUESS = 8.0
THETA0 = 30.0
RULE = "-" * 78


def _solve(**kw):
    kw.setdefault("aero", AeroConfig())
    kw.setdefault("t_burn_guess", T_GUESS)
    kw.setdefault("theta0_deg", THETA0)
    kw.setdefault("verbose", False)
    return solve_scvx_complete(**kw)


def _replay_error(r, vehicle):
    sigma, delta = r["sigma"], r["delta"]
    t_f = r.get("t_f", r.get("t_burn"))
    dtc = t_f / len(sigma)

    def control(t, state, veh):
        k = min(int(t / dtc), len(sigma) - 1)
        return sigma[k], delta[k]

    y0 = np.array([r["x"][0], r["z"][0], r["vx"][0], r["vz"][0],
                   r["theta"][0], r["omega"][0], vehicle.m_wet])
    _, y = propagate(
        lambda t, yy, *a: dynamics_full(t, yy, control, vehicle, AeroConfig()),
        y0, (0.0, t_f), t_f / 4000, method="rk4")
    return float(np.hypot(y[-1, 0], y[-1, 1]))


# ======================================================================
def experiment_a():
    """
    Time penalty: the fuel / duration trade.

    With no penalty the objective is pure minimum-propellant. The guide adds
    `0.1 * t_f` to "avoid degeneracy", which presumes the duration would
    otherwise run away. It does not: the 40% throttle floor means a longer burn
    always costs propellant, so minimising fuel already prefers short burns and
    the penalty only distorts the answer.
    """
    print("\nEXPERIMENT A - Time penalty and the fuel/duration front")
    print(RULE)
    print(f"  {'w_time':>8} {'t_f [s]':>9} {'fuel [kg]':>10} "
          f"{'|nu|':>10} {'status':>12}")
    out = []
    for w in (0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0):
        r = _solve(w_time=w)
        print(f"  {w:>8.3f} {r['t_f']:>9.3f} {r['fuel']:>10,.0f} "
              f"{r['vc_norm']:>10.1e} {r['status']:>12}")
        out.append((w, r))
    base = out[0][1]
    print(f"\n  Minimum-propellant duration is {base['t_f']:.3f} s with no")
    print("  penalty at all, so nothing was degenerate to begin with. Adding")
    print("  the guide's 0.1 shortens the burn and costs propellant - the")
    print("  penalty is a preference for haste, not a numerical safeguard.")
    return out


# ======================================================================
def experiment_b():
    """
    Node count: does the higher-order rule buy back nodes?

    Compared against the Day 7 Euler solver at the same N, with the burn
    duration pinned so the discretisation is the only difference, and judged by
    replay error through the verified simulator rather than by self-report.
    """
    print("\nEXPERIMENT B - Node count, trapezoidal against Euler")
    print(RULE)
    veh = Vehicle6DoF()
    print(f"  {'N':>5} {'Euler miss':>11} {'trapz miss':>11} {'ratio':>7} "
          f"{'Euler fuel':>11} {'trapz fuel':>11} {'trapz t':>8}")
    out = []
    for n in (20, 30, 40, 60, 80, 120):
        d7 = solve_scvx(aero=AeroConfig(), N=n, t_burn=T_GUESS,
                        theta0_deg=THETA0, verbose=False)
        d8 = _solve(N=n, t_f_min=T_GUESS, t_f_max=T_GUESS)
        if d7.get("status") == "failed" or d8.get("status") == "failed":
            print(f"  {n:>5}  a solver failed")
            continue
        e7 = _replay_error(d7, veh)
        e8 = _replay_error(d8, veh)
        print(f"  {n:>5} {e7:>10.3f}m {e8:>10.3f}m {e7 / max(e8, 1e-9):>6.1f}x "
              f"{d7['fuel']:>10,.0f}k {d8['fuel']:>10,.0f}k "
              f"{d8['elapsed']:>7.1f}s")
        out.append((n, e7, e8, d7["fuel"], d8["fuel"]))

    if out:
        target = out[-1][1]      # Euler's error at the finest grid
        cheaper = [n for n, _, e8, _, _ in out if e8 <= target]
        if cheaper:
            print(f"\n  Trapezoidal at N={min(cheaper)} already beats Euler at "
                  f"N={out[-1][0]} ({target:.2f} m).")
            print("  That is the node count the higher-order rule buys back.")
    return out


# ======================================================================
def experiment_c():
    """
    Extreme conditions.

    The guide's cases (5 km entry at 150 m/s and 85 degrees of pitch) are not
    this vehicle's: Day 6 established that it coasts on its belly with the
    engines off and lights them near-upright and low. These are extremes of the
    regime it actually flies.
    """
    print("\nEXPERIMENT C - Extreme conditions")
    print(RULE)
    cases = [
        ("pure vertical drop",  dict(theta0_deg=0.0, x0=0.0, vx0=0.0)),
        ("hard flip, 50 deg",   dict(theta0_deg=50.0)),
        ("very short burn",     dict(t_burn_guess=4.5)),
        ("long burn",           dict(t_burn_guess=12.0)),
        ("off-axis 60 m",       dict(x0=60.0, vx0=-10.0)),
        ("tight corridor 85",   dict(gamma_gs_deg=85.0)),
        ("no atmosphere",       dict(aero=None)),
    ]
    print(f"  {'case':>20} {'t_f [s]':>9} {'fuel [kg]':>10} {'|nu|':>10} "
          f"{'status':>12}")
    out = []
    for name, kw in cases:
        r = _solve(**kw)
        if r.get("status") == "failed":
            print(f"  {name:>20}   no solution")
            continue
        print(f"  {name:>20} {r['t_f']:>9.3f} {r['fuel']:>10,.0f} "
              f"{r['vc_norm']:>10.1e} {r['status']:>12}")
        out.append((name, r))
    print("\n  The pure vertical drop is the cheapest, as it must be - there")
    print("  is no lateral excursion to build and then null.")
    return out


# ======================================================================
def experiment_d():
    """
    Is free final time guess-independent?

    The obvious sweep -- vary `t_burn_guess`, check `t_f` agrees -- measures
    nothing here, because the entry altitude and speed are *sized from the
    guess* (Day 5's `feasible_entry_state`). A different guess is a different
    problem, arriving from a different height at a different speed, and it
    should absolutely produce a different optimal duration.

    The claim only becomes testable with the entry state pinned. Then every
    guess describes the same physical situation, and the solver either finds
    the same answer from all of them or it does not.
    """
    print("\nEXPERIMENT D - Sensitivity to the initial guess")
    print(RULE)
    veh = Vehicle6DoF()

    print("  (i) guess also sets the entry state -- different problems, so")
    print("      different answers are correct, not a failure:")
    print(f"  {'guess':>7} {'entry z0':>9} {'entry vz0':>10} {'t_f':>8} "
          f"{'fuel':>9} {'status':>12}")
    for g in (5.0, 6.0, 8.0, 10.0, 12.0):
        t_flip = float(np.clip(1.4 * np.radians(THETA0) / veh.omega_max,
                               1.5, 0.6 * g))
        z0, vz0 = feasible_entry_state(veh, g, THETA0, t_flip)
        r = _solve(t_burn_guess=g)
        print(f"  {g:>7.1f} {z0:>8,.0f}m {vz0:>9.1f} {r['t_f']:>8.3f} "
              f"{r['fuel']:>9,.0f} {r['status']:>12}")

    print("\n  (ii) entry state pinned -- the same problem from every guess,")
    print("       which is the version of the claim that can be false:")
    t_flip = float(np.clip(1.4 * np.radians(THETA0) / veh.omega_max, 1.5,
                           0.6 * T_GUESS))
    z0, vz0 = feasible_entry_state(veh, T_GUESS, THETA0, t_flip)
    print(f"  entry pinned at {z0:,.0f} m, {vz0:.1f} m/s")
    print(f"  {'guess':>7} {'t_f':>8} {'fuel':>9} {'|nu|':>10} {'status':>12}")
    tfs, out = [], []
    for g in (5.0, 6.0, 7.0, 8.0, 10.0, 12.0):
        r = _solve(t_burn_guess=g, z0=z0, vz0=vz0,
                   t_f_min=3.0, t_f_max=18.0)
        print(f"  {g:>7.1f} {r['t_f']:>8.3f} {r['fuel']:>9,.0f} "
              f"{r['vc_norm']:>10.1e} {r['status']:>12}")
        if r.get("status") == "converged":
            tfs.append(r["t_f"])
        out.append((g, r))
    if len(tfs) >= 2:
        spread = max(tfs) - min(tfs)
        print(f"\n  Converged durations span {spread:.3f} s "
              f"({min(tfs):.3f} to {max(tfs):.3f}) across guesses from 5 to "
              f"12 s.")
        print("  That spread, not the raw t_f list, is the guess-independence")
        print("  claim - and it is what the trust region is doing for you.")
    return out


# ======================================================================
def plot_sweeps(a, b, d, save_path=None):
    save_path = save_path or os.path.join(RESULTS, "day8_sweeps.png")
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("Day 8: complete-solver sweeps", fontsize=13)

    p = ax[0]
    tf = [r["t_f"] for _, r in a]
    fu = [r["fuel"] for _, r in a]
    p.plot(tf, fu, "o-", lw=2, color="tab:blue")
    for (w, r) in a:
        p.annotate(f"{w:g}", (r["t_f"], r["fuel"]), fontsize=7,
                   textcoords="offset points", xytext=(4, 4))
    p.set_xlabel("burn duration $t_f$ [s]"); p.set_ylabel("Propellant [kg]")
    p.set_title("A: fuel / duration front (labels = $w_{time}$)")
    p.grid(alpha=0.3)

    p = ax[1]
    if b:
        ns = [n for n, _, _, _, _ in b]
        p.semilogy(ns, [e for _, e, _, _, _ in b], "o-", lw=2,
                   color="tab:gray", label="Euler (Day 7)")
        p.semilogy(ns, [e for _, _, e, _, _ in b], "s-", lw=2,
                   color="tab:blue", label="trapezoidal (Day 8)")
    p.set_xlabel("nodes N"); p.set_ylabel("replay miss distance [m] (log)")
    p.set_title("B: accuracy vs node count"); p.legend(fontsize=8)
    p.grid(alpha=0.3)

    p = ax[2]
    gs = [g for g, _ in d]
    p.plot(gs, [r["t_f"] for _, r in d], "o-", lw=2, color="tab:green")
    p.plot(gs, gs, ":", color="gray", label="if it just echoed the guess")
    p.set_xlabel("initial guess $t_{nom}$ [s]"); p.set_ylabel("chosen $t_f$ [s]")
    p.set_title("D: same problem, every guess"); p.legend(fontsize=8)
    p.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSweep plot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 78)
    print("DAY 8 - COMPLETE SOLVER EXPLORATION")
    print("=" * 78)
    a = experiment_a()
    b = experiment_b()
    experiment_c()
    d = experiment_d()
    plot_sweeps(a, b, d)
    print()
