"""
Day 7 exploration: four sweeps over the SCvx solver.

A  SCvx against the Day 5/6 ad-hoc loop across a set of initial conditions
B  trust-region radius        eta_0
C  virtual-control penalty    w_vc
D  node count                 N

Kept in the repository rather than run and discarded, because three of the four
answers contradict what the guide predicts and the numbers are the argument.

Run:  python src/scvx_experiments.py
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

from src.scvx import solve_scvx                          # noqa: E402
from src.scvx_params import SCvxParams                   # noqa: E402
from src.aero import AeroConfig                          # noqa: E402
from src.landing_flip import solve_flip_landing          # noqa: E402

warnings.filterwarnings("ignore")

T_BURN = 8.0
THETA0 = 30.0
RULE = "-" * 78


def _scvx(**kw):
    kw.setdefault("aero", AeroConfig())
    kw.setdefault("t_burn", T_BURN)
    kw.setdefault("theta0_deg", THETA0)
    kw.setdefault("verbose", False)
    return solve_scvx(**kw)


# ======================================================================
def experiment_a():
    """
    SCvx against the Day 5/6 loop on identical problems.

    Both solve the same physics with the same discretisation; the only
    difference is how they iterate. What is being compared is not really fuel,
    it is whether the answer can be trusted -- so the defect column matters
    more than the kilograms.
    """
    print("\nEXPERIMENT A - SCvx vs. the Day 5/6 ad-hoc loop")
    print(RULE)
    cases = [
        ("nominal",          dict()),
        ("upright entry",    dict(theta0_deg=0.0)),
        ("shallow entry",    dict(theta0_deg=20.0)),
        ("steep entry",      dict(theta0_deg=40.0)),
        ("short burn",       dict(t_burn=6.0)),
        ("long burn",        dict(t_burn=10.0)),
    ]
    print(f"  {'case':<16} {'SCvx':>22}   {'Day 5/6':>22}")
    print(f"  {'':<16} {'fuel':>8} {'defect':>7} {'it':>5}   "
          f"{'fuel':>8} {'defect':>7} {'it':>5}   {'saving':>7}")
    rows = []
    for name, kw in cases:
        r = _scvx(**kw)
        d5 = solve_flip_landing(
            aero=AeroConfig(), N=kw.get("N", 80),
            t_burn=kw.get("t_burn", T_BURN),
            theta0_deg=kw.get("theta0_deg", THETA0), verbose=False)
        ok5 = str(d5.get("status", "")).startswith("optimal")
        f5 = d5["fuel"] if ok5 else float("nan")
        d5d = d5.get("final_defect", float("nan")) if ok5 else float("nan")
        saving = (100 * (f5 - r["fuel"]) / f5) if ok5 else float("nan")
        print(f"  {name:<16} {r['fuel']:>8,.0f} {r['thrust_defect']:>7.1e} "
              f"{r['iterations']:>5}   "
              f"{f5:>8,.0f} {d5d:>7.1e} {d5.get('iterations', 0):>5}   "
              f"{saving:>6.1f}%")
        rows.append((name, r, d5 if ok5 else None))

    n_better = sum(1 for _, r, d in rows
                   if d is not None and r["thrust_defect"] < d["final_defect"])
    print(f"\n  SCvx linearisation is tighter in {n_better} of "
          f"{sum(1 for _, _, d in rows if d is not None)} comparable cases.")
    print("  The ad-hoc loop stops when its objective stops moving, which can")
    print("  happen while the linear model is still 6e-2 away from the true")
    print("  dynamics. SCvx stops on a measured quantity instead.")
    return rows


# ======================================================================
def experiment_b():
    """
    Trust-region radius.

    The guide predicts small radii converge slowly but reliably, large radii
    quickly but with a risk of oscillation, and a sweet spot in between.
    """
    print("\nEXPERIMENT B - Trust-region radius eta_0")
    print(RULE)
    etas = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    print(f"  {'eta_0':>7} {'iters':>6} {'rejected':>9} {'fuel':>8} "
          f"{'|nu|':>10} {'thrust_d':>9} {'time':>7} {'status':>12}")
    out = []
    for e in etas:
        r = _scvx(params=SCvxParams(eta_0=e))
        rej = sum(1 for a in r["history"]["accepted"] if not a)
        print(f"  {e:>7} {r['iterations']:>6} {rej:>9} {r['fuel']:>8,.0f} "
              f"{r['vc_norm']:>10.1e} {r['thrust_defect']:>9.1e} "
              f"{r['elapsed']:>6.1f}s {r['status']:>12}")
        out.append((e, r))
    return out


# ======================================================================
def experiment_c():
    """
    Virtual-control penalty.

    The guide predicts the solver "cheats" at low w_vc -- buying slack instead
    of finding a real trajectory -- and that early iterations go infeasible at
    high w_vc, with 1e5 the sweet spot.

    Half right, and the half that is right is worth seeing. The weight has to
    be swept with adaptive growth *disabled*, or the sweep measures nothing:
    the growth rule lifts any starting weight to the level the problem needs,
    which is exactly what it is for.
    """
    print("\nEXPERIMENT C - Virtual-control penalty w_vc")
    print(RULE)
    print("  With adaptive growth on (the default), the starting weight is")
    print("  almost irrelevant -- the rule finds the level it needs:")
    print(f"  {'w_vc_0':>8} {'iters':>6} {'fuel':>8} {'|nu|':>10} "
          f"{'thrust_d':>9} {'time':>7} {'status':>12}")
    out = []
    for w in (1e0, 1e1, 1e2, 1e3, 1e4, 1e5):
        r = _scvx(params=SCvxParams(w_vc=w))
        print(f"  {w:>8.0e} {r['iterations']:>6} {r['fuel']:>8,.0f} "
              f"{r['vc_norm']:>10.1e} {r['thrust_defect']:>9.1e} "
              f"{r['elapsed']:>6.1f}s {r['status']:>12}")
        out.append((w, r))

    print("\n  Pinning the weight (w_vc_grow = 1) isolates it, and the guide's")
    print("  cheating prediction shows up sharply at the bottom of the range:")
    print(f"  {'w_vc':>8} {'fuel':>8} {'|nu|':>12} {'status':>12}   verdict")
    fixed = []
    for w in (1e0, 1e1, 1e2, 1e3, 1e4, 1e5):
        r = _scvx(params=SCvxParams(w_vc=w, w_vc_grow=1.0))
        cheat = r["vc_norm"] > 1e-3
        print(f"  {w:>8.0e} {r['fuel']:>8,.0f} {r['vc_norm']:>12.4e} "
              f"{r['status']:>12}   "
              f"{'buys a fake trajectory' if cheat else 'honest'}")
        fixed.append((w, r))

    print("\n  ...and on a problem with no solution (100 m off-axis entry),")
    print("  the deficit is the same number to five figures over four decades")
    print("  of weight -- a price cannot buy down a shortfall that is real:")
    print(f"  {'w_vc':>8} {'|nu|':>12} {'fuel':>8}")
    for w in (1e0, 1e1, 1e2, 1e3, 1e4, 1e5):
        r = _scvx(x0=100.0, vx0=-15.0,
                  params=SCvxParams(w_vc=w, w_vc_grow=1.0, max_iter=40))
        print(f"  {w:>8.0e} {r['vc_norm']:>12.4e} {r['fuel']:>8,.0f}")

    print("\n  So the weight has a floor, not a sweet spot: below it the")
    print("  optimiser lies, above it nothing changes. The guide's upper")
    print("  failure mode is real but sits far higher than it suggests -- at")
    print("  1e7 the penalty wrecks the conditioning and both CLARABEL and")
    print("  SCS return `unbounded` on a problem bounded below by -1.")
    return out, fixed


# ======================================================================
def experiment_d():
    """
    Node count.

    More nodes means a finer discretisation and a smaller Euler error, at the
    cost of a larger subproblem. The guide expects diminishing returns around
    N = 80 to 100.
    """
    print("\nEXPERIMENT D - Node count N")
    print(RULE)
    print(f"  {'N':>5} {'iters':>6} {'fuel':>8} {'d(fuel)':>8} {'|nu|':>10} "
          f"{'thrust_d':>9} {'time':>7} {'status':>12}")
    out = []
    prev = None
    for n in (20, 40, 60, 80, 100, 120, 160):
        r = _scvx(N=n)
        d = f"{r['fuel'] - prev:>8,.0f}" if prev is not None else f"{'-':>8}"
        print(f"  {n:>5} {r['iterations']:>6} {r['fuel']:>8,.0f} {d} "
              f"{r['vc_norm']:>10.1e} {r['thrust_defect']:>9.1e} "
              f"{r['elapsed']:>6.1f}s {r['status']:>12}")
        prev = r["fuel"]
        out.append((n, r))
    return out


# ======================================================================
def plot_sweeps(b, c, d, save_path=None):
    """Three panels: what each knob actually controls."""
    save_path = save_path or os.path.join(RESULTS, "day7_scvx_sweeps.png")
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("Day 7: SCvx parameter sweeps", fontsize=13)

    a = ax[0]
    etas = [e for e, _ in b]
    a.semilogx(etas, [r["iterations"] for _, r in b], "o-", lw=2,
               color="tab:blue", label="iterations")
    a.set_xlabel("$\\eta_0$"); a.set_ylabel("iterations to stop")
    a2 = a.twinx()
    a2.semilogx(etas, [max(r["thrust_defect"], 1e-18) for _, r in b], "s--",
                lw=2, color="tab:red")
    a2.set_yscale("log")
    a2.set_ylabel("thrust defect", color="tab:red")
    a.set_title("B: trust-region radius"); a.grid(alpha=0.3)
    a.legend(fontsize=8, loc="upper left")

    # Panel C shows the *pinned* sweep. With adaptive growth on there is
    # nothing to see, which is the point of the growth rule but makes a dull
    # plot; pinning the weight exposes the floor below which the optimiser
    # buys itself a trajectory that does not obey the dynamics.
    a = ax[1]
    ws = [w for w, _ in c]
    a.loglog(ws, [max(r["vc_norm"], 1e-18) for _, r in c], "o-", lw=2,
             color="tab:red", label="$\\|\\nu\\|_1$")
    a.axhline(1e-6, color="green", ls=":", alpha=0.7, label="tolerance")
    a.set_xlabel("$w_{vc}$ (pinned, no growth)")
    a.set_ylabel("residual slack")
    a.set_title("C: virtual-control penalty")
    a.legend(fontsize=8, loc="lower left"); a.grid(alpha=0.3)
    a4 = a.twinx()
    a4.semilogx(ws, [r["fuel"] for _, r in c], "s--", lw=2, color="tab:blue")
    a4.set_ylabel("reported propellant [kg]", color="tab:blue")
    a4.annotate("solver buys a\nfake trajectory", xy=(ws[0], c[0][1]["fuel"]),
                xytext=(0.18, 0.42), textcoords="axes fraction", fontsize=8,
                color="tab:blue",
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1))

    a = ax[2]
    ns = [n for n, _ in d]
    a.plot(ns, [r["fuel"] for _, r in d], "o-", lw=2, color="tab:purple")
    a.set_xlabel("nodes N"); a.set_ylabel("Propellant [kg]")
    a.set_title("D: node count"); a.grid(alpha=0.3)
    a3 = a.twinx()
    a3.plot(ns, [r["elapsed"] for _, r in d], "s--", lw=2, color="tab:gray")
    a3.set_ylabel("solve time [s]", color="tab:gray")

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSweep plot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 78)
    print("DAY 7 - SCvx EXPLORATION")
    print("=" * 78)
    experiment_a()
    b = experiment_b()
    _, c_fixed = experiment_c()
    d = experiment_d()
    plot_sweeps(b, c_fixed, d)
    print()
