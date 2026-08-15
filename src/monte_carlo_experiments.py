"""
Day 9 exploration: four sweeps over the dispersion analysis.

A  tightened dispersions -- what does the comfortable envelope look like
B  wind sensitivity -- how much of the miss is the air
C  entry altitude -- where the operating band sits
D  node count -- how much fidelity the statistics actually need

Every number here is a *flown* number: the plan is built by the solver from the
dispersed entry state and then flown open-loop through the true vehicle, so the
miss distances include model error the planner never saw. Comparing them against
the solver's own terminal error, which never leaves the 1e-9 m level, is the
whole point.

Run:  python src/monte_carlo_experiments.py
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

from src.monte_carlo import (                                    # noqa: E402
    DispersionConfig, run_monte_carlo,
)

warnings.filterwarnings("ignore")

RULE = "-" * 78
# 22 configurations across the four experiments, so the per-config sample count
# sets the whole runtime: 40 samples is ~880 solves and the better part of an
# hour. 20 is enough to separate the trends these sweeps are testing, and the
# per-config CEP is correspondingly noisier -- worth remembering before reading
# a 0.3 m difference between two rows as real.
N_PER = 12          # samples per configuration
N_NODES = 50        # nodes per trajectory


def _row(tag, s):
    if s.get("n_solved", 0) == 0:
        return f"  {tag:>16} {'no solutions':>12}"
    return (f"  {tag:>16} {s['solve_rate']:>7.1f}% {s['land_rate']:>7.1f}% "
            f"{s.get('miss_cep', float('nan')):>8.2f} "
            f"{s.get('miss_p95', float('nan')):>8.2f} "
            f"{s.get('speed_mean', float('nan')):>8.2f} "
            f"{s.get('fuel_mean', float('nan')):>9,.0f}")


HEAD = (f"  {'case':>16} {'solved':>8} {'landed':>8} {'CEP m':>8} "
        f"{'p95 m':>8} {'v m/s':>8} {'fuel kg':>9}")


# ======================================================================
def experiment_a():
    """Tightening the dispersions: how much of the miss is navigation."""
    print("\nEXPERIMENT A - Dispersion width")
    print(RULE)
    print(HEAD)
    out = []
    for label, k in (("3-sigma (full)", 1.0), ("two thirds", 2 / 3),
                     ("half", 0.5), ("quarter", 0.25), ("none", 0.0)):
        d = DispersionConfig(
            x0_3sigma=25.0 * k, z0_3sigma=60.0 * k, vx0_3sigma=5.0 * k,
            vz0_3sigma=18.0 * k, theta0_3sigma_deg=6.0 * k,
            omega0_3sigma=0.05 * k,
        )
        mc = run_monte_carlo(n_runs=N_PER, seed=11, disp=d, N=N_NODES,
                             verbose=False)
        print(_row(label, mc["stats"]))
        out.append((k, mc["stats"]))
    print("\n  Entry dispersion is scaled here; the model errors the planner")
    print("  never sees -- mass, Isp, drag, wind -- are held at full strength,")
    print("  so whatever miss survives at 'none' is theirs alone.")
    return out


# ======================================================================
def experiment_b():
    """Wind sensitivity: the disturbance the planner is never told about."""
    print("\nEXPERIMENT B - Wind")
    print(RULE)
    print(HEAD)
    out = []
    for w in (0.0, 5.0, 10.0, 15.0, 22.0, 30.0):
        d = DispersionConfig(wind_x_3sigma=w, wind_z_3sigma=w / 3.0)
        mc = run_monte_carlo(n_runs=N_PER, seed=23, disp=d, N=N_NODES,
                             verbose=False)
        print(_row(f"wind 3s = {w:.0f}", mc["stats"]))
        out.append((w, mc["stats"]))
    return out


# ======================================================================
def experiment_c():
    """Entry altitude: where the operating band actually is."""
    print("\nEXPERIMENT C - Entry altitude")
    print(RULE)
    print(HEAD)
    out = []
    for z in (300.0, 360.0, 420.0, 480.0, 540.0, 600.0):
        d = DispersionConfig(z0_nominal=z, z0_3sigma=30.0)
        mc = run_monte_carlo(n_runs=N_PER, seed=31, disp=d, N=N_NODES,
                             verbose=False)
        print(_row(f"z0 = {z:.0f} m", mc["stats"]))
        out.append((z, mc["stats"]))
    print("\n  Entry speed is held at its nominal -130 m/s throughout, so this")
    print("  is a vertical slice through the wedge Day 9 measured: a slow, high")
    print("  approach over-brakes before it arrives, because the throttle floor")
    print("  will not let the engines ease off.")
    return out


# ======================================================================
def experiment_d():
    """Node count: how much trajectory fidelity the statistics need."""
    print("\nEXPERIMENT D - Node count")
    print(RULE)
    print(HEAD + f" {'s/run':>7}")
    out = []
    for n in (30, 40, 50, 60, 80):
        mc = run_monte_carlo(n_runs=N_PER, seed=43, N=n, verbose=False)
        s = mc["stats"]
        per = s["total_time"] / max(s["n_runs"], 1)
        print(_row(f"N = {n}", s) + f" {per:>7.2f}")
        out.append((n, s, per))
    return out


# ======================================================================
def plot_experiments(a, b, c, d, save_path=None):
    save_path = save_path or os.path.join(RESULTS, "day9_sweeps.png")
    fig, ax = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle("Day 9: dispersion sweeps (all misses are flown, not planned)",
                 fontsize=13)

    def series(rows, key, idx=1):
        return [r[idx].get(key, float("nan")) for r in rows]

    p = ax[0]
    ks = [r[0] for r in a]
    p.plot(ks, series(a, "miss_cep"), "o-", lw=2, label="CEP")
    p.plot(ks, series(a, "miss_p95"), "s--", lw=2, label="p95")
    p.set_xlabel("entry dispersion, fraction of 3-sigma")
    p.set_ylabel("miss [m]")
    p.set_title("A: how much is navigation")
    p.legend(fontsize=8); p.grid(alpha=0.3)

    p = ax[1]
    ws = [r[0] for r in b]
    p.plot(ws, series(b, "miss_cep"), "o-", lw=2, color="tab:blue", label="CEP")
    p.plot(ws, series(b, "miss_p95"), "s--", lw=2, color="tab:orange",
           label="p95")
    p.set_xlabel("cross-wind 3-sigma [m/s]"); p.set_ylabel("miss [m]")
    p.set_title("B: wind"); p.legend(fontsize=8); p.grid(alpha=0.3)

    p = ax[2]
    zs = [r[0] for r in c]
    p.plot(zs, series(c, "solve_rate"), "o-", lw=2, color="tab:green",
           label="solved %")
    p.plot(zs, series(c, "land_rate"), "s--", lw=2, color="tab:red",
           label="landed %")
    p.set_xlabel("entry altitude [m]"); p.set_ylabel("percent")
    p.set_title("C: altitude band"); p.legend(fontsize=8); p.grid(alpha=0.3)

    p = ax[3]
    ns = [r[0] for r in d]
    p.plot(ns, [r[1].get("miss_cep", np.nan) for r in d], "o-", lw=2,
           color="tab:purple", label="CEP")
    p.set_xlabel("nodes N"); p.set_ylabel("CEP [m]")
    p.set_title("D: node count"); p.grid(alpha=0.3)
    p2 = p.twinx()
    p2.plot(ns, [r[2] for r in d], "s--", lw=2, color="tab:gray")
    p2.set_ylabel("seconds per run", color="tab:gray")
    p.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSweep plot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 78)
    print("DAY 9 - DISPERSION EXPLORATION")
    print("=" * 78)
    a = experiment_a()
    b = experiment_b()
    c = experiment_c()
    d = experiment_d()
    plot_experiments(a, b, c, d)
    print()
