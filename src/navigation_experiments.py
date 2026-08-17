"""
Day 11 exploration: what the filter is worth, and where it stops working.

A  process noise      Q too small ignores the wind, too large chases the sensor
B  sensor rate        the slowest nav update the guidance loop can live with
C  attitude only      dead reckoning, to show why position aiding is required
D  gyro bias          an error the filter cannot remove, because it does not
                      estimate it

Each figure is a median over several wind and sensor realisations, because a
single run of a stochastic system is an anecdote -- the lesson Day 9 paid for.

Run:  python src/navigation_experiments.py
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

from src.navigation_loop import run_navigation                 # noqa: E402
from src.sensors import SensorConfig                           # noqa: E402

warnings.filterwarnings("ignore")

RULE = "-" * 78
SEEDS = (0, 1, 2, 3, 4)


def _sweep(label, **kw):
    """Median over seeds of one configuration."""
    runs = []
    for s in SEEDS:
        r = run_navigation(wind_seed=s, sensor_seed=s + 40, verbose=False, **kw)
        if r.get("status") == "flown":
            runs.append(r)
    if not runs:
        print(f"  {label:>18}   nothing completed")
        return None
    out = {
        "est": float(np.median([r["mean_est_pos_err"] for r in runs])),
        "est_max": float(np.median([r["max_est_pos_err"] for r in runs])),
        "miss": float(np.median([r["miss"] for r in runs])),
        "speed": float(np.median([r["speed"] for r in runs])),
        "fuel": float(np.median([r["fuel"] for r in runs])),
        "n": len(runs),
    }
    print(f"  {label:>18} {out['est']:>9.2f} {out['est_max']:>9.2f} "
          f"{out['miss']:>8.2f} {out['speed']:>8.2f} {out['fuel']:>9,.0f}")
    return out


def _head(title):
    print(f"\n{title}")
    print(RULE)
    print(f"  {'case':>18} {'est mean':>9} {'est max':>9} {'miss':>8} "
          f"{'arrival':>8} {'fuel':>9}")


def experiment_a():
    _head("EXPERIMENT A - Process noise")
    out = []
    for q in (0.01, 0.1, 1.0, 10.0, 100.0):
        out.append((q, _sweep(f"Q x {q:g}", q_scale=q)))
    print("\n  Q is the filter's distrust of its own dynamics. Too small and "
          "it\n  ignores the gusts it cannot see; too large and it chases "
          "sensor noise.")
    return out


def experiment_b():
    _head("EXPERIMENT B - Nav sensor rate")
    out = []
    for hz in (1.0, 2.0, 5.0, 10.0, 20.0):
        out.append((hz, _sweep(f"nav {hz:g} Hz",
                               sensors=SensorConfig(nav_rate_hz=hz))))
    print("\n  The rate below which guidance is steering on stale position.")
    return out


def experiment_c():
    _head("EXPERIMENT C - Attitude only, no position aiding")
    out = [("full fusion", _sweep("full fusion")),
           ("attitude only",
            _sweep("attitude only",
                   sensors=SensorConfig(nav_enabled=False)))]
    print("\n  With the nav sensor gone the filter is dead-reckoning position "
          "from\n  its own dynamics model, and nothing bounds the drift. "
          "Attitude alone\n  cannot observe where the vehicle is.")
    return out


def experiment_d():
    _head("EXPERIMENT D - Gyro bias the filter does not estimate")
    out = []
    for b in (0.0, 0.5, 1.0, 2.0):
        out.append((b, _sweep(f"bias {b:g} deg/s",
                              sensors=SensorConfig(
                                  omega_bias=np.radians(b)))))
    print("\n  A constant offset on the rate channel is not noise, and "
          "averaging\n  cannot remove it. A filter with no bias state treats "
          "it as truth --\n  which is the textbook argument for augmenting "
          "the state vector.")
    return out


def plot_sweeps(a, b, c, d, save_path=None):
    save_path = save_path or os.path.join(RESULTS, "day11_sweeps.png")
    fig, ax = plt.subplots(1, 4, figsize=(23, 5))
    fig.suptitle("Day 11: estimator sweeps (medians over "
                 f"{len(SEEDS)} realisations)", fontsize=13)

    p = ax[0]
    qs = [q for q, r in a if r]
    p.semilogx(qs, [r["est"] for _, r in a if r], "o-", lw=2,
               color="tab:blue", label="mean")
    p.semilogx(qs, [r["est_max"] for _, r in a if r], "s--", lw=2,
               color="tab:cyan", label="worst")
    p.set_xlabel("Q scale"); p.set_ylabel("position estimation error [m]")
    p.set_title("A: process noise"); p.legend(fontsize=8); p.grid(alpha=0.3)

    p = ax[1]
    hz = [h for h, r in b if r]
    p.plot(hz, [r["est"] for _, r in b if r], "o-", lw=2, color="tab:blue",
           label="estimation error")
    p.set_xlabel("nav rate [Hz]"); p.set_ylabel("[m]")
    p.set_title("B: sensor rate"); p.grid(alpha=0.3)
    p2 = p.twinx()
    p2.plot(hz, [r["miss"] for _, r in b if r], "s--", lw=2, color="tab:red")
    p2.set_ylabel("miss [m]", color="tab:red")
    p.legend(fontsize=8, loc="upper right")

    p = ax[2]
    names = [n for n, r in c if r]
    p.bar(names, [r["est"] for _, r in c if r],
          color=["tab:blue", "tab:red"])
    for i, (_, r) in enumerate([x for x in c if x[1]]):
        p.text(i, r["est"], f"{r['est']:.1f}", ha="center", va="bottom",
               fontsize=9)
    p.set_ylabel("[m]"); p.set_title("C: position aiding")
    p.grid(alpha=0.3, axis="y")

    p = ax[3]
    bs = [x for x, r in d if r]
    p.plot(bs, [r["est"] for _, r in d if r], "o-", lw=2, color="tab:blue",
           label="estimation error")
    p.plot(bs, [r["miss"] for _, r in d if r], "s--", lw=2, color="tab:red",
           label="miss")
    p.set_xlabel("gyro bias [deg/s]"); p.set_ylabel("[m]")
    p.set_title("D: unestimated bias"); p.legend(fontsize=8); p.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSweep plot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 78)
    print("DAY 11 - ESTIMATOR EXPLORATION")
    print("=" * 78)
    a = experiment_a()
    b = experiment_b()
    c = experiment_c()
    d = experiment_d()
    plot_sweeps(a, b, c, d)
    print()
