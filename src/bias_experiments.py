"""
Day 12 exploration: what augmenting the state is worth, and where it runs out.

A  bias magnitude    how the blind/aware gap grows with the bias
B  attitude rate     how fast the bias is resolved
D  misspecified walk a filter told the bias is steadier than it is

C in the guide extends the state again with an accelerometer bias. It is the
same pattern one dimension larger and is left undone rather than half-done.

Run:  python src/bias_experiments.py
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
SEEDS = (0, 1, 2, 3)


def _med(bias_deg, aware, **kw):
    runs = [run_navigation(mode="ekf", wind_seed=s, sensor_seed=s + 60,
                           bias0_deg_s=bias_deg, bias_aware=aware,
                           verbose=False, **kw)
            for s in SEEDS]
    ok = [r for r in runs if r.get("status") == "flown"]
    if not ok:
        return None
    return {
        "miss": float(np.median([r["miss"] for r in ok])),
        "speed": float(np.median([r["speed"] for r in ok])),
        "est": float(np.median([r["mean_est_pos_err"] for r in ok])),
        "b_err": float(np.median([np.degrees(r["b_err_final"]) for r in ok])),
        "n": len(ok),
    }


def _row(tag, r):
    if r is None:
        return print(f"  {tag:>22}   nothing completed")
    print(f"  {tag:>22} {r['miss']:>8.2f} {r['speed']:>8.2f} "
          f"{r['est']:>8.2f} {r['b_err']:>9.3f}")


def _head(title):
    print(f"\n{title}")
    print(RULE)
    print(f"  {'case':>22} {'miss':>8} {'arrival':>8} {'est err':>8} "
          f"{'bias err':>9}")


def experiment_a():
    _head("EXPERIMENT A - Bias magnitude")
    out = []
    for b in (0.0, 0.5, 1.0, 2.0, 4.0):
        blind = _med(b, False)
        aware = _med(b, True)
        _row(f"{b:g} deg/s, blind", blind)
        _row(f"{b:g} deg/s, aware", aware)
        out.append((b, blind, aware))
    print("\n  The gap is what an uncalibrated gyro costs a filter with no")
    print("  state for it. The descent is about five seconds, which is most")
    print("  of what limits how much of that gap augmentation can recover.")
    return out


def experiment_b():
    _head("EXPERIMENT B - Attitude sensor rate")
    out = []
    for hz in (5.0, 10.0, 20.0, 50.0):
        r = _med(1.5, True, sensors=SensorConfig(att_rate_hz=hz))
        _row(f"{hz:g} Hz, aware", r)
        out.append((hz, r))
    print("\n  More readings separate the bias from a genuine rotation sooner.")
    return out


def experiment_d():
    _head("EXPERIMENT D - Misspecified bias walk")
    out = []
    for w in (0.0002, 0.002, 0.02, 0.2):
        r = _med(1.5, True, filter_bias_walk_deg_s=w)
        _row(f"walk {w:g} deg/s", r)
        out.append((w, r))
    print("\n  The true walk is 0.01. Told the bias is far steadier than it")
    print("  is, the filter stops adapting and its estimate lags; told it is")
    print("  far noisier, it chases the rate channel instead of tracking.")
    return out


def plot(a, b, d, save_path=None):
    save_path = save_path or os.path.join(RESULTS, "day12_bias.png")
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("Day 12: gyro-bias estimation "
                 f"(medians over {len(SEEDS)} realisations)", fontsize=13)

    p = ax[0]
    bs = [x[0] for x in a]
    p.plot(bs, [x[1]["miss"] if x[1] else np.nan for x in a], "o-", lw=2,
           color="tab:red", label="bias-blind")
    p.plot(bs, [x[2]["miss"] if x[2] else np.nan for x in a], "o-", lw=2,
           color="tab:blue", label="bias-aware")
    p.set_xlabel("true gyro bias [deg/s]"); p.set_ylabel("median miss [m]")
    p.set_title("A: bias magnitude"); p.legend(fontsize=8); p.grid(alpha=0.3)

    p = ax[1]
    hz = [x[0] for x in b if x[1]]
    p.plot(hz, [x[1]["b_err"] for x in b if x[1]], "o-", lw=2,
           color="tab:blue")
    p.set_xlabel("attitude sensor rate [Hz]")
    p.set_ylabel("final bias error [deg/s]")
    p.set_title("B: how fast the bias is resolved"); p.grid(alpha=0.3)

    p = ax[2]
    ws = [x[0] for x in d if x[1]]
    p.semilogx(ws, [x[1]["b_err"] for x in d if x[1]], "o-", lw=2,
               color="tab:blue")
    p.axvline(0.01, color="k", ls="--", alpha=0.7, label="true walk")
    p.set_xlabel("filter's assumed walk [deg/s per sqrt(s)]")
    p.set_ylabel("final bias error [deg/s]")
    p.set_title("D: misspecified Q"); p.legend(fontsize=8); p.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 78)
    print("DAY 12 - BIAS EXPLORATION")
    print("=" * 78)
    a = experiment_a()
    b = experiment_b()
    d = experiment_d()
    plot(a, b, d)
    print()
