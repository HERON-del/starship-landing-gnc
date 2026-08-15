"""
Day 10 exploration: what closing the loop is actually worth.

A  guidance rate          guidance_dt
B  wind severity          closed against open at each level
C  navigation noise       does the loop still help when steering on an estimate
D  many seeds             the statistically defensible version of the claim

D is the one that counts. A single seed says nothing -- Day 9 spent a whole
day establishing that -- so the headline comparison is a paired sweep over
wind seeds, closed and open flying the identical gust sequence each time.

Run:  python src/closed_loop_experiments.py
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

from src.closed_loop import (                                  # noqa: E402
    run_closed_loop, run_open_loop, MISS_TOL_M, SPEED_TOL_MS,
)

warnings.filterwarnings("ignore")

RULE = "-" * 78
SEEDS = tuple(range(12))


def _pair(seed, **kw):
    """Closed and open loop on one identical gust sequence."""
    cl = run_closed_loop(wind_seed=seed, verbose=False, **kw)
    ol = run_open_loop(wind_seed=seed, verbose=False, **kw)
    return cl, ol


def _agg(runs):
    ok = [r for r in runs if r.get("status") == "flown"]
    if not ok:
        return None
    miss = np.array([r["miss"] for r in ok])
    speed = np.array([r["speed"] for r in ok])
    fuel = np.array([r["fuel"] for r in ok])
    good = sum(1 for r in ok if r["good"])
    return {
        "n": len(ok), "good": good, "good_pct": 100.0 * good / len(ok),
        "miss_med": float(np.median(miss)), "miss_max": float(miss.max()),
        "speed_med": float(np.median(speed)), "speed_max": float(speed.max()),
        "fuel_med": float(np.median(fuel)),
    }


def _row(tag, a):
    if a is None:
        return f"  {tag:>16}   no runs completed"
    return (f"  {tag:>16} {a['good_pct']:>7.0f}% {a['miss_med']:>8.2f} "
            f"{a['miss_max']:>8.2f} {a['speed_med']:>8.2f} "
            f"{a['speed_max']:>8.2f} {a['fuel_med']:>9,.0f}")


def _head(title):
    print(f"\n{title}")
    print(RULE)
    print(f"  {'case':>16} {'landed':>8} {'miss med':>8} {'miss max':>8} "
          f"{'v med':>8} {'v max':>8} {'fuel med':>9}")


# ======================================================================
def experiment_d():
    """Paired over wind seeds -- the claim that can survive scrutiny."""
    _head("EXPERIMENT D - Closed against open over 12 wind seeds")
    cls, ols = [], []
    for s in SEEDS:
        cl, ol = _pair(s)
        cls.append(cl)
        ols.append(ol)
    a_cl, a_ol = _agg(cls), _agg(ols)
    print(_row("open loop", a_ol))
    print(_row("closed loop", a_cl))

    pairs = [(c, o) for c, o in zip(cls, ols)
             if c.get("status") == "flown" and o.get("status") == "flown"]
    if pairs:
        m_better = sum(1 for c, o in pairs if c["miss"] < o["miss"])
        s_better = sum(1 for c, o in pairs if c["speed"] < o["speed"])
        dfuel = np.median([c["fuel"] - o["fuel"] for c, o in pairs])
        print(f"\n  Paired on identical gusts, {len(pairs)} seeds:")
        print(f"    closed loop lands nearer in {m_better}/{len(pairs)}")
        print(f"    closed loop arrives slower in {s_better}/{len(pairs)}")
        print(f"    median propellant difference {dfuel:+,.0f} kg")
        print(f"\n  Scoring is Day 9's: within {MISS_TOL_M:.0f} m and "
              f"{SPEED_TOL_MS:.0f} m/s counts as landed.")
    return cls, ols


def experiment_a():
    """Guidance rate."""
    _head("EXPERIMENT A - Guidance rate")
    out = []
    for dt in (1.0, 0.5, 0.25, 0.125):
        runs = [run_closed_loop(wind_seed=s, guidance_dt=dt, verbose=False)
                for s in SEEDS[:8]]
        a = _agg(runs)
        print(_row(f"dt = {dt:.3f}s", a))
        solve = np.median([r["mean_solve_time"] for r in runs
                           if r.get("status") == "flown"])
        out.append((dt, a, float(solve)))
        print(f"  {'':>16} median replan {solve:.3f}s against a "
              f"{dt:.3f}s cycle"
              + ("   -- does not fit" if solve > dt else ""))
    return out


def experiment_b():
    """Wind severity, closed against open at each level."""
    _head("EXPERIMENT B - Wind severity")
    out = []
    for sx in (0.0, 3.0, 6.0, 10.0, 15.0):
        cls, ols = [], []
        for s in SEEDS[:8]:
            cl, ol = _pair(s, wind_sigma_x=sx)
            cls.append(cl)
            ols.append(ol)
        a_cl, a_ol = _agg(cls), _agg(ols)
        print(_row(f"open,  sx={sx:.0f}", a_ol))
        print(_row(f"closed,sx={sx:.0f}", a_cl))
        out.append((sx, a_cl, a_ol))
    return out


def experiment_c():
    """Navigation noise: the loop steers on an estimate, not the truth."""
    _head("EXPERIMENT C - Navigation noise")
    out = []
    for pos, vel in ((0.0, 0.0), (1.0, 0.2), (3.0, 0.5), (8.0, 1.5)):
        runs = [run_closed_loop(wind_seed=s, nav_sigma_pos=pos,
                                nav_sigma_vel=vel, verbose=False)
                for s in SEEDS[:8]]
        a = _agg(runs)
        print(_row(f"{pos:.0f} m, {vel:.1f} m/s", a))
        out.append((pos, a))
    return out


# ======================================================================
def plot_sweeps(d_cls, d_ols, a, b, save_path=None):
    save_path = save_path or os.path.join(RESULTS, "day10_sweeps.png")
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("Day 10: closed-loop guidance sweeps", fontsize=13)

    pairs = [(c, o) for c, o in zip(d_cls, d_ols)
             if c.get("status") == "flown" and o.get("status") == "flown"]
    p = ax[0]
    if pairs:
        om = [o["speed"] for _, o in pairs]
        cm = [c["speed"] for c, _ in pairs]
        lim = max(max(om), max(cm)) * 1.1
        p.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6)
        p.scatter(om, cm, s=45, color="tab:blue", zorder=3)
        p.axhline(SPEED_TOL_MS, color="g", ls=":", alpha=0.7)
        p.axvline(SPEED_TOL_MS, color="g", ls=":", alpha=0.7)
        p.set_xlim(0, lim); p.set_ylim(0, lim)
    p.set_xlabel("open-loop arrival [m/s]")
    p.set_ylabel("closed-loop arrival [m/s]")
    p.set_title("D: paired by seed (below the line = better)")
    p.grid(alpha=0.3)

    p = ax[1]
    dts = [x[0] for x in a]
    p.semilogx(dts, [x[1]["speed_med"] if x[1] else np.nan for x in a],
               "o-", lw=2, color="tab:blue", label="median arrival")
    p.set_xlabel("guidance cycle [s]"); p.set_ylabel("[m/s]")
    p.set_title("A: guidance rate"); p.grid(alpha=0.3)
    p2 = p.twinx()
    p2.semilogx(dts, [x[2] for x in a], "s--", lw=2, color="tab:gray")
    p2.plot(dts, dts, ":", color="r", lw=1)
    p2.set_ylabel("replan cost [s]", color="tab:gray")
    p.legend(fontsize=8, loc="upper left")

    p = ax[2]
    sx = [x[0] for x in b]
    p.plot(sx, [x[2]["miss_med"] if x[2] else np.nan for x in b], "o-", lw=2,
           color="tab:red", label="open loop")
    p.plot(sx, [x[1]["miss_med"] if x[1] else np.nan for x in b], "o-", lw=2,
           color="tab:blue", label="closed loop")
    p.axhline(MISS_TOL_M, color="k", ls=":", alpha=0.6, label="tolerance")
    p.set_xlabel("cross-wind 3-sigma [m/s]"); p.set_ylabel("median miss [m]")
    p.set_title("B: wind severity"); p.legend(fontsize=8); p.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSweep plot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 78)
    print("DAY 10 - CLOSED-LOOP EXPLORATION")
    print("=" * 78)
    d_cls, d_ols = experiment_d()
    a = experiment_a()
    b = experiment_b()
    experiment_c()
    plot_sweeps(d_cls, d_ols, a, b)
    print()
