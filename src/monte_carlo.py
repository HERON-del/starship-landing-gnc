"""
Monte Carlo dispersion analysis for the complete SCvx landing solver.

One trajectory is a demo; a few hundred are evidence. This throws dispersed
entry states, mass, engine and aerodynamic errors, and wind at the Day 8 solver
and measures what comes out.

Where this differs from the guide, and why it has to
---------------------------------------------------
The guide measures landing accuracy by reading `x_f, z_f` off the solver's own
solution. Those are hard equality constraints here -- `x[N] == 0`, `z[N] == 0` --
so that number is between 5e-10 and 1e-7 m on every dispersed run, and a CEP
computed from it would be zero by construction. The guide's own expected output
quotes a CEP of 2.05 m, which is only possible for a solver whose terminal
conditions are soft. Measuring the optimiser against its own constraint tells
you nothing about robustness; it tells you the constraint was enforced.

So accuracy here is measured where the error actually lives: **the plan is flown
open-loop through the independently verified nonlinear simulator, with the
dispersions actually applied.** That splits the perturbations into two kinds,
and the split is the whole design:

    told      the entry state. Navigation error puts the vehicle somewhere
              other than nominal, and the solver is given that state and plans
              from it.

    not told  mass, Isp, drag coefficient and wind. The solver plans with the
              nominal vehicle in calm air, because that is what a real onboard
              planner has. The truth differs, and nobody tells it.

The gap between the two is the entire robustness question. A solver that lands
perfectly on its own model and 40 m away in reality is not robust; one that is
never asked the question cannot be shown to be either way.

This is also why Day 10 is closed-loop guidance. Everything measured here is
open-loop: plan once, fly it blind. The miss distances below are the argument
for replanning.

Usage
-----
    python src/monte_carlo.py                    # 100 runs
    python src/monte_carlo.py --n 300 --N 60     # fuller sweep
    python src/monte_carlo.py --n 50 --seed 7    # reproducible subset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as timer
from dataclasses import dataclass, asdict

import matplotlib
import numpy as np

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
RESULTS = os.path.join(REPO_ROOT, "results")

from src.scvx_complete import solve_scvx_complete            # noqa: E402
from src.scvx_params import SCvxParams                       # noqa: E402
from src.dynamics_6dof import Vehicle6DoF, G0, G_EARTH       # noqa: E402
from src.aero import AeroConfig, aero_acceleration           # noqa: E402
from src.integrators import propagate                        # noqa: E402
from src.landing_flip import feasible_entry_state            # noqa: E402

# What counts as a landing rather than an arrival. Both are reported either
# way; these only set the headline success figure.
MISS_TOL_M = 5.0
SPEED_TOL_MS = 5.0


# ======================================================================
# Dispersion model
# ======================================================================
@dataclass
class DispersionConfig:
    """
    3-sigma dispersions.

    Centred on the regime Day 6 concluded the vehicle actually flies and Day 7
    measured as feasible with drag on: coast on the belly, ignite near-upright,
    burn briefly. The guide centres on a 2,500 m, 90 m/s, 70-degree entry, which
    for this vehicle is the case Day 7 measured as having no solution at all --
    a Monte Carlo about that centre reports a failure rate, not a robustness
    figure.

    **The centre is not the output of `feasible_entry_state`, deliberately.**
    That helper sizes an entry the burn can *just* null, with a 1.25 margin, so
    what it returns is a point on the feasibility boundary by construction.
    Measured one axis at a time about its (473 m, -118.3 m/s, 30 deg) answer,
    the solver converges at 473 m and fails at 500; converges at -118.3 m/s and
    fails at -112; converges at 30 deg and fails at 35. Every axis is one-sided.
    Dispersing about that point put a third of the samples outside the feasible
    set and would have been reported as a 33% success rate -- a statement about
    where the centre was chosen, not about the solver.

    The centre below sits inside the envelope instead: at (420 m, -130 m/s,
    25 deg) the solver still converges at +/-80 m of altitude, +/-20 m/s of
    descent rate and +/-8 deg of pitch. The envelope itself is a wedge -- a
    faster entry permits a higher one, because the throttle floor means a slow,
    high approach over-brakes before it arrives.
    """

    # --- told to the solver: where the vehicle actually is ---
    x0_nominal: float = 0.0
    x0_3sigma: float = 25.0
    z0_nominal: float = 420.0
    z0_3sigma: float = 60.0
    vx0_nominal: float = 0.0
    vx0_3sigma: float = 5.0
    vz0_nominal: float = -130.0
    vz0_3sigma: float = 18.0
    theta0_nominal_deg: float = 25.0
    theta0_3sigma_deg: float = 6.0
    omega0_3sigma: float = 0.05

    # --- not told: model error the planner never sees ---
    m_prop_nominal: float = 30_000.0
    m_prop_3sigma: float = 1_500.0
    isp_nominal: float = 327.0
    isp_3sigma: float = 3.0
    Cd_scale_3sigma: float = 0.15
    wind_x_3sigma: float = 15.0
    wind_z_3sigma: float = 5.0

    def summary(self) -> str:
        return "\n".join([
            "Dispersion configuration (3 sigma)",
            "  told to the solver -- it plans from these:",
            f"    x0       : {self.x0_nominal:>9.1f} +/- {self.x0_3sigma:.1f} m",
            f"    z0       : {self.z0_nominal:>9.1f} +/- {self.z0_3sigma:.1f} m",
            f"    vx0      : {self.vx0_nominal:>9.1f} +/- {self.vx0_3sigma:.1f} m/s",
            f"    vz0      : {self.vz0_nominal:>9.1f} +/- {self.vz0_3sigma:.1f} m/s",
            f"    theta0   : {self.theta0_nominal_deg:>9.1f} +/- "
            f"{self.theta0_3sigma_deg:.1f} deg",
            f"    omega0   : {0.0:>9.3f} +/- {self.omega0_3sigma:.3f} rad/s",
            "  not told -- it plans nominal, the vehicle flies these:",
            f"    m_prop   : {self.m_prop_nominal:>9.0f} +/- "
            f"{self.m_prop_3sigma:.0f} kg",
            f"    Isp      : {self.isp_nominal:>9.1f} +/- {self.isp_3sigma:.1f} s",
            f"    Cd scale : {1.0:>9.2f} +/- {self.Cd_scale_3sigma:.2f}",
            f"    wind x   : {0.0:>9.1f} +/- {self.wind_x_3sigma:.1f} m/s",
            f"    wind z   : {0.0:>9.1f} +/- {self.wind_z_3sigma:.1f} m/s",
        ])


def sample_dispersions(rng, disp: DispersionConfig) -> dict:
    """
    Draw one sample.

    Returns the entry state the solver will be given, and separately the truth
    it will not be given. Physical floors are applied where a Gaussian tail
    would otherwise produce a vehicle that cannot exist.
    """
    def g(nominal, three_sigma):
        return nominal + rng.normal() * (three_sigma / 3.0)

    return {
        # told
        "x0": g(disp.x0_nominal, disp.x0_3sigma),
        "z0": max(g(disp.z0_nominal, disp.z0_3sigma), 100.0),
        "vx0": g(disp.vx0_nominal, disp.vx0_3sigma),
        "vz0": min(g(disp.vz0_nominal, disp.vz0_3sigma), -20.0),
        "theta0_deg": float(np.clip(
            g(disp.theta0_nominal_deg, disp.theta0_3sigma_deg), 0.0, 70.0)),
        "omega0": g(0.0, disp.omega0_3sigma),
        # not told
        "m_prop": max(g(disp.m_prop_nominal, disp.m_prop_3sigma), 5_000.0),
        "isp": max(g(disp.isp_nominal, disp.isp_3sigma), 250.0),
        "Cd_scale": max(g(1.0, disp.Cd_scale_3sigma), 0.4),
        "wind_x": g(0.0, disp.wind_x_3sigma),
        "wind_z": g(0.0, disp.wind_z_3sigma),
    }


# ======================================================================
# Truth dynamics: the same 6-DoF model, in moving air
# ======================================================================
def dynamics_in_wind(t, state, control_fn, vehicle, aero, wind):
    """
    State derivative with a steady wind field.

    Identical to `dynamics_aero.dynamics_full` except that the aerodynamic
    forces see the velocity of the vehicle *relative to the air*, which is what
    makes wind a disturbance rather than a relabelling of the initial condition.
    The guide injects wind by shifting `vx0` and `vz0`, but the solver is then
    handed the shifted state and plans around it -- so the wind is not
    unmodelled at all, it is just more navigation dispersion. Here the planner
    never sees it.
    """
    x, z, vx, vz, theta, omega, m = state
    wx, wz = wind

    T, delta = control_fn(t, state, vehicle)
    T = float(np.clip(T, 0.0, vehicle.T_max))
    delta = float(np.clip(delta, -vehicle.delta_max, vehicle.delta_max))

    if m <= vehicle.m_dry:
        m = vehicle.m_dry
        Tx = Tz = tau = 0.0
        mdot = 0.0
    else:
        Tx = T * np.sin(theta + delta)
        Tz = T * np.cos(theta + delta)
        tau = T * vehicle.L_engine * np.sin(delta)
        mdot = -T / (vehicle.isp * G0)

    if aero is not None and aero.enabled:
        ax, az = aero_acceleration(vx - wx, vz - wz, z, theta, m, aero)
    else:
        ax = az = 0.0

    return np.array([
        vx, vz,
        Tx / m + float(ax),
        Tz / m + float(az) - G_EARTH,
        omega,
        tau / vehicle.I_pitch,
        mdot,
    ])


def fly_the_plan(plan, truth_vehicle, truth_aero, wind, steps=4000,
                 return_path=False):
    """
    Fly the commanded throttle and gimbal open-loop through the true vehicle.

    The miss is the distance from the pad at the end of the flown plan, which
    is the same quantity Day 8's replay test reports. A ground-crossing test
    alone would be wrong here: the plan puts the vehicle at exactly `z = 0` at
    `t_f`, and the RK4 replay of it bottoms out at `z = +0.097 m` on the nominal
    case, so `z <= 0` never fires and every run would be scored as a failure to
    land. Day 8's 0.502 m was almost all downrange -- 0.486 m of it.

    Both cases are handled: if the trajectory does cross the ground the state is
    interpolated to the crossing and the miss is horizontal, and if it does not,
    the miss is the straight-line distance from the pad at `t_f` and the
    residual altitude is reported alongside it.

    With `return_path`, the flown state history is returned too, truncated at
    the ground crossing where there is one. The viewer needs it to draw the
    trajectory the vehicle actually flew rather than the one that was planned,
    which is the entire point of the day.
    """
    sigma, delta = plan["sigma"], plan["delta"]
    t_f = float(plan["t_f"])
    dt_ctrl = t_f / len(sigma)

    def control(t, state, vehicle):
        k = min(int(t / dt_ctrl), len(sigma) - 1)
        return sigma[k], delta[k]

    y0 = np.array([plan["x"][0], plan["z"][0], plan["vx"][0], plan["vz"][0],
                   plan["theta"][0], plan["omega"][0], truth_vehicle.m_wet])
    t_hist, y = propagate(
        lambda tt, yy, *a: dynamics_in_wind(tt, yy, control, truth_vehicle,
                                            truth_aero, wind),
        y0, (0.0, t_f), t_f / steps, method="rk4")

    below = np.flatnonzero(y[:, 1] <= 0.0)
    if below.size:
        i = int(below[0])
        if i == 0:
            s, frac = y[0], 0.0
        else:
            prev, cur = y[i - 1], y[i]
            dz = prev[1] - cur[1]
            frac = prev[1] / dz if abs(dz) > 1e-12 else 0.0
            s = prev + frac * (cur - prev)
        out = {
            "touched_down": True,
            "miss": float(abs(s[0])),
            "speed": float(np.hypot(s[2], s[3])),
            "pitch_deg": float(np.degrees(s[4])),
            "altitude_left": 0.0,
            "mass": float(s[6]),
            "t_end": float(t_hist[max(i - 1, 0)] + frac * (t_f / steps)),
        }
        if return_path:
            out["t_path"] = np.append(t_hist[:i], out["t_end"])
            out["y_path"] = np.vstack([y[:i], s])
        return out

    end = y[-1]
    out = {
        "touched_down": False,
        "miss": float(np.hypot(end[0], end[1])),
        "speed": float(np.hypot(end[2], end[3])),
        "pitch_deg": float(np.degrees(end[4])),
        "altitude_left": float(end[1]),
        "mass": float(end[6]),
        "t_end": float(t_hist[-1]),
    }
    if return_path:
        out["t_path"] = t_hist
        out["y_path"] = y
    return out


# ======================================================================
# One dispersed case
# ======================================================================
def run_single(sample: dict, N: int = 60, disp: DispersionConfig = None,
               max_iter: int = 25, keep_path: bool = False) -> dict:
    """
    Plan with the nominal vehicle in calm air, then fly the true one.

    Note what the planner is handed: `Vehicle6DoF()` and `AeroConfig()` at their
    nominal values, never the sampled ones. Handing it the truth would measure
    the optimiser, not the guidance.
    """
    disp = disp or DispersionConfig()
    t0 = timer.time()

    plan_vehicle = Vehicle6DoF(m_prop_initial=disp.m_prop_nominal,
                               isp=disp.isp_nominal)
    plan_aero = AeroConfig()

    # Duration guess from the sampled state, the same constant-deceleration
    # estimate that sizes the entry state in the first place. Free final time
    # then moves it, which is exactly what Day 8 bought.
    t_guess = float(np.clip(2.0 * sample["z0"] / abs(sample["vz0"]),
                            3.0, 20.0))

    try:
        plan = solve_scvx_complete(
            vehicle=plan_vehicle, aero=plan_aero,
            params=SCvxParams(max_iter=max_iter),
            N=N, t_burn_guess=t_guess,
            x0=sample["x0"], z0=sample["z0"],
            vx0=sample["vx0"], vz0=sample["vz0"],
            theta0_deg=sample["theta0_deg"], omega0=sample["omega0"],
            verbose=False,
        )
    except Exception as exc:            # noqa: BLE001
        return _blank(sample, timer.time() - t0, f"error: {type(exc).__name__}")

    if plan.get("status") != "converged":
        return _blank(sample, timer.time() - t0,
                      plan.get("status", "failed"),
                      iterations=plan.get("iterations", 0))

    # --- the truth the planner was not told ---------------------------
    truth_vehicle = Vehicle6DoF(m_prop_initial=sample["m_prop"],
                                isp=sample["isp"])
    truth_aero = AeroConfig()
    truth_aero.Cd_belly *= sample["Cd_scale"]
    truth_aero.Cd_nose *= sample["Cd_scale"]

    flown = fly_the_plan(plan, truth_vehicle, truth_aero,
                         (sample["wind_x"], sample["wind_z"]),
                         return_path=keep_path)

    fuel = truth_vehicle.m_wet - flown["mass"]
    margin = flown["mass"] - truth_vehicle.m_dry
    reasons = []
    if flown["miss"] >= MISS_TOL_M:
        reasons.append("missed the pad")
    if flown["speed"] >= SPEED_TOL_MS:
        reasons.append("arrived too fast")
    if margin <= 0.0:
        reasons.append("out of propellant")
    good = not reasons

    return {
        "converged": True,
        "good": bool(good),
        "fail_reason": " + ".join(reasons) if reasons else "landed",
        "touched_down": bool(flown["touched_down"]),
        "status": "converged",
        "miss": flown["miss"],
        "speed": flown["speed"],
        "pitch_deg": abs(flown["pitch_deg"]),
        "altitude_left": flown["altitude_left"],
        "fuel": float(fuel),
        "margin": float(margin),
        "t_f": float(plan["t_f"]),
        "t_end": flown["t_end"],
        "iterations": int(plan["iterations"]),
        # what the solver thought it had achieved, for the contrast
        "planned_miss": float(np.hypot(plan["x"][-1], plan["z"][-1])),
        "elapsed": timer.time() - t0,
        "sample": sample,
        # Only when asked: the viewer draws the flown path against the plan,
        # and carrying these for every run of a 250-sample sweep would hold
        # the whole fleet's trajectories in memory for no reason.
        **({"plan": plan,
            "t_path": flown["t_path"],
            "y_path": flown["y_path"]} if keep_path else {}),
    }


def _blank(sample, elapsed, status, iterations=0) -> dict:
    nan = float("nan")
    return {
        "converged": False, "good": False, "touched_down": False,
        "status": status, "fail_reason": "no plan",
        "miss": nan, "speed": nan, "pitch_deg": nan, "altitude_left": nan,
        "fuel": nan, "margin": nan, "t_f": nan, "t_end": nan,
        "iterations": iterations, "planned_miss": nan,
        "elapsed": elapsed, "sample": sample,
    }


# ======================================================================
# The sweep
# ======================================================================
def run_monte_carlo(n_runs: int = 100, seed: int = 42,
                    disp: DispersionConfig = None, N: int = 60,
                    max_iter: int = 25, verbose: bool = True) -> dict:
    """Run `n_runs` dispersed cases and reduce them to statistics."""
    disp = disp or DispersionConfig()
    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 72)
        print(f"MONTE CARLO -- {n_runs} runs, N = {N}, seed = {seed}")
        print("=" * 72)
        print(disp.summary())
        print()

    samples = [sample_dispersions(rng, disp) for _ in range(n_runs)]
    t_start = timer.time()
    results = []
    for i, s in enumerate(samples):
        results.append(run_single(s, N=N, disp=disp, max_iter=max_iter))
        if verbose and (i + 1) % 10 == 0:
            ok = sum(1 for r in results if r["converged"])
            good = sum(1 for r in results if r["good"])
            avg = float(np.mean([r["elapsed"] for r in results]))
            print(f"  [{i + 1:>4}/{n_runs}]  solved {100 * ok / len(results):5.1f}%"
                  f"  landed {100 * good / len(results):5.1f}%"
                  f"  {avg:.1f}s/run  ETA {avg * (n_runs - i - 1):.0f}s")

    stats = summarize(results, n_runs, timer.time() - t_start)
    if verbose:
        print(format_stats(stats))
    return {"results": results, "stats": stats,
            "config": asdict(disp), "seed": seed, "N": N}


def summarize(results, n_runs, total_time) -> dict:
    solved = [r for r in results if r["converged"]]
    landed = solved      # every converged plan is flown; miss is always defined
    good = [r for r in results if r["good"]]

    stats = {
        "n_runs": n_runs,
        "n_solved": len(solved),
        "n_good": len(good),
        "n_touched_down": sum(1 for r in results if r["touched_down"]),
        "solve_rate": 100.0 * len(solved) / n_runs if n_runs else 0.0,
        "land_rate": 100.0 * len(good) / n_runs if n_runs else 0.0,
        "miss_tol_m": MISS_TOL_M,
        "speed_tol_ms": SPEED_TOL_MS,
        "total_time": total_time,
    }
    reasons = {}
    for r in results:
        reasons[r["fail_reason"]] = reasons.get(r["fail_reason"], 0) + 1
    stats["outcomes"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    if solved:
        stats["iter_mean"] = float(np.mean([r["iterations"] for r in solved]))
        stats["iter_max"] = int(np.max([r["iterations"] for r in solved]))
        stats["t_f_mean"] = float(np.mean([r["t_f"] for r in solved]))
        stats["t_f_min"] = float(np.min([r["t_f"] for r in solved]))
        stats["t_f_max"] = float(np.max([r["t_f"] for r in solved]))
        stats["planned_miss_max"] = float(
            np.max([r["planned_miss"] for r in solved]))
    if landed:
        miss = np.array([r["miss"] for r in landed])
        spd = np.array([r["speed"] for r in landed])
        pit = np.array([r["pitch_deg"] for r in landed])
        fuel = np.array([r["fuel"] for r in landed])
        marg = np.array([r["margin"] for r in landed])
        stats.update({
            "miss_mean": float(np.mean(miss)),
            "miss_std": float(np.std(miss)),
            "miss_max": float(np.max(miss)),
            "miss_cep": float(np.percentile(miss, 50)),
            "miss_p95": float(np.percentile(miss, 95)),
            "miss_p997": float(np.percentile(miss, 99.7)),
            "speed_mean": float(np.mean(spd)),
            "speed_max": float(np.max(spd)),
            "pitch_mean": float(np.mean(pit)),
            "pitch_max": float(np.max(pit)),
            "fuel_mean": float(np.mean(fuel)),
            "fuel_std": float(np.std(fuel)),
            "fuel_min": float(np.min(fuel)),
            "fuel_max": float(np.max(fuel)),
            "margin_mean": float(np.mean(marg)),
            "margin_min": float(np.min(marg)),
            "margin_p03": float(np.percentile(marg, 0.3)),
            "n_margin_negative": int(np.sum(marg < 0)),
        })
        # The arrival-speed distribution is bimodal, and the ground-crossing
        # flag is what separates the two modes: a plan that reaches the pad
        # early is still moving, one that runs out of trajectory above it has
        # already stopped. Reported split so the mean is never read as typical.
        early = np.array([r["speed"] for r in landed if r["touched_down"]])
        short = np.array([r["speed"] for r in landed if not r["touched_down"]])
        if early.size:
            stats["speed_mean_touchdown"] = float(np.mean(early))
        if short.size:
            stats["speed_mean_stopped_short"] = float(np.mean(short))
    return stats


def format_stats(s) -> str:
    out = ["", "=" * 72,
           f"RESULTS -- {s['n_runs']} runs in {s['total_time']:.0f}s",
           "=" * 72, ""]
    out.append(f"  Solver converged : {s['solve_rate']:5.1f}%  "
               f"({s['n_solved']}/{s['n_runs']})")
    out.append(f"  Landed well      : {s['land_rate']:5.1f}%  "
               f"({s['n_good']}/{s['n_runs']})   "
               f"[within {s['miss_tol_m']:.0f} m and "
               f"{s['speed_tol_ms']:.0f} m/s, flown open-loop "
               f"through the true vehicle]")
    if "miss_mean" in s:
        out += [
            "",
            "  Miss distance at touchdown (the plan flown blind):",
            f"    mean {s['miss_mean']:>8.2f} m     "
            f"std {s['miss_std']:>7.2f} m     max {s['miss_max']:>8.2f} m",
            f"    CEP  {s['miss_cep']:>8.2f} m     "
            f"p95 {s['miss_p95']:>7.2f} m     p99.7 {s['miss_p997']:>7.2f} m",
            "",
            f"  For contrast, the solver's own terminal error never exceeded "
            f"{s.get('planned_miss_max', float('nan')):.2e} m --",
            "  it is a hard equality constraint, which is why it cannot be the "
            "accuracy metric.",
            "",
            "  Touchdown speed (bimodal -- the mean is not a typical case):",
            f"    mean {s['speed_mean']:>8.2f} m/s   max {s['speed_max']:>7.2f} m/s",
            f"    reached the pad early, still moving : "
            f"{s.get('speed_mean_touchdown', float('nan')):>6.2f} m/s "
            f"({s['n_touched_down']} runs)",
            f"    ran out of trajectory above it      : "
            f"{s.get('speed_mean_stopped_short', float('nan')):>6.2f} m/s "
            f"({s['n_solved'] - s['n_touched_down']} runs)",
            "  Pitch at touchdown:",
            f"    mean {s['pitch_mean']:>8.2f} deg   max {s['pitch_max']:>7.2f} deg",
            "",
            "  Propellant used:",
            f"    mean {s['fuel_mean']:>8,.0f} kg    std {s['fuel_std']:>6,.0f} kg"
            f"    range [{s['fuel_min']:,.0f}, {s['fuel_max']:,.0f}] kg",
            "  Margin remaining:",
            f"    mean {s['margin_mean']:>8,.0f} kg    min {s['margin_min']:>7,.0f} kg"
            f"    p0.3 {s['margin_p03']:>7,.0f} kg",
            f"    runs finishing below dry mass: {s['n_margin_negative']}",
        ]
    if "t_f_mean" in s:
        out += ["",
                f"  Burn duration chosen: mean {s['t_f_mean']:.2f} s, "
                f"range [{s['t_f_min']:.2f}, {s['t_f_max']:.2f}] s",
                f"  SCvx iterations: mean {s['iter_mean']:.1f}, "
                f"max {s['iter_max']}"]
    if s.get("outcomes"):
        out += ["", "  Outcome breakdown:"]
        for k, v in s["outcomes"].items():
            out.append(f"    {v:>4}  {100 * v / s['n_runs']:5.1f}%   {k}")
    return "\n".join(out)


# ======================================================================
# Plots
# ======================================================================
def plot_monte_carlo(mc, save_path=None):
    """Six panels: where it lands, what it costs, and what it had left."""
    save_path = save_path or os.path.join(RESULTS, "day9_monte_carlo.png")
    res = mc["results"]
    landed = [r for r in res if r["converged"]]
    s = mc["stats"]
    if not landed:
        print("Nothing landed - no plot.")
        return

    miss = np.array([r["miss"] for r in landed])
    spd = np.array([r["speed"] for r in landed])
    fuel = np.array([r["fuel"] for r in landed])
    marg = np.array([r["margin"] for r in landed])
    t_f = np.array([r["t_f"] for r in landed])
    ventry = np.array([abs(r["sample"]["vz0"]) for r in landed])
    windx = np.array([r["sample"]["wind_x"] for r in landed])

    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle(f"Day 9: {mc['stats']['n_runs']} dispersed landings, "
                 f"flown open-loop through the true vehicle", fontsize=14)

    a = ax[0, 0]
    a.scatter(np.array([r["miss"] for r in landed])
              * np.sign(windx + 1e-12), np.zeros(len(landed)),
              s=18, alpha=0.5, color="tab:blue")
    for r_c, c, lab in ((s["miss_cep"], "tab:green", "CEP"),
                        (s["miss_p95"], "tab:orange", "p95"),
                        (s["miss_max"], "tab:red", "max")):
        a.add_patch(plt.Circle((0, 0), r_c, fill=False, color=c, ls="--",
                               alpha=0.8, label=f"{lab} {r_c:.1f} m"))
    a.plot(0, 0, "k^", ms=12)
    lim = max(s["miss_max"] * 1.2, 1.0)
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_aspect("equal")
    a.set_xlabel("downrange miss [m]")
    a.set_title("Where it actually landed")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.hist(miss, bins=25, color="tab:blue", alpha=0.8)
    a.axvline(s["miss_cep"], color="tab:green", ls="--",
              label=f"CEP {s['miss_cep']:.1f} m")
    a.axvline(s["miss_p95"], color="tab:orange", ls="--",
              label=f"p95 {s['miss_p95']:.1f} m")
    a.set_xlabel("miss distance [m]"); a.set_ylabel("runs")
    a.set_title("Miss distribution"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 2]
    a.hist(spd, bins=25, color="tab:purple", alpha=0.8)
    a.axvline(s["speed_mean"], color="k", ls="--",
              label=f"mean {s['speed_mean']:.1f} m/s")
    a.set_xlabel("touchdown speed [m/s]"); a.set_ylabel("runs")
    a.set_title("Arrival speed"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.hist(fuel, bins=25, color="tab:red", alpha=0.8)
    a.axvline(s["fuel_mean"], color="k", ls="--",
              label=f"mean {s['fuel_mean']:,.0f} kg")
    a.set_xlabel("propellant used [kg]"); a.set_ylabel("runs")
    a.set_title("Propellant"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.hist(marg / 1000, bins=25, color="tab:green", alpha=0.8)
    a.axvline(0, color="tab:red", lw=2, label="dry mass")
    a.axvline(s["margin_p03"] / 1000, color="k", ls="--",
              label=f"p0.3 {s['margin_p03'] / 1000:.1f} t")
    a.set_xlabel("margin remaining [tonnes]"); a.set_ylabel("runs")
    a.set_title("Propellant margin"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 2]
    sc = a.scatter(ventry, t_f, c=miss, cmap="viridis", s=22, alpha=0.85)
    plt.colorbar(sc, ax=a, label="miss [m]")
    a.set_xlabel("entry descent rate [m/s]")
    a.set_ylabel("burn duration chosen [s]")
    a.set_title("Free final time adapting to the entry")
    a.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nMonte Carlo plot -> {save_path}")
    plt.close()


def plot_failures(mc, save_path=None):
    """Where in the dispersion space the solver gives up, and why."""
    save_path = save_path or os.path.join(RESULTS, "day9_failures.png")
    res = mc["results"]
    ok = [r for r in res if r["good"]]
    bad = [r for r in res if not r["good"]]

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("Day 9: failure modes", fontsize=13)

    def sc(a, key_x, key_y, xlabel, ylabel, title):
        if ok:
            a.scatter([r["sample"][key_x] for r in ok],
                      [r["sample"][key_y] for r in ok],
                      s=18, alpha=0.45, color="tab:green", label="landed")
        if bad:
            a.scatter([r["sample"][key_x] for r in bad],
                      [r["sample"][key_y] for r in bad],
                      s=34, alpha=0.9, color="tab:red", marker="x",
                      label="did not")
        a.set_xlabel(xlabel); a.set_ylabel(ylabel); a.set_title(title)
        a.legend(fontsize=8); a.grid(alpha=0.3)

    sc(ax[0], "theta0_deg", "vz0", "entry pitch [deg]",
       "entry descent rate [m/s]", "Attitude against speed")
    sc(ax[1], "z0", "vz0", "entry altitude [m]",
       "entry descent rate [m/s]", "Altitude against speed")

    a = ax[2]
    modes = {}
    for r in res:
        key = r["fail_reason"].replace(" + ", "\n+ ")
        modes[key] = modes.get(key, 0) + 1
    keys = sorted(modes, key=lambda k: -modes[k])
    bars = a.bar(range(len(keys)), [modes[k] for k in keys],
                 color=["tab:green" if k == "landed" else "tab:red"
                        for k in keys])
    a.set_xticks(range(len(keys)))
    a.set_xticklabels(keys, rotation=20, ha="right", fontsize=8)
    a.set_ylabel("runs"); a.set_title("Outcome breakdown")
    for b, k in zip(bars, keys):
        a.text(b.get_x() + b.get_width() / 2, b.get_height(),
               f"{modes[k]}", ha="center", va="bottom", fontsize=9)
    a.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Failure plot -> {save_path}")
    plt.close()


def plot_dispersion(mc, save_path=None):
    """Which unmodelled error actually drives the miss."""
    save_path = save_path or os.path.join(RESULTS, "day9_dispersion.png")
    landed = [r for r in mc["results"] if r["converged"]]
    if len(landed) < 5:
        print("Too few landings for a sensitivity plot.")
        return
    miss = np.array([r["miss"] for r in landed])

    drivers = [
        ("wind_x", "cross-wind [m/s]"),
        ("Cd_scale", "drag coefficient scale"),
        ("m_prop", "true propellant load [kg]"),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("Day 9: what the miss distance is actually made of",
                 fontsize=13)
    for a, (key, label) in zip(ax, drivers):
        v = np.array([r["sample"][key] for r in landed])
        a.scatter(v, miss, s=20, alpha=0.6, color="tab:blue")
        if np.std(v) > 1e-12:
            r_p = float(np.corrcoef(v, miss)[0, 1])
            k, b = np.polyfit(v, miss, 1)
            xs = np.linspace(v.min(), v.max(), 20)
            a.plot(xs, k * xs + b, "r--", lw=2,
                   label=f"corr {r_p:+.2f}")
            a.legend(fontsize=9)
        a.set_xlabel(label); a.set_ylabel("miss distance [m]")
        a.set_title(label.split("[")[0].strip()); a.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Dispersion plot -> {save_path}")
    plt.close()


def save_stats(mc, save_path=None):
    save_path = save_path or os.path.join(RESULTS, "day9_stats.json")
    os.makedirs(RESULTS, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump({"stats": mc["stats"], "config": mc["config"],
                   "seed": mc["seed"], "N": mc["N"]}, fh, indent=2)
    print(f"Statistics -> {save_path}")


# ======================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Monte Carlo dispersion analysis")
    ap.add_argument("--n", type=int, default=100, help="number of runs")
    ap.add_argument("--N", type=int, default=60, help="nodes per trajectory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=25)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    print()
    mc = run_monte_carlo(n_runs=args.n, seed=args.seed, N=args.N,
                         max_iter=args.max_iter, verbose=True)
    if not args.no_plots:
        plot_monte_carlo(mc)
        plot_failures(mc)
        plot_dispersion(mc)
    save_stats(mc)
    print()
