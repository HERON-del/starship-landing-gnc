"""
Closed-loop guidance: warm-started SCvx replanning against truth dynamics.

Day 9 measured what one plan is worth when the vehicle does not fly it exactly.
The answer was 29.6% good landings out of 250 dispersed runs, and the failure
was almost never position or propellant -- it was arrival speed, because a
minimum-fuel trajectory is bang-bang and arrives with no slack, so any error in
net deceleration puts the vehicle on one side of the pad or the other with
nothing open-loop to restore it. There were 22 tonnes of propellant aboard that
never got spent on the problem.

This is the layer that spends it. The Day 8 solver stops being a thing you run
once and becomes a subroutine called every `guidance_dt` seconds from wherever
the vehicle actually is, warm-started from the previous answer.

What warm starting is actually worth here
-----------------------------------------
Not what the guide claims. It measures speedup by capping warm solves at four
iterations and comparing against an uncapped cold solve, which guarantees a
speedup of at least the cap whether or not warm starting does anything. Run
both to the same convergence tolerance and the effect is absent or negative:
measured at three replan points, warm took 16, 28 and 26 iterations against
cold's 20, 21 and 23. Tightening the trust region to "exploit" the good
reference makes it worse still, because the solver then cannot move far enough
per iteration to absorb the tracking error.

The reason is structural. This solver's iteration count is set by the
trust-region schedule annealing down from `eta_0` and by the convergence test,
not by how far the initial reference sits from the answer. A better starting
point does not shorten a schedule that does not know about it.

What warm starting *does* buy is the thing a guidance loop actually needs:
a usable command inside a fixed compute budget. Given one iteration from a
3 m tracking gap, the warm solve commands a gimbal 5.9 degrees from the
converged answer and the cold solve is 24.1 degrees off -- saturated the wrong
way. Given three, warm is 0.30 degrees off and cold is still 5.9. Commanded
thrust is identical in every case; it is the steering that is wrong when cold.

So the loop runs a small fixed budget per cycle rather than solving to
convergence, which is what a real guidance computer does anyway. Three
iterations costs about 0.26 s against a 0.5 s cycle.
"""

import os
import sys
import time as timer

import matplotlib
import numpy as np

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
RESULTS = os.path.join(REPO_ROOT, "results")

from src.scvx_complete import solve_scvx_complete              # noqa: E402
from src.scvx_params import SCvxParams                         # noqa: E402
from src.warm_start import shift_reference                     # noqa: E402
from src.dynamics_6dof import Vehicle6DoF                      # noqa: E402
from src.aero import AeroConfig                                # noqa: E402
from src.monte_carlo import dynamics_in_wind                   # noqa: E402
from src.integrators import propagate                          # noqa: E402

# Day 9's operating point, and its scoring thresholds, so the two days'
# numbers mean the same thing.
Z0_NOM, VZ0_NOM, THETA0_NOM = 420.0, -130.0, 25.0
MISS_TOL_M, SPEED_TOL_MS = 5.0, 5.0

# Ceiling on the duration a replan may choose, as a multiple of the time the
# previous plan had left. `None` lets the solver pick freely within its usual
# half-to-double band. This matters more than it looks: each replan
# re-optimises minimum fuel, and the cheapest plan is always to brake as late
# as possible, so a loop free to re-decide the duration every cycle can
# postpone the burn indefinitely and arrive fast.
TF_CAP = None


class WindGusts:
    """
    Ornstein-Uhlenbeck gust field.

    White noise would be the wrong model and would also flatter the guidance:
    a disturbance that reverses every step averages itself out over a cycle and
    barely perturbs the vehicle. Real gusts persist for seconds, which is long
    enough to push the trajectory somewhere the plan did not expect, and that
    is the case the loop has to answer.
    """

    def __init__(self, sigma_x=6.0, sigma_z=2.0, tau=2.0, seed=0):
        self.sigma_x, self.sigma_z, self.tau = sigma_x, sigma_z, tau
        self.rng = np.random.default_rng(seed)
        self.wx = self.rng.normal() * sigma_x
        self.wz = self.rng.normal() * sigma_z

    def step(self, dt):
        a = float(np.exp(-dt / self.tau))
        s = float(np.sqrt(max(1.0 - a * a, 0.0)))
        self.wx = a * self.wx + s * self.sigma_x * self.rng.normal()
        self.wz = a * self.wz + s * self.sigma_z * self.rng.normal()
        return self.wx, self.wz


def _fly(state, plan, t_offset, dt, vehicle, aero, wind, steps=200):
    """
    One guidance interval of truth, following the current plan's schedule.

    The vehicle tracks the plan's zero-order-hold control profile from
    `t_offset` onward, rather than freezing its first sample for the whole
    cycle. That distinction is not cosmetic: the plan's control interval is
    about 0.13 s and the guidance cycle is 0.5 s, so holding one gimbal command
    across a cycle mis-steers the attitude badly -- pitch is the fastest state
    in the vehicle. Measured, holding it drove the replan infeasible within two
    cycles while position still matched the plan to 0.3 m.

    This is the ordinary guidance/control split: guidance re-plans at a low
    rate, an inner loop follows the plan it was given at a much higher one.
    """
    sig, dlt = plan["sigma"], plan["delta"]
    dt_ctrl = float(plan["t_f"]) / len(sig)

    def control(t, s, veh):
        k = int(np.clip((t_offset + t) / dt_ctrl, 0, len(sig) - 1))
        return float(sig[k]), float(dlt[k])

    _, y = propagate(
        lambda t, s, *a: dynamics_in_wind(t, s, control, vehicle, aero, wind),
        np.asarray(state, dtype=float), (0.0, dt), dt / steps, method="rk4")
    return y


def _ground_crossing(y):
    """Index and interpolated state at the first z <= 0, or None."""
    below = np.flatnonzero(y[:, 1] <= 0.0)
    if not below.size:
        return None
    i = int(below[0])
    if i == 0:
        return y[0]
    prev, cur = y[i - 1], y[i]
    dz = prev[1] - cur[1]
    f = prev[1] / dz if abs(dz) > 1e-12 else 0.0
    return prev + f * (cur - prev)


def _plan(vehicle, aero, N, t_guess, state, max_iter, ref=None,
          gamma_gs_deg=75.0, tf_cap=None):
    """One SCvx solve from the current state, warm if a reference is given."""
    return solve_scvx_complete(
        vehicle=vehicle, aero=aero,
        params=SCvxParams(max_iter=max_iter, min_iter=1),
        N=N, t_burn_guess=t_guess,
        t_f_max=None if tf_cap is None else tf_cap * t_guess,
        x0=float(state["x"]), z0=float(state["z"]),
        vx0=float(state["vx"]), vz0=float(state["vz"]),
        theta0_deg=float(np.degrees(state["theta"])),
        omega0=float(state["omega"]), m0=float(state["m"]),
        gamma_gs_deg=gamma_gs_deg, initial_ref=ref, verbose=False,
    )


def _as_state(y):
    return {"x": y[0], "z": y[1], "vx": y[2], "vz": y[3],
            "theta": y[4], "omega": y[5], "m": y[6]}


def _outcome(final, vehicle, m_start):
    miss = float(abs(final[0]))
    speed = float(np.hypot(final[2], final[3]))
    reasons = []
    if miss >= MISS_TOL_M:
        reasons.append("missed the pad")
    if speed >= SPEED_TOL_MS:
        reasons.append("arrived too fast")
    if final[6] - vehicle.m_dry <= 0.0:
        reasons.append("out of propellant")
    return {
        "miss": miss, "speed": speed,
        "pitch_deg": float(abs(np.degrees(final[4]))),
        "fuel": float(m_start - final[6]),
        "margin": float(final[6] - vehicle.m_dry),
        "good": not reasons,
        "fail_reason": " + ".join(reasons) if reasons else "landed",
    }


# ======================================================================
def run_closed_loop(
    vehicle=None, aero=None, N=40, guidance_dt=0.5, budget=3,
    x0=0.0, z0=Z0_NOM, vx0=0.0, vz0=VZ0_NOM, theta0_deg=THETA0_NOM,
    omega0=0.0, gamma_gs_deg=75.0,
    wind=None, wind_sigma_x=6.0, wind_sigma_z=2.0, wind_tau=2.0, wind_seed=0,
    nav_sigma_pos=0.0, nav_sigma_vel=0.0, nav_seed=None,
    max_steps=200, keep_path=False, verbose=True,
):
    """
    Replan every `guidance_dt` seconds, warm-started, and fly the result.

    `budget` is the per-cycle SCvx iteration allowance -- the compute a real
    guidance computer would have, rather than however long convergence takes.
    """
    vehicle = vehicle or Vehicle6DoF()
    aero = aero if aero is not None else AeroConfig()
    gusts = wind or WindGusts(wind_sigma_x, wind_sigma_z, wind_tau, wind_seed)
    nav = np.random.default_rng(wind_seed if nav_seed is None else nav_seed)

    truth = np.array([x0, z0, vx0, vz0, np.radians(theta0_deg), omega0,
                      vehicle.m_wet], dtype=float)
    m_start = float(truth[6])
    t_guess = float(np.clip(2.0 * z0 / abs(vz0), 3.0, 20.0))

    t_solve = timer.time()
    plan = _plan(vehicle, aero, N, t_guess, _as_state(truth),
                 max_iter=30, gamma_gs_deg=gamma_gs_deg)
    cold_time, cold_iters = timer.time() - t_solve, plan.get("iterations", 0)
    if plan.get("status") == "failed":
        return {"status": "no initial plan"}

    if verbose:
        print(f"  cold start: {cold_iters} iterations, {cold_time:.2f}s, "
              f"t_f {plan['t_f']:.2f}s")
        print(f"\n  {'t':>6} {'z':>8} {'|v|':>8} {'gap':>7} {'it':>3} "
              f"{'solve':>7} {'warm':>5}")

    log = {"t": [0.0], "x": [truth[0]], "z": [truth[1]], "vx": [truth[2]],
           "vz": [truth[3]], "theta": [truth[4]], "omega": [truth[5]],
           "m": [truth[6]]}
    replan = {"t": [], "iterations": [], "solve_time": [], "gap": [],
              "warm": [], "t_f": [], "wind_x": []}
    # The truth log is one sample per guidance cycle, which is far too coarse
    # to draw. When asked, keep the integrator's own sub-steps as well.
    fine_t, fine_y = ([0.0], [truth.copy()]) if keep_path else (None, None)

    t_sim, final, n_replans = 0.0, None, 0
    sigma_cmd = float(plan["sigma"][0])
    delta_cmd = float(plan["delta"][0])
    # Age of the plan currently in hand. Zero right after a solve, since that
    # plan's own clock starts at the state it was solved from -- shifting a
    # freshly computed plan forward by a cycle would hand the solver a
    # reference for where the vehicle is about to be, against a state where it
    # actually is, and the resulting gap is pure bookkeeping error.
    age = 0.0
    n_failed = 0

    for step in range(max_steps):
        state = _as_state(truth)
        # Navigation error: the loop steers on an estimate, not on the truth.
        if nav_sigma_pos or nav_sigma_vel:
            state = dict(state)
            state["x"] += nav.normal() * nav_sigma_pos
            state["z"] += nav.normal() * nav_sigma_pos
            state["vx"] += nav.normal() * nav_sigma_vel
            state["vz"] += nav.normal() * nav_sigma_vel

        ref, remaining, gap = shift_reference(plan, age, state, N, vehicle)
        t0 = timer.time()
        if ref is not None:
            new = _plan(vehicle, aero, N, remaining, state, budget, ref=ref,
                        gamma_gs_deg=gamma_gs_deg, tf_cap=TF_CAP)
            used_warm = True
            n_iters = new.get("iterations", 0)
            if new.get("status") != "failed":
                plan = new
                sigma_cmd = float(plan["sigma"][0])
                delta_cmd = float(plan["delta"][0])
                age = 0.0
            else:
                n_failed += 1
        else:
            # Too little horizon left to replan; ride out the last command.
            used_warm, n_iters = False, 0
        solve_time = timer.time() - t0
        n_replans += 1

        replan["t"].append(t_sim)
        replan["iterations"].append(n_iters)
        replan["solve_time"].append(solve_time)
        replan["gap"].append(gap)
        replan["warm"].append(used_warm)
        replan["t_f"].append(float(plan.get("t_f", np.nan)))

        wx, wz = gusts.step(guidance_dt)
        replan["wind_x"].append(wx)
        if verbose and step % 2 == 0:
            print(f"  {t_sim:>6.2f} {truth[1]:>8.1f} "
                  f"{np.hypot(truth[2], truth[3]):>8.2f} {gap:>7.2f} "
                  f"{n_iters:>3} {solve_time:>6.3f}s "
                  f"{'yes' if used_warm else 'no':>5}")

        y = _fly(truth, plan, age, guidance_dt, vehicle, aero, (wx, wz))
        if keep_path:
            sub = np.linspace(t_sim, t_sim + guidance_dt, len(y))
            fine_t.extend(sub[1:].tolist())
            fine_y.extend(list(y[1:]))
        hit = _ground_crossing(y)
        if hit is not None:
            final = hit
            t_sim += guidance_dt
            break

        truth = y[-1]
        t_sim += guidance_dt
        age += guidance_dt
        for k, v in zip(("t", "x", "z", "vx", "vz", "theta", "omega", "m"),
                        (t_sim, *truth)):
            log[k].append(v)

    if final is None:
        final = truth
    for k in log:
        log[k] = np.asarray(log[k])

    out = _outcome(final, vehicle, m_start)
    out.update({
        "status": "flown", "truth": log, "replan": replan,
        "cold_iters": cold_iters, "cold_time": cold_time,
        "n_replans": n_replans, "n_failed_replans": n_failed,
        "sim_time": t_sim,
        "mean_solve_time": float(np.mean(replan["solve_time"]))
        if replan["solve_time"] else float("nan"),
        "max_solve_time": float(np.max(replan["solve_time"]))
        if replan["solve_time"] else float("nan"),
        "mean_gap": float(np.nanmean(replan["gap"])) if replan["gap"]
        else float("nan"),
        "guidance_dt": guidance_dt, "budget": budget,
    })
    if keep_path:
        y_arr = np.asarray(fine_y)
        # Trim anything the integrator carried below the pad on the last cycle.
        above = np.flatnonzero(y_arr[:, 1] >= 0.0)
        cut = int(above[-1]) + 1 if above.size else len(y_arr)
        out["path_t"] = np.asarray(fine_t)[:cut]
        out["path_y"] = y_arr[:cut]
    if verbose:
        print(f"\n  {out['fail_reason']}: {out['miss']:.2f} m at "
              f"{out['speed']:.2f} m/s, {out['fuel']:,.0f} kg burned, "
              f"{n_replans} replans, mean solve {out['mean_solve_time']:.3f}s")
    return out


# ======================================================================
def run_open_loop(
    vehicle=None, aero=None, N=40, guidance_dt=0.5,
    x0=0.0, z0=Z0_NOM, vx0=0.0, vz0=VZ0_NOM, theta0_deg=THETA0_NOM,
    omega0=0.0, gamma_gs_deg=75.0,
    wind=None, wind_sigma_x=6.0, wind_sigma_z=2.0, wind_tau=2.0, wind_seed=0,
    max_steps=200, keep_path=False, verbose=True, **_,
):
    """
    Day 9's strategy under the same gusts: plan once, fly it, never look again.

    Stepped on the same `guidance_dt` grid as the closed loop so the two see an
    identical gust sequence -- otherwise the comparison would be measuring the
    integration schedule rather than the guidance.
    """
    vehicle = vehicle or Vehicle6DoF()
    aero = aero if aero is not None else AeroConfig()
    gusts = wind or WindGusts(wind_sigma_x, wind_sigma_z, wind_tau, wind_seed)

    truth = np.array([x0, z0, vx0, vz0, np.radians(theta0_deg), omega0,
                      vehicle.m_wet], dtype=float)
    m_start = float(truth[6])
    t_guess = float(np.clip(2.0 * z0 / abs(vz0), 3.0, 20.0))

    plan = _plan(vehicle, aero, N, t_guess, _as_state(truth), max_iter=30,
                 gamma_gs_deg=gamma_gs_deg)
    if plan.get("status") == "failed":
        return {"status": "no initial plan"}

    t_f = float(plan["t_f"])
    log = {"t": [0.0], "x": [truth[0]], "z": [truth[1]], "vx": [truth[2]],
           "vz": [truth[3]], "theta": [truth[4]], "omega": [truth[5]],
           "m": [truth[6]]}

    t_sim, final = 0.0, None
    fine_t, fine_y = ([0.0], [truth.copy()]) if keep_path else (None, None)
    for _ in range(max_steps):
        if t_sim >= t_f - 1e-9:
            break
        dt = min(guidance_dt, t_f - t_sim)
        wx, wz = gusts.step(dt)
        y = _fly(truth, plan, t_sim, dt, vehicle, aero, (wx, wz))
        if keep_path:
            sub = np.linspace(t_sim, t_sim + dt, len(y))
            fine_t.extend(sub[1:].tolist())
            fine_y.extend(list(y[1:]))
        hit = _ground_crossing(y)
        if hit is not None:
            final = hit
            t_sim += dt
            break
        truth = y[-1]
        t_sim += dt
        for key, v in zip(("t", "x", "z", "vx", "vz", "theta", "omega", "m"),
                          (t_sim, *truth)):
            log[key].append(v)

    if final is None:
        final = truth
    for key in log:
        log[key] = np.asarray(log[key])

    out = _outcome(final, vehicle, m_start)
    out.update({"status": "flown", "truth": log, "sim_time": t_sim,
                "plan": plan, "n_replans": 0})
    if keep_path:
        y_arr = np.asarray(fine_y)
        above = np.flatnonzero(y_arr[:, 1] >= 0.0)
        cut = int(above[-1]) + 1 if above.size else len(y_arr)
        out["path_t"] = np.asarray(fine_t)[:cut]
        out["path_y"] = y_arr[:cut]
    if verbose:
        print(f"  open loop: {out['fail_reason']}: {out['miss']:.2f} m at "
              f"{out['speed']:.2f} m/s, {out['fuel']:,.0f} kg burned")
    return out


# ======================================================================
def plot_comparison(cl, ol, save_path=None):
    """Six panels: the two strategies under one gust sequence."""
    save_path = save_path or os.path.join(RESULTS, "day10_closed_loop.png")
    c, o = cl["truth"], ol["truth"]
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle("Day 10: closed-loop guidance against one plan flown blind, "
                 "identical gusts", fontsize=14)

    a = ax[0, 0]
    a.plot(o["x"], o["z"], lw=2, color="tab:red", label="open loop")
    a.plot(c["x"], c["z"], lw=2, color="tab:blue", label="closed loop")
    a.plot(0, 0, "k^", ms=13, label="pad")
    a.set_xlabel("Downrange [m]"); a.set_ylabel("Altitude [m]")
    a.set_title("Trajectory"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot(o["t"], np.hypot(o["vx"], o["vz"]), lw=2, color="tab:red",
           label="open loop")
    a.plot(c["t"], np.hypot(c["vx"], c["vz"]), lw=2, color="tab:blue",
           label="closed loop")
    a.axhline(SPEED_TOL_MS, color="k", ls=":", alpha=0.6, label="tolerance")
    a.set_xlabel("Time [s]"); a.set_ylabel("Speed [m/s]")
    a.set_title("Speed -- Day 9's failure mode"); a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[0, 2]
    labels = ["miss [m]", "arrival [m/s]"]
    xs = np.arange(2)
    a.bar(xs - 0.18, [ol["miss"], ol["speed"]], 0.36, color="tab:red",
          label="open loop")
    a.bar(xs + 0.18, [cl["miss"], cl["speed"]], 0.36, color="tab:blue",
          label="closed loop")
    a.axhline(MISS_TOL_M, color="k", ls=":", alpha=0.6)
    a.set_xticks(xs); a.set_xticklabels(labels)
    a.set_title("Touchdown"); a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")
    for i, (ov, cv) in enumerate(((ol["miss"], cl["miss"]),
                                  (ol["speed"], cl["speed"]))):
        a.text(i - 0.18, ov, f"{ov:.1f}", ha="center", va="bottom", fontsize=8)
        a.text(i + 0.18, cv, f"{cv:.1f}", ha="center", va="bottom", fontsize=8)

    r = cl["replan"]
    a = ax[1, 0]
    a.plot(r["t"], r["gap"], "o-", lw=1.5, ms=3, color="tab:purple")
    a.set_xlabel("Time [s]"); a.set_ylabel("[m]")
    a.set_title("Tracking gap the replan absorbs"); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(r["t"], r["solve_time"], "o-", lw=1.5, ms=3, color="tab:green")
    a.axhline(cl["guidance_dt"], color="r", ls="--", alpha=0.7,
              label=f"cycle {cl['guidance_dt']}s")
    a.set_xlabel("Time [s]"); a.set_ylabel("[s]")
    a.set_title(f"Replan cost, {cl['budget']}-iteration budget")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 2]
    a.plot(r["t"], r["wind_x"], lw=1.5, color="tab:orange")
    a.axhline(0, color="k", lw=0.5)
    a.set_xlabel("Time [s]"); a.set_ylabel("[m/s]")
    a.set_title("Cross-wind gust (both flew this)"); a.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("CLOSED-LOOP GUIDANCE")
    print("=" * 70)
    cl = run_closed_loop(wind_seed=7, verbose=True)
    print()
    ol = run_open_loop(wind_seed=7, verbose=True)
    if cl.get("status") == "flown" and ol.get("status") == "flown":
        print(f"\n  miss    {ol['miss']:8.2f} m   -> {cl['miss']:8.2f} m")
        print(f"  arrival {ol['speed']:8.2f} m/s -> {cl['speed']:8.2f} m/s")
        print(f"  fuel    {ol['fuel']:8,.0f} kg  -> {cl['fuel']:8,.0f} kg "
              f"({cl['fuel'] - ol['fuel']:+,.0f})")
        plot_comparison(cl, ol)
    print()
