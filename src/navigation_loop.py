"""
Guidance flown on an estimate rather than on the truth.

Day 10 closed the loop but handed the solver the exact state, which is the one
thing a real vehicle never has. It also recorded what happens when you remove
that privilege naively: feeding raw noisy readings straight into a
re-optimisation that is bang-bang by construction, 3 m of position noise
produced a 109 m worst-case miss and 8 m produced an 84 m median miss and ten
tonnes of propellant against a nominal six. The conclusion was that there needs
to be a filter between the estimate and the solver. This is that filter, wired
in.

Three modes fly the identical wind and the identical sensor noise, so the
comparison isolates the estimator and nothing else:

    truth   guidance reads the true state -- Day 10's privileged case, kept
            as the ceiling that no estimator can beat
    ekf     guidance reads the filter's estimate
    naive   guidance reads the newest raw sensor reading, held between
            updates -- no fusion, no dynamics, just the last thing measured

The filter runs faster than guidance, at `ekf_dt`, so it can absorb 20 Hz
attitude readings between 2 Hz replans. Guidance samples whatever the estimator
believes at the instant it needs it.
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

from src.closed_loop import (                                  # noqa: E402
    WindGusts, _plan, _as_state, _fly, _ground_crossing, _outcome,
    Z0_NOM, VZ0_NOM, THETA0_NOM, MISS_TOL_M, SPEED_TOL_MS,
)
from src.warm_start import shift_reference                     # noqa: E402
from src.sensors import SensorConfig, SensorSuite              # noqa: E402
from src.ekf import EKF, default_process_noise                 # noqa: E402
from src.dynamics_6dof import Vehicle6DoF                      # noqa: E402
from src.aero import AeroConfig                                # noqa: E402

MODES = ("truth", "ekf", "naive")


def run_navigation(
    mode="ekf", vehicle=None, aero=None, N=40, guidance_dt=0.5, budget=3,
    ekf_dt=0.05,
    x0=0.0, z0=Z0_NOM, vx0=0.0, vz0=VZ0_NOM, theta0_deg=THETA0_NOM,
    omega0=0.0, gamma_gs_deg=75.0,
    wind_sigma_x=6.0, wind_sigma_z=2.0, wind_tau=2.0, wind_seed=0,
    sensors=None, sensor_seed=0, q_scale=1.0,
    max_steps=200, keep_path=False, verbose=True,
):
    """
    Fly the descent with guidance reading `mode`.

    Wind and sensor streams are seeded independently of the mode, so the three
    modes see identical realisations and any difference between them is the
    estimator.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    vehicle = vehicle or Vehicle6DoF()
    aero = aero if aero is not None else AeroConfig()
    cfg = sensors or SensorConfig()
    gusts = WindGusts(wind_sigma_x, wind_sigma_z, wind_tau, wind_seed)
    suite = SensorSuite(cfg, seed=sensor_seed)

    truth = np.array([x0, z0, vx0, vz0, np.radians(theta0_deg), omega0,
                      vehicle.m_wet], dtype=float)
    m_start = float(truth[6])

    # The filter starts where the vehicle says it is, which is itself a noisy
    # reading rather than the truth -- otherwise the estimator would be handed
    # the answer at t = 0.
    first = suite.due(0.0, truth)
    x_hat = truth[:6].copy()
    if "nav" in first:
        x_hat[:4] = first["nav"]
    if "att" in first:
        x_hat[4:6] = first["att"]
    ekf = EKF(x_hat, truth[6], vehicle, aero, Q=default_process_noise(q_scale))
    last_raw = x_hat.copy()

    def estimate():
        if mode == "truth":
            return _as_state(truth)
        if mode == "ekf":
            return ekf.state()
        s = _as_state(np.concatenate([last_raw, [ekf.m]]))
        return s

    t_guess = float(np.clip(2.0 * z0 / abs(vz0), 3.0, 20.0))
    plan = _plan(vehicle, aero, N, t_guess, estimate(), max_iter=30,
                 gamma_gs_deg=gamma_gs_deg)
    if plan.get("status") == "failed":
        return {"status": "no initial plan", "mode": mode}

    log = {"t": [0.0], "err_pos": [float(np.hypot(*(x_hat[:2] - truth[:2])))],
           "err_vel": [float(np.hypot(*(x_hat[2:4] - truth[2:4])))],
           "err_theta": [float(abs(x_hat[4] - truth[4]))],
           "sigma_pos": [ekf.position_sigma()]}
    replan = {"t": [], "solve_time": [], "gap": []}
    fine_t, fine_y = ([0.0], [truth.copy()]) if keep_path else (None, None)

    t_sim, age, final, n_replans = 0.0, 0.0, None, 0
    for _ in range(max_steps):
        est = estimate()
        ref, remaining, gap = shift_reference(plan, age, est, N, vehicle)
        t0 = timer.time()
        if ref is not None:
            new = _plan(vehicle, aero, N, remaining, est, budget, ref=ref,
                        gamma_gs_deg=gamma_gs_deg)
            if new.get("status") != "failed":
                plan, age = new, 0.0
        solve_time = timer.time() - t0
        n_replans += 1
        replan["t"].append(t_sim)
        replan["solve_time"].append(solve_time)
        replan["gap"].append(gap)

        wx, wz = gusts.step(guidance_dt)
        y = _fly(truth, plan, age, guidance_dt, vehicle, aero, (wx, wz))
        if keep_path:
            sub = np.linspace(t_sim, t_sim + guidance_dt, len(y))
            fine_t.extend(sub[1:].tolist())
            fine_y.extend(list(y[1:]))

        # Run the estimator across the interval the vehicle just flew, at its
        # own rate, folding in whatever the instruments reported on the way.
        n_sub = max(int(round(guidance_dt / ekf_dt)), 1)
        dt_sub = guidance_dt / n_sub
        k_ctrl = int(np.clip(age / (plan["t_f"] / len(plan["sigma"])), 0,
                             len(plan["sigma"]) - 1))
        sig_c, del_c = float(plan["sigma"][k_ctrl]), float(plan["delta"][k_ctrl])
        for i in range(n_sub):
            ekf.predict(sig_c, del_c, dt_sub)
            t_read = t_sim + (i + 1) * dt_sub
            true_here = y[min(int((i + 1) * (len(y) - 1) / n_sub), len(y) - 1)]
            got = suite.due(t_read, true_here)
            if "att" in got:
                ekf.update_attitude(got["att"], cfg.R_att())
                last_raw[4:6] = got["att"]
            if "nav" in got:
                ekf.update_nav(got["nav"], cfg.R_nav())
                last_raw[:4] = got["nav"]

        hit = _ground_crossing(y)
        truth_end = hit if hit is not None else y[-1]
        t_sim += guidance_dt
        age += guidance_dt
        est_now = ekf.x if mode != "naive" else last_raw
        log["t"].append(t_sim)
        log["err_pos"].append(float(np.hypot(*(est_now[:2] - truth_end[:2]))))
        log["err_vel"].append(float(np.hypot(*(est_now[2:4] - truth_end[2:4]))))
        log["err_theta"].append(float(abs(est_now[4] - truth_end[4])))
        log["sigma_pos"].append(ekf.position_sigma())

        if hit is not None:
            final = hit
            break
        truth = y[-1]

    if final is None:
        final = truth
    for k in log:
        log[k] = np.asarray(log[k])

    out = _outcome(final, vehicle, m_start)
    out.update({
        "status": "flown", "mode": mode, "est": log, "replan": replan,
        "n_replans": n_replans, "sim_time": t_sim,
        "mean_est_pos_err": float(np.mean(log["err_pos"])),
        "max_est_pos_err": float(np.max(log["err_pos"])),
        "mean_est_vel_err": float(np.mean(log["err_vel"])),
        "final_est_pos_err": float(log["err_pos"][-1]),
        "mean_solve_time": float(np.mean(replan["solve_time"])),
        "guidance_dt": guidance_dt,
    })
    if keep_path:
        y_arr = np.asarray(fine_y)
        above = np.flatnonzero(y_arr[:, 1] >= 0.0)
        cut = int(above[-1]) + 1 if above.size else len(y_arr)
        out["path_t"] = np.asarray(fine_t)[:cut]
        out["path_y"] = y_arr[:cut]
    if verbose:
        print(f"  {mode:>6}: {out['fail_reason']:<34} "
              f"{out['miss']:6.2f} m at {out['speed']:6.2f} m/s, "
              f"{out['fuel']:7,.0f} kg, mean est error "
              f"{out['mean_est_pos_err']:6.2f} m")
    return out


def compare(seed=0, sensor_seed=0, verbose=True, **kw):
    """All three modes on one wind and one sensor realisation."""
    return {m: run_navigation(mode=m, wind_seed=seed, sensor_seed=sensor_seed,
                              keep_path=True, verbose=verbose, **kw)
            for m in MODES}


# ======================================================================
def plot_navigation(runs, save_path=None):
    """Six panels: what the estimator does and what it costs."""
    save_path = save_path or os.path.join(RESULTS, "day11_navigation.png")
    colours = {"truth": "tab:green", "ekf": "tab:blue", "naive": "tab:red"}
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle("Day 11: guidance flown on an estimate, identical wind and "
                 "sensor noise", fontsize=14)

    a = ax[0, 0]
    for m, r in runs.items():
        if r.get("status") == "flown":
            y = r["path_y"]
            a.plot(y[:, 0], np.maximum(y[:, 1], 0), lw=2, color=colours[m],
                   label=m)
    a.plot(0, 0, "k^", ms=13)
    a.set_xlabel("Downrange [m]"); a.set_ylabel("Altitude [m]")
    a.set_title("Trajectory"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 1]
    for m, r in runs.items():
        if r.get("status") == "flown" and m != "truth":
            a.plot(r["est"]["t"], r["est"]["err_pos"], lw=2, color=colours[m],
                   label=f"{m} error")
    r_ekf = runs.get("ekf", {})
    if r_ekf.get("status") == "flown":
        a.plot(r_ekf["est"]["t"], r_ekf["est"]["sigma_pos"], "--", lw=1.5,
               color="k", alpha=0.7, label="EKF 1-sigma")
    a.set_xlabel("Time [s]"); a.set_ylabel("[m]")
    a.set_title("Estimation error, and whether the filter knows it")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 2]
    for m, r in runs.items():
        if r.get("status") == "flown" and m != "truth":
            a.plot(r["est"]["t"], np.degrees(r["est"]["err_theta"]), lw=2,
                   color=colours[m], label=m)
    a.set_xlabel("Time [s]"); a.set_ylabel("[deg]")
    a.set_title("Attitude estimation error"); a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 0]
    ms = [m for m in MODES if runs.get(m, {}).get("status") == "flown"]
    xs = np.arange(len(ms))
    a.bar(xs - 0.2, [runs[m]["miss"] for m in ms], 0.4, color="tab:purple",
          label="miss [m]")
    a.bar(xs + 0.2, [runs[m]["speed"] for m in ms], 0.4, color="tab:orange",
          label="arrival [m/s]")
    a.axhline(MISS_TOL_M, color="k", ls=":", alpha=0.6)
    a.set_xticks(xs); a.set_xticklabels(ms)
    a.set_title("Touchdown"); a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")
    for i, m in enumerate(ms):
        a.text(i - 0.2, runs[m]["miss"], f"{runs[m]['miss']:.1f}",
               ha="center", va="bottom", fontsize=8)
        a.text(i + 0.2, runs[m]["speed"], f"{runs[m]['speed']:.1f}",
               ha="center", va="bottom", fontsize=8)

    a = ax[1, 1]
    for m in ms:
        if m != "truth":
            a.bar(m, runs[m]["mean_est_pos_err"], color=colours[m])
            a.text(m, runs[m]["mean_est_pos_err"],
                   f"{runs[m]['mean_est_pos_err']:.2f}", ha="center",
                   va="bottom", fontsize=9)
    a.set_ylabel("[m]"); a.set_title("Mean position estimation error")
    a.grid(alpha=0.3, axis="y")

    a = ax[1, 2]
    a.bar(ms, [runs[m]["fuel"] for m in ms],
          color=[colours[m] for m in ms])
    for i, m in enumerate(ms):
        a.text(i, runs[m]["fuel"], f"{runs[m]['fuel']:,.0f}", ha="center",
               va="bottom", fontsize=9)
    a.set_ylabel("[kg]"); a.set_title("Propellant used")
    a.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nNavigation plot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("GUIDANCE ON AN ESTIMATE")
    print("=" * 70)
    print(SensorConfig().summary())
    print()
    runs = compare(seed=7, sensor_seed=3)
    ok = {m: r for m, r in runs.items() if r.get("status") == "flown"}
    if ok:
        print(f"\n  {'mode':>6} {'miss':>8} {'arrival':>9} {'fuel':>9} "
              f"{'mean est err':>13}")
        for m, r in ok.items():
            print(f"  {m:>6} {r['miss']:>7.2f}m {r['speed']:>8.2f}m/s "
                  f"{r['fuel']:>8,.0f}kg {r['mean_est_pos_err']:>12.2f}m")
        plot_navigation(ok)
    print()
