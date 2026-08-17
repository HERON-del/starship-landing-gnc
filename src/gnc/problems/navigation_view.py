"""
Day 11 — flying on an estimate, in the viewer.

Day 10's entry lets guidance read the true state, which is the one thing a real
vehicle never has. This one takes that privilege away and puts a filter in its
place. Three modes fly identical wind and identical sensor noise, so switching
between them isolates the estimator and nothing else:

    truth   guidance reads the true state -- the ceiling, kept for reference
    ekf     guidance reads the filter's fused estimate
    naive   guidance reads the newest raw sensor reading, held between updates

**The naive mode is Day 10's recorded failure, reproduced.** That day fed raw
estimates straight into a re-optimisation that is bang-bang by construction and
watched it produce a 109 m worst-case miss and ten tonnes of propellant against
a nominal six. Select `naive` here and you can watch it happen: on the default
seed it misses by 92 m and burns 11 t, against 1.7 m and 5.8 t filtered.

**What the filter is worth is the tail, not the median.** Over four wind and
sensor realisations the EKF estimates three to four times better every single
time, and the worst-case miss falls from 210 m unfiltered to 6.7 m filtered.
The median landing is a closer-run thing, because the naive error is close to
zero-mean sensor noise and successive replans average much of it out, while a
filter's error is correlated -- a filter lags, and a lag biases every replan the
same way.

**The process-noise slider is the one to move, and it does not do what its
name suggests.** `Q` is how much the filter distrusts its own dynamics. Tuned
a hundred times too tight it lands 34 m out; at the measured optimum it lands
3 m out; and the mean estimation error barely moves between the two, 2.22 m
against 1.60 m. A lagging estimate and a noisy one look alike by that measure
and behave nothing alike in a control loop. That was a real defect in this
code, found by the sweep and fixed, rather than a hypothetical.

Every solve here flies a full descent per mode, so it costs a few seconds.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.navigation_loop import run_navigation, MODES
from src.sensors import SensorConfig
from src.closed_loop import (
    MISS_TOL_M, SPEED_TOL_MS, Z0_NOM, VZ0_NOM, THETA0_NOM,
)
from src.dynamics_6dof import Vehicle6DoF

from ..registry import Problem, register
from ..types import Param, Series, Trajectory, quats_from_pitch


@register
class NavigationGuidance(Problem):
    slug = "navigation"
    title = "Navigation: Guidance on an Estimate (EKF)"
    summary = ("The vehicle no longer knows where it is. A Kalman filter has "
               "to work it out from noisy sensors.")
    phase = "Day 11"
    scene_scale = 700.0
    # A flown trajectory, allowed to miss and allowed to arrive moving.
    enforces_terminal_state = False

    def params(self) -> list[Param]:
        return [
            Param("mode", "Guidance reads", "ekf", kind="choice",
                  choices=["ekf", "naive", "truth"], group="Navigation",
                  help="'truth' is Day 10's privilege and the ceiling. "
                       "'naive' is the raw newest reading, which is the "
                       "failure Day 10 recorded. 'ekf' is the filter."),
            Param("q_scale", "Process noise Q", 1.0, min=0.01, max=100.0,
                  step=0.01, group="Navigation",
                  help="How much the filter distrusts its own dynamics. The "
                       "default is the measured optimum; a hundred times "
                       "tighter lands 34 m out instead of 3, while the mean "
                       "estimation error barely changes."),
            Param("ekf_dt", "Filter step", 0.05, min=0.01, max=0.25,
                  step=0.01, unit="s", group="Navigation",
                  help="The filter runs faster than guidance so it can absorb "
                       "20 Hz attitude readings between 2 Hz replans."),

            Param("nav_rate_hz", "Nav sensor rate", 5.0, min=1.0, max=20.0,
                  step=1.0, unit="Hz", group="Sensors",
                  help="Position aiding. Below 2 Hz the loop is steering on "
                       "stale position and the miss jumps to 175 m; above it, "
                       "landing accuracy saturates almost at once."),
            Param("sigma_x", "Nav noise, downrange", 3.0, min=0.0, max=15.0,
                  step=0.5, unit="m", group="Sensors"),
            Param("sigma_z", "Nav noise, altitude", 2.0, min=0.0, max=15.0,
                  step=0.5, unit="m", group="Sensors"),
            Param("nav_enabled", "Position aiding", True, kind="bool",
                  group="Sensors",
                  help="Switch off to dead-reckon position from the dynamics "
                       "alone. Estimation error goes from 1.6 m to 7.4 m - "
                       "attitude cannot observe where the vehicle is."),
            Param("att_rate_hz", "Attitude sensor rate", 20.0, min=2.0,
                  max=50.0, step=1.0, unit="Hz", group="Sensors"),
            Param("omega_bias", "Gyro bias", 0.0, min=0.0, max=3.0, step=0.1,
                  unit="deg/s", group="Sensors",
                  help="A constant offset the filter has no state for, so it "
                       "cannot remove it. Miss runs 3.1, 3.4, 10.2 and 13.7 m "
                       "at 0, 0.5, 1 and 2 deg/s while the position estimate "
                       "still looks healthy - the argument for a bias state."),

            Param("wind_sigma_x", "Cross-wind (3 sigma)", 6.0, min=0.0,
                  max=15.0, step=1.0, unit="m/s", group="Disturbance"),
            Param("wind_seed", "Wind seed", 7, kind="int", min=0, max=999,
                  step=1, group="Disturbance"),
            Param("sensor_seed", "Sensor seed", 3, kind="int", min=0, max=999,
                  step=1, group="Disturbance"),

            Param("z0", "Entry altitude", Z0_NOM, min=300.0, max=560.0,
                  step=20.0, unit="m", group="Entry state"),
            Param("vz0", "Entry descent rate", VZ0_NOM, min=-160.0,
                  max=-100.0, step=5.0, unit="m/s", group="Entry state"),
            Param("theta0_deg", "Entry pitch", THETA0_NOM, min=0.0, max=45.0,
                  step=5.0, unit="deg", group="Entry state"),

            Param("N", "Nodes per replan", 40, kind="int", min=25, max=70,
                  step=5, group="Solver"),
            Param("guidance_dt", "Guidance cycle", 0.5, min=0.25, max=1.0,
                  step=0.25, unit="s", group="Solver"),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)
        cfg = SensorConfig(
            nav_rate_hz=float(p["nav_rate_hz"]),
            sigma_x=float(p["sigma_x"]), sigma_z=float(p["sigma_z"]),
            att_rate_hz=float(p["att_rate_hz"]),
            omega_bias=np.radians(float(p["omega_bias"])),
            nav_enabled=bool(p["nav_enabled"]),
        )
        kw = dict(N=int(p["N"]), z0=float(p["z0"]), vz0=float(p["vz0"]),
                  theta0_deg=float(p["theta0_deg"]),
                  guidance_dt=float(p["guidance_dt"]),
                  ekf_dt=float(p["ekf_dt"]),
                  q_scale=float(p["q_scale"]),
                  wind_sigma_x=float(p["wind_sigma_x"]),
                  wind_seed=int(p["wind_seed"]),
                  sensors=cfg, sensor_seed=int(p["sensor_seed"]),
                  verbose=False)

        t0 = time.perf_counter()
        try:
            shown = run_navigation(mode=str(p["mode"]), keep_path=True, **kw)
            # The other two, for the comparison in the panel. Same wind, same
            # sensor stream -- otherwise this would be comparing realisations.
            others = {m: run_navigation(mode=m, keep_path=False, **kw)
                      for m in MODES if m != str(p["mode"])}
        except Exception as exc:      # noqa: BLE001
            return _failed("error", [f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        if shown.get("status") != "flown":
            return _failed("infeasible", [
                "No initial plan exists from this entry state. The feasible "
                "band is narrow and one-sided - see the Day 9 problem."
            ])
        return _trajectory(shown, others, p, elapsed)


# ----------------------------------------------------------------------
def _trajectory(r, others, p, elapsed) -> Trajectory:
    t = np.asarray(r["path_t"])
    y = np.asarray(r["path_y"])
    x, z, vx, vz, th, om, m = (y[:, i] for i in range(7))
    z = np.maximum(z, 0.0)
    n = len(t)
    veh = Vehicle6DoF()

    mag = m[:-1] * 9.80665
    thrust = np.column_stack([mag * np.sin(th[:-1]), mag * np.cos(th[:-1]),
                              np.zeros(n - 1)])

    # Estimation error is logged per guidance cycle; stretch it onto the fine
    # grid so it can ride alongside the trajectory rather than needing its own.
    est = r["est"]
    err_pos = np.interp(t, est["t"], est["err_pos"])
    err_vel = np.interp(t, est["t"], est["err_vel"])
    sig_pos = np.interp(t, est["t"], est["sigma_pos"])

    diag = {
        "mode": r["mode"],
        "miss_m": float(r["miss"]),
        "arrival_ms": float(r["speed"]),
        "fuel_kg": float(r["fuel"]),
        "margin_kg": float(r["margin"]),
        "mean_est_pos_err_m": float(r["mean_est_pos_err"]),
        "max_est_pos_err_m": float(r["max_est_pos_err"]),
        "mean_est_vel_err_ms": float(r["mean_est_vel_err"]),
        "replans": int(r["n_replans"]),
        "mean_replan_s": float(r["mean_solve_time"]),
    }
    for k, o in others.items():
        if o.get("status") == "flown":
            diag[f"{k}_miss_m"] = float(o["miss"])
            diag[f"{k}_arrival_ms"] = float(o["speed"])
            diag[f"{k}_fuel_kg"] = float(o["fuel"])
            diag[f"{k}_mean_est_err_m"] = float(o["mean_est_pos_err"])

    return Trajectory(
        t_state=t.tolist(),
        t_control=t[:-1].tolist(),
        position=np.column_stack([x, z, np.zeros(n)]).tolist(),
        velocity=np.column_stack([vx, vz, np.zeros(n)]).tolist(),
        thrust=thrust.tolist(),
        attitude=quats_from_pitch(th).tolist(),
        series=[
            Series("altitude", "Altitude", "m", z.tolist()),
            Series("downrange", "Downrange", "m", x.tolist()),
            Series("speed", "Speed", "m/s", np.hypot(vx, vz).tolist()),
            Series("pitch", "Pitch from vertical", "deg",
                   np.degrees(th).tolist()),
            Series("est_err_pos", "Position estimate error", "m",
                   err_pos.tolist()),
            Series("est_sigma_pos", "Filter's own 1-sigma", "m",
                   sig_pos.tolist()),
            Series("est_err_vel", "Velocity estimate error", "m/s",
                   err_vel.tolist()),
            Series("mass", "Vehicle mass", "kg", m.tolist()),
        ],
        status=r["fail_reason"],
        feasible=True,
        cost=float(r["fuel"]),
        solve_time_ms=elapsed,
        solver=f"guidance on {r['mode']}, {r['n_replans']} replans",
        thrust_max=veh.T_max,
        notes=_notes(r, others, p),
        diagnostics=diag,
    )


def _notes(r, others, p) -> list[str]:
    mode = r["mode"]
    notes = [
        f"Guidance is reading the {mode} state. Outcome: {r['fail_reason']} -- "
        f"{r['miss']:.2f} m from the pad at {r['speed']:.2f} m/s, "
        f"{r['fuel']:,.0f} kg burned. Scoring is Day 9's: within "
        f"{MISS_TOL_M:.0f} m and {SPEED_TOL_MS:.0f} m/s counts as landed."
    ]

    comp = "  ".join(
        f"{k} {o['miss']:.2f} m / {o['speed']:.1f} m/s"
        for k, o in others.items() if o.get("status") == "flown")
    if comp:
        notes.append(
            f"Same wind, same sensor noise, the other ways: {comp}. Switch the "
            f"'guidance reads' control to fly them."
        )

    if mode != "truth":
        notes.append(
            f"Estimation error averaged {r['mean_est_pos_err']:.2f} m and "
            f"peaked at {r['max_est_pos_err']:.2f} m. The telemetry strip "
            f"carries it alongside the filter's own 1-sigma, which is the "
            f"more interesting pair: a filter that is wrong and knows it is "
            f"recoverable, one that is wrong and confident is not."
        )
    if mode == "naive":
        notes.append(
            "This is Day 10's recorded failure reproduced. There is no filter "
            "here at all -- guidance steers on whatever the sensor said most "
            "recently, and that goes straight into a re-optimisation that is "
            "bang-bang by construction. Day 10 measured 3 m of position noise "
            "producing a 109 m worst case and ten tonnes of propellant."
        )
    if mode == "ekf" and float(p["q_scale"]) < 0.05:
        notes.append(
            f"Q is at {p['q_scale']:.2f} of its tuned value, so the filter is "
            f"over-trusting its own dynamics through gusts it cannot see. The "
            f"error that produces is a lag rather than noise, and a lag biases "
            f"every replan the same way where noise partly averages out. "
            f"Measured at 0.01, this lands 34 m out against 3 m tuned -- while "
            f"the mean estimation error barely moves."
        )
    if not bool(p["nav_enabled"]):
        notes.append(
            "Position aiding is off, so the filter is dead-reckoning from its "
            "own dynamics model with only attitude to correct it. Nothing "
            "bounds the drift: estimation error goes from 1.6 m to 7.4 m. "
            "Attitude is not observable into position."
        )
    if float(p["omega_bias"]) > 0.0:
        notes.append(
            f"The rate gyro carries a {p['omega_bias']:.1f} deg/s bias and the "
            f"filter has no state for it, so it cannot be averaged away and is "
            f"taken as truth. Miss runs 3.1, 3.4, 10.2 and 13.7 m at 0, 0.5, 1 "
            f"and 2 deg/s while the position estimate still looks healthy - "
            f"which is exactly the argument for augmenting the state vector."
        )
    return notes


def _failed(status: str, notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status=status, feasible=False,
        solver="navigation", notes=notes,
    )
