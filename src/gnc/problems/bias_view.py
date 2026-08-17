"""
Day 12 — estimating the sensor's own error, in the viewer.

Day 11's filter assumed its instruments were unbiased on average. Real rate
gyros are not: they carry a slowly drifting offset that reads `omega + b` every
single time, and no amount of averaging removes it, because it is not random
relative to itself. Day 11 measured what that costs and left it unfixed — the
miss ran 3.1, 3.4, 10.2 and 13.7 m at 0, 0.5, 1 and 2 deg/s of bias while the
position estimate stayed flat near 1.5 m, the filter reporting good health the
whole way down.

This entry adds the bias as a state and lets you watch the filter work it out.

**The telemetry strip is where the day lives.** It carries the true bias, the
filter's estimate of it, and the filter's own one-sigma on that estimate.
Switch the filter to `bias-blind` and the estimate stays flat at zero while the
truth sits at whatever you set — the blind filter's bias error simply *is* the
bias, because it estimates nothing. Switch to `bias-aware` and watch the
estimate climb to meet the truth over the first second or two, with its
uncertainty collapsing behind it.

**It pays downstream only when the bias is large.** At 4 deg/s the blind loop
misses by 17.5 m against 3.5 m aware; below about 2 deg/s the two are inside
the seed-to-seed noise. The descent lasts about five seconds and the bias takes
one to two of them to resolve, which is most of the reason — a third of the
flight is spent learning a constant.

**The filter's assumed drift rate is worth breaking.** Tell it the bias is ten
times steadier than it really is and it stops adapting; tell it the bias is ten
times noisier and it chases the rate channel instead of tracking. Both show up
as a bias estimate that never settles on the truth.

One thing this entry cannot show: Day 12 also estimates a second, simultaneous
sensor error — a velocity-channel bias — in an eight-state filter, resolving
both at once because they are seen by different instruments. That runs in
`tests/test_imu_bias.py` rather than here, because the closed-loop path carries
the gyro bias only.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.navigation_loop import run_navigation
from src.sensors import SensorConfig
from src.closed_loop import (
    MISS_TOL_M, SPEED_TOL_MS, Z0_NOM, VZ0_NOM, THETA0_NOM,
)
from src.dynamics_6dof import Vehicle6DoF

from ..registry import Problem, register
from ..types import Param, Series, Trajectory, quats_from_pitch


@register
class IMUBias(Problem):
    slug = "imu-bias"
    title = "IMU Bias Estimation"
    summary = ("The gyro is wrong by a constant. The filter has to notice, "
               "and estimate it.")
    phase = "Day 12"
    scene_scale = 700.0
    enforces_terminal_state = False

    def params(self) -> list[Param]:
        return [
            Param("bias_aware", "Filter", "bias-aware", kind="choice",
                  choices=["bias-aware", "bias-blind"], group="Filter",
                  help="Blind is Day 11's six-state filter, which has no "
                       "concept of a bias and folds it into its rate "
                       "estimate. Aware adds a seventh state for it."),
            Param("bias0_deg_s", "True gyro bias", 1.5, min=0.0, max=5.0,
                  step=0.25, unit="deg/s", group="Filter",
                  help="An uncalibrated MEMS gyro carries 0.5 to 2. The "
                       "downstream benefit only becomes clear above about 2 - "
                       "below that the two filters are inside the noise."),
            Param("bias_walk_deg_s", "True drift rate", 0.01, min=0.0,
                  max=0.2, step=0.01, group="Filter",
                  help="How fast the true bias wanders, per sqrt(s). Thermal "
                       "drift is slow, which is what makes it estimable."),
            Param("filter_bias_walk_deg_s", "Filter's assumed drift", 0.02,
                  min=0.0002, max=0.5, step=0.0002, group="Filter",
                  help="What the filter believes about that drift. Set it far "
                       "below the truth and the estimate stops adapting; far "
                       "above and it chases the rate channel."),

            Param("att_rate_hz", "Attitude sensor rate", 20.0, min=5.0,
                  max=50.0, step=5.0, unit="Hz", group="Sensors",
                  help="More readings separate a bias from a genuine rotation "
                       "sooner. Bias error runs 0.217 at 5 Hz and 0.117 at "
                       "20, and stops improving above that."),
            Param("nav_rate_hz", "Nav sensor rate", 5.0, min=1.0, max=20.0,
                  step=1.0, unit="Hz", group="Sensors"),

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
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)
        aware = str(p["bias_aware"]) == "bias-aware"
        cfg = SensorConfig(nav_rate_hz=float(p["nav_rate_hz"]),
                           att_rate_hz=float(p["att_rate_hz"]))
        kw = dict(mode="ekf", N=int(p["N"]), z0=float(p["z0"]),
                  vz0=float(p["vz0"]), theta0_deg=float(p["theta0_deg"]),
                  wind_sigma_x=float(p["wind_sigma_x"]),
                  wind_seed=int(p["wind_seed"]),
                  sensors=cfg, sensor_seed=int(p["sensor_seed"]),
                  bias0_deg_s=float(p["bias0_deg_s"]),
                  bias_walk_deg_s=float(p["bias_walk_deg_s"]),
                  filter_bias_walk_deg_s=float(
                      p["filter_bias_walk_deg_s"]),
                  verbose=False)

        t0 = time.perf_counter()
        try:
            shown = run_navigation(bias_aware=aware, keep_path=True, **kw)
            other = run_navigation(bias_aware=not aware, keep_path=False, **kw)
        except Exception as exc:      # noqa: BLE001
            return _failed("error", [f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        if shown.get("status") != "flown":
            return _failed("infeasible", [
                "No initial plan exists from this entry state. The feasible "
                "band is narrow and one-sided - see the Day 9 problem."
            ])
        return _trajectory(shown, other, aware, p, elapsed)


# ----------------------------------------------------------------------
def _trajectory(r, other, aware, p, elapsed) -> Trajectory:
    t = np.asarray(r["path_t"])
    y = np.asarray(r["path_y"])
    x, z, vx, vz, th, om, m = (y[:, i] for i in range(7))
    z = np.maximum(z, 0.0)
    n = len(t)
    veh = Vehicle6DoF()

    mag = m[:-1] * 9.80665
    thrust = np.column_stack([mag * np.sin(th[:-1]), mag * np.cos(th[:-1]),
                              np.zeros(n - 1)])

    est = r["est"]
    def onto(key):
        return np.interp(t, est["t"], est[key])

    b_true = np.degrees(onto("b_true"))
    b_est = np.degrees(onto("b_est"))
    b_sig = np.degrees(onto("b_sigma"))

    return Trajectory(
        t_state=t.tolist(),
        t_control=t[:-1].tolist(),
        position=np.column_stack([x, z, np.zeros(n)]).tolist(),
        velocity=np.column_stack([vx, vz, np.zeros(n)]).tolist(),
        thrust=thrust.tolist(),
        attitude=quats_from_pitch(th).tolist(),
        series=[
            Series("altitude", "Altitude", "m", z.tolist()),
            Series("speed", "Speed", "m/s", np.hypot(vx, vz).tolist()),
            Series("pitch", "Pitch from vertical", "deg",
                   np.degrees(th).tolist()),
            Series("bias_true", "Gyro bias, true", "deg/s", b_true.tolist()),
            Series("bias_est", "Gyro bias, estimated", "deg/s",
                   b_est.tolist()),
            Series("bias_sigma", "Bias estimate 1-sigma", "deg/s",
                   b_sig.tolist()),
            Series("est_err_pos", "Position estimate error", "m",
                   onto("err_pos").tolist()),
            Series("mass", "Vehicle mass", "kg", m.tolist()),
        ],
        status=r["fail_reason"],
        feasible=True,
        cost=float(r["fuel"]),
        solve_time_ms=elapsed,
        solver=("7-state, bias estimated" if aware
                else "6-state, bias invisible"),
        thrust_max=veh.T_max,
        notes=_notes(r, other, aware, p),
        diagnostics={
            "filter": "bias-aware" if aware else "bias-blind",
            "miss_m": float(r["miss"]),
            "arrival_ms": float(r["speed"]),
            "fuel_kg": float(r["fuel"]),
            "bias_true_final_deg_s": float(np.degrees(r["b_true_final"])),
            "bias_est_final_deg_s": float(np.degrees(r["b_est_final"])),
            "bias_err_final_deg_s": float(np.degrees(r["b_err_final"])),
            "bias_err_mean_deg_s": float(np.degrees(r["b_err_mean"])),
            "mean_est_pos_err_m": float(r["mean_est_pos_err"]),
            "other_filter_miss_m": float(other["miss"])
            if other.get("status") == "flown" else float("nan"),
            "other_filter_bias_err_deg_s": float(
                np.degrees(other["b_err_final"]))
            if other.get("status") == "flown" else float("nan"),
        },
    )


def _notes(r, other, aware, p) -> list[str]:
    which = "bias-aware" if aware else "bias-blind"
    b_true = np.degrees(r["b_true_final"])
    b_err = np.degrees(r["b_err_final"])
    notes = [
        f"Filter: {which}. Outcome: {r['fail_reason']} -- {r['miss']:.2f} m "
        f"from the pad at {r['speed']:.2f} m/s, {r['fuel']:,.0f} kg burned. "
        f"Scoring is Day 9's: within {MISS_TOL_M:.0f} m and "
        f"{SPEED_TOL_MS:.0f} m/s counts as landed."
    ]

    if aware:
        notes.append(
            f"The true bias ended at {b_true:.3f} deg/s and the filter "
            f"estimated it to within {b_err:.3f}. Watch the three bias traces "
            f"in the telemetry strip: the estimate climbs to meet the truth "
            f"over the first second or two and its one-sigma collapses behind "
            f"it. That is the bias becoming observable -- omega and the bias "
            f"look identical in any single reading and behave differently "
            f"over time, which is the only reason they can be separated."
        )
    else:
        notes.append(
            f"The estimate stays at zero because this filter has no state for "
            f"it, so its bias error simply *is* the bias: {b_err:.3f} deg/s "
            f"against a true {b_true:.3f}. The rate error goes into the "
            f"omega estimate instead, where nothing distinguishes it from the "
            f"vehicle genuinely rotating a little faster than the model says."
        )

    if other.get("status") == "flown":
        notes.append(
            f"The other filter, same wind and same sensor noise: "
            f"{other['miss']:.2f} m at {other['speed']:.2f} m/s, bias error "
            f"{np.degrees(other['b_err_final']):.3f} deg/s. Switch the filter "
            f"control to fly it."
        )

    if float(p["bias0_deg_s"]) < 2.0:
        notes.append(
            "At this bias the two filters are close, and that is the honest "
            "result rather than a disappointing one: the benefit only becomes "
            "clear above about 2 deg/s, where the blind loop misses by 17.5 m "
            "against 3.5 m aware. The descent is five seconds and the bias "
            "takes one to two of them to resolve, so a third of the flight "
            "goes on learning a constant."
        )

    ratio = float(p["filter_bias_walk_deg_s"]) / max(
        float(p["bias_walk_deg_s"]), 1e-9)
    if aware and (ratio < 0.1 or ratio > 20.0):
        notes.append(
            f"The filter's assumed drift is {ratio:.3g}x the true one, which "
            f"is a deliberate misspecification. Too low and the filter stops "
            f"adapting and its estimate lags; too high and it chases the rate "
            f"channel rather than tracking. Measured, telling it the bias "
            f"drifts twenty times too fast takes the bias error from 0.117 to "
            f"0.913 deg/s."
        )

    notes.append(
        "Not shown here: the same pattern extends to a second, simultaneous "
        "sensor error. An eight-state filter resolves a gyro bias and a "
        "velocity-channel bias at once -- 0.135 deg/s and 0.100 m/s -- because "
        "they are seen by different instruments. Two errors on the same "
        "channel would not be separable at all."
    )
    return notes


def _failed(status: str, notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status=status, feasible=False,
        solver="bias estimation", notes=notes,
    )
