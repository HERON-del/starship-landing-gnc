"""
Week 1 — 3-DoF convex powered-descent guidance.

Full translational motion in 3-D: the vehicle starts downrange and off-axis and
must fly to the pad at the origin with zero velocity. Everything here stays
convex, which is the whole point — it solves in milliseconds and the answer is
provably global.

Constraints, and why each one is still convex:

    ||a[k]|| <= a_max                       second-order cone
    a_y[k]   >= ||a[k]|| cos(theta_max)     thrust-tilt cone (gimbal limit)
    r_y[k]   >= tan(gamma) ||r_xz[k]||      glideslope cone, keeps the approach
                                            above terrain and inside the
                                            sensor's field of view

What is deliberately *missing* is the minimum-throttle bound ||a|| >= a_min.
That one is genuinely non-convex, and the change of variables that rescues it
(lossless convexification, Acikmese & Ploen 2007) is the Week 2 topic.

Note the contrast with the 1-D problem: minimising sum(||a||) here is *not*
degenerate. The vector sum of accelerations is pinned by the terminal
constraint, but sum(||a||) >= ||sum(a)|| leaves real slack, so the optimiser
genuinely chooses.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np

from ..registry import Problem, register
from ..types import Param, Series, Trajectory, attitudes_from_thrust

G = 9.81


@register
class Landing3DoF(Problem):
    slug = "landing-3dof"
    title = "3-DoF Powered Descent"
    summary = "Full 3-D translation with thrust-tilt and glideslope cones."
    phase = "Week 1"
    scene_scale = 900.0

    def params(self) -> list[Param]:
        return [
            Param("x0", "Downrange X", 400.0, min=-1500.0, max=1500.0, step=10.0,
                  unit="m", group="Initial state"),
            Param("y0", "Altitude", 700.0, min=100.0, max=2000.0, step=10.0,
                  unit="m", group="Initial state"),
            Param("z0", "Crossrange Z", -250.0, min=-1500.0, max=1500.0, step=10.0,
                  unit="m", group="Initial state"),
            Param("vx0", "Velocity X", -40.0, min=-150.0, max=150.0, step=1.0,
                  unit="m/s", group="Initial state"),
            Param("vy0", "Velocity Y", -60.0, min=-200.0, max=20.0, step=1.0,
                  unit="m/s", group="Initial state"),
            Param("vz0", "Velocity Z", 20.0, min=-150.0, max=150.0, step=1.0,
                  unit="m/s", group="Initial state"),

            Param("T_final", "Time of flight", 22.0, min=5.0, max=60.0, step=0.5,
                  unit="s", group="Mission"),
            Param("N", "Discretisation steps", 60, kind="int", min=15, max=200,
                  step=5, group="Mission"),

            Param("a_max", "Max thrust accel", 30.0, min=10.0, max=80.0, step=0.5,
                  unit="m/s^2", group="Vehicle limits"),
            Param("tilt_max", "Max thrust tilt", 25.0, min=2.0, max=89.0, step=1.0,
                  unit="deg", group="Vehicle limits",
                  help="Gimbal cone half-angle from vertical."),
            Param("use_tilt", "Enforce tilt cone", True, kind="bool",
                  group="Vehicle limits"),

            Param("glideslope", "Glideslope angle", 25.0, min=0.0, max=80.0,
                  step=1.0, unit="deg", group="Approach corridor",
                  help="Minimum elevation angle of the vehicle seen from the pad."),
            Param("use_glideslope", "Enforce glideslope", True, kind="bool",
                  group="Approach corridor"),

            Param("objective", "Objective", "min-fuel", kind="choice",
                  choices=["min-fuel", "min-energy"], group="Optimisation"),
            Param("solver", "Solver", "CLARABEL", kind="choice",
                  choices=["CLARABEL", "SCS"], group="Optimisation",
                  help="Second-order cone problem, so it needs an SOCP solver."),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)

        r0 = np.array([float(p["x0"]), float(p["y0"]), float(p["z0"])])
        v0 = np.array([float(p["vx0"]), float(p["vy0"]), float(p["vz0"])])
        T = float(p["T_final"])
        N = int(p["N"])
        a_max = float(p["a_max"])
        dt = T / N
        g_vec = np.array([0.0, -G, 0.0])

        r = cp.Variable((N + 1, 3))
        v = cp.Variable((N + 1, 3))
        a = cp.Variable((N, 3))

        cons = [r[0] == r0, v[0] == v0, r[N] == np.zeros(3), v[N] == np.zeros(3)]

        # Exact double-integrator discretisation over each zero-order-hold step.
        for k in range(N):
            acc = a[k] + g_vec
            cons += [v[k + 1] == v[k] + acc * dt]
            cons += [r[k + 1] == r[k] + v[k] * dt + 0.5 * acc * dt ** 2]

        # Thrust magnitude cone
        cons += [cp.norm(a, axis=1) <= a_max]

        # Thrust tilt cone: vertical component dominates
        if bool(p["use_tilt"]):
            ct = float(np.cos(np.deg2rad(float(p["tilt_max"]))))
            cons += [a[:, 1] >= ct * cp.norm(a, axis=1)]
        else:
            cons += [a[:, 1] >= 0]

        # Glideslope cone: stay above a cone rising from the pad
        if bool(p["use_glideslope"]):
            gamma = np.deg2rad(float(p["glideslope"]))
            if gamma > 1e-6:
                tg = float(np.tan(gamma))
                cons += [r[:, 1] >= tg * cp.norm(r[:, [0, 2]], axis=1)]
            else:
                cons += [r[:, 1] >= 0]
        else:
            cons += [r[:, 1] >= 0]

        if p["objective"] == "min-energy":
            obj = cp.Minimize(cp.sum_squares(a) * dt)
        else:
            obj = cp.Minimize(cp.sum(cp.norm(a, axis=1)) * dt)

        prob = cp.Problem(obj, cons)
        solver = str(p["solver"])
        try:
            prob.solve(solver=getattr(cp, solver))
        except Exception as exc:
            return Trajectory(
                t_state=[], t_control=[], position=[], velocity=[],
                thrust=[], attitude=[], status="error", feasible=False,
                solver=solver, notes=[f"{type(exc).__name__}: {exc}"],
            )

        if prob.status not in ("optimal", "optimal_inaccurate"):
            return Trajectory(
                t_state=[], t_control=[], position=[], velocity=[],
                thrust=[], attitude=[], status=prob.status, feasible=False,
                solver=solver,
                notes=[
                    f"No solution exists for these limits (status: {prob.status}). "
                    "Common causes: tilt cone too tight to kill the crossrange "
                    "velocity, glideslope too steep for the starting position, or "
                    "not enough thrust for the time of flight."
                ],
            )

        rv = np.asarray(r.value)
        vv = np.asarray(v.value)
        av = np.asarray(a.value)

        a_mag = np.linalg.norm(av, axis=1)
        speed = np.linalg.norm(vv, axis=1)
        downrange = np.linalg.norm(rv[:, [0, 2]], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            tilt = np.degrees(np.arccos(np.clip(av[:, 1] / np.maximum(a_mag, 1e-9),
                                                -1.0, 1.0)))

        delta_v = float(np.sum(a_mag) * dt)
        notes = [
            f"Delta-v {delta_v:.1f} m/s. Unlike the 1-D case this objective is "
            f"non-degenerate: sum(||a||) >= ||sum(a)||, so the optimiser has real "
            f"freedom and the answer is unique.",
            f"Peak thrust {a_mag.max():.1f} m/s^2 of {a_max:.1f} allowed; "
            f"peak tilt {np.nanmax(tilt):.1f} deg.",
        ]

        t_state = np.linspace(0.0, T, N + 1)
        t_ctrl = np.linspace(0.0, T - dt, N)

        return Trajectory(
            t_state=t_state.tolist(),
            t_control=t_ctrl.tolist(),
            position=rv.tolist(),
            velocity=vv.tolist(),
            thrust=av.tolist(),
            attitude=attitudes_from_thrust(av).tolist(),
            series=[
                Series("altitude", "Altitude", "m", rv[:, 1].tolist()),
                Series("speed", "Speed", "m/s", speed.tolist()),
                Series("downrange", "Horizontal distance", "m", downrange.tolist()),
                Series("thrust", "Thrust accel", "m/s^2", a_mag.tolist(), on="control"),
                Series("tilt", "Thrust tilt", "deg", tilt.tolist(), on="control"),
            ],
            status=prob.status,
            feasible=True,
            cost=float(prob.value),
            solve_time_ms=(prob.solver_stats.solve_time * 1000.0
                           if prob.solver_stats and prob.solver_stats.solve_time
                           else None),
            solver=solver,
            thrust_max=a_max,
            notes=notes,
            diagnostics={
                "delta_v": delta_v,
                "peak_thrust": float(a_mag.max()),
                "peak_tilt_deg": float(np.nanmax(tilt)),
                "final_position_error_m": float(np.linalg.norm(rv[-1])),
                "final_velocity_error_ms": float(np.linalg.norm(vv[-1])),
                "glideslope_deg": float(p["glideslope"]) if p["use_glideslope"] else None,
            },
        )
