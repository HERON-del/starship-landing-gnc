"""
Day 1 — minimum-fuel 1-D soft landing.

The vertical-only ancestor of the full problem: a point mass under gravity with
a bounded, non-negative thruster, required to reach the pad at zero velocity.

A caution this module reports at runtime: with **fixed final time and a fixed
terminal velocity**, minimum-fuel is degenerate. Summing the velocity dynamics
telescopes to

    dt * sum(a) = v[N] - v[0] + g * T

so total thrust is pinned by the constraints and *every* feasible trajectory
ties. Interior-point solvers return a smooth interior point of that optimal
face; simplex solvers return a bang-bang vertex. Both are "optimal". Switch the
objective to minimum-energy to get a problem where the optimiser actually
chooses.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np

from ..registry import Problem, register
from ..types import Param, Series, Trajectory

G = 9.81


@register
class Landing1D(Problem):
    slug = "landing-1d"
    title = "1-D Soft Landing"
    summary = "Vertical descent, bounded non-negative thrust, fixed time of flight."
    phase = "Day 1"
    scene_scale = 120.0

    def params(self) -> list[Param]:
        return [
            Param("h0", "Initial altitude", 100.0, min=10.0, max=500.0, step=5.0,
                  unit="m", group="Initial state"),
            Param("v0", "Initial velocity", -20.0, min=-80.0, max=0.0, step=1.0,
                  unit="m/s", group="Initial state",
                  help="Negative is falling."),
            Param("T_final", "Time of flight", 10.0, min=2.0, max=30.0, step=0.5,
                  unit="s", group="Mission"),
            Param("N", "Discretisation steps", 50, kind="int", min=10, max=300,
                  step=10, group="Mission",
                  help="More steps = finer control resolution, slower solve."),
            Param("a_max", "Max thrust accel", 25.0, min=5.0, max=60.0, step=0.5,
                  unit="m/s^2", group="Vehicle limits"),
            Param("objective", "Objective", "min-fuel", kind="choice",
                  choices=["min-fuel", "min-energy"], group="Optimisation",
                  help="min-fuel is degenerate at fixed final time; "
                       "min-energy genuinely discriminates."),
            Param("ground", "Enforce h >= 0", True, kind="bool",
                  group="Optimisation",
                  help="Turn off to let the trajectory dip below the pad."),
            Param("solver", "Solver", "CLARABEL", kind="choice",
                  choices=["CLARABEL", "SCS", "OSQP", "HIGHS"],
                  group="Optimisation",
                  help="HIGHS is simplex and returns a bang-bang vertex; "
                       "the rest are interior-point and return smooth profiles."),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)

        h0 = float(p["h0"])
        v0 = float(p["v0"])
        T = float(p["T_final"])
        N = int(p["N"])
        a_max = float(p["a_max"])
        dt = T / N

        h = cp.Variable(N + 1)
        v = cp.Variable(N + 1)
        a = cp.Variable(N)

        cons = [h[0] == h0, v[0] == v0, h[N] == 0.0, v[N] == 0.0,
                a >= 0.0, a <= a_max]
        if bool(p["ground"]):
            cons += [h >= 0.0]
        for k in range(N):
            cons += [v[k + 1] == v[k] + (a[k] - G) * dt,
                     h[k + 1] == h[k] + v[k] * dt]

        if p["objective"] == "min-energy":
            obj = cp.Minimize(cp.sum_squares(a) * dt)
        else:
            obj = cp.Minimize(cp.sum(a) * dt)

        prob = cp.Problem(obj, cons)
        solver = str(p["solver"])
        try:
            prob.solve(solver=getattr(cp, solver))
        except Exception as exc:  # solver blew up rather than proving infeasible
            return Trajectory(
                t_state=[], t_control=[], position=[], velocity=[],
                thrust=[], attitude=[], status="error", feasible=False,
                solver=solver, notes=[f"{type(exc).__name__}: {exc}"],
            )

        feasible = prob.status in ("optimal", "optimal_inaccurate")
        notes: list[str] = []
        diagnostics: dict[str, Any] = {}

        if not feasible:
            notes.append(
                f"No solution exists for these limits (status: {prob.status}). "
                f"Raise max thrust or lengthen the time of flight."
            )
            return Trajectory(
                t_state=[], t_control=[], position=[], velocity=[],
                thrust=[], attitude=[], status=prob.status, feasible=False,
                solver=solver, notes=notes,
            )

        hv = np.asarray(h.value).ravel()
        vv = np.asarray(v.value).ravel()
        av = np.asarray(a.value).ravel()

        # The degeneracy check, reported live rather than buried in a comment.
        pinned = -v0 + G * T
        impulse = float(np.sum(av) * dt)
        diagnostics["total_delta_v"] = impulse
        diagnostics["delta_v_forced_by_constraints"] = float(pinned)
        at_bounds = float(np.mean((av < 1e-4) | (av > a_max - 1e-4)))
        diagnostics["fraction_of_steps_at_a_bound"] = at_bounds

        if p["objective"] == "min-fuel":
            notes.append(
                f"Total delta-v is {impulse:.2f} m/s, exactly the "
                f"-v0 + g*T = {pinned:.2f} m/s forced by the terminal velocity "
                f"constraint. At fixed final time every feasible trajectory ties, "
                f"so this objective selects nothing."
            )
            shape = "Bang-bang" if at_bounds > 0.3 else "Smooth"
            notes.append(
                f"{shape} profile: {at_bounds * 100:.0f}% of steps sit on a "
                "thrust bound. That reflects the solver's vertex vs "
                "interior-point answer, not a physical result."
            )
        else:
            notes.append(
                "Minimum-energy penalises large accelerations quadratically, so "
                "this objective does discriminate and the profile is unique."
            )

        t_state = np.linspace(0.0, T, N + 1)
        t_ctrl = np.linspace(0.0, T - dt, N)

        pos = np.zeros((N + 1, 3))
        pos[:, 1] = hv
        vel = np.zeros((N + 1, 3))
        vel[:, 1] = vv
        thr = np.zeros((N, 3))
        thr[:, 1] = av

        # Purely vertical flight: the vehicle stays upright.
        att = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (N + 1, 1))

        return Trajectory(
            t_state=t_state.tolist(),
            t_control=t_ctrl.tolist(),
            position=pos.tolist(),
            velocity=vel.tolist(),
            thrust=thr.tolist(),
            attitude=att.tolist(),
            series=[
                Series("altitude", "Altitude", "m", hv.tolist()),
                Series("velocity", "Vertical velocity", "m/s", vv.tolist()),
                Series("thrust", "Thrust accel", "m/s^2", av.tolist(), on="control"),
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
            diagnostics=diagnostics,
        )
