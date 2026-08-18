"""
Day 17 -- validating the 3-D stack, in the viewer.

Day 16 put a solver on the site that does not converge. This entry is the day
that works out *why*, and the answer splits cleanly in two: the physics from
Days 13 to 15 is right, and the problem statement is wrong.

**The planar switch is the first half.** Set the initial condition to planar --
no cross-range, no out-of-plane velocity, no roll -- and every out-of-plane
quantity in the answer comes back as **exactly zero**. Not small: zero, in the
position, the velocity, both the roll and yaw rates, and the side thrust. This
solver's Jacobians come from central differences where Day 16's were derived by
hand, so two independent derivations of the same physics agree there is no
leakage between the in-plane and out-of-plane subspaces. That is what clears
Days 13 to 15.

Switch to the 3-D initial condition and the solver genuinely uses the third
dimension -- around 190 m of cross-range and 20 deg/s of roll and yaw. Both
halves are needed: a solver that had quietly stayed planar would pass the first
test without doing anything at all.

**The burn duration is the second half, and it is the day's real finding.**
This vehicle cannot throttle below 40 per cent, so a lit engine always produces
at least 21.23 m/s^2 against gravity's 9.81 -- a net *upward* floor of
11.42 m/s^2. The engine cannot push the vehicle down. From 80 m/s of descent
there is therefore exactly one burn length that arrives at rest,
80 / 11.42 = 7.00 s, and anything longer turns the descent into a climb that no
throttle setting can prevent.

Move the duration slider and watch that happen. Arrival speed bottoms out at
7 s and climbs monotonically after: 19.6 m/s at 7, 28.3 at 8, 54.2 at 10,
122.2 at 14, 201.8 at 18. The Day 17 guide picks 18 seconds.

**It still does not converge**, and the entry says so. Even at the right
horizon the plan misses by 334 m and the virtual control sits at 18.3 against a
1e-1 target. The residual is roughly flat across duration, so the throttle
floor was a large part of the problem and not all of it. As on Day 16, the
default view is the trajectory that actually gets flown rather than the one the
optimiser drew.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.scvx_3d_validate import (
    solve_scvx_validate, replay, planar_ic, threed_ic,
    out_of_plane_extremes, gimbal_angles_deg,
)
from src.dynamics_3d import Vehicle3D, G_EARTH, tilt_from_vertical
from src.aero_3d import AeroConfig3D
from src.quaternion import quat_to_rotmatrix

from ..registry import Problem, register
from ..types import Param, Series, Trajectory
from .rigid_body_view import _to_view_vec, _to_view_quat


@register
class Validate3D(Problem):
    slug = "validate-3d"
    title = "3-D Validation"
    summary = ("A planar case that stays exactly planar, and a burn duration "
               "the throttle floor decides for you.")
    phase = "Day 17"
    scene_scale = 900.0
    enforces_terminal_state = False

    def params(self) -> list[Param]:
        return [
            Param("case", "Initial condition", "planar", kind="choice",
                  choices=["planar", "3-D"], group="What to solve",
                  help="Planar has no cross-range, no out-of-plane velocity "
                       "and no roll, so a correct solver has no reason to "
                       "leave the plane -- and this one does not, to exactly "
                       "zero. 3-D adds all three back."),
            Param("view", "Show", "flown", kind="choice",
                  choices=["flown", "plan"], group="What to solve",
                  help="Flown is the solver's control replayed through Day "
                       "15's model. Plan is what the optimiser drew. The plan "
                       "always lands; the flown one does not."),

            Param("t_f", "Burn duration", 7.0, min=4.0, max=18.0, step=0.5,
                  unit="s", group="The finding",
                  help="The one control worth moving. The throttle floor puts "
                       "minimum acceleration at 21.23 m/s^2 against gravity's "
                       "9.81, so from 80 m/s of descent exactly 7.00 s "
                       "arrives at rest. Longer turns the descent into a "
                       "climb. The Day 17 guide picks 18."),

            Param("K", "Nodes", 30, kind="int", min=15, max=45, step=1,
                  group="Solver"),
            Param("max_iter", "Iteration cap", 10, kind="int", min=3, max=20,
                  step=1, group="Solver"),
            Param("trust0", "Initial trust radius", 5.0, min=0.5, max=15.0,
                  step=0.5, group="Solver",
                  help="Shrinks geometrically each iteration. Both this "
                       "solver and Day 16's end the same way -- the radius "
                       "eventually squeezes the sub-problem infeasible."),
            Param("gs_half_angle_deg", "Glideslope half-angle", 60.0,
                  min=30.0, max=85.0, step=5.0, unit="deg", group="Solver",
                  help="Measured from the vertical. The guide's first "
                       "documented bug was taking this from the horizontal "
                       "instead, which made its own initial condition "
                       "violate the constraint before iteration 1."),
            Param("cross_range", "Cross-range offset", 180.0, min=50.0,
                  max=400.0, step=10.0, unit="m", group="Solver",
                  help="Only used by the 3-D case."),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)
        v, cfg = Vehicle3D(), AeroConfig3D()
        planar = str(p["case"]) == "planar"
        s0 = (planar_ic(v) if planar
              else threed_ic(v, cross_range=float(p["cross_range"])))

        t0 = time.perf_counter()
        try:
            r = solve_scvx_validate(
                s0, v, K=int(p["K"]), tf=float(p["t_f"]),
                max_iter=int(p["max_iter"]), trust0=float(p["trust0"]),
                gs_half_angle_deg=float(p["gs_half_angle_deg"]),
                verbose=False)
            r["replay"] = replay(r, v, cfg)
        except Exception as exc:                                # noqa: BLE001
            return _failed([f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        if not r["ever_solved"]:
            return _failed([
                "No sub-problem solved at these settings, so there is nothing "
                "to draw. Worth knowing that the Day 17 guide's loop returns "
                "its straight-line initial guess in exactly this situation -- "
                "and that guess lands at the origin, upright, at rest, with "
                "zero gimbal, because that is how it was built. This entry "
                "refuses instead."])
        return _trajectory(r, p, planar, elapsed)


# ======================================================================
def _trajectory(r, p, planar, elapsed) -> Trajectory:
    v, cfg = Vehicle3D(), AeroConfig3D()
    flown = str(p["view"]) == "flown"
    rp = r["replay"]

    if flown:
        h = rp["hist"]
        pos, vel = h[:, 0:3].copy(), h[:, 3:6]
        quats, omega, mass = h[:, 6:10], h[:, 10:13], h[:, 13]
    else:
        s = r["s"]
        pos, vel = s[:, 0:3].copy(), s[:, 3:6]
        quats, omega, mass = s[:, 6:10], s[:, 10:13], s[:, 13]
    pos[:, 2] = np.maximum(pos[:, 2], 0.0)
    n = len(pos)
    ts = np.linspace(0.0, r["tf"], n)

    F = r["F"][:n - 1]
    thrust = np.array([quat_to_rotmatrix(quats[k]) @ F[k]
                       for k in range(n - 1)])
    gim = gimbal_angles_deg(r)[:n - 1]
    tilt = np.degrees([tilt_from_vertical(q) for q in quats])
    e = out_of_plane_extremes(r)

    a_min = v.T_min / v.m_wet - G_EARTH
    t_ceiling = 80.0 / a_min

    return Trajectory(
        t_state=ts.tolist(),
        t_control=ts[:-1].tolist(),
        position=[_to_view_vec(x) for x in pos],
        velocity=[_to_view_vec(x) for x in vel],
        thrust=[_to_view_vec(x) for x in thrust],
        attitude=[_to_view_quat(q) for q in quats],
        series=[
            Series("altitude", "Altitude", "m", pos[:, 2].tolist()),
            Series("speed", "Speed", "m/s",
                   np.linalg.norm(vel, axis=1).tolist()),
            Series("vz", "Vertical velocity", "m/s", vel[:, 2].tolist()),
            Series("cross_range", "Out-of-plane position", "m",
                   pos[:, 1].tolist()),
            Series("vy", "Out-of-plane velocity", "m/s", vel[:, 1].tolist()),
            Series("roll_rate", "Roll rate", "deg/s",
                   np.degrees(omega[:, 0]).tolist()),
            Series("yaw_rate", "Yaw rate", "deg/s",
                   np.degrees(omega[:, 2]).tolist()),
            Series("tilt", "Tilt from vertical", "deg", tilt.tolist()),
            Series("gimbal", "Gimbal deflection", "deg", gim.tolist(),
                   on="control"),
            Series("thrust_mag", "Thrust magnitude", "MN",
                   (np.linalg.norm(F, axis=1) / 1e6).tolist(), on="control"),
            Series("mass", "Vehicle mass", "kg", mass.tolist()),
        ],
        status="flown" if flown else "plan",
        feasible=True,
        cost=float(rp["fuel_kg"] if flown else r["fuel"]),
        solve_time_ms=elapsed,
        solver=("validation solver, replayed through the true model" if flown
                else "validation solver, as it drew the plan"),
        thrust_max=v.T_max,
        notes=_notes(r, rp, p, planar, e, gim, t_ceiling, a_min),
        diagnostics={
            "case": "planar" if planar else "3-D",
            "showing": "flown" if flown else "plan",
            "converged": False,
            "virtual_control": r["nu"],
            "iterations": r["iterations"],
            "out_of_plane_position_max_m": e["y"],
            "out_of_plane_velocity_max_ms": e["vy"],
            "roll_rate_max": e["roll_rate"],
            "yaw_rate_max": e["yaw_rate"],
            "side_thrust_max_N": e["Fy"],
            "plan_miss_m": float(np.linalg.norm(r["s"][-1, 0:3])),
            "flown_miss_m": rp["miss_m"],
            "flown_speed_ms": rp["speed_ms"],
            "burn_duration_ceiling_s": t_ceiling,
            "peak_gimbal_deg": float(gim.max()),
            "fuel_kg": float(r["fuel"]),
        },
    )


def _notes(r, rp, p, planar, e, gim, t_ceiling, a_min) -> list:
    v = Vehicle3D()
    tf = float(p["t_f"])
    notes = []
    rate = np.degrees(max(e["roll_rate"], e["yaw_rate"]))

    if planar:
        worst = max(e.values())
        notes.append(
            f"**Planar in, planar out -- exactly.** Out-of-plane position, "
            f"out-of-plane velocity, roll rate, yaw rate and side thrust all "
            f"come back at {worst:.2e}. Not small; zero. A planar initial "
            f"condition gives the optimiser no reason to leave the plane, and "
            f"any leakage would be a sign error or a frame mix-up bleeding "
            f"into a degree of freedom it should not touch. This solver's "
            f"Jacobians are central differences where Day 16's were derived "
            f"by hand, so two independent derivations agree. That is what "
            f"clears Days 13 to 15."
        )
    else:
        notes.append(
            f"The 3-D case genuinely uses the third dimension -- "
            f"{e['y']:.0f} m of out-of-plane travel, {e['vy']:.0f} m/s of "
            f"out-of-plane velocity, and {rate:.1f} deg/s of roll and yaw. "
            f"That check matters as much as the planar "
            f"one: a solver that had quietly stayed planar would pass the "
            f"reduction test without doing anything at all. Switch the "
            f"initial condition to see the other half."
        )

    notes.append(
        f"**The burn duration is not yours to choose.** This vehicle cannot "
        f"throttle below {v.throttle_min * 100:.0f} per cent, so a lit engine "
        f"always makes at least {v.T_min / v.m_wet:.2f} m/s^2 against "
        f"gravity's {G_EARTH:.2f} -- a net *upward* floor of {a_min:.2f}. The "
        f"engine cannot push the vehicle down. From 80 m/s of descent exactly "
        f"{t_ceiling:.2f} s arrives at rest, and this run uses {tf:.1f}. "
        f"Longer does not land softly; it turns the descent into a climb."
    )
    if tf > t_ceiling + 0.6:
        notes.append(
            f"You are {tf - t_ceiling:.1f} s past that ceiling, which is why "
            f"the vertical-velocity trace turns positive before touchdown and "
            f"the flown arrival speed is {rp['speed_ms']:.1f} m/s. Measured "
            f"across the slider: 19.6 m/s at 7 s, 28.3 at 8, 54.2 at 10, "
            f"122.2 at 14, 201.8 at 18. The Day 17 guide picks 18."
        )
    elif tf < t_ceiling - 0.6:
        notes.append(
            f"You are {t_ceiling - tf:.1f} s short of the ceiling, so the "
            f"vehicle arrives still descending rather than overshooting into "
            f"a climb -- the other side of the same wall."
        )

    notes.append(
        f"**It still does not converge.** Virtual control is at "
        f"{r['nu']:.2e} against a 1e-1 target after {r['iterations']} "
        f"iterations, and the plan that lands "
        f"{float(np.linalg.norm(r['s'][-1, 0:3])):.1e} m from the pad misses "
        f"by {rp['miss_m']:,.0f} m at {rp['speed_ms']:.1f} m/s when flown. "
        f"The residual is roughly flat across burn duration, so the throttle "
        f"floor was a large part of the problem and not all of it."
    )
    notes.append(
        f"Where the slack sits is how that was found: 88 to 90 per cent of it "
        f"lands in the *velocity* rows on the early iterations, not the "
        f"attitude or the quaternion. That is the translational dynamics "
        f"being unsatisfiable, which is exactly what a throttle floor fighting "
        f"a fixed horizon looks like. Day 16 guessed at an over-constrained "
        f"sub-problem; this locates it."
    )
    tau_cmd = v.T_max * v.L_engine * np.sin(np.radians(float(gim.max())))
    notes.append(
        f"The gimbal tells the same story from the other end. Peak deflection "
        f"here is {gim.max():.3f} deg of {v.delta_max_deg:.0f} available, "
        f"which commands "
        f"{tau_cmd / 1e3:,.0f} "
        f"kN m of torque against {v.tau_max / 1e3:,.0f} -- while the plan asks "
        f"the vehicle to rotate. The rotation is coming from slack rather than "
        f"from torque, which is another face of the same non-convergence."
    )
    return notes


def _failed(notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status="error", feasible=False,
        solver="3-D validation", notes=notes,
    )
