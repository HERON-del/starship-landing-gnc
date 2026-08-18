"""
Day 16 -- the 3-D SCvx solver, and the gap between a plan and a trajectory.

This is the first entry in the viewer for a solver that **does not converge**,
and it is built to show that rather than hide it. The default view is the
trajectory the vehicle would actually fly, not the one the optimiser drew.

What works, and it is most of the day. Making the decision variable the
body-frame thrust *force vector* rather than a magnitude and two angles leaves
four things exactly convex with no linearisation near them: the thrust bound,
the gimbal cone, the torque -- which is linear in the force -- and the 3-D
glideslope. Day 14 needed a sweep of 10,201 deflection pairs to establish that
this gimbal cannot roll the vehicle; here it is a structural property of a
cross product with a fixed body-x vector. Every linearisation that *is* needed
checks out against finite differences, and the returned plan meets its terminal
conditions to nanometres, inside every cone, on a third of the propellant.

What does not work is the outer loop. The virtual control -- the slack the
solver is allowed to add to its own dynamics -- stalls around 0.42 against a
tolerance of 1e-6. A plan that pays slack is not a trajectory, and the
**Flown** view is what happens when you hand it to the real vehicle: it misses
by about 250 m.

**Switch between Flown and Plan.** That difference is the entire point of the
entry. The plan lands perfectly because the solver is checking it against its
own linearised, Euler-discretised model with slack available; the flown one
does not, because Day 15's model does not offer slack.

Two causes were ruled out by measurement rather than argument, and both are
worth knowing before anyone tries to fix it. It is not the Euler step -- the
miss shows no trend with node count. And it is not an under-sized trust region
-- the defect *falls* as the radius grows, which is the opposite signature.
What fits is an over-constrained sub-problem: hard terminal equalities on all
four state blocks, a throttle floor that puts minimum deceleration at 21 m/s^2
against gravity's 9.8, and a fixed horizon. Lengthening the horizon eases the
defect without improving the flight.

The iteration cap here is low on purpose. Running the full thirty iterations
changes neither the defect nor the miss by a digit, which is its own evidence:
the loop is stalled, not still working.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.scvx_3d import solve_scvx_3d, force_to_gimbal
from src.scvx_params import SCvxParams
from src.dynamics_3d import Vehicle3D, tilt_from_vertical
from src.aero_3d import AeroConfig3D, angle_history
from src.quaternion import quat_to_rotmatrix

from ..registry import Problem, register
from ..types import Param, Series, Trajectory
from .rigid_body_view import _to_view_vec, _to_view_quat


@register
class SCvx3D(Problem):
    slug = "scvx-3d"
    title = "3-D SCvx Solver"
    summary = ("The convex sub-problem is right; the outer loop does not "
               "converge, and this shows the difference.")
    phase = "Day 16"
    scene_scale = 700.0
    enforces_terminal_state = False

    def params(self) -> list[Param]:
        return [
            Param("view", "Show", "flown", kind="choice",
                  choices=["flown", "plan"], group="What to draw",
                  help="Flown is the solver's control replayed through Day "
                       "15's true model -- what the vehicle would do. Plan is "
                       "what the optimiser drew. The plan lands perfectly "
                       "because it is checked against a linearised model with "
                       "slack available; the flown one is not."),

            Param("N", "Nodes", 25, kind="int", min=12, max=45, step=1,
                  group="Solver",
                  help="Raising this does not close the gap -- the miss shows "
                       "no trend with node count, which is how the Euler step "
                       "was ruled out as the cause."),
            Param("max_iter", "Iteration cap", 10, kind="int", min=3, max=30,
                  step=1, group="Solver",
                  help="Deliberately low. Running the full thirty changes "
                       "neither the defect nor the miss by a digit, which is "
                       "its own evidence that the loop is stalled."),
            Param("eta_0", "Initial trust radius", 0.5, min=0.05, max=2.0,
                  step=0.05, group="Solver",
                  help="Worth moving. The dynamics defect FALLS as this grows "
                       "-- 0.557 at 0.2, 0.416 at 0.5, 0.310 at 1.0 -- which "
                       "is the opposite of what an invalid linearisation "
                       "looks like, and is why the trust region is not the "
                       "problem."),

            Param("t_f", "Burn duration", 8.0, min=5.0, max=16.0, step=0.5,
                  unit="s", group="Problem",
                  help="Fixed, not solved for -- Day 8's free-time extension "
                       "was dropped for this day. Lengthening it eases the "
                       "defect without improving the flight, which is part of "
                       "why the fixed horizon looks like the culprit."),
            Param("gamma_gs_deg", "Glideslope half-angle", 80.0, min=45.0,
                  max=89.0, step=1.0, unit="deg", group="Problem"),
            Param("x0", "Entry downrange", 300.0, min=0.0, max=800.0,
                  step=25.0, unit="m", group="Problem"),
            Param("y0", "Entry cross-range", 0.0, min=-400.0, max=400.0,
                  step=25.0, unit="m", group="Problem",
                  help="Nonzero makes the problem genuinely 3-D rather than a "
                       "planar one solved in 3-D variables."),
            Param("z0", "Entry altitude", 420.0, min=250.0, max=900.0,
                  step=20.0, unit="m", group="Problem"),
            Param("vz0", "Entry descent rate", -130.0, min=-180.0, max=-60.0,
                  step=5.0, unit="m/s", group="Problem"),
            Param("theta0_deg", "Entry pitch", 25.0, min=0.0, max=90.0,
                  step=5.0, unit="deg", group="Problem"),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)
        sp = SCvxParams()
        sp.max_iter = int(p["max_iter"])
        sp.eta_0 = float(p["eta_0"])

        t0 = time.perf_counter()
        try:
            r = solve_scvx_3d(
                N=int(p["N"]), t_f=float(p["t_f"]),
                pos0=(float(p["x0"]), float(p["y0"]), float(p["z0"])),
                vel0=(-30.0, 0.0, float(p["vz0"])),
                theta0_deg=float(p["theta0_deg"]),
                gamma_gs_deg=float(p["gamma_gs_deg"]),
                params=sp, verbose=False)
        except Exception as exc:                                # noqa: BLE001
            return _failed([f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        if r.get("status") != "converged":
            return _failed([
                "The sub-problem did not return a usable answer at these "
                "settings. That is a different failure from the one this "
                "entry is about -- see the notes on any working setting."])
        return _trajectory(r, p, elapsed)


# ======================================================================
def _trajectory(r, p, elapsed) -> Trajectory:
    v, cfg = Vehicle3D(), AeroConfig3D()
    flown = str(p["view"]) == "flown"
    rp = r["replay"]
    hist = rp["hist"]

    if flown:
        pos, vel = hist[:, 0:3].copy(), hist[:, 3:6]
        quats, omega, mass = hist[:, 6:10], hist[:, 10:13], hist[:, 13]
    else:
        pos, vel = r["pos"].copy(), r["vel"]
        quats, omega, mass = r["q"], r["omega"], r["m"]
    pos[:, 2] = np.maximum(pos[:, 2], 0.0)
    n = len(pos)

    # Thrust the solver commanded, rotated into whichever attitude is drawn.
    thrust = np.array([quat_to_rotmatrix(quats[k]) @ r["F"][k]
                       for k in range(n - 1)])
    tilt = np.degrees([tilt_from_vertical(q) for q in quats])
    gim = np.degrees(np.arctan2(np.linalg.norm(r["F"][:, 1:], axis=1),
                                r["F"][:, 0]))
    gs = np.degrees(np.arctan2(np.linalg.norm(pos[:, :2], axis=1),
                               np.maximum(pos[:, 2], 1e-9)))
    ang = angle_history(hist, cfg=cfg) if flown else None

    series = [
        Series("altitude", "Altitude", "m", pos[:, 2].tolist()),
        Series("speed", "Speed", "m/s",
               np.linalg.norm(vel, axis=1).tolist()),
        Series("cross_range", "Cross-range", "m", pos[:, 1].tolist()),
        Series("tilt", "Tilt from vertical", "deg", tilt.tolist()),
        Series("glideslope", "Glideslope angle", "deg", gs.tolist()),
        Series("gimbal", "Gimbal deflection", "deg", gim.tolist(),
               on="control"),
        Series("sigma", "Thrust bound sigma", "MN",
               (r["sigma"] / 1e6).tolist(), on="control"),
        Series("thrust_mag", "Thrust magnitude", "MN",
               (np.linalg.norm(r["F"], axis=1) / 1e6).tolist(), on="control"),
        Series("mass", "Vehicle mass", "kg", mass.tolist()),
    ]
    if ang is not None:
        series.append(Series("alpha", "Angle of attack", "deg",
                             np.degrees(ang[:, 0]).tolist()))

    return Trajectory(
        t_state=r["t"].tolist(),
        t_control=r["t"][:-1].tolist(),
        position=[_to_view_vec(x) for x in pos],
        velocity=[_to_view_vec(x) for x in vel],
        thrust=[_to_view_vec(x) for x in thrust],
        attitude=[_to_view_quat(q) for q in quats],
        series=series,
        status="flown" if flown else "plan",
        feasible=True,
        cost=float(rp["fuel_kg"] if flown else r["fuel"]),
        solve_time_ms=elapsed,
        solver=("SCvx plan, replayed through the true model" if flown
                else "SCvx plan, as the optimiser drew it"),
        thrust_max=v.T_max,
        notes=_notes(r, rp, p, flown, gim, gs),
        diagnostics={
            "showing": "flown through the true model" if flown else "the plan",
            "converged": False,
            "virtual_control": r["vc_norm"],
            "vc_tolerance": 1e-6,
            "iterations": r["iterations"],
            "plan_miss_m": float(np.linalg.norm(r["pos"][-1])),
            "flown_miss_m": rp["miss_m"],
            "flown_speed_ms": rp["speed_ms"],
            "flown_tilt_deg": rp["tilt_deg"],
            "peak_plan_vs_flown_m": rp["pos_err_m"],
            "quaternion_norm_drift": r["q_norm_drift"],
            "lcvx_gap": r["lcvx_gap"],
            "peak_gimbal_deg": float(gim.max()),
            "peak_glideslope_deg": float(gs.max()),
            "fuel_kg": float(r["fuel"]),
        },
    )


def _notes(r, rp, p, flown, gim, gs) -> list:
    v = Vehicle3D()
    notes = [
        f"**This solver does not converge.** The virtual control -- the slack "
        f"it may add to its own dynamics -- sits at {r['vc_norm']:.3e} against "
        f"a tolerance of 1e-6 after {r['iterations']} iterations. A plan that "
        f"pays slack is not a trajectory, and the entry is built to show that "
        f"rather than hide it."
    ]

    if flown:
        notes.append(
            f"You are looking at the **flown** trajectory: the solver's own "
            f"control, handed to Day 15's model. It misses by "
            f"{rp['miss_m']:,.1f} m at {rp['speed_ms']:.1f} m/s, ending "
            f"{rp['tilt_deg']:.1f} deg from vertical, and departs from the "
            f"plan by up to {rp['pos_err_m']:,.0f} m along the way. Switch to "
            f"**plan** to see the same run as the optimiser drew it."
        )
    else:
        notes.append(
            f"You are looking at the **plan**. It lands at "
            f"{float(np.linalg.norm(r['pos'][-1])):.2e} m from the pad at "
            f"{float(np.linalg.norm(r['vel'][-1])):.2e} m/s, upright, inside "
            f"every cone -- and none of that is worth much, because the model "
            f"it satisfies is linearised, Euler-discretised, and has slack "
            f"available. Switch to **flown** to see it handed to physics that "
            f"offers none."
        )

    notes.append(
        f"What is genuinely right here is the convex sub-problem. Choosing "
        f"the body-frame thrust *force vector* as the decision variable leaves "
        f"the thrust bound, the gimbal cone, the torque and the glideslope all "
        f"exactly convex -- no reference trajectory, no Taylor expansion. This "
        f"run respects them: gimbal peaks at {gim.max():.1f} deg of "
        f"{v.delta_max_deg:.0f}, glideslope at {gs.max():.1f} of "
        f"{float(p['gamma_gs_deg']):.0f}. Day 14 needed a sweep of 10,201 "
        f"deflection pairs to show this gimbal cannot roll the vehicle; here "
        f"the roll torque is structurally absent, because a cross product with "
        f"a fixed body-x vector has no x component."
    )
    notes.append(
        "Every linearisation that IS needed was checked against finite "
        "differences: the Hamilton matrices to 1e-12, dR/dq to 2.8e-10, the "
        "gyroscopic Jacobian to 1.7e-09, and the quaternion kinematics exact "
        "at the reference with error that divides by 4.00 each time the step "
        "halves -- which is what a product rule on a bilinear term must do. "
        "The failure is in the outer loop, not the algebra."
    )
    notes.append(
        f"Two causes ruled out by measurement rather than argument. It is not "
        f"the Euler step: the miss shows no trend with node count (141 m at "
        f"N=15, 247 at 25, 418 at 40, 240 at 60, 417 at 90). And it is not an "
        f"under-sized trust region: the defect FALLS as the radius grows "
        f"(0.557 at eta 0.2, 0.416 at 0.5, 0.310 at 1.0), which is the "
        f"opposite of what an invalid linearisation looks like. Move the trust "
        f"radius control and watch the defect go the wrong way."
    )
    notes.append(
        f"What fits is an over-constrained sub-problem. Hard terminal "
        f"equalities on position, velocity, attitude and rate; a "
        f"{v.throttle_min * 100:.0f}% throttle floor that puts minimum "
        f"deceleration at {v.T_min / v.m_wet:.0f} m/s^2 against gravity's 9.8, "
        f"so the vehicle cannot fly a gentle approach; and a fixed horizon. "
        f"Lengthening the burn eases the defect (0.42 at 8 s, 0.26 at 11, 0.18 "
        f"at 14) without improving the flight, which fits that reading and "
        f"nothing else tried. Free final time is the next thing to try."
    )

    if r["lcvx_gap"] > 0.05:
        notes.append(
            f"The lossless-convexification gap is {r['lcvx_gap']:.2f}. The "
            f"relaxation ||F|| <= sigma is only honest if it is tight at the "
            f"solution, and here it is not -- sigma sits above the thrust it "
            f"is supposed to bound. Day 4 learned to check this rather than "
            f"assume it; the same check belongs here."
        )

    if abs(float(p["y0"])) < 1e-9:
        notes.append(
            "Entry cross-range is zero, so this is a planar problem being "
            "solved in 3-D variables. Set it nonzero to make the solver "
            "actually leave the plane -- the out-of-plane states are exact "
            "zeros until you do."
        )
    return notes


def _failed(notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status="error", feasible=False,
        solver="3-D SCvx", notes=notes,
    )
