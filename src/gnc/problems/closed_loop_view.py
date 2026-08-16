"""
Day 10 — guidance, in the viewer.

Day 9's entry renders one plan flown blind. This one renders the same descent
flown by a guidance loop that re-solves the whole landing problem every half
second from wherever the vehicle actually is, warm-started from its previous
answer. Both strategies fly the identical gust sequence, so the `strategy`
control switches between them and nothing else changes.

**The panel is the point, and it does not say what you would expect.** Closing
the loop makes the landing *worse* by Day 9's scoring. Over twelve wind seeds
it lands nearer in eleven of them -- median miss 3.45 m down to 0.60 m -- and
arrives nearly three times faster, 5.76 m/s up to 15.31 m/s. Good landings fall
from 33% to 8%. Day 9 established that position was never the failure and
arrival speed was, so this loop fixes the error that did not matter and worsens
the one that did.

**It is a rate problem, not a concept problem, and the guidance-cycle slider
shows it.** Median arrival runs 21.6, 15.3, 12.2 and 7.85 m/s at cycles of 1.0,
0.5, 0.25 and 0.125 s -- improving all the way down. The descent lasts about
five seconds and nearly all the braking is in the last one, so a half-second
cycle leaves the final command half a second stale exactly where precision is
needed. Position is a slow state and gets corrected; velocity is fast, and on a
bang-bang trajectory with no slack it does not. Take the cycle to 0.125 s and
the arrival is 7.85 m/s and still falling -- but the replan itself costs about
0.22 s, so that rate does not fit and the panel says so.

**Navigation noise breaks it.** There is no filter between the state estimate
and the solver, so noise goes straight into a re-optimisation that is bang-bang
by construction. One metre of position noise is harmless; three produces a
109 m worst case; eight gives an 84 m median miss and burns 10 tonnes against a
nominal six. Worth turning up to see a guidance loop fail honestly.

Each solve here runs a full descent -- a cold start plus a replan every cycle,
and the open-loop baseline alongside it for the comparison -- so it costs a few
seconds rather than milliseconds.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.closed_loop import (
    run_closed_loop, run_open_loop, MISS_TOL_M, SPEED_TOL_MS,
    Z0_NOM, VZ0_NOM, THETA0_NOM,
)
from src.dynamics_6dof import Vehicle6DoF

from ..registry import Problem, register
from ..types import Param, Series, Trajectory, quats_from_pitch


@register
class ClosedLoopGuidance(Problem):
    slug = "closed-loop"
    title = "Closed-Loop Guidance (MPC)"
    summary = ("Replanning every half second against gusts. It fixes the "
               "error Day 9 said did not matter.")
    phase = "Day 10"
    scene_scale = 700.0
    # A flown trajectory, not an optimiser's promise: allowed to miss, allowed
    # to arrive moving. That is what is being measured.
    enforces_terminal_state = False

    def params(self) -> list[Param]:
        return [
            Param("strategy", "Strategy", "closed loop", kind="choice",
                  choices=["closed loop", "open loop"], group="Guidance",
                  help="Both fly the identical gust sequence. Open loop is "
                       "Day 9's approach: plan once, never look again."),
            Param("guidance_dt", "Guidance cycle", 0.5, min=0.125, max=2.0,
                  step=0.125, unit="s", group="Guidance",
                  help="The control that matters. Arrival speed improves all "
                       "the way down to 0.125 s - but the replan costs about "
                       "0.22 s, so below roughly 0.25 s it no longer fits."),
            Param("budget", "Iterations per replan", 3, kind="int", min=1,
                  max=10, step=1, group="Guidance",
                  help="A real guidance computer has a compute budget, not a "
                       "convergence tolerance. Three is enough because the "
                       "warm start puts the command within a degree."),

            Param("wind_sigma_x", "Cross-wind (3 sigma)", 6.0, min=0.0,
                  max=15.0, step=1.0, unit="m/s", group="Disturbance",
                  help="Ornstein-Uhlenbeck gusts, correlated over seconds. "
                       "White noise would average itself out inside a cycle "
                       "and flatter the guidance."),
            Param("wind_tau", "Gust correlation time", 2.0, min=0.5, max=6.0,
                  step=0.5, unit="s", group="Disturbance"),
            Param("wind_seed", "Wind seed", 7, kind="int", min=0, max=999,
                  step=1, group="Disturbance"),
            Param("nav_sigma_pos", "Navigation noise, position", 0.0, min=0.0,
                  max=10.0, step=0.5, unit="m", group="Disturbance",
                  help="Fed straight into the solver - there is no filter. "
                       "Three metres already produces a 109 m worst case."),
            Param("nav_sigma_vel", "Navigation noise, velocity", 0.0, min=0.0,
                  max=3.0, step=0.1, unit="m/s", group="Disturbance"),

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
        kw = dict(N=int(p["N"]), z0=float(p["z0"]), vz0=float(p["vz0"]),
                  theta0_deg=float(p["theta0_deg"]),
                  guidance_dt=float(p["guidance_dt"]),
                  wind_sigma_x=float(p["wind_sigma_x"]),
                  wind_tau=float(p["wind_tau"]),
                  wind_seed=int(p["wind_seed"]),
                  verbose=False)

        t0 = time.perf_counter()
        try:
            cl = run_closed_loop(budget=int(p["budget"]),
                                 nav_sigma_pos=float(p["nav_sigma_pos"]),
                                 nav_sigma_vel=float(p["nav_sigma_vel"]),
                                 keep_path=True, **kw)
            ol = run_open_loop(keep_path=True, **kw)
        except Exception as exc:      # noqa: BLE001
            return _failed("error", [f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        if cl.get("status") != "flown" or ol.get("status") != "flown":
            return _failed("infeasible", [
                "No initial plan exists from this entry state, so neither "
                "strategy has anything to fly. The feasible band is narrow "
                "and one-sided - see the Day 9 problem for the measurement."
            ])

        closed = str(p["strategy"]) == "closed loop"
        shown, other = (cl, ol) if closed else (ol, cl)
        return _trajectory(shown, other, closed, cl, ol, p, elapsed)


# ----------------------------------------------------------------------
def _trajectory(r, other, closed, cl, ol, p, elapsed) -> Trajectory:
    t = np.asarray(r["path_t"])
    y = np.asarray(r["path_y"])
    x, z, vx, vz, th, om, m = (y[:, i] for i in range(7))
    z = np.maximum(z, 0.0)
    veh = Vehicle6DoF()

    n = len(t)
    pos = np.column_stack([x, z, np.zeros(n)])
    vel = np.column_stack([vx, vz, np.zeros(n)])
    # Thrust is not logged per sub-step; show the direction the vehicle points,
    # scaled by weight, so the plume reads sensibly without inventing data.
    mag = m[:-1] * 9.80665
    thrust = np.column_stack([mag * np.sin(th[:-1]), mag * np.cos(th[:-1]),
                              np.zeros(n - 1)])

    diag = {
        "shown_miss_m": float(r["miss"]),
        "shown_speed_ms": float(r["speed"]),
        "shown_pitch_deg": float(r["pitch_deg"]),
        "shown_fuel_kg": float(r["fuel"]),
        "shown_margin_kg": float(r["margin"]),
        "closed_miss_m": float(cl["miss"]),
        "closed_speed_ms": float(cl["speed"]),
        "closed_fuel_kg": float(cl["fuel"]),
        "open_miss_m": float(ol["miss"]),
        "open_speed_ms": float(ol["speed"]),
        "open_fuel_kg": float(ol["fuel"]),
    }
    if closed:
        diag.update({
            "replans": int(cl["n_replans"]),
            "failed_replans": int(cl["n_failed_replans"]),
            "mean_replan_s": float(cl["mean_solve_time"]),
            "max_replan_s": float(cl["max_solve_time"]),
            "guidance_cycle_s": float(cl["guidance_dt"]),
            "replan_fits_cycle": bool(cl["max_solve_time"]
                                      < cl["guidance_dt"]),
            "mean_tracking_gap_m": float(cl["mean_gap"]),
            "cold_start_iters": int(cl["cold_iters"]),
        })

    return Trajectory(
        t_state=t.tolist(),
        t_control=t[:-1].tolist(),
        position=pos.tolist(),
        velocity=vel.tolist(),
        thrust=thrust.tolist(),
        attitude=quats_from_pitch(th).tolist(),
        series=[
            Series("altitude", "Altitude", "m", z.tolist()),
            Series("downrange", "Downrange", "m", x.tolist()),
            Series("speed", "Speed", "m/s", np.hypot(vx, vz).tolist()),
            Series("pitch", "Pitch from vertical", "deg",
                   np.degrees(th).tolist()),
            Series("rate", "Pitch rate", "deg/s", np.degrees(om).tolist()),
            Series("mass", "Vehicle mass", "kg", m.tolist()),
        ],
        status=r["fail_reason"],
        feasible=True,
        cost=float(r["fuel"]),
        solve_time_ms=elapsed,
        solver=("warm-started SCvx, "
                f"{cl['n_replans']} replans" if closed
                else "single plan, flown blind"),
        thrust_max=veh.T_max,
        notes=_notes(r, closed, cl, ol, p),
        diagnostics=diag,
    )


def _notes(r, closed, cl, ol, p) -> list[str]:
    which = "closed loop" if closed else "open loop"
    notes = [
        f"Showing the {which}: {r['fail_reason']} -- {r['miss']:.2f} m from "
        f"the pad at {r['speed']:.2f} m/s, {r['fuel']:,.0f} kg burned, "
        f"{r['margin']:,.0f} kg still aboard. Scoring is Day 9's: within "
        f"{MISS_TOL_M:.0f} m and {SPEED_TOL_MS:.0f} m/s counts as landed.",
        f"Same gusts, the other way: open loop {ol['miss']:.2f} m at "
        f"{ol['speed']:.2f} m/s, closed loop {cl['miss']:.2f} m at "
        f"{cl['speed']:.2f} m/s. Switch the strategy control to fly it.",
    ]

    if cl["miss"] < ol["miss"] and cl["speed"] > ol["speed"]:
        notes.append(
            "This seed shows the trade the whole day is about. Replanning "
            "corrects position decisively and costs arrival speed. Over "
            "twelve seeds it lands nearer in eleven and arrives slower in "
            "one, taking good landings from 33% down to 8% -- Day 9 having "
            "established that position was never the failure and speed was."
        )

    if closed:
        fits = cl["max_solve_time"] < cl["guidance_dt"]
        notes.append(
            f"Guidance ran {cl['n_replans']} times, {cl['n_failed_replans']} "
            f"of them abandoned, at {cl['mean_solve_time']:.3f}s mean and "
            f"{cl['max_solve_time']:.3f}s worst against a "
            f"{cl['guidance_dt']:.3f}s cycle"
            + ("." if fits else
               " -- which does not fit, so this rate is not real-time.")
            + f" The cold start alone took {cl['cold_iters']} "
              f"iterations; each replan is given "
              f"{int(p['budget'])}, because warm starting puts the command "
              f"within about a degree of converged by the third."
        )
        notes.append(
            f"Mean tracking gap {cl['mean_gap']:.2f} m -- the "
            f"distance between where the previous plan said the vehicle "
            f"would be and where it was, which is the error each replan "
            f"exists to remove."
        )
        if float(p["guidance_dt"]) > 0.25:
            notes.append(
                "Try shortening the guidance cycle. Median arrival over the "
                "sweeps runs 21.6, 15.3, 12.2 and 7.85 m/s at 1.0, 0.5, 0.25 "
                "and 0.125 s, so the loop is converging on the right answer "
                "and is simply sampled too slowly at the end, where all the "
                "braking happens."
            )
        if float(p["nav_sigma_pos"]) >= 3.0:
            notes.append(
                "Navigation noise is on and there is no filter between the "
                "estimate and the solver, so it feeds straight into a "
                "re-optimisation that is bang-bang by construction. At three "
                "metres this produces a 109 m worst case across seeds; at "
                "eight, an 84 m median miss and ten tonnes of propellant."
            )
    else:
        notes.append(
            "Open loop: one plan, computed once, flown to the ground whatever "
            "the air does. This is exactly Day 9's strategy, and the reason "
            "its 250-sample sweep landed only 29.6% of the time."
        )
    return notes


def _failed(status: str, notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status=status, feasible=False,
        solver="closed-loop guidance", notes=notes,
    )
