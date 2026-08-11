"""
Day 4 — free-final-time landing, in the viewer.

Wraps `src.landing_free_time.solve_landing_free_time`. The burn duration is no
longer a slider: it is the answer. Everything else is a control, and the panel
reports which duration the search picked, how many it rejected, and why.

Two things this problem exists to make visible.

**The duration is searched, not declared.** Time enters the dynamics
multiplicatively, so it cannot be a variable in a convex program. Instead the
convex problem is solved at many fixed durations and the best is kept. The
`t_f` bounds slider sets the interval that gets searched.

**Losslessness is checked, not assumed.** A duration whose relaxation has gone
slack is rejected outright: it burns propellant at the sigma rate while
commanding less force than that, so it is cheap on paper and unflyable in fact.
Turn `require_lossless` off to see the rejected trajectories — the status chip
will warn rather than reassure.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.landing_free_time import solve_landing_free_time
from src.landing_problem import feasible_entry_state, max_downrange
from src.dynamics import Vehicle, G_EARTH

from ..registry import Problem, register
from ..types import Param, Series, Trajectory, attitudes_from_thrust


@register
class FreeTimeLanding(Problem):
    slug = "landing-free-time"
    title = "Free-Final-Time Landing"
    summary = "The optimiser picks the burn duration; losslessness is enforced."
    phase = "Day 4"
    scene_scale = 1600.0

    def params(self) -> list[Param]:
        return [
            Param("method", "Discretisation", "trapz", kind="choice",
                  choices=["trapz", "euler"], group="Method",
                  help="Trapezoidal is second-order. Euler reports LESS fuel "
                       "because its model is wrong, not because it is better - "
                       "compare the replay error, not the cost."),
            Param("N", "Nodes", 40, kind="int", min=20, max=70, step=5,
                  group="Method",
                  help="Each duration evaluated is a full convex solve, so "
                       "this drives response time."),

            Param("t_f_min", "Search from", 8.0, min=5.0, max=20.0, step=0.5,
                  unit="s", group="Duration search"),
            Param("t_f_max", "Search to", 34.0, min=20.0, max=60.0, step=1.0,
                  unit="s", group="Duration search"),
            Param("require_lossless", "Reject slack relaxations", True,
                  kind="bool", group="Duration search",
                  help="Off: the cheapest duration wins even if its thrust "
                       "command is below the throttle floor."),

            Param("t_nominal", "Entry sizing duration", 20.0, min=10.0,
                  max=30.0, step=0.5, unit="s", group="Entry state",
                  help="Sets the entry altitude and speed. Held fixed while "
                       "the burn duration is searched - otherwise the search "
                       "would be moving the problem, not solving it."),
            Param("x_frac", "Downrange (fraction of corridor)", 0.75,
                  min=0.0, max=1.15, step=0.01, group="Entry state"),
            Param("vx0", "Entry horizontal speed", -40.0, min=-150.0,
                  max=150.0, step=1.0, unit="m/s", group="Entry state"),

            Param("gamma_gs_deg", "Glideslope angle", 80.0, min=40.0, max=88.0,
                  step=1.0, unit="deg", group="Constraints"),
            Param("theta_max_deg", "Max thrust tilt", 30.0, min=6.0, max=70.0,
                  step=1.0, unit="deg", group="Constraints",
                  help="Where this saturates, lossless convexification stops "
                       "being lossless - the magnitude-only proof does not "
                       "cover an active pointing constraint."),

            Param("m_prop", "Landing propellant", 30000.0, min=10000.0,
                  max=60000.0, step=1000.0, unit="kg", group="Vehicle"),
            Param("n_engines", "Engines lit", 3, kind="int", min=1, max=6,
                  step=1, group="Vehicle"),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)

        vehicle = Vehicle(m_prop_initial=float(p["m_prop"]),
                          n_engines=int(p["n_engines"]))
        gamma = float(p["gamma_gs_deg"])
        theta = float(p["theta_max_deg"])
        t_nom = float(p["t_nominal"])

        try:
            z0, vz0 = feasible_entry_state(vehicle, t_nom, theta)
        except ValueError as exc:
            return _failed("error", [str(exc)])
        x0 = float(p["x_frac"]) * max_downrange(z0, gamma)

        t_lo = float(p["t_f_min"])
        t_hi = max(float(p["t_f_max"]), t_lo + 2.0)

        t_start = time.perf_counter()
        try:
            r = solve_landing_free_time(
                vehicle=vehicle, N=int(p["N"]),
                t_f_min=t_lo, t_f_max=t_hi,
                x0=x0, z0=z0, vx0=float(p["vx0"]), vz0=vz0,
                gamma_gs_deg=gamma, theta_max_deg=theta,
                method=str(p["method"]), t_nominal=t_nom,
                n_scan=9, n_refine=8,
                require_lossless=bool(p["require_lossless"]),
                verbose=False,
            )
        except Exception as exc:      # noqa: BLE001
            return _failed("error", [f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t_start) * 1000.0

        if not str(r.get("status", "")).startswith("optimal"):
            return _failed(r.get("status", "infeasible"),
                           _diagnose(r, vehicle, t_lo, t_hi, theta, gamma,
                                     x0, z0, bool(p["require_lossless"])))

        # Trapezoidal carries a control at every node; the viewer's contract
        # wants one per interval, so the terminal command is dropped for
        # rendering only. It still shaped the collocation.
        n_int = len(r["t"]) - 1
        Tx, Tz, sig = r["Tx"][:n_int], r["Tz"][:n_int], r["sigma"][:n_int]

        pos = np.column_stack([r["x"], r["z"], np.zeros(len(r["t"]))])
        vel = np.column_stack([r["vx"], r["vz"], np.zeros(len(r["t"]))])
        thrust = np.column_stack([Tx, Tz, np.zeros(n_int)])

        T_mag = np.hypot(Tx, Tz)
        tilt = np.degrees(np.arctan2(np.abs(Tx), np.maximum(Tz, 1.0)))
        speed = np.hypot(r["vx"], r["vz"])
        twr = T_mag / (r["m"][:n_int] * G_EARTH)

        sweep = r.get("sweep", [])
        ok_pts = [(t, f) for t, f, k in sweep if k == "ok"]
        slack_pts = [(t, f) for t, f, k in sweep if k == "slack"]
        feas = [t for t, _ in ok_pts]

        notes = [
            f"Searched {r.get('n_solves', 0)} durations in "
            f"[{t_lo:.0f}, {t_hi:.0f}] s and chose {r['t_f']:.2f} s for "
            f"{r['fuel']:,.0f} kg. Burn duration cannot be a variable in a "
            f"convex program - time multiplies the states - so it is searched "
            f"with a convex solve at every point.",
        ]
        if feas:
            notes.append(
                f"Flyable window is {min(feas):.1f} - {max(feas):.1f} s. "
                f"Shorter runs out of thrust authority; longer runs out of "
                f"entry energy, because minimum throttle decelerates harder "
                f"than the arrival speed can absorb."
            )
        if slack_pts:
            cheapest = min(f for _, f in slack_pts)
            notes.append(
                f"{len(slack_pts)} duration(s) solved but were rejected: the "
                f"lossless relaxation went slack, the cheapest at "
                f"{cheapest:,.0f} kg. Those trajectories burn propellant at "
                f"the sigma rate while commanding less force than that - "
                f"cheaper on paper, unflyable. Every slack case has the "
                f"pointing constraint at its limit, though saturation alone "
                f"does not force a gap."
            )
        if not bool(p["require_lossless"]) and r.get("relaxation_gap", 0) > 0.01 * vehicle.T_min:
            notes.append(
                f"WARNING: losslessness is not being enforced and this "
                f"solution is slack by {r['relaxation_gap']:,.0f} N. Minimum "
                f"commanded thrust is {r['min_thrust_over_Tmin']:.2f}x T_min. "
                f"Do not fly it."
            )

        return Trajectory(
            t_state=r["t"].tolist(),
            t_control=r["t"][:-1].tolist(),
            position=pos.tolist(),
            velocity=vel.tolist(),
            thrust=thrust.tolist(),
            attitude=attitudes_from_thrust(thrust).tolist(),
            series=[
                Series("altitude", "Altitude", "m", r["z"].tolist()),
                Series("downrange", "Downrange", "m", r["x"].tolist()),
                Series("speed", "Speed", "m/s", speed.tolist()),
                Series("mass", "Vehicle mass", "kg", r["m"].tolist()),
                Series("thrust", "Thrust", "N", T_mag.tolist(), on="control"),
                Series("sigma", "Sigma (slack)", "N", sig.tolist(), on="control"),
                Series("tilt", "Thrust tilt", "deg", tilt.tolist(), on="control"),
                Series("twr", "Thrust/weight", "-", twr.tolist(), on="control"),
            ],
            status=r["status"],
            feasible=True,
            cost=float(r["fuel"]),
            solve_time_ms=elapsed,
            solver=f"{r['method']} + duration search",
            thrust_max=vehicle.T_max,
            notes=notes,
            diagnostics={
                "chosen_burn_time_s": float(r["t_f"]),
                "fuel_kg": float(r["fuel"]),
                "convex_solves": int(r.get("n_solves", 0)),
                "rejected_slack": int(r.get("n_rejected_slack", 0)),
                "flyable_window_lo_s": float(min(feas)) if feas else None,
                "flyable_window_hi_s": float(max(feas)) if feas else None,
                "relaxation_gap_N": float(r.get("relaxation_gap", 0.0)),
                "peak_tilt_deg": float(r.get("max_tilt_deg", tilt.max())),
                "entry_altitude_m": float(z0),
                "entry_speed_ms": float(np.hypot(p["vx0"], vz0)),
                "final_position_error_m": float(np.hypot(r["x"][-1], r["z"][-1])),
                "final_velocity_error_ms": float(np.hypot(r["vx"][-1], r["vz"][-1])),
                "glideslope_deg": gamma,
            },
        )


def _failed(status: str, notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status=status, feasible=False,
        solver="duration search", notes=notes,
    )


def _diagnose(r, vehicle, t_lo, t_hi, theta, gamma, x0, z0, strict) -> list[str]:
    rejected = int(r.get("n_rejected_slack", 0))
    notes = [f"No flyable duration in [{t_lo:.0f}, {t_hi:.0f}] s."]

    if rejected and strict:
        notes.append(
            f"{rejected} duration(s) DID solve, but every one had a slack "
            f"relaxation and was rejected. The pointing limit "
            f"({theta:.0f} deg) is saturated across the whole window - relax "
            f"it and the relaxation becomes lossless again."
        )
    corridor = max_downrange(z0, gamma)
    if abs(x0) > corridor:
        notes.append(
            f"GEOMETRY: entry is {abs(x0):,.0f} m downrange but the "
            f"{gamma:.0f} deg corridor allows {corridor:,.0f} m at "
            f"{z0:,.0f} m altitude."
        )
    if not rejected:
        notes.append(
            "Nothing solved at all, so the entry state is unreachable across "
            "the searched interval. Widen the bounds, or change the entry "
            "sizing duration - it sets the altitude and speed the burn has "
            "to absorb."
        )
    return notes
