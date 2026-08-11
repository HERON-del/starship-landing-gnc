"""
Day 3 — constrained minimum-fuel landing, in the viewer.

Wraps `src.landing_problem.solve_landing` so the constrained optimiser can be
driven from sliders. The three constraints that make this problem realistic are
all live controls:

    glideslope   the approach corridor, drawn as the translucent cone
    pointing     the gimbal/tilt limit on the thrust vector
    throttle     T_min <= ||T|| <= T_max, via lossless convexification

Two design notes on the controls.

Entry state is derived from the burn duration by default rather than dialled in
directly. Minimum throttle gives TWR 2.16, so a lit engine can only decelerate,
and an arbitrary (altitude, speed) pair is almost always unreachable — the
sliders would spend most of their travel on "infeasible". Turn the toggle off to
set them by hand and watch how narrow the reachable band actually is.

Downrange is expressed as a *fraction of the glideslope corridor* rather than in
metres, for the same reason: the corridor width depends on both altitude and
cone angle, so a fixed metre value goes in and out of the cone as you move the
other sliders. At a fraction above 1.0 the entry point is outside the cone and
the problem is infeasible on geometry alone — which is worth seeing.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.landing_problem import (
    solve_landing,
    feasible_entry_state,
    max_downrange,
    min_arrestable_speed,
)
from src.dynamics import Vehicle, G_EARTH

from ..registry import Problem, register
from ..types import Param, Series, Trajectory, attitudes_from_thrust


@register
class ConstrainedLanding(Problem):
    slug = "landing-constrained"
    title = "Constrained Landing (2-D)"
    summary = "Glideslope cone, throttle bounds and gimbal limit, minimum fuel."
    phase = "Day 3"
    scene_scale = 1600.0

    def params(self) -> list[Param]:
        return [
            Param("t_burn", "Burn time", 20.0, min=8.0, max=34.0, step=0.5,
                  unit="s", group="Mission",
                  help="Above ~34.9 s the vehicle runs dry: minimum throttle "
                       "flows 861 kg/s against a 30 t load."),
            Param("N", "Discretisation nodes", 40, kind="int", min=20, max=90,
                  step=5, group="Mission",
                  help="Each solve runs several mass-reference iterations, so "
                       "this drives the response time."),

            Param("auto_entry", "Derive entry from burn", True, kind="bool",
                  group="Entry state",
                  help="On: altitude and speed are sized to what the burn can "
                       "actually null. Off: set them by hand."),
            Param("margin", "Entry energy margin", 1.42, min=1.0, max=1.6,
                  step=0.01, group="Entry state",
                  help="Multiple of the minimum arrestable speed. Below ~1.4 "
                       "the lossless relaxation goes slack — see the notes."),
            Param("z0", "Entry altitude (manual)", 2900.0, min=200.0,
                  max=9000.0, step=50.0, unit="m", group="Entry state"),
            Param("vz0", "Entry vertical speed (manual)", -280.0, min=-700.0,
                  max=-50.0, step=5.0, unit="m/s", group="Entry state"),

            Param("x_frac", "Downrange (fraction of corridor)", 0.75,
                  min=0.0, max=1.2, step=0.01, group="Entry state",
                  help="Above 1.0 the entry point is outside the glideslope "
                       "cone and no trajectory exists."),
            Param("vx0", "Entry horizontal speed", -40.0, min=-150.0, max=150.0,
                  step=1.0, unit="m/s", group="Entry state"),

            Param("gamma_gs_deg", "Glideslope angle", 80.0, min=40.0, max=88.0,
                  step=1.0, unit="deg", group="Constraints",
                  help="From horizontal. Larger is steeper and tighter; the "
                       "corridor at altitude z is |x| <= z / tan(gamma)."),
            Param("theta_max_deg", "Max thrust tilt", 30.0, min=4.0, max=70.0,
                  step=1.0, unit="deg", group="Constraints",
                  help="Gimbal plus vehicle tilt, from vertical."),

            Param("m_prop", "Landing propellant", 30000.0, min=10000.0,
                  max=60000.0, step=1000.0, unit="kg", group="Vehicle"),
            Param("n_engines", "Engines lit", 3, kind="int", min=1, max=6,
                  step=1, group="Vehicle",
                  help="Fewer engines lowers the minimum-throttle floor and "
                       "widens the reachable entry band considerably."),

            Param("damping", "Mass-reference damping", 0.5, min=0.1, max=1.0,
                  step=0.05, group="Solver",
                  help="1.0 takes the new mass profile outright and oscillates, "
                       "because bang-bang switching times flip between "
                       "iterations. This is a crude trust region."),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)

        vehicle = Vehicle(
            m_prop_initial=float(p["m_prop"]),
            n_engines=int(p["n_engines"]),
        )
        t_burn = float(p["t_burn"])
        theta = float(p["theta_max_deg"])
        gamma = float(p["gamma_gs_deg"])

        # --- entry state -------------------------------------------------
        if bool(p["auto_entry"]):
            try:
                z0, vz0 = feasible_entry_state(
                    vehicle, t_burn, theta, margin=float(p["margin"])
                )
            except ValueError as exc:
                return _failed("error", [str(exc)])
        else:
            z0, vz0 = float(p["z0"]), float(p["vz0"])

        x0 = float(p["x_frac"]) * max_downrange(z0, gamma)

        # --- solve -------------------------------------------------------
        t_start = time.perf_counter()
        try:
            r = solve_landing(
                vehicle=vehicle, N=int(p["N"]), t_burn=t_burn,
                x0=x0, z0=z0, vx0=float(p["vx0"]), vz0=vz0,
                gamma_gs_deg=gamma, theta_max_deg=theta,
                damping=float(p["damping"]), verbose=False,
            )
        except Exception as exc:      # noqa: BLE001 - surface, do not crash
            return _failed("error", [f"{type(exc).__name__}: {exc}"])
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        if not r["status"].startswith("optimal"):
            return _failed(r["status"], _diagnose(
                vehicle, t_burn, theta, gamma, x0, z0, vz0
            ))

        # --- repackage for the viewer ------------------------------------
        n_state = len(r["t"])
        pos = np.column_stack([r["x"], r["z"], np.zeros(n_state)])
        vel = np.column_stack([r["vx"], r["vz"], np.zeros(n_state)])
        thrust = np.column_stack([r["Tx"], r["Tz"], np.zeros(len(r["Tx"]))])

        T_mag = np.hypot(r["Tx"], r["Tz"])
        sigma = r["sigma"]
        tilt = np.degrees(np.arctan2(np.abs(r["Tx"]),
                                     np.maximum(r["Tz"], 1.0)))
        speed = np.hypot(r["vx"], r["vz"])
        twr = T_mag / (r["m"][:-1] * G_EARTH)
        gap = float(np.max(sigma - T_mag))
        rel_gap = gap / vehicle.T_min

        notes = [
            f"Fuel {r['fuel']:,.0f} kg, {100 * r['fuel'] / vehicle.m_prop_initial:.0f}% "
            f"of the landing load. Peak thrust {T_mag.max() / vehicle.T_max * 100:.0f}% "
            f"of maximum, peak tilt {tilt.max():.1f} deg.",
        ]
        if rel_gap < 0.01:
            notes.append(
                f"Lossless convexification is tight: max(sigma - ||T||) is "
                f"{gap:,.0f} N, and ||T|| never drops below "
                f"{T_mag.min() / vehicle.T_min:.2f}x T_min. The relaxation is "
                f"standing in for the real minimum-throttle bound correctly."
            )
        else:
            notes.append(
                f"WARNING: the relaxation has gone slack. max(sigma - ||T||) is "
                f"{gap:,.0f} N ({100 * rel_gap:.1f}% of T_min) and ||T|| falls to "
                f"{T_mag.min() / vehicle.T_min:.2f}x T_min. This trajectory burns "
                f"minimum-throttle propellant while commanding less than "
                f"minimum-throttle force — it is not flyable. Raise the entry "
                f"energy margin."
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
                Series("sigma", "Sigma (slack)", "N", sigma.tolist(), on="control"),
                Series("tilt", "Thrust tilt", "deg", tilt.tolist(), on="control"),
                Series("twr", "Thrust/weight", "-", twr.tolist(), on="control"),
            ],
            status=r["status"],
            feasible=True,
            cost=float(r["fuel"]),
            solve_time_ms=elapsed_ms,
            solver="CLARABEL + mass-ref iteration",
            thrust_max=vehicle.T_max,
            notes=notes,
            diagnostics={
                "fuel_kg": float(r["fuel"]),
                "entry_speed_ms": float(np.hypot(p["vx0"], vz0)),
                "entry_altitude_m": float(z0),
                "entry_downrange_m": float(x0),
                "corridor_at_entry_m": float(max_downrange(z0, gamma)),
                "peak_thrust_frac": float(T_mag.max() / vehicle.T_max),
                "peak_tilt_deg": float(tilt.max()),
                "relaxation_gap_N": gap,
                "min_thrust_over_Tmin": float(T_mag.min() / vehicle.T_min),
                "final_position_error_m": float(np.hypot(r["x"][-1], r["z"][-1])),
                "final_velocity_error_ms": float(np.hypot(r["vx"][-1], r["vz"][-1])),
                # Draws the approach corridor in the 3-D scene.
                "glideslope_deg": gamma,
            },
        )


def _failed(status: str, notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status=status, feasible=False,
        solver="CLARABEL", notes=notes,
    )


def _diagnose(vehicle, t_burn, theta, gamma, x0, z0, vz0) -> list[str]:
    """Work the Day 3 checklist and say which constraint is the binding one."""
    notes = ["No trajectory exists for these settings. Working the checklist:"]

    corridor = max_downrange(z0, gamma)
    if abs(x0) > corridor:
        notes.append(
            f"GEOMETRY: the entry point is {abs(x0):,.0f} m downrange but the "
            f"{gamma:.0f} deg corridor at {z0:,.0f} m only allows "
            f"{corridor:,.0f} m. Outside the cone before the dynamics are even "
            f"considered — no amount of thrust or time fixes this."
        )

    try:
        v_req, drop_min, _ = min_arrestable_speed(vehicle, t_burn, theta)
    except ValueError:
        v_req = drop_min = None

    if v_req:
        if abs(vz0) < v_req:
            notes.append(
                f"ENERGY: entering at {abs(vz0):,.0f} m/s, but a {t_burn:.0f} s "
                f"burn cannot null less than {v_req:,.0f} m/s — minimum throttle "
                f"decelerates harder than that, so the vehicle would arrive "
                f"climbing. Enter faster or shorten the burn."
            )
        if z0 < drop_min * 0.95:
            notes.append(
                f"ALTITUDE: {z0:,.0f} m is below the {drop_min:,.0f} m the burn "
                f"needs to consume. The vehicle reaches the pad still moving."
            )

    mdot_min = vehicle.T_min / (vehicle.isp * 9.80665)
    max_burn = vehicle.m_prop_initial / mdot_min
    if t_burn > max_burn:
        notes.append(
            f"PROPELLANT: minimum throttle flows {mdot_min:,.0f} kg/s, so "
            f"{vehicle.m_prop_initial:,.0f} kg buys only {max_burn:.1f} s of "
            f"burn. A {t_burn:.0f} s burn runs dry."
        )

    if theta < 8.0:
        notes.append(
            f"POINTING: a {theta:.0f} deg tilt limit leaves almost no lateral "
            f"authority. Try relaxing it."
        )

    if len(notes) == 1:
        notes.append(
            "None of the cheap geometric checks fire, so the binding constraint "
            "is a combination. Relax one at a time: burn time, glideslope, "
            "pointing limit."
        )
    return notes
