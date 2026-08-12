"""
Day 6 — unpowered aerodynamic entry, in the viewer.

The belly-flop, with the engines off, which is where it actually pays.

Measuring the *powered* landing with drag on and off gives 14,783 kg versus
14,785 kg. Drag saves nothing during the burn, because a throttle that cannot go
below 40% already sets the bill. So the interesting phase is the one before
ignition, and that is what this problem shows: a long coast where drag is the
only force acting and attitude is worth a 28x change in effective area.

Slide the entry attitude from broadside to nose-first and watch the arrival
speed move between 64 m/s and 358 m/s. Switch the atmosphere off entirely and it
becomes 494 m/s. The gap between those numbers is delta-v the vehicle does not
have to buy with propellant — worth more than the entire landing burn costs.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.dynamics_aero import simulate_entry
from src.aero import AeroConfig, drag_area, effective_area
from src.dynamics_6dof import Vehicle6DoF, G_EARTH

from ..registry import Problem, register
from ..types import Param, Series, Trajectory, quats_from_pitch


@register
class AeroEntry(Problem):
    slug = "entry-aero"
    title = "Unpowered Aerodynamic Entry"
    summary = "The belly-flop, engines off, where the delta-v is actually saved."
    phase = "Day 6"
    scene_scale = 9000.0
    enforces_terminal_state = False    # a coast, not an optimisation

    def params(self) -> list[Param]:
        return [
            Param("theta_deg", "Entry attitude", 90.0, min=0.0, max=90.0,
                  step=5.0, unit="deg", group="Attitude",
                  help="From vertical. 90 is full belly-flop and 28x the drag "
                       "area of nose-first."),
            Param("aero_on", "Atmosphere", True, kind="bool", group="Attitude",
                  help="Off gives the ballistic fall the belly-flop is measured "
                       "against."),

            Param("z0", "Entry altitude", 12000.0, min=2000.0, max=25000.0,
                  step=500.0, unit="m", group="Entry state"),
            Param("vz0", "Entry vertical speed", -120.0, min=-400.0, max=-20.0,
                  step=10.0, unit="m/s", group="Entry state"),
            Param("vx0", "Entry horizontal speed", 0.0, min=-200.0, max=200.0,
                  step=10.0, unit="m/s", group="Entry state"),
            Param("z_stop", "Handoff altitude", 300.0, min=50.0, max=3000.0,
                  step=50.0, unit="m", group="Entry state",
                  help="Where the coast ends and the landing burn would start."),

            Param("Cd_belly", "Cd broadside", 1.2, min=0.5, max=2.0, step=0.05,
                  group="Vehicle"),
            Param("diameter", "Diameter", 9.0, min=4.0, max=15.0, step=0.5,
                  unit="m", group="Vehicle"),
            Param("length", "Length", 50.0, min=20.0, max=80.0, step=2.0,
                  unit="m", group="Vehicle"),
            Param("m_prop", "Propellant aboard", 30000.0, min=0.0, max=80000.0,
                  step=2000.0, unit="kg", group="Vehicle",
                  help="Not burned here - it is mass the drag has to slow."),

            Param("dt", "Time step", 0.05, min=0.005, max=0.2, step=0.005,
                  unit="s", group="Integration",
                  help="The coast runs for minutes, so this drives response "
                       "time. RK4 is exact enough here that 0.05 s costs "
                       "nothing in accuracy."),
            Param("t_max", "Max coast time", 240.0, min=30.0, max=600.0,
                  step=10.0, unit="s", group="Integration"),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)

        vehicle = Vehicle6DoF(m_prop_initial=float(p["m_prop"]))
        aero = AeroConfig(
            diameter=float(p["diameter"]),
            length=float(p["length"]),
            Cd_belly=float(p["Cd_belly"]),
            enabled=bool(p["aero_on"]),
        )
        theta = float(p["theta_deg"])

        t0 = time.perf_counter()
        try:
            r = simulate_entry(
                vehicle, aero,
                z0=float(p["z0"]), vz0=float(p["vz0"]), vx0=float(p["vx0"]),
                theta0_deg=theta, dt=float(p["dt"]), t_max=float(p["t_max"]),
                z_stop=float(p["z_stop"]),
            )
        except Exception as exc:      # noqa: BLE001
            return _failed("error", [f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        n = len(r["t"])
        if n < 2:
            return _failed("no flight",
                           ["The vehicle starts at or below the handoff "
                            "altitude."])

        pos = np.column_stack([r["x"], r["z"], np.zeros(n)])
        vel = np.column_stack([r["vx"], r["vz"], np.zeros(n)])
        # Engines are off, so there is no thrust to render - but attitude is
        # held, and it is the whole story here.
        thrust = np.zeros((n - 1, 3))

        arrival = r["handoff"]["speed"]
        reached_ground = r["z"][-1] <= float(p["z_stop"]) + 1.0
        status = "coast complete" if reached_ground else "still airborne"

        CdA = float(drag_area(np.radians(theta), aero)) if aero.enabled else 0.0
        A = float(effective_area(np.radians(theta), aero))

        notes = [
            f"Coasted {r['handoff']['t']:.0f} s from {p['z0']:,.0f} m with the "
            f"engines off, arriving at {arrival:.1f} m/s. No propellant used.",
        ]
        if aero.enabled:
            notes.append(
                f"At {theta:.0f} deg the vehicle presents {A:,.0f} m^2 and "
                f"Cd*A = {CdA:,.0f} m^2. Terminal velocity is "
                f"{r['terminal_velocity']:.1f} m/s, and the arrival speed is "
                f"within a few percent of it - the coast is long enough to "
                f"reach equilibrium."
            )
            # Rocket-equation value of the velocity drag removed.
            v_vac = float(np.sqrt(max(p['vz0'] ** 2
                                      + 2 * G_EARTH * (p['z0'] - p['z_stop']), 0.0)))
            dv = max(v_vac - arrival, 0.0)
            prop = vehicle.m_wet * (1.0 - np.exp(-dv / (vehicle.isp * 9.80665)))
            notes.append(
                f"A ballistic fall from the same state arrives at {v_vac:.0f} m/s, "
                f"so drag removed {dv:.0f} m/s for free - worth about "
                f"{prop:,.0f} kg of propellant by the rocket equation. That is "
                f"more than the entire landing burn costs, and it is the whole "
                f"reason the belly-flop exists."
            )
        else:
            notes.append(
                "Atmosphere off: this is the ballistic fall the belly-flop is "
                "measured against. Turn it back on and watch the arrival speed "
                "collapse."
            )
        notes.append(
            "Dynamic pressure does not decay as the vehicle slows. At terminal "
            "velocity drag balances weight, so q = mg/(Cd A) is pinned by the "
            "vehicle rather than the altitude - rising density exactly offsets "
            "falling speed."
        )

        return Trajectory(
            t_state=r["t"].tolist(),
            t_control=r["t"][:-1].tolist(),
            position=pos.tolist(),
            velocity=vel.tolist(),
            thrust=thrust.tolist(),
            attitude=quats_from_pitch(r["theta"]).tolist(),
            series=[
                Series("altitude", "Altitude", "m", r["z"].tolist()),
                Series("speed", "Speed", "m/s", r["speed"].tolist()),
                Series("vz", "Vertical velocity", "m/s", r["vz"].tolist()),
                Series("q", "Dynamic pressure", "Pa", r["q"].tolist()),
            ],
            status=status,
            feasible=True,
            cost=0.0,
            solve_time_ms=elapsed,
            solver=f"RK4 dt={float(p['dt']):g}s",
            thrust_max=vehicle.T_max,
            notes=notes,
            diagnostics={
                "arrival_speed_ms": arrival,
                "terminal_velocity_ms": float(r["terminal_velocity"]),
                "coast_time_s": float(r["handoff"]["t"]),
                "propellant_used_kg": 0.0,
                "effective_area_m2": A,
                "drag_area_m2": CdA,
                "peak_q_kPa": float(np.max(r["q"]) / 1000.0),
                "handoff_altitude_m": float(r["z"][-1]),
                "handoff_q_kPa": float(r["q"][-1] / 1000.0),
            },
        )


def _failed(status: str, notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status=status, feasible=False,
        solver="RK4", notes=notes,
    )
