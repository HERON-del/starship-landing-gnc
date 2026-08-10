"""
Day 2 — open-loop powered-descent simulation.

This is the Day 2 physics engine made interactive: no optimiser, just the
verified 3-DoF variable-mass model propagated forward under a chosen guidance
law with a chosen integrator. It exists so the Day 2 experiments can be run by
dragging a slider instead of editing a notebook.

Each exploration experiment maps onto a control on the panel:

    A  suicide burn altitude   -> guidance "suicide burn" + ignition altitude
    B  cannot hover            -> guidance "hover" + enforce throttle limits ON
    C  propellant budget       -> guidance "hover", watch the propellant readout
    D  integrator breakdown    -> integrator "euler" + a large time step

Unlike the optimiser problems, this one can fail *physically* — it will happily
fly the vehicle into the ground at 200 m/s and report the crash. That is the
point: it shows what the guidance actually does, not what you wish it did.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.dynamics import (
    Vehicle,
    dynamics_3dof,
    G_EARTH,
)
from src.integrators import propagate

from ..registry import Problem, register
from ..types import Param, Series, Trajectory, attitudes_from_thrust

SOFT_TOUCHDOWN = 2.0     # m/s, below this counts as a landing
HARD_TOUCHDOWN = 10.0    # m/s, above this counts as a crash


def _make_control(mode: str, vehicle: Vehicle, throttle: float,
                  ignite_alt: float, enforce_limits: bool):
    """Build a control_fn(t, state, vehicle) -> [Tx, Tz] for the chosen law."""
    T_cmd = vehicle.T_max * throttle

    def raw(t, state, veh):
        if mode == "ballistic":
            return np.array([0.0, 0.0])
        if mode == "hover":
            return np.array([0.0, state[4] * G_EARTH])
        if mode == "suicide burn":
            return np.array([0.0, T_cmd]) if state[1] <= ignite_alt else np.zeros(2)
        if mode == "closed-loop":
            # Null the velocity exactly at the pad. Constant deceleration from
            # (z, vz) to (0, 0) needs a = vz^2 / 2z, plus g to cancel weight.
            # Below minimum throttle the engine simply cannot be lit, so the
            # vehicle coasts until the demand rises above T_min -- which is the
            # ignition trigger, computed rather than guessed.
            z, vz = state[1], state[3]
            if z <= 0.0 or vz >= 0.0:
                return np.array([0.0, state[4] * G_EARTH])   # hold once arrested
            T_req = state[4] * (vz * vz / (2.0 * z) + G_EARTH)
            if enforce_limits and T_req < veh.T_min:
                return np.zeros(2)                            # cannot throttle that low
            return np.array([0.0, min(T_req, veh.T_max)])
        # "constant throttle"
        return np.array([0.0, T_cmd])

    if not enforce_limits:
        return raw

    def clamped(t, state, veh):
        u = raw(t, state, veh)
        mag = float(np.hypot(u[0], u[1]))
        if mag <= 0.0:
            return u                       # engines off is always allowed
        # A lit engine cannot go below minimum throttle. This is exactly what
        # makes hovering impossible near dry mass.
        target = min(max(mag, veh.T_min), veh.T_max)
        return u * (target / mag)

    return clamped


@register
class DescentSim3DoF(Problem):
    slug = "descent-sim"
    title = "Powered Descent Simulation"
    summary = "Open-loop propagation of the verified variable-mass model."
    phase = "Day 2"
    scene_scale = 1200.0
    enforces_terminal_state = False   # open-loop: crashing is a valid outcome

    def params(self) -> list[Param]:
        return [
            Param("z0", "Initial altitude", 3000.0, min=200.0, max=8000.0,
                  step=50.0, unit="m", group="Initial state"),
            Param("vz0", "Initial vertical velocity", -200.0, min=-400.0, max=0.0,
                  step=5.0, unit="m/s", group="Initial state",
                  help="Negative is descending. Set to 0 for the hover "
                       "experiment - hover cancels weight, not velocity."),
            Param("x0", "Downrange offset", 0.0, min=-2000.0, max=2000.0,
                  step=25.0, unit="m", group="Initial state"),
            Param("vx0", "Horizontal velocity", 0.0, min=-100.0, max=100.0,
                  step=1.0, unit="m/s", group="Initial state"),

            Param("guidance", "Guidance law", "closed-loop", kind="choice",
                  choices=["closed-loop", "suicide burn", "constant throttle",
                           "hover", "ballistic"],
                  group="Guidance",
                  help="Closed-loop throttles to null velocity at the pad and "
                       "lands. Suicide burn is open-loop: it coasts, then goes to "
                       "full commanded thrust below the ignition altitude - and "
                       "essentially cannot land, which is the point."),
            Param("ignite_alt", "Ignition altitude", 450.0, min=50.0, max=4000.0,
                  step=10.0, unit="m", group="Guidance",
                  help="Experiment A: find the value that lands at 0 m/s."),
            Param("throttle", "Commanded throttle", 1.0, min=0.1, max=1.0,
                  step=0.01, group="Guidance"),
            Param("enforce_limits", "Enforce throttle limits", True, kind="bool",
                  group="Guidance",
                  help="Experiment B: a lit engine cannot go below 40%. Try "
                       "hovering with this on."),

            Param("m_prop", "Landing propellant", 30000.0, min=5000.0,
                  max=80000.0, step=1000.0, unit="kg", group="Vehicle"),
            Param("n_engines", "Engines lit", 3, kind="int", min=1, max=6,
                  step=1, group="Vehicle"),

            Param("method", "Integrator", "rk4", kind="choice",
                  choices=["rk4", "euler"], group="Integration",
                  help="Experiment D: switch to euler and raise the time step."),
            Param("dt", "Time step", 0.05, min=0.005, max=2.0, step=0.005,
                  unit="s", group="Integration"),
            Param("t_max", "Max sim time", 60.0, min=10.0, max=180.0, step=5.0,
                  unit="s", group="Integration"),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)

        vehicle = Vehicle(
            m_prop_initial=float(p["m_prop"]),
            n_engines=int(p["n_engines"]),
        )

        y0 = np.array([
            float(p["x0"]), float(p["z0"]),
            float(p["vx0"]), float(p["vz0"]),
            vehicle.m_wet,
        ])

        ctrl = _make_control(
            str(p["guidance"]), vehicle, float(p["throttle"]),
            float(p["ignite_alt"]), bool(p["enforce_limits"]),
        )

        dt = float(p["dt"])
        t_max = float(p["t_max"])

        try:
            t, y = propagate(dynamics_3dof, y0, (0.0, t_max), dt,
                             ctrl, vehicle, method=str(p["method"]))
        except Exception as exc:
            return Trajectory(
                t_state=[], t_control=[], position=[], velocity=[],
                thrust=[], attitude=[], status="error", feasible=False,
                solver=str(p["method"]),
                notes=[f"{type(exc).__name__}: {exc}"],
            )

        if not np.all(np.isfinite(y)):
            return Trajectory(
                t_state=[], t_control=[], position=[], velocity=[],
                thrust=[], attitude=[], status="diverged", feasible=False,
                solver=str(p["method"]),
                notes=["Integration diverged and produced non-finite values. "
                       "Reduce the time step."],
            )

        t, y, touchdown = _truncate_at_ground(t, y)

        if len(t) < 2:
            return Trajectory(
                t_state=[], t_control=[], position=[], velocity=[],
                thrust=[], attitude=[], status="no flight", feasible=False,
                solver=str(p["method"]),
                notes=["The vehicle starts at or below ground level."],
            )

        # Re-evaluate the control that was actually applied on each interval,
        # so the plume and the path colour reflect the real commanded thrust.
        thrust = np.array([ctrl(t[k], y[k], vehicle) for k in range(len(t) - 1)])
        thrust_3d = np.column_stack([thrust[:, 0],
                                     thrust[:, 1],
                                     np.zeros(len(thrust))])
        t_mag = np.linalg.norm(thrust_3d, axis=1)

        pos = np.column_stack([y[:, 0], y[:, 1], np.zeros(len(t))])
        vel = np.column_stack([y[:, 2], y[:, 3], np.zeros(len(t))])
        speed = np.linalg.norm(vel, axis=1)
        mass = y[:, 4]

        prop_used = float(vehicle.m_wet - mass[-1])
        prop_left = float(mass[-1] - vehicle.m_dry)
        v_touch = float(abs(y[-1, 3]))
        twr = t_mag / (mass[:-1] * G_EARTH)

        notes, status = _assess(
            touchdown, v_touch, y, vehicle, prop_left, p, twr,
        )

        return Trajectory(
            t_state=t.tolist(),
            t_control=t[:-1].tolist(),
            position=pos.tolist(),
            velocity=vel.tolist(),
            thrust=thrust_3d.tolist(),
            attitude=attitudes_from_thrust(thrust_3d).tolist(),
            series=[
                Series("altitude", "Altitude", "m", y[:, 1].tolist()),
                Series("vz", "Vertical velocity", "m/s", y[:, 3].tolist()),
                Series("speed", "Speed", "m/s", speed.tolist()),
                Series("mass", "Vehicle mass", "kg", mass.tolist()),
                Series("thrust", "Thrust", "N", t_mag.tolist(), on="control"),
                Series("twr", "Thrust/weight", "-", twr.tolist(), on="control"),
            ],
            status=status,
            feasible=True,
            cost=prop_used,
            solve_time_ms=None,
            solver=f"{p['method']}  dt={dt:g}s",
            thrust_max=vehicle.T_max,
            notes=notes,
            diagnostics={
                "touchdown_speed_ms": v_touch,
                "flight_time_s": float(t[-1]),
                "propellant_used_kg": prop_used,
                "propellant_remaining_kg": prop_left,
                "peak_twr": float(twr.max()) if len(twr) else 0.0,
                "min_twr_lit": float(twr[t_mag > 0].min()) if np.any(t_mag > 0) else 0.0,
                "final_altitude_m": float(y[-1, 1]),
                "steps": int(len(t) - 1),
            },
        )


def _truncate_at_ground(t: np.ndarray, y: np.ndarray):
    """
    Cut the trajectory at the first ground contact and interpolate the last
    sample onto z = 0, so the vehicle lands on the pad instead of sinking
    through it or stopping short.
    """
    below = np.nonzero(y[:, 1] <= 0.0)[0]
    if len(below) == 0:
        return t, y, False

    i = int(below[0])
    if i == 0:
        return t[:1], y[:1], True

    z_prev, z_now = y[i - 1, 1], y[i, 1]
    f = z_prev / (z_prev - z_now) if z_prev != z_now else 0.0

    y_land = y[i - 1] + f * (y[i] - y[i - 1])
    y_land[1] = 0.0
    t_land = t[i - 1] + f * (t[i] - t[i - 1])

    return (np.append(t[:i], t_land),
            np.vstack([y[:i], y_land]),
            True)


def _assess(touchdown, v_touch, y, vehicle, prop_left, p, twr):
    """Turn the raw numbers into the sentences the flight-data panel shows."""
    notes: list[str] = []

    if not touchdown:
        status = "still airborne"
        notes.append(
            f"Never reached the ground within the {float(p['t_max']):.0f} s sim "
            f"window — final altitude {y[-1, 1]:,.0f} m, vertical velocity "
            f"{y[-1, 3]:+.1f} m/s."
        )
    elif v_touch <= SOFT_TOUCHDOWN:
        status = "soft landing"
        notes.append(f"Soft landing at {v_touch:.2f} m/s.")
    elif v_touch <= HARD_TOUCHDOWN:
        status = "hard landing"
        notes.append(f"Hard landing at {v_touch:.1f} m/s — survivable, not good.")
    else:
        status = "crash"
        notes.append(f"Crashed at {v_touch:.1f} m/s.")

    if prop_left <= 1.0:
        notes.append(
            "Propellant exhausted before touchdown; the engines cut out and the "
            "rest of the descent was ballistic."
        )

    if p["guidance"] == "suicide burn":
        notes.append(
            "Open-loop suicide burn: thrust is fixed, so the only tuning knob is "
            "the ignition altitude. Ignite a metre too low and it crashes; a metre "
            "too high and it arrests the descent above the pad and climbs away on "
            "TWR ~6. There is no setting that reliably lands, because the trigger "
            "is only tested once per step. Switch to closed-loop and compare - "
            "that gap is the entire reason guidance is computed rather than "
            "scheduled."
        )

    if p["guidance"] == "closed-loop":
        notes.append(
            "Closed-loop law: thrust tracks a = v^2/2z + g, the constant "
            "deceleration that nulls velocity exactly at the pad. Ignition is not "
            "scheduled - the engine lights itself the moment the demand exceeds "
            "minimum throttle."
        )

    if p["guidance"] == "hover":
        twr_min = vehicle.T_min / (vehicle.m_wet * G_EARTH)
        twr_dry = vehicle.T_min / (vehicle.m_dry * G_EARTH)

        if abs(float(p["vz0"])) > 1e-6:
            notes.append(
                f"Hover thrust cancels weight, not velocity. Starting at "
                f"{float(p['vz0']):+.0f} m/s the net acceleration is zero, so the "
                f"vehicle holds that descent rate all the way down. Set the "
                f"initial vertical velocity to 0 to see a true hover."
            )

        if p["enforce_limits"] and twr_min > 1.0:
            notes.append(
                f"Thrust-to-weight at minimum throttle is {twr_min:.2f} at wet "
                f"mass and {twr_dry:.2f} at dry mass — above 1 throughout. With "
                f"throttle limits enforced the vehicle cannot hover at any point "
                f"in the burn, so it climbs away. Turn the limits off and it "
                f"holds altitude exactly."
            )
        elif not p["enforce_limits"]:
            notes.append(
                "Ideal hover with throttle limits disabled: thrust tracks weight "
                "exactly, so altitude is conserved. Endurance is "
                f"Isp*ln(m_wet/m_dry) = "
                f"{vehicle.isp * np.log(vehicle.m_wet / vehicle.m_dry):.1f} s."
            )

    if p["method"] == "euler" and float(p["dt"]) >= 0.5:
        notes.append(
            f"Euler at dt = {float(p['dt']):g} s accumulates first-order error. "
            "Switch to RK4 at the same step and compare the touchdown numbers — "
            "that difference is the whole argument for a 4th-order integrator."
        )

    return notes, status
