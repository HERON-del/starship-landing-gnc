"""
Combined 6-DoF dynamics with aerodynamic forces, and the unpowered entry phase.

Day 5's `dynamics_6dof` models thrust, gravity and gimbal torque. This adds the
air: drag and a simplified lift, with a reference area that depends on attitude.
The result is the model the real vehicle actually flies.

The unpowered phase is the point
--------------------------------
Measuring the powered landing with drag on and off gives essentially the same
propellant - 14,783 kg versus 14,785 kg at a 60 degree entry. Drag saves nothing
there, because a throttle that cannot go below 40% already sets the bill: the
engines must burn for the whole descent whatever the air is doing.

So the belly-flop's value is not in the burn. It is in the minutes *before* the
burn, with the engines off, where drag is the only thing acting and it is worth
a 28x change in effective area. That is what `simulate_entry` models, and it is
why the real vehicle flips immediately before ignition rather than during it.
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.dynamics_6dof import (            # noqa: E402
    Vehicle6DoF, G0, G_EARTH, control_zero_6dof,
)
from src.aero import (                     # noqa: E402
    AeroConfig, aero_acceleration, dynamic_pressure, terminal_velocity,
)
from src.integrators import propagate      # noqa: E402


def dynamics_full(t, state, control_fn, vehicle: Vehicle6DoF,
                  aero: AeroConfig = None):
    """
    State derivative for the planar 6-DoF vehicle including air forces.

    State is `[x, z, vx, vz, theta, omega, m]`, identical to `dynamics_6dof`;
    the aerodynamic acceleration is simply added to the translational rows.
    """
    x, z, vx, vz, theta, omega, m = state

    T, delta = control_fn(t, state, vehicle)
    T = float(np.clip(T, 0.0, vehicle.T_max))
    delta = float(np.clip(delta, -vehicle.delta_max, vehicle.delta_max))

    if m <= vehicle.m_dry:
        m = vehicle.m_dry
        Tx = Tz = tau = 0.0
        mdot = 0.0
    else:
        Tx = T * np.sin(theta + delta)
        Tz = T * np.cos(theta + delta)
        tau = T * vehicle.L_engine * np.sin(delta)
        mdot = -T / (vehicle.isp * G0)

    if aero is not None and aero.enabled:
        ax, az = aero_acceleration(vx, vz, z, theta, m, aero)
    else:
        ax = az = 0.0

    return np.array([
        vx,
        vz,
        Tx / m + float(ax),
        Tz / m + float(az) - G_EARTH,
        omega,
        tau / vehicle.I_pitch,
        mdot,
    ])


def simulate_entry(
    vehicle: Vehicle6DoF = None,
    aero: AeroConfig = None,
    z0: float = 12_000.0,
    vz0: float = -120.0,
    vx0: float = 0.0,
    x0: float = 0.0,
    theta0_deg: float = 90.0,
    t_max: float = 240.0,
    dt: float = 0.02,
    z_stop: float = 300.0,
):
    """
    Unpowered belly-flop descent: engines off, drag doing all the work.

    Propagates until the vehicle drops below `z_stop` or the clock runs out.
    Attitude is held fixed — the flip belongs to the powered phase — so this is
    a pure terminal-velocity problem.

    Returns
    -------
    dict with the state history and the handoff conditions.
    """
    vehicle = vehicle or Vehicle6DoF()
    aero = aero or AeroConfig()
    theta0 = np.radians(theta0_deg)

    y0 = np.array([x0, z0, vx0, vz0, theta0, 0.0, vehicle.m_wet])
    t, y = propagate(
        lambda tt, yy, *a: dynamics_full(tt, yy, control_zero_6dof, vehicle, aero),
        y0, (0.0, t_max), dt, method="rk4",
    )

    below = np.flatnonzero(y[:, 1] <= z_stop)
    end = int(below[0]) if len(below) else len(t) - 1
    t, y = t[:end + 1], y[:end + 1]

    speed = np.hypot(y[:, 2], y[:, 3])
    q = np.asarray(dynamic_pressure(y[:, 2], y[:, 3], y[:, 1]))
    v_term = terminal_velocity(theta0, vehicle.m_wet, float(y[-1, 1]), aero)

    return {
        "t": t,
        "x": y[:, 0], "z": y[:, 1],
        "vx": y[:, 2], "vz": y[:, 3],
        "theta": y[:, 4], "omega": y[:, 5], "m": y[:, 6],
        "speed": speed,
        "q": q,
        "terminal_velocity": v_term,
        "handoff": {
            "t": float(t[-1]),
            "x": float(y[-1, 0]), "z": float(y[-1, 1]),
            "vx": float(y[-1, 2]), "vz": float(y[-1, 3]),
            "theta_deg": float(np.degrees(y[-1, 4])),
            "speed": float(speed[-1]),
            "q": float(q[-1]),
        },
        "propellant_used": 0.0,
        "theta0_deg": theta0_deg,
    }


def freefall_comparison(vehicle=None, aero=None, z0=12_000.0, vz0=-120.0,
                        t_max=240.0, dt=0.02):
    """
    What the belly-flop is worth, in one table.

    Drops the vehicle from the same state at several fixed attitudes and reports
    the speed it arrives with. With no air at all it is a straight ballistic
    fall, which is the number the aerodynamic entry is being measured against.
    """
    vehicle = vehicle or Vehicle6DoF()
    aero = aero or AeroConfig()
    rows = []

    for label, theta_deg, ae in (
        ("no atmosphere", 90.0, AeroConfig(enabled=False)),
        ("nose-first (0 deg)", 0.0, aero),
        ("45 deg", 45.0, aero),
        ("belly-flop (90 deg)", 90.0, aero),
    ):
        r = simulate_entry(vehicle, ae, z0=z0, vz0=vz0,
                           theta0_deg=theta_deg, t_max=t_max, dt=dt,
                           z_stop=300.0)
        h = r["handoff"]
        rows.append({
            "case": label,
            "arrival_speed": h["speed"],
            "arrival_q": h["q"],
            "time": h["t"],
            "terminal_velocity": r["terminal_velocity"],
        })
    return rows


if __name__ == "__main__":
    print("Unpowered descent from 12 km to 300 m, engines off\n")
    print(f"{'case':<22}{'arrival [m/s]':>15}{'q [kPa]':>10}{'t [s]':>9}"
          f"{'v_term [m/s]':>14}")
    for row in freefall_comparison():
        print(f"{row['case']:<22}{row['arrival_speed']:>15.1f}"
              f"{row['arrival_q'] / 1e3:>10.1f}{row['time']:>9.1f}"
              f"{row['terminal_velocity']:>14.1f}")
