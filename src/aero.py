"""
Aerodynamic forces for a belly-flopping vehicle.

The whole point of Starship's entry attitude is that reference area is a
*control*. Flat to the airflow the vehicle presents roughly `diameter x length`;
nose- or tail-first it presents only its base disc. Rotating is therefore not
just an attitude change, it is a 7x change in drag area — the flip is the
vehicle deliberately switching its own air brake off.

    A_eff(theta)  = A_base + (A_belly - A_base) |sin theta|
    Cd_eff(theta) = Cd_nose + (Cd_belly - Cd_nose) |sin theta|

Drag opposes the velocity vector. Written per component this is

    F_x = -0.5 rho Cd A |v| vx
    F_z = -0.5 rho Cd A |v| vz

using `v^2 * v_hat_x = v * vx`, which avoids a division by |v| and stays
well-behaved as the vehicle slows.

Lift is included in a deliberately crude form. For a blunt body at high angle of
attack L/D is small (~0.2-0.4) and the coefficient peaks near 45°, so
`Cl = Cl_max sin(2 alpha)` captures the shape without pretending to more
fidelity than a 2-D model can carry. `alpha` is the angle between the velocity
vector and the body axis.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.atmosphere import density   # noqa: E402


@dataclass
class AeroConfig:
    """Aerodynamic configuration for the vehicle."""

    # Geometry
    diameter: float = 9.0        # vehicle diameter [m]
    length: float = 50.0         # vehicle length [m]

    # Drag coefficients
    Cd_belly: float = 1.2        # blunt body broadside
    Cd_nose: float = 0.3         # nose-first / tail-first

    # Lift
    Cl_max: float = 0.4          # peak lift coefficient, near 45 deg AoA

    # Below this speed aero forces are negligible and the direction of the
    # velocity vector is numerically meaningless.
    v_min: float = 5.0           # [m/s]

    enabled: bool = True

    @property
    def A_belly(self) -> float:
        """Reference area broadside to the flow [m^2]."""
        return self.diameter * self.length

    @property
    def A_base(self) -> float:
        """Reference area nose- or tail-first [m^2]."""
        return np.pi * (self.diameter / 2.0) ** 2

    def summary(self) -> str:
        return "\n".join([
            "Aerodynamic configuration",
            f"  Diameter          : {self.diameter:.1f} m",
            f"  Length            : {self.length:.1f} m",
            f"  A_belly           : {self.A_belly:.0f} m^2",
            f"  A_base            : {self.A_base:.0f} m^2",
            f"  A_belly / A_base  : {self.A_belly / self.A_base:.1f}x",
            f"  Cd_belly          : {self.Cd_belly:.2f}",
            f"  Cd_nose           : {self.Cd_nose:.2f}",
            f"  Cl_max            : {self.Cl_max:.2f}",
            f"  Cd*A belly / base : "
            f"{(self.Cd_belly * self.A_belly) / (self.Cd_nose * self.A_base):.1f}x",
        ])


def effective_area(theta, aero: AeroConfig = None):
    """
    Reference area as a function of pitch from vertical [m^2].

    theta = 0 is upright (base area), theta = pi/2 is broadside (belly area).
    """
    aero = aero or AeroConfig()
    s = np.abs(np.sin(np.asarray(theta, dtype=float)))
    return aero.A_base + (aero.A_belly - aero.A_base) * s


def effective_Cd(theta, aero: AeroConfig = None):
    """Drag coefficient as a function of pitch from vertical [-]."""
    aero = aero or AeroConfig()
    s = np.abs(np.sin(np.asarray(theta, dtype=float)))
    return aero.Cd_nose + (aero.Cd_belly - aero.Cd_nose) * s


def drag_area(theta, aero: AeroConfig = None):
    """`Cd * A`, the product that actually sets the force [m^2]."""
    return effective_Cd(theta, aero) * effective_area(theta, aero)


def dynamic_pressure(vx, vz, z):
    """Dynamic pressure `q = 0.5 rho v^2` [Pa]."""
    v2 = np.asarray(vx, dtype=float) ** 2 + np.asarray(vz, dtype=float) ** 2
    return 0.5 * density(z) * v2


def angle_of_attack(vx, vz, theta):
    """
    Angle between the velocity vector and the body axis [rad].

    The body axis points along `(sin theta, cos theta)`. Returns 0 where the
    vehicle is essentially stationary, since the velocity direction is then
    meaningless.
    """
    vx = np.asarray(vx, dtype=float)
    vz = np.asarray(vz, dtype=float)
    theta = np.asarray(theta, dtype=float)

    speed = np.hypot(vx, vz)
    safe = speed > 1e-6
    v_ang = np.arctan2(np.where(safe, vx, 0.0), np.where(safe, vz, 1.0))
    alpha = v_ang - theta
    # wrap to [-pi, pi]
    alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
    return np.where(safe, alpha, 0.0)


def aero_force(vx, vz, z, theta, aero: AeroConfig = None):
    """
    Aerodynamic force in the inertial frame [N].

    Returns
    -------
    (Fx, Fz) : tuple of float or ndarray
        Drag opposing the velocity, plus a simplified lift perpendicular to it.
    """
    aero = aero or AeroConfig()
    vx = np.asarray(vx, dtype=float)
    vz = np.asarray(vz, dtype=float)

    if not aero.enabled:
        return np.zeros_like(vx), np.zeros_like(vz)

    speed = np.hypot(vx, vz)
    rho = density(z)
    CdA = drag_area(theta, aero)

    # Drag: -0.5 rho Cd A |v| * v_component. Gated below v_min so that a
    # near-stationary vehicle does not pick up force from a meaningless
    # velocity direction.
    live = speed > aero.v_min
    k = np.where(live, 0.5 * rho * CdA * speed, 0.0)
    Fx = -k * vx
    Fz = -k * vz

    # Lift: perpendicular to velocity, Cl peaking near 45 deg angle of attack.
    alpha = angle_of_attack(vx, vz, theta)
    Cl = aero.Cl_max * np.sin(2.0 * alpha)
    A = effective_area(theta, aero)
    q = 0.5 * rho * speed ** 2
    L = np.where(live, q * Cl * A, 0.0)

    # Rotate the velocity direction by +90 deg to get the lift direction.
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(live, vx / np.maximum(speed, 1e-9), 0.0)
        uz = np.where(live, vz / np.maximum(speed, 1e-9), 0.0)
    Fx = Fx + L * (-uz)
    Fz = Fz + L * ux

    return Fx, Fz


def aero_acceleration(vx, vz, z, theta, m, aero: AeroConfig = None):
    """Aerodynamic acceleration in the inertial frame [m/s^2]."""
    Fx, Fz = aero_force(vx, vz, z, theta, aero)
    m = np.maximum(np.asarray(m, dtype=float), 1.0)
    return Fx / m, Fz / m


def terminal_velocity(theta, m, z=0.0, aero: AeroConfig = None, g=9.80665):
    """
    Steady-state fall speed at a given attitude [m/s].

    Useful as a sanity number: it is what the belly-flop buys you before the
    engines are lit at all.
    """
    aero = aero or AeroConfig()
    CdA = drag_area(theta, aero)
    return float(np.sqrt(2.0 * m * g / (density(z) * CdA)))


if __name__ == "__main__":
    cfg = AeroConfig()
    print(cfg.summary())
    print()
    m = 130_000.0
    print(f"{'pitch':>7} {'A [m^2]':>9} {'Cd':>6} {'Cd*A':>8} "
          f"{'v_term [m/s]':>13}")
    for deg in (0, 15, 30, 45, 60, 75, 90):
        th = np.radians(deg)
        print(f"{deg:6d}d {float(effective_area(th, cfg)):9.0f} "
              f"{float(effective_Cd(th, cfg)):6.2f} "
              f"{float(drag_area(th, cfg)):8.0f} "
              f"{terminal_velocity(th, m, 2000.0, cfg):13.1f}")
