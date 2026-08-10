"""
3-DoF planar rocket dynamics with variable mass.

State vector (5 elements):
    y = [x, z, vx, vz, m]
    x   downrange position   [m]
    z   altitude             [m]     (z = 0 is the landing pad)
    vx  horizontal velocity  [m/s]
    vz  vertical velocity    [m/s]   (positive = climbing)
    m   total vehicle mass   [kg]

Control vector (2 elements):
    u = [Tx, Tz]
    Tx  horizontal thrust component [N]
    Tz  vertical thrust component   [N]

Equations of motion:
    dx/dt  = vx
    dz/dt  = vz
    dvx/dt = Tx / m
    dvz/dt = Tz / m - g
    dm/dt  = -||T|| / (Isp * g0)          (Tsiolkovsky mass flow)

This 3-DoF model is extended to full 6-DoF with quaternion attitude in Week 3.
"""

from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------
G0 = 9.80665       # standard gravity, used in the Isp definition [m/s^2]
G_EARTH = 9.80665  # local gravitational acceleration [m/s^2]


# ----------------------------------------------------------------------
# Vehicle definition
# ----------------------------------------------------------------------
@dataclass
class Vehicle:
    """
    Vehicle mass and propulsion properties.

    Defaults approximate a SpaceX Starship upper stage during the
    landing burn. Public figures vary between sources; these are
    order-of-magnitude representative values suitable for a
    guidance study, not official specifications.
    """

    m_dry: float = 100_000.0           # structural (empty) mass [kg]
    m_prop_initial: float = 30_000.0   # landing propellant available [kg]
    n_engines: int = 3                 # Raptor sea-level engines used to land
    thrust_per_engine: float = 2.3e6   # sea-level thrust per engine [N]
    throttle_min: float = 0.40         # minimum throttle fraction [-]
    throttle_max: float = 1.00         # maximum throttle fraction [-]
    isp: float = 327.0                 # sea-level specific impulse [s]

    @property
    def m_wet(self) -> float:
        """Total mass at the start of the landing burn [kg]."""
        return self.m_dry + self.m_prop_initial

    @property
    def T_max(self) -> float:
        """Maximum total thrust from all landing engines [N]."""
        return self.n_engines * self.thrust_per_engine * self.throttle_max

    @property
    def T_min(self) -> float:
        """Minimum total thrust once engines are lit [N]."""
        return self.n_engines * self.thrust_per_engine * self.throttle_min

    def twr(self, mass: float) -> float:
        """Thrust-to-weight ratio at maximum thrust for a given mass [-]."""
        return self.T_max / (mass * G_EARTH)

    def summary(self) -> str:
        return (
            f"Vehicle configuration\n"
            f"  Dry mass          : {self.m_dry:>12,.0f} kg\n"
            f"  Landing propellant: {self.m_prop_initial:>12,.0f} kg\n"
            f"  Wet mass          : {self.m_wet:>12,.0f} kg\n"
            f"  Engines           : {self.n_engines:>12d}\n"
            f"  Max thrust        : {self.T_max:>12,.0f} N\n"
            f"  Min thrust        : {self.T_min:>12,.0f} N\n"
            f"  Isp               : {self.isp:>12.1f} s\n"
            f"  TWR at wet mass   : {self.twr(self.m_wet):>12.2f}\n"
            f"  TWR at dry mass   : {self.twr(self.m_dry):>12.2f}"
        )


# ----------------------------------------------------------------------
# Equations of motion
# ----------------------------------------------------------------------
def dynamics_3dof(
    t: float,
    state: np.ndarray,
    control_fn,
    vehicle: Vehicle,
) -> np.ndarray:
    """
    Compute state derivatives for the planar variable-mass rocket.

    Parameters
    ----------
    t : float
        Current time [s].
    state : np.ndarray, shape (5,)
        [x, z, vx, vz, m].
    control_fn : callable
        Function with signature control_fn(t, state, vehicle) -> np.array([Tx, Tz]).
        Thrust components in newtons.
    vehicle : Vehicle
        Vehicle parameters.

    Returns
    -------
    np.ndarray, shape (5,)
        Time derivative of the state.
    """
    x, z, vx, vz, m = state

    # Guard against integrating past propellant depletion. Once the
    # vehicle is at dry mass it can produce no thrust.
    if m <= vehicle.m_dry:
        m = vehicle.m_dry
        thrust = np.zeros(2)
    else:
        thrust = np.asarray(control_fn(t, state, vehicle), dtype=float)

    Tx, Tz = thrust
    thrust_magnitude = np.hypot(Tx, Tz)

    dx = vx
    dz = vz
    dvx = Tx / m
    dvz = Tz / m - G_EARTH
    dm = -thrust_magnitude / (vehicle.isp * G0)

    return np.array([dx, dz, dvx, dvz, dm])


# ----------------------------------------------------------------------
# Control laws for testing
# ----------------------------------------------------------------------
def control_zero(t, state, vehicle):
    """No thrust. Used for the ballistic free-fall verification test."""
    return np.array([0.0, 0.0])


def control_hover(t, state, vehicle):
    """
    Ideal hover: thrust exactly cancels instantaneous weight.

    Because propellant is consumed, the required thrust decreases over
    time. A vehicle under this law holds altitude exactly, which makes
    it a clean analytical test case.
    """
    m = state[4]
    return np.array([0.0, m * G_EARTH])


def exact_constant_thrust(
    t: np.ndarray,
    y0: np.ndarray,
    Tx: float,
    Tz: float,
    vehicle: Vehicle,
) -> np.ndarray:
    """
    Closed-form solution for constant thrust with variable mass.

    Used as the reference in the convergence-order study. Deriving a
    *nonlinear* exact solution matters: integrating a coarse numerical
    reference instead puts a round-off floor under the measurement and
    corrupts the measured order.

    With constant thrust the mass flow is constant, so m(t) = m0 - mdot*t
    and the velocity integral is a logarithm:

        u(t)  = m(t) / m0
        L(t)  = -ln u                     (the Tsiolkovsky delta-v factor)
        S(t)  = (m0/mdot) * (u ln u + 1 - u)      (integral of L)

        vx(t) = vx0 + (Tx/mdot) * L
        vz(t) = vz0 - g*t + (Tz/mdot) * L
        x(t)  = x0 + vx0*t + (Tx/mdot) * S
        z(t)  = z0 + vz0*t - 0.5*g*t^2 + (Tz/mdot) * S

    Valid only while m(t) > m_dry and thrust is genuinely constant.

    Returns
    -------
    np.ndarray, shape (len(t), 5)
        Exact state history [x, z, vx, vz, m].
    """
    t = np.asarray(t, dtype=float)
    x0, z0, vx0, vz0, m0 = y0

    thrust_magnitude = float(np.hypot(Tx, Tz))

    if thrust_magnitude == 0.0:
        # Ballistic: the logarithms degenerate, fall back to kinematics.
        return np.column_stack([
            x0 + vx0 * t,
            z0 + vz0 * t - 0.5 * G_EARTH * t**2,
            np.full_like(t, vx0),
            vz0 - G_EARTH * t,
            np.full_like(t, m0),
        ])

    mdot = thrust_magnitude / (vehicle.isp * G0)
    m = m0 - mdot * t
    u = m / m0

    L = -np.log(u)
    S = (m0 / mdot) * (u * np.log(u) + 1.0 - u)

    return np.column_stack([
        x0 + vx0 * t + (Tx / mdot) * S,
        z0 + vz0 * t - 0.5 * G_EARTH * t**2 + (Tz / mdot) * S,
        vx0 + (Tx / mdot) * L,
        vz0 - G_EARTH * t + (Tz / mdot) * L,
        m,
    ])


def control_constant(Tx: float, Tz: float):
    """
    Factory returning a control law with fixed thrust components.

    Example
    -------
    >>> ctrl = control_constant(0.0, 5.0e6)
    """
    def _control(t, state, vehicle):
        return np.array([Tx, Tz])

    return _control


# ----------------------------------------------------------------------
# Manual check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    v = Vehicle()
    print(v.summary())
