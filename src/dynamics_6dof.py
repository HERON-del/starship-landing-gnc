"""
Planar 6-DoF dynamics for powered descent with rotation.

Extends the 3-DoF translational model from dynamics.py to include pitch angle,
pitch rate, and engine gimbal torque.

State vector: [x, z, vx, vz, theta, omega, m]

    x, z      position in the inertial frame [m]
    vx, vz    velocity in the inertial frame [m/s]
    theta     pitch angle from vertical [rad] (0 = upright, pi/2 = horizontal)
    omega     pitch rate [rad/s]
    m         mass [kg]

Control vector: [T, delta]

    T         thrust magnitude [N]
    delta     gimbal angle from the body axis [rad]

The engine is bolted to the vehicle, so the thrust direction is *not* free —
it is the body axis plus whatever the gimbal adds:

    Tx  = T sin(theta + delta)
    Tz  = T cos(theta + delta)
    tau = T L_engine sin(delta)

That coupling is the whole difficulty of the flip. Gimbaling to generate the
torque that rotates the vehicle simultaneously tilts the thrust that decelerates
it; you cannot buy one without paying for the other. The optimiser in
landing_flip.py has to linearise this, but here in the simulator it is exact.

Conventions
-----------
- theta = 0     vehicle upright (engines down, nose up)
- theta = pi/2  vehicle horizontal (belly-flop, entry attitude)
- positive omega is counter-clockwise
- gravity acts in -z

References
----------
[1] Szmuk, M. et al., "Successive Convexification for 6-DoF Mars Rocket
    Powered Landing with Free-Final-Time," AIAA, 2018.
"""

from dataclasses import dataclass

import numpy as np

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
G0 = 9.80665
G_EARTH = 9.80665


@dataclass
class Vehicle6DoF:
    """Vehicle parameters for planar 6-DoF powered descent."""

    # Mass
    m_dry: float = 100_000.0
    m_prop_initial: float = 30_000.0

    # Propulsion
    n_engines: int = 3
    thrust_per_engine: float = 2.3e6
    throttle_min: float = 0.40
    throttle_max: float = 1.00
    isp: float = 327.0

    # Geometry
    length: float = 50.0          # vehicle length [m]
    L_engine: float = 25.0        # CG to engine gimbal point [m]
    delta_max_deg: float = 15.0   # max gimbal deflection [deg]

    # Rotational
    I_pitch: float = 2.7e7        # pitch moment of inertia [kg m^2]
    omega_max: float = 0.5        # max pitch rate [rad/s] (~28 deg/s)

    @property
    def m_wet(self) -> float:
        return self.m_dry + self.m_prop_initial

    @property
    def T_max(self) -> float:
        return self.n_engines * self.thrust_per_engine * self.throttle_max

    @property
    def T_min(self) -> float:
        return self.n_engines * self.thrust_per_engine * self.throttle_min

    @property
    def delta_max(self) -> float:
        return np.radians(self.delta_max_deg)

    @property
    def tau_max(self) -> float:
        """Maximum torque [N m] at full thrust and full gimbal."""
        return self.T_max * self.L_engine * np.sin(self.delta_max)

    @property
    def twr(self) -> float:
        """Thrust-to-weight ratio at wet mass."""
        return self.T_max / (self.m_wet * G_EARTH)

    @property
    def alpha_max(self) -> float:
        """Peak angular acceleration [rad/s^2] at full thrust and gimbal."""
        return self.tau_max / self.I_pitch

    def summary(self) -> str:
        return "\n".join([
            "Vehicle configuration (6-DoF)",
            f"  Dry mass          : {self.m_dry:>12,.0f} kg",
            f"  Propellant        : {self.m_prop_initial:>12,.0f} kg",
            f"  Wet mass          : {self.m_wet:>12,.0f} kg",
            f"  Engines           : {self.n_engines:>12d}",
            f"  T_max             : {self.T_max:>12,.0f} N "
            f"({self.T_max / 1e6:.2f} MN)",
            f"  T_min             : {self.T_min:>12,.0f} N "
            f"({self.T_min / 1e6:.2f} MN)",
            f"  TWR (wet)         : {self.twr:>12.2f}",
            f"  Isp               : {self.isp:>12.0f} s",
            f"  Length            : {self.length:>12.0f} m",
            f"  CG-to-engine      : {self.L_engine:>12.0f} m",
            f"  Gimbal max        : {self.delta_max_deg:>12.0f} deg",
            f"  I_pitch           : {self.I_pitch:>12.2e} kg m^2",
            f"  omega_max         : {self.omega_max:>12.2f} rad/s "
            f"({np.degrees(self.omega_max):.1f} deg/s)",
            f"  tau_max           : {self.tau_max:>12,.0f} N m",
            f"  alpha_max         : {self.alpha_max:>12.3f} rad/s^2 "
            f"({np.degrees(self.alpha_max):.1f} deg/s^2)",
        ])


def dynamics_6dof(t, state, control_fn, vehicle: Vehicle6DoF):
    """
    Equations of motion for planar 6-DoF powered descent.

    Parameters
    ----------
    t : float
        Time [s].
    state : array-like, shape (7,)
        [x, z, vx, vz, theta, omega, m].
    control_fn : callable
        control_fn(t, state, vehicle) -> (T, delta), thrust [N] and gimbal [rad].
    vehicle : Vehicle6DoF

    Returns
    -------
    np.ndarray, shape (7,)
        State derivative.
    """
    x, z, vx, vz, theta, omega, m = state

    T, delta = control_fn(t, state, vehicle)
    T = float(np.clip(T, 0.0, vehicle.T_max))
    delta = float(np.clip(delta, -vehicle.delta_max, vehicle.delta_max))

    # Engine is fixed to the body: thrust points along the body axis, offset
    # by the gimbal. Torque and thrust tilt come from the same deflection.
    Tx = T * np.sin(theta + delta)
    Tz = T * np.cos(theta + delta)
    tau = T * vehicle.L_engine * np.sin(delta)

    # Once at dry mass there is no propellant left to burn.
    if m <= vehicle.m_dry:
        m = vehicle.m_dry
        Tx = Tz = tau = 0.0
        mdot = 0.0
    else:
        mdot = -T / (vehicle.isp * G0)

    return np.array([
        vx,
        vz,
        Tx / m,
        Tz / m - G_EARTH,
        omega,
        tau / vehicle.I_pitch,
        mdot,
    ])


# -----------------------------------------------------------------------
# Example control laws, for verification rather than flight
# -----------------------------------------------------------------------
def control_zero_6dof(t, state, vehicle):
    """No thrust: ballistic translation and torque-free rotation."""
    return 0.0, 0.0


def control_hover_6dof(t, state, vehicle):
    """Thrust balances weight along the vertical, no gimbal."""
    return state[6] * G_EARTH, 0.0


def control_constant_gimbal(T: float, delta: float):
    """Factory for a fixed thrust and gimbal, used in the angular-impulse test."""
    def _control(t, state, vehicle):
        return T, delta
    return _control


def control_flip_bang_bang(t, state, vehicle, t_flip_start, t_flip_mid):
    """
    Bang-bang flip control for simulation testing.

    Gimbal hard one way to start the rotation, then hard the other way to stop
    it. This is not optimal and makes no attempt to land — it exists to show
    that the dynamics produce a plausible flip before the optimiser is trusted
    with them.
    """
    theta = state[4]
    m = state[6]

    # Hold roughly enough thrust to support the vehicle's vertical component.
    T = m * G_EARTH / max(np.cos(theta), 0.1)
    T = float(np.clip(T, vehicle.T_min, vehicle.T_max))

    if t < t_flip_start:
        delta = 0.0
    elif t < t_flip_mid:
        delta = -vehicle.delta_max
    else:
        delta = vehicle.delta_max
    return T, delta


if __name__ == "__main__":
    print(Vehicle6DoF().summary())
