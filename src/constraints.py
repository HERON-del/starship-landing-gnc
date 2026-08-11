"""
Constraint functions for the powered-descent guidance problem.

Each constraint returns CVXPY-compatible expressions that can be appended
to a constraint list. This module encodes the physical limitations of the
vehicle and the operational rules of the landing problem.

Naming convention follows Açıkmeşe & Ploen (2007):

    sigma  — thrust magnitude slack variable (the lossless convexification trick)
    gamma  — glideslope half-angle
    theta  — maximum thrust pointing angle from vertical

References
----------
[1] Açıkmeşe, B. and Ploen, S., "Convex Programming Approach to Powered
    Descent Guidance for Mars Landing," JGCD, Vol. 30, No. 5, 2007.
"""

import cvxpy as cp
import numpy as np

# -----------------------------------------------------------------------
# Physical constants (must match dynamics.py exactly)
# -----------------------------------------------------------------------
G0 = 9.80665       # standard gravity for Isp definition [m/s^2]
G_EARTH = 9.80665  # local gravitational acceleration [m/s^2]


def glideslope_constraint(x_k, z_k, gamma_gs_deg=80.0):
    """
    Glideslope cone: the vehicle must stay above a cone whose tip
    is at the landing site, keeping the approach steep enough to
    recover from wind or thrust perturbations.

    In 2-D this is:  |x| <= z / tan(gamma_gs)

    Note the division. `gamma_gs` is measured from the *horizontal*, so a
    large angle means a steep, tightly constrained approach. Writing
    `|x| <= z * tan(gamma)` would inflate the allowed corridor as the angle
    grows, which is backwards — see the note in the module tests.

    Parameters
    ----------
    x_k : cvxpy.Variable or float
        Horizontal position at time step k [m].
    z_k : cvxpy.Variable or float
        Altitude at time step k [m].
    gamma_gs_deg : float
        Glideslope angle from the horizontal [deg].
        80° means the approach must be within 10° of vertical.

    Returns
    -------
    list of cvxpy constraints
    """
    tan_gs = np.tan(np.radians(gamma_gs_deg))
    # |x| <= z / tan(gamma_gs), as two linear inequalities.
    return [
        x_k * tan_gs <= z_k,
        -x_k * tan_gs <= z_k,
    ]


def thrust_magnitude_constraint(Tx_k, Tz_k, sigma_k, T_min, T_max):
    """
    Lossless convexification of thrust magnitude bounds.

    Replaces the non-convex ||T|| >= T_min with:

        ||T|| <= sigma            (second-order cone)
        T_min <= sigma <= T_max   (box)

    At the optimal solution, ||T|| = sigma exactly.

    Parameters
    ----------
    Accepts scalars (one time step) or equal-length vectors (the whole horizon
    at once); the norm is taken down the component axis either way. Vectorising
    matters more than it looks — building this constraint N times in a Python
    loop dominates the solve time once CVXPY has to compile it.

    Parameters
    ----------
    Tx_k, Tz_k : cvxpy.Variable
        Thrust components at time step k, or over the horizon [N].
    sigma_k : cvxpy.Variable
        Thrust magnitude slack variable, same shape [N].
    T_min, T_max : float
        Minimum and maximum allowable thrust magnitude [N].

    Returns
    -------
    list of cvxpy constraints
    """
    T_vec = cp.vstack([Tx_k, Tz_k])          # (2,) -> (2,1);  (N,) -> (2,N)
    return [
        cp.norm(T_vec, axis=0) <= sigma_k,   # SOC: thrust inside the sigma ball
        sigma_k >= T_min,                    # minimum thrust (was non-convex!)
        sigma_k <= T_max,                    # maximum thrust
    ]


def pointing_constraint(Tx_k, Tz_k, sigma_k, theta_max_deg=30.0):
    """
    Thrust pointing: the thrust vector cannot tilt more than theta_max
    from vertical. This represents the combined effect of vehicle tilt
    and engine gimbal limits.

        Tz >= ||T|| * cos(theta_max)  =>  Tz >= sigma * cos(theta_max)

    Parameters
    ----------
    Tx_k, Tz_k : cvxpy.Variable
        Thrust components at step k [N].
    sigma_k : cvxpy.Variable
        Thrust magnitude slack [N].
    theta_max_deg : float
        Maximum angle from vertical [deg].

    Returns
    -------
    list of cvxpy constraints
    """
    cos_theta = np.cos(np.radians(theta_max_deg))
    return [
        Tz_k >= sigma_k * cos_theta,  # thrust mostly points up
    ]


def mass_dynamics_linear(m_k, m_k1, sigma_k, dt, isp, coeff=None):
    """
    Linear mass dynamics between consecutive time steps.

        dm/dt = -sigma / (Isp * g0)

    Discretized with forward Euler for the mass state:

        m[k+1] = m[k] - dt * sigma[k] / (Isp * g0)

    This is exact when sigma is constant over the interval, which is a
    good approximation for small dt. The log-mass SCvx formulation
    (Week 2) removes this approximation entirely.

    Parameters
    ----------
    m_k, m_k1 : cvxpy.Variable
        Mass at step k and k+1 [kg].
    sigma_k : cvxpy.Variable
        Thrust magnitude slack at step k [N].
    dt : float
        Time step [s].
    isp : float
        Specific impulse [s].
    coeff : float, optional
        Override for `dt / (Isp * g0)`. The landing problem solves in
        non-dimensional variables and supplies the rescaled coefficient here.

    Returns
    -------
    list of cvxpy constraints (equality)
    """
    mdot_coeff = dt / (isp * G0) if coeff is None else coeff
    return [
        m_k1 == m_k - mdot_coeff * sigma_k,
    ]


def velocity_dynamics(vx_k, vz_k, vx_k1, vz_k1, Tx_k, Tz_k, m_k_ref, dt):
    """
    Euler-discretized translational dynamics for a powered descent.

        dvx/dt = Tx / m
        dvz/dt = Tz / m - g

    `m_k_ref` must be a **float**, not a cvxpy Variable. T/m with both as
    variables is bilinear and not DCP; the landing problem supplies a mass
    reference and iterates until it agrees with the solved mass profile.

    NOTE: This formulation is approximate. The full SCvx loop (Day 7–8)
    handles the T/m nonlinearity properly via successive linearization.

    Returns
    -------
    list of cvxpy equality constraints
    """
    return [
        vx_k1 == vx_k + dt * Tx_k / m_k_ref,
        vz_k1 == vz_k + dt * Tz_k / m_k_ref - dt * G_EARTH,
    ]


def position_dynamics(x_k, z_k, x_k1, z_k1, vx_k, vz_k, dt):
    """
    Euler-discretized position kinematics.

        dx/dt = vx
        dz/dt = vz

    Returns
    -------
    list of cvxpy equality constraints
    """
    return [
        x_k1 == x_k + dt * vx_k,
        z_k1 == z_k + dt * vz_k,
    ]


def log_mass_bounds(z_mass_k, m_dry, m_wet):
    """
    Bounds on the log-mass variable for future SCvx use.

        z_mass = ln(m),  so  ln(m_dry) <= z_mass <= ln(m_wet)

    Not used in today's direct formulation but placed here so
    the constraint library is complete for Week 2.

    Parameters
    ----------
    z_mass_k : cvxpy.Variable
        Log-mass at step k [-].
    m_dry, m_wet : float
        Dry and wet mass bounds [kg].

    Returns
    -------
    list of cvxpy constraints
    """
    return [
        z_mass_k >= np.log(m_dry),
        z_mass_k <= np.log(m_wet),
    ]
