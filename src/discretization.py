"""
Discretization schemes for embedding dynamics into convex programs.

Provides Euler and trapezoidal methods that return CVXPY-compatible equality
constraints. These replace the hand-coded dynamics constraints from Day 3's
landing_problem.py with reusable, testable functions.

The trapezoidal method is second-order accurate — it uses dynamics information
at both endpoints of each interval. In simulation that would require iteration
(it is an implicit method); in optimization it is free, because the state at
k+1 is already a decision variable.

Two implementation notes that matter more than they look.

**Vectorised, not looped.** Each function takes whole state and control vectors
and emits a handful of vector equalities rather than 5N scalar ones. Building
these in a Python loop dominates CVXPY compile time — on the Day 3 problem it
was the difference between 2.9 s and 0.1 s per solve.

**Coefficients are passed in, not computed here.** The caller supplies
`Coeffs`, whose fields may be floats *or* `cp.Parameter`. That is what lets a
single compiled problem be re-solved across a whole line search over the burn
duration: the time step changes the coefficients, not the problem structure.
Each coefficient must appear linearly for the problem to stay DPP-compliant,
which is why the mass reference is folded into `vel` rather than dividing by a
separate mass parameter.

References
----------
[1] Betts, J.T., "Practical Methods for Optimal Control and Estimation Using
    Nonlinear Programming," SIAM, 2010.
[2] Szmuk, M. and Açıkmeşe, B., "Successive Convexification for Fuel-Optimal
    Powered Landing with Aerodynamic Drag and Non-Convex Constraints,"
    AIAA, 2016.
"""

from dataclasses import dataclass
from typing import Any

import cvxpy as cp

G0 = 9.80665
G_EARTH = 9.80665


@dataclass
class Coeffs:
    """
    Pre-scaled discretization coefficients.

    Working in non-dimensional variables, one Euler step reads

        x[k+1] = x[k] + pos * vx[k]
        vx[k+1] = vx[k] + vel[k] * Tx[k]
        vz[k+1] = vz[k] + vel[k] * Tz[k] - grav
        m[k+1] = m[k] - mass * sigma[k]

    Attributes
    ----------
    pos : float or cp.Parameter
        dt * V / L — velocity's contribution to position.
    vel : array-like or cp.Parameter, shape (n_ctrl,)
        dt * F / (M * V * m_ref[k]) — thrust's contribution to velocity, with
        the mass reference already divided in. Folding mass in here keeps the
        expression affine in the parameters.
    grav : float or cp.Parameter
        dt * g / V — gravity's contribution to vertical velocity.
    mass : float or cp.Parameter
        dt * F / (M * Isp * g0) — mass flow per unit sigma.
    """

    pos: Any
    vel: Any
    grav: Any
    mass: Any


def euler_dynamics_constraints(x, z, vx, vz, m, Tx, Tz, sigma, c: Coeffs):
    """
    Forward Euler discretization of 2-D powered-descent dynamics.

    First-order: local error O(dt^2), global O(dt).

    Parameters
    ----------
    x, z, vx, vz, m : cvxpy.Variable, shape (N+1,)
        State at every node.
    Tx, Tz, sigma : cvxpy.Variable, shape (N,)
        Control on every interval, held constant across it.
    c : Coeffs
        Discretization coefficients; `c.vel` has shape (N,).

    Returns
    -------
    list of cvxpy equality constraints
    """
    return [
        x[1:] == x[:-1] + c.pos * vx[:-1],
        z[1:] == z[:-1] + c.pos * vz[:-1],
        vx[1:] == vx[:-1] + cp.multiply(c.vel, Tx),
        vz[1:] == vz[:-1] + cp.multiply(c.vel, Tz) - c.grav,
        m[1:] == m[:-1] - c.mass * sigma,
    ]


def trapz_dynamics_constraints(x, z, vx, vz, m, Tx, Tz, sigma, c: Coeffs):
    """
    Trapezoidal discretization of 2-D powered-descent dynamics.

    Averages the derivative at both ends of each interval:

        y[k+1] = y[k] + dt/2 * (f(y[k]) + f(y[k+1]))

    Second-order: local error O(dt^3), global O(dt^2).

    Unlike Euler this needs a control value at *every node*, including the last
    one, so `Tx`, `Tz` and `sigma` have length N+1 rather than N. The final
    control is the thrust at the instant of touchdown; the optimiser picks it,
    and it is a real command a vehicle would have to issue.

    Parameters
    ----------
    x, z, vx, vz, m : cvxpy.Variable, shape (N+1,)
        State at every node.
    Tx, Tz, sigma : cvxpy.Variable, shape (N+1,)
        Control at every node.
    c : Coeffs
        Discretization coefficients; `c.vel` has shape (N+1,).

    Returns
    -------
    list of cvxpy equality constraints
    """
    half_pos = c.pos / 2.0
    half_mass = c.mass / 2.0

    # vel already carries the per-node mass reference, so halve after the
    # multiply rather than before.
    a_x = cp.multiply(c.vel, Tx)
    a_z = cp.multiply(c.vel, Tz)

    return [
        x[1:] == x[:-1] + half_pos * (vx[:-1] + vx[1:]),
        z[1:] == z[:-1] + half_pos * (vz[:-1] + vz[1:]),
        vx[1:] == vx[:-1] + 0.5 * (a_x[:-1] + a_x[1:]),
        vz[1:] == vz[:-1] + 0.5 * (a_z[:-1] + a_z[1:]) - c.grav,
        m[1:] == m[:-1] - half_mass * (sigma[:-1] + sigma[1:]),
    ]


def n_control_nodes(method: str, N: int) -> int:
    """Control vector length for a given scheme: N for Euler, N+1 for trapezoidal."""
    return N + 1 if is_trapz(method) else N


def is_trapz(method: str) -> bool:
    return str(method).lower().startswith("trap")


def dynamics_constraints(method, x, z, vx, vz, m, Tx, Tz, sigma, c: Coeffs):
    """Dispatch to the requested scheme."""
    if is_trapz(method):
        return trapz_dynamics_constraints(x, z, vx, vz, m, Tx, Tz, sigma, c)
    return euler_dynamics_constraints(x, z, vx, vz, m, Tx, Tz, sigma, c)
