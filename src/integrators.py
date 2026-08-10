"""
Numerical integrators for rigid-body and point-mass dynamics.

Day 2 deliverable. These advance a state vector forward in time given a
function that returns state derivatives.

The same integrators are reused unchanged in Week 3 when the state grows
from 5 elements (3-DoF) to 14 elements (6-DoF with quaternions), because
they are written generically over NumPy arrays.
"""

from typing import Callable

import numpy as np


def euler_step(
    f: Callable,
    t: float,
    y: np.ndarray,
    dt: float,
    *args,
) -> np.ndarray:
    """
    Advance one step with forward (explicit) Euler.

    First-order accurate: global error is O(dt).
    Included only as a baseline to demonstrate why RK4 is worth the cost.

    Parameters
    ----------
    f : callable
        Derivative function with signature f(t, y, *args) -> dy/dt.
    t : float
        Current time [s].
    y : np.ndarray
        Current state vector.
    dt : float
        Time step [s].

    Returns
    -------
    np.ndarray
        State at time t + dt.
    """
    return y + dt * f(t, y, *args)


def rk4_step(
    f: Callable,
    t: float,
    y: np.ndarray,
    dt: float,
    *args,
) -> np.ndarray:
    """
    Advance one step with the classical 4th-order Runge-Kutta method.

    Fourth-order accurate: global error is O(dt^4). Halving the step size
    reduces error by a factor of ~16.

    Parameters
    ----------
    f : callable
        Derivative function with signature f(t, y, *args) -> dy/dt.
    t : float
        Current time [s].
    y : np.ndarray
        Current state vector.
    dt : float
        Time step [s].

    Returns
    -------
    np.ndarray
        State at time t + dt.
    """
    k1 = f(t,             y,                 *args)
    k2 = f(t + 0.5 * dt,  y + 0.5 * dt * k1, *args)
    k3 = f(t + 0.5 * dt,  y + 0.5 * dt * k2, *args)
    k4 = f(t + dt,        y + dt * k3,       *args)

    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate(
    f: Callable,
    y0: np.ndarray,
    t_span: tuple,
    dt: float,
    *args,
    method: str = "rk4",
):
    """
    Integrate a system of ODEs over a time interval with a fixed step.

    A fixed step (rather than an adaptive one like scipy's solve_ivp) is
    used deliberately: the trajectory optimizer in Week 2 requires the
    state to be available on a known, uniform time grid.

    Parameters
    ----------
    f : callable
        Derivative function f(t, y, *args) -> dy/dt.
    y0 : np.ndarray
        Initial state vector.
    t_span : tuple of (float, float)
        (t_start, t_end) in seconds.
    dt : float
        Fixed time step [s].
    method : {"rk4", "euler"}
        Integration scheme.

    Returns
    -------
    t_history : np.ndarray, shape (N+1,)
        Time at each node.
    y_history : np.ndarray, shape (N+1, len(y0))
        State at each node.
    """
    t_start, t_end = t_span
    n_steps = int(np.round((t_end - t_start) / dt))
    stepper = {"rk4": rk4_step, "euler": euler_step}[method]

    t_history = np.zeros(n_steps + 1)
    y_history = np.zeros((n_steps + 1, len(y0)))

    t_history[0] = t_start
    y_history[0] = np.asarray(y0, dtype=float)

    for k in range(n_steps):
        t_history[k + 1] = t_history[k] + dt
        y_history[k + 1] = stepper(f, t_history[k], y_history[k], dt, *args)

    return t_history, y_history
