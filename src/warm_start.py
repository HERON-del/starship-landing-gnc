"""
Warm starting for closed-loop SCvx guidance.

A guidance loop re-solves the whole landing problem every cycle, and it has a
fraction of a second to do it in. Solving from a cold seed takes the Day 8
solver twenty-odd iterations, which is nowhere near fast enough. But
consecutive problems in an MPC loop are nearly identical -- same pad, same
vehicle, a slightly different starting state and a slightly shorter horizon --
so the previous answer, shifted forward in time, is already close to the next
one.

`shift_reference` does that shifting. It takes the solution computed one
guidance step ago, drops the part that has already been flown, re-samples what
remains onto a fresh grid, and replaces node zero with where the vehicle
actually is rather than where the old plan said it would be. That last
substitution is the entire point: the gap between those two is the tracking
error the new solve exists to remove.

One caveat worth stating, because it is easy to fool yourself here. The guide
measures the warm-start speedup by capping warm solves at four iterations and
comparing against an uncapped cold solve, which guarantees a speedup of at
least the cap regardless of whether warm starting works. The comparison is only
meaningful if both are run to the same convergence criterion and the iterations
are counted, which is what `tests/test_closed_loop.py` does.
"""

import numpy as np

# Below this much remaining flight there is no useful trajectory left to
# re-plan: the horizon is shorter than the flip takes, and the subproblem
# spends its effort on a boundary layer rather than on guidance.
MIN_HORIZON_S = 0.8


def shift_reference(prev_sol, elapsed, current_state, N, vehicle):
    """
    Build a warm-start reference from the previous solution.

    Parameters
    ----------
    prev_sol : dict
        A converged `solve_scvx_complete` result.
    elapsed : float
        Seconds flown since `prev_sol` was computed.
    current_state : dict
        Where the vehicle actually is: x, z, vx, vz, theta, omega, m.
    N : int
        Interval count for the new reference.
    vehicle : Vehicle6DoF
        Needed for `m_wet`, since the reference carries log-mass normalised
        by it.

    Returns
    -------
    (ref, t_remaining, gap)
        `ref` is in the format `initialize_reference` returns, so it can be
        handed straight to `solve_scvx_complete(initial_ref=...)`, and is
        `None` when there is too little flight left to replan. `gap` is the
        distance between where the old plan predicted the vehicle would be by
        now and where it actually is -- the tracking error this solve exists
        to remove, and the honest predictor of how much the warm start saves.
    """
    t_f_prev = float(prev_sol["t_f"])
    t_remaining = t_f_prev - elapsed
    if t_remaining < MIN_HORIZON_S:
        return None, t_remaining, float("nan")

    t_old = np.asarray(prev_sol["t"])
    # Sample the old solution across the window it has not yet flown, then
    # relabel that window as a fresh [0, t_remaining] grid.
    t_query = np.linspace(elapsed, t_f_prev, N + 1)

    def at_nodes(field):
        return np.interp(t_query, t_old, np.asarray(prev_sol[field]))

    x = at_nodes("x")
    z = at_nodes("z")
    vx = at_nodes("vx")
    vz = at_nodes("vz")
    theta = at_nodes("theta")
    omega = at_nodes("omega")
    m = at_nodes("m")

    # Measured before the substitution below overwrites the prediction.
    gap = float(np.hypot(x[0] - current_state["x"],
                         z[0] - current_state["z"]))

    # Node zero is the measurement, not the prediction.
    x[0] = current_state["x"]
    z[0] = current_state["z"]
    vx[0] = current_state["vx"]
    vz[0] = current_state["vz"]
    theta[0] = current_state["theta"]
    omega[0] = current_state["omega"]
    m[0] = current_state["m"]

    # Controls are zero-order held on intervals, so they are sampled at
    # interval midpoints of the new grid rather than at its nodes.
    t_ctrl_old = 0.5 * (t_old[:-1] + t_old[1:])
    edges = np.linspace(elapsed, t_f_prev, N + 1)
    t_ctrl_query = 0.5 * (edges[:-1] + edges[1:])

    def at_intervals(field):
        return np.interp(t_ctrl_query, t_ctrl_old, np.asarray(prev_sol[field]))

    ref = {
        "x": x, "z": np.maximum(z, 0.0),
        "vx": vx, "vz": vz,
        "theta": theta, "omega": omega,
        "zm": np.log(np.maximum(m, vehicle.m_dry) / vehicle.m_wet),
        "sigma": at_intervals("sigma"),
        "tau": at_intervals("tau"),
        # The new nominal duration *is* the remaining time, so the time-scale
        # factor starts at 1 again and the solver moves it from there.
        "kt": 1.0,
    }
    return ref, t_remaining, gap
