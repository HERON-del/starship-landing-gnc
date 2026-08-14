"""
Successive Convex Programming for 6-DoF powered descent with aerodynamics.

Days 3-6 reached a working optimiser by ad-hoc reference iteration: linearise
about the last solution, re-solve, repeat, stop when the answer stops moving.
Day 5 added a trust region on the attitude because without one the iteration
did not converge at all. What was still missing is the other half of SCvx, and
the reason the day 5/6 loop needed a six-step homotopy to switch drag on:

    a subproblem that is always feasible.

That is what virtual control buys. Slack variables `nu` are added to every
dynamics row and penalised in L1:

    x[k+1] = A x[k] + B u[k] + c + nu[k]
    minimise   fuel  +  w_vc ||nu||_1

Any state sequence is now reachable, so the subproblem cannot be infeasible
for want of a good reference -- the optimiser simply pays for the parts of the
dynamics it cannot honour yet. As the reference improves, the price stops being
worth paying and `nu` falls to zero on its own. **The decay of ||nu|| is the
convergence proof**: it is the linear model measuring its own disagreement with
the dynamics it claims to represent, and driving it to nothing.

Three things here differ from the Day 7 guide, each because the guide's version
was built and measured first.

**The thrust vector is not free.** The guide models control as `(Tx, Tz)` with
`||T|| <= sigma`, plus an independent torque bounded by `sigma L sin(delta_max)`
-- lossless convexification, as on Days 3-4. That is exactly the model Day 5
rejected. The engine is bolted to the vehicle, so the thrust direction *is* the
attitude, plus at most a 15 degree gimbal. Transcribing the guide and measuring
the result: the thrust vector sits a mean of 43 degrees and a maximum of 115
degrees off the body axis, at **every one of the 80 nodes**. It lands exactly on
the pad at exactly zero speed by thrusting sideways relative to where it points.
So the coupling is kept, linearised as on Day 5, and the torque enters through
the gimbal that produces it.

**Everything is non-dimensional.** The guide works in SI and then needs a
hand-picked multiplier per state to make one trust-region radius mean anything
(`eta * max(|x_ref|, 100)`, `eta * 5000` for mass, bare `eta` for pitch). Worse,
its L1 penalty adds metres to kilograms to radians, so no single `w_vc` can be
right for all seven rows. Measured: the virtual-control norm grows monotonically
from 54 to 2.8e6 across 14 iterations while the solver reports `optimal` every
time; with the guide's own trust-region logic the subproblem instead goes
numerically unbounded at iteration 2 and the radius collapses to its floor.
Scaled, one `eta` and one `w_vc` serve every row.

**The step quality is the real one.** The guide computes `rho` from a ladder of
hand-tuned thresholds on the step size and the virtual-control norm, explaining
that the true metric "would require forward-propagating the dynamics." It would,
and this module does it -- the exact 6-DoF coupling and the full aerodynamic
model are both already available here, so there is no reason to guess:

    rho = (J(prev) - J(new)) / (J(prev) - L(new))

`J` is the penalised cost under the *true* dynamics, `L` the value the convex
subproblem predicted. rho near 1 means the linear model told the truth about
its own step, and the trust region can grow; rho near 0 means it did not, and
the step is rejected. This is the metric from [1], and it makes the radius
adapt to the problem instead of to a threshold table.

State:   [x, z, vx, vz, theta, omega, m]   at N+1 nodes
Control: [sigma, tau]                       at N nodes
Virtual: [nu_x, nu_z, nu_vx, nu_vz, nu_theta, nu_omega, nu_m]  at N nodes

References
----------
[1] Mao, Szmuk, Acikmese, "Successive Convexification: A Superlinearly
    Convergent Algorithm for Non-convex Optimal Control Problems," 2020.
[2] Szmuk, Acikmese, "Successive Convexification for 6-DoF Powered Descent
    Guidance with Free-Final-Time," AIAA, 2018.
"""

import os
import sys
import time as timer

import cvxpy as cp
import matplotlib
import numpy as np

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
RESULTS = os.path.join(REPO_ROOT, "results")

from src.dynamics_6dof import Vehicle6DoF, G0, G_EARTH        # noqa: E402
from src.aero import (                                        # noqa: E402
    AeroConfig, aero_acceleration, dynamic_pressure, aero_force,
)
from src.landing_flip import feasible_entry_state             # noqa: E402
from src.scvx_params import SCvxParams                        # noqa: E402


# ======================================================================
# Scaling
# ======================================================================
class Scales:
    """
    Non-dimensionalisation constants.

    Same choice as Day 3 and Day 5: divide each quantity by the largest value
    it can plausibly take, so every variable in the problem is O(1) and the
    coefficient matrix stops spanning a dozen orders of magnitude.
    """

    def __init__(self, vehicle, t_burn, x0, z0):
        self.L = max(abs(z0), abs(x0), 1.0)
        self.V = self.L / t_burn
        self.M = vehicle.m_wet
        self.F = vehicle.T_max
        self.TAU = vehicle.tau_max
        self.W = vehicle.omega_max


# ======================================================================
# Reference trajectory
# ======================================================================
def initialize_reference(N, t_burn, x0, z0, vx0, vz0, theta0, omega0,
                         vehicle, seed="flip"):
    """
    Build the initial reference. It need not be dynamically feasible.

    Two seeds, kept switchable because the difference between them is a
    measurable claim about what virtual control is worth.

    `flip` is the Day 5 seed: the rotation is rate-limited, so the vehicle
    cannot sweep its pitch linearly across a 30 second burn -- it flips fast and
    then holds. Day 5 recorded that linearising about the slow linear sweep gave
    an infeasible first subproblem even where the true problem was fine.

    `linear` is that known-bad sweep, kept so the claim can be re-run. With
    virtual control the subproblem cannot be infeasible, so this should now
    converge to the same answer, only more slowly.
    """
    t_grid = np.linspace(0.0, t_burn, N + 1)

    if seed == "flip":
        t_flip = float(np.clip(1.4 * abs(theta0) / vehicle.omega_max,
                               1.5, 0.6 * t_burn))
        theta_ref = theta0 * np.clip(1.0 - t_grid / t_flip, 0.0, 1.0)
    elif seed == "linear":
        theta_ref = np.linspace(theta0, 0.0, N + 1)
    else:
        raise ValueError(f"unknown seed {seed!r}")

    mdot_est = 0.7 * vehicle.T_max / (vehicle.isp * G0)
    m_ref = np.linspace(vehicle.m_wet,
                        max(vehicle.m_wet - mdot_est * t_burn,
                            vehicle.m_dry + 1000.0), N + 1)

    return {
        "t": t_grid,
        "x": np.linspace(x0, 0.0, N + 1),
        "z": np.linspace(z0, 0.0, N + 1),
        "vx": np.linspace(vx0, 0.0, N + 1),
        "vz": np.linspace(vz0, 0.0, N + 1),
        "theta": theta_ref,
        "omega": np.linspace(omega0, 0.0, N + 1),
        "m": m_ref,
        "sigma": np.full(N, 0.7 * vehicle.T_max),
        "tau": np.zeros(N),
    }


# ======================================================================
# True dynamics residual
# ======================================================================
def nonlinear_defect(sol, vehicle, aero, sc, dt, N, aero_scale=1.0):
    """
    The residual the virtual control was standing in for.

    Propagates one Euler step from every node using the *exact* thrust-attitude
    coupling and the full aerodynamic model, and reports how far that lands from
    the node the optimiser actually chose. Same discretisation as the
    subproblem, deliberately, so what is measured is linearisation error and not
    integration error.

    Returns
    -------
    (rows, total)
        `rows` is the per-row L1 defect in scaled units and `total` their sum --
        the honest counterpart to ||nu||_1. Where `nu` is what the linear model
        *believes* it needs to close the dynamics, this is what closing them
        actually costs.
    """
    x, z = sol["x"] / sc.L, sol["z"] / sc.L
    vx, vz = sol["vx"] / sc.V, sol["vz"] / sc.V
    th, w = sol["theta"], sol["omega"] / sc.W
    m = sol["m"] / sc.M
    s = sol["sigma"] / sc.F
    u = sol["tau"] / sc.TAU

    c_pos = dt * sc.V / sc.L
    c_vel = dt * sc.F / (sc.M * sc.V)
    c_grav = dt * G_EARTH / sc.V
    c_mass = dt * sc.F / (sc.M * vehicle.isp * G0)
    c_th = dt * sc.W
    c_w = dt * (sc.TAU / vehicle.I_pitch) / sc.W

    # True gimbal angle from the torque that was commanded.
    sin_dmax = float(np.sin(vehicle.delta_max))
    sin_delta = np.clip(u * sin_dmax / np.maximum(s, 1e-9), -1.0, 1.0)
    delta = np.arcsin(sin_delta)
    phi = th[:N] + delta                      # true thrust direction

    m_safe = np.maximum(m[:N], vehicle.m_dry / sc.M)
    a_thrust_x = c_vel * s * np.sin(phi) / m_safe
    a_thrust_z = c_vel * s * np.cos(phi) / m_safe

    if aero is not None and aero.enabled and aero_scale > 0.0:
        ax, az = aero_acceleration(sol["vx"][:N], sol["vz"][:N],
                                   np.maximum(sol["z"][:N], 0.0),
                                   th[:N], sol["m"][:N], aero)
        a_aero_x = aero_scale * dt * np.asarray(ax) / sc.V
        a_aero_z = aero_scale * dt * np.asarray(az) / sc.V
    else:
        a_aero_x = np.zeros(N)
        a_aero_z = np.zeros(N)

    rows = {
        "x": np.abs(x[:N] + c_pos * vx[:N] - x[1:]),
        "z": np.abs(z[:N] + c_pos * vz[:N] - z[1:]),
        "vx": np.abs(vx[:N] + a_thrust_x + a_aero_x - vx[1:]),
        "vz": np.abs(vz[:N] + a_thrust_z + a_aero_z - c_grav - vz[1:]),
        "theta": np.abs(th[:N] + c_th * w[:N] - th[1:]),
        "omega": np.abs(w[:N] + c_w * u - w[1:]),
        "m": np.abs(m[:N] - c_mass * s - m[1:]),
    }
    total = float(sum(r.sum() for r in rows.values()))
    return rows, total


# ======================================================================
# Main solver
# ======================================================================
def solve_scvx(
    vehicle: Vehicle6DoF = None,
    aero: AeroConfig = None,
    params: SCvxParams = None,
    N: int = 80,
    t_burn: float = 8.0,
    x0: float = 0.0,
    z0: float = None,
    vx0: float = 0.0,
    vz0: float = None,
    theta0_deg: float = 30.0,
    omega0: float = 0.0,
    gamma_gs_deg: float = 75.0,
    verbose: bool = True,
):
    """
    Solve the 6-DoF powered descent problem by successive convexification.

    Replaces the ad-hoc reference iteration of Days 3-6 with a subproblem that
    is always feasible and a trust region sized by measured linearisation error.

    Parameters
    ----------
    N : int
        Number of time intervals.
    t_burn : float
        Burn duration [s]. Fuel is very nearly proportional to it, because the
        40% throttle floor sets the flow rate whatever the optimiser wants.
    x0, z0, vx0, vz0 : float
        Initial translational state. `z0` and `vz0` default to the entry state a
        burn of this length can actually null (Day 5's sizing).
    theta0_deg : float
        Initial pitch from vertical [deg]. 90 is a full belly-flop.

    Notes
    -----
    The defaults are the regime Day 6 concluded the vehicle actually flies:
    coast on the belly with the engines off, ignite near-upright, burn briefly.
    That is not a stylistic choice -- with drag active, entry pitch is what
    decides whether the problem has a solution at all. Measured, sweeping entry
    pitch at fixed burn time, the surviving virtual control at convergence is

        0 deg   1e-12      30 deg  1e-8 .. 2e-2
        10 deg  1e-12      60 deg  0.28 .. 1.63

    and at 60 degrees it scales linearly with how much of the aerodynamic
    forcing is switched on, which is the signature of a deficit rather than a
    linearisation error. The cause is geometric: `Cd A` belly-on is 28x its
    upright value, so the drag term the subproblem holds fixed from the
    reference is both large and swinging fast while the vehicle rotates. Day 5
    sized the entry state on thrust alone, deliberately; a burn sized that way
    over-brakes once drag is added, and cannot reach the pad at the specified
    time from the specified altitude. Virtual control does not repair that --
    it *reports* it, which is the improvement over a solver that returned
    `infeasible` and left you guessing.
    gamma_gs_deg : float
        Glideslope half-angle from horizontal [deg].

    Returns
    -------
    dict with the trajectory, the control, the convergence history and status.
    """
    vehicle = vehicle or Vehicle6DoF()
    params = params or SCvxParams()
    theta0 = np.radians(theta0_deg)
    t_start = timer.time()

    if z0 is None or vz0 is None:
        t_flip = float(np.clip(1.4 * theta0 / vehicle.omega_max,
                               1.5, 0.6 * t_burn))
        z_auto, vz_auto = feasible_entry_state(vehicle, t_burn, theta0_deg,
                                               t_flip)
        z0 = z_auto if z0 is None else z0
        vz0 = vz_auto if vz0 is None else vz0

    dt = t_burn / N
    t_grid = np.linspace(0.0, t_burn, N + 1)
    sc = Scales(vehicle, t_burn, x0, z0)
    sin_dmax = float(np.sin(vehicle.delta_max))

    if verbose:
        print("=" * 70)
        print("SCvx TRAJECTORY OPTIMIZATION")
        print("=" * 70)
        print(vehicle.summary())
        print()
        if aero is not None:
            print(aero.summary())
            print()
        print(params.summary())
        print(f"\n  Nodes             : {N + 1}   (dt = {dt:.4f} s)")
        print(f"  Burn time         : {t_burn:.1f} s")
        print(f"  Entry state       : ({x0:,.0f}, {z0:,.0f}) m, "
              f"({vx0:.1f}, {vz0:.1f}) m/s")
        print(f"  Entry pitch       : {theta0_deg:.0f} deg from vertical")
        print(f"  Glideslope        : {gamma_gs_deg:.0f} deg")
        print()

    # ------------------------------------------------------------------
    # Scaled dynamics coefficients
    # ------------------------------------------------------------------
    c_pos = dt * sc.V / sc.L
    c_vel = dt * sc.F / (sc.M * sc.V)
    c_grav = dt * G_EARTH / sc.V
    c_mass = dt * sc.F / (sc.M * vehicle.isp * G0)
    c_th = dt * sc.W
    c_w = dt * (sc.TAU / vehicle.I_pitch) / sc.W

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    x = cp.Variable(N + 1, name="x")
    z = cp.Variable(N + 1, name="z")
    vx = cp.Variable(N + 1, name="vx")
    vz = cp.Variable(N + 1, name="vz")
    th = cp.Variable(N + 1, name="theta")     # radians, already O(1)
    w = cp.Variable(N + 1, name="omega")      # omega / omega_max
    m = cp.Variable(N + 1, name="m")
    s = cp.Variable(N, name="sigma")          # thrust / T_max
    u = cp.Variable(N, name="tau")            # torque / tau_max

    # Virtual control: one slack per dynamics row per interval. Scaled units,
    # so a single L1 weight is meaningful across all seven.
    nu = {k: cp.Variable(N, name=f"nu_{k}")
          for k in ("x", "z", "vx", "vz", "theta", "omega", "m")}

    # ------------------------------------------------------------------
    # Parameters. Every product of references is pre-multiplied in numpy so the
    # problem stays DPP and CVXPY caches the compilation across iterations.
    # ------------------------------------------------------------------
    q_sin = cp.Parameter(N)      # sigma -> vx
    q_cos = cp.Parameter(N)      # sigma -> vz
    q_a = cp.Parameter(N)        # theta -> vx
    q_b = cp.Parameter(N)        # theta -> vz
    q_c = cp.Parameter(N)        # constant offset -> vx
    q_d = cp.Parameter(N)        # constant offset -> vz
    qu_x = cp.Parameter(N)       # tau -> vx
    qu_z = cp.Parameter(N)       # tau -> vz
    p_ax = cp.Parameter(N)       # aerodynamic forcing, from the reference
    p_az = cp.Parameter(N)

    p_xref = cp.Parameter(N + 1)
    p_zref = cp.Parameter(N + 1)
    p_vxref = cp.Parameter(N + 1)
    p_vzref = cp.Parameter(N + 1)
    p_thref = cp.Parameter(N + 1)
    p_uref = cp.Parameter(N)
    p_trust = cp.Parameter(nonneg=True)
    p_trust_u = cp.Parameter(nonneg=True)
    p_wvc = cp.Parameter(nonneg=True)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    tan_gs = float(np.tan(np.radians(gamma_gs_deg)))
    cons = [
        # Entry state.
        x[0] == x0 / sc.L, z[0] == z0 / sc.L,
        vx[0] == vx0 / sc.V, vz[0] == vz0 / sc.V,
        th[0] == theta0, w[0] == omega0 / sc.W,
        m[0] == 1.0,
        # Terminal: on the pad, at rest, upright, not rotating. Held hard --
        # virtual control absorbs infeasible *dynamics*, and letting it absorb
        # a missed landing site instead would be the solver lying to itself.
        x[N] == 0.0, z[N] == 0.0,
        vx[N] == 0.0, vz[N] == 0.0,
        th[N] == 0.0, w[N] == 0.0,
    ]

    # --- dynamics, each row carrying its slack ------------------------
    cons += [
        x[1:] == x[:-1] + c_pos * vx[:-1] + nu["x"],
        z[1:] == z[:-1] + c_pos * vz[:-1] + nu["z"],
        th[1:] == th[:-1] + c_th * w[:-1] + nu["theta"],
        w[1:] == w[:-1] + c_w * u + nu["omega"],
        m[1:] == m[:-1] - c_mass * s + nu["m"],
        # Linearised thrust-attitude coupling, mass reference already divided
        # in. Affine in (sigma, theta, tau), so the subproblem stays convex and
        # the torque carries its own thrust-tilt cost as it must.
        vx[1:] == vx[:-1] + cp.multiply(q_sin, s) + cp.multiply(q_a, th[:-1])
        - q_c + cp.multiply(qu_x, u) + p_ax + nu["vx"],
        vz[1:] == vz[:-1] + cp.multiply(q_cos, s) - cp.multiply(q_b, th[:-1])
        + q_d - cp.multiply(qu_z, u) - c_grav + p_az + nu["vz"],
    ]

    # --- bounds and path constraints ----------------------------------
    cons += [
        z >= 0.0,
        m >= vehicle.m_dry / sc.M, m <= 1.0,
        s >= vehicle.T_min / sc.F, s <= 1.0,
        w >= -1.0, w <= 1.0,
        # |tau| <= sigma L sin(delta_max) is exactly |u| <= s once both are
        # normalised by their maxima. Linear, and exact.
        u <= s, u >= -s,
        x * tan_gs <= z, -x * tan_gs <= z,
        # Trust region. One radius for every scaled state, which is only
        # meaningful because they are scaled.
        cp.abs(x - p_xref) <= p_trust,
        cp.abs(z - p_zref) <= p_trust,
        cp.abs(vx - p_vxref) <= p_trust,
        cp.abs(vz - p_vzref) <= p_trust,
        cp.abs(th - p_thref) <= p_trust,
        cp.abs(u - p_uref) <= p_trust_u,
    ]

    fuel_obj = -m[N]
    vc_obj = sum(cp.norm1(v) for v in nu.values())
    problem = cp.Problem(cp.Minimize(fuel_obj + p_wvc * vc_obj), cons)

    # ------------------------------------------------------------------
    # Iteration state
    # ------------------------------------------------------------------
    ref = initialize_reference(N, t_burn, x0, z0, vx0, vz0, theta0, omega0,
                               vehicle, seed=params.seed)
    eta = params.eta_0
    eta_u = params.eta_u_0
    w_vc = params.w_vc

    aero_on = aero is not None and aero.enabled
    ramp_steps = max(int(params.aero_ramp_iters), 0)
    aero_scale = 0.0 if (aero_on and ramp_steps) else 1.0

    def _linear_coeffs():
        """First-order expansion of the thrust direction about the reference.

        Expanded about the *previous* gimbal angle rather than about zero: at
        the 15 degree limit the neglected 0.5 sin(theta) delta^2 term is 0.034
        of maximum thrust, which would put a floor under the defect that no
        amount of iterating could clear.
        """
        th_r = ref["theta"][:N]
        s_r = np.maximum(ref["sigma"] / sc.F, 1e-6)
        u_r = ref["tau"] / sc.TAU
        d_ref = np.arcsin(np.clip(u_r * sin_dmax / s_r, -1.0, 1.0))
        phi_r = th_r + d_ref
        K = sin_dmax / np.maximum(np.cos(d_ref), 1e-6)
        return th_r, s_r, u_r, phi_r, K

    def push_params():
        th_r, s_r, u_r, phi_r, K = _linear_coeffs()
        m_r = np.maximum(ref["m"][:N] / sc.M, vehicle.m_dry / sc.M)
        inv_m = c_vel / m_r
        sin_p, cos_p = np.sin(phi_r), np.cos(phi_r)

        q_sin.value = inv_m * sin_p
        q_cos.value = inv_m * cos_p
        q_a.value = inv_m * s_r * cos_p
        q_b.value = inv_m * s_r * sin_p
        q_c.value = inv_m * (s_r * cos_p * th_r + K * cos_p * u_r)
        q_d.value = inv_m * (s_r * sin_p * th_r + K * sin_p * u_r)
        qu_x.value = inv_m * K * cos_p
        qu_z.value = inv_m * K * sin_p

        if aero_on and aero_scale > 0.0:
            ax, az = aero_acceleration(
                ref["vx"][:N], ref["vz"][:N], np.maximum(ref["z"][:N], 0.0),
                ref["theta"][:N], ref["m"][:N], aero,
            )
            p_ax.value = aero_scale * dt * np.asarray(ax) / sc.V
            p_az.value = aero_scale * dt * np.asarray(az) / sc.V
        else:
            p_ax.value = np.zeros(N)
            p_az.value = np.zeros(N)

        p_xref.value = ref["x"] / sc.L
        p_zref.value = ref["z"] / sc.L
        p_vxref.value = ref["vx"] / sc.V
        p_vzref.value = ref["vz"] / sc.V
        p_thref.value = ref["theta"]
        p_uref.value = u_r
        p_trust.value = float(eta)
        p_trust_u.value = float(eta_u)
        p_wvc.value = float(w_vc)

    def solve_once():
        for name in (params.solver, params.solver_fallback):
            try:
                problem.solve(solver=getattr(cp, name),
                              verbose=params.solver_verbose)
                if problem.status in ("optimal", "optimal_inaccurate"):
                    return problem.status
            except Exception:      # noqa: BLE001 - try the next solver
                continue
        return problem.status or "solver_error"

    def unpack():
        """Solution in physical units."""
        return {
            "x": np.asarray(x.value) * sc.L,
            "z": np.asarray(z.value) * sc.L,
            "vx": np.asarray(vx.value) * sc.V,
            "vz": np.asarray(vz.value) * sc.V,
            "theta": np.asarray(th.value),
            "omega": np.asarray(w.value) * sc.W,
            "m": np.asarray(m.value) * sc.M,
            "sigma": np.asarray(s.value) * sc.F,
            "tau": np.asarray(u.value) * sc.TAU,
        }

    def penalised_cost(sol, scale):
        """
        J: the objective under the *true* dynamics.

        Returned as its two pieces rather than a number, because `w_vc` moves
        between iterations. rho compares a predicted cost against an incumbent,
        and if the two are priced at different weights the ratio is meaningless
        -- so the incumbent is re-priced at the current weight each iteration
        instead of being cached as a total.
        """
        _, defect = nonlinear_defect(sol, vehicle, aero, sc, dt, N, scale)
        return -sol["m"][-1] / sc.M, defect

    def thrust_linearisation_defect(sol):
        """
        Day 5's honesty check, in thrust units.

        Recomputes the true thrust vector from the solved throttle, attitude and
        torque, and returns the largest disagreement with what the linear model
        predicted at this reference, as a fraction of maximum thrust. Virtual
        control going to zero proves the linear dynamics were *satisfied*; this
        is what proves they were the right dynamics.
        """
        th_r, s_r, u_r, phi_r, K = _linear_coeffs()
        sin_p, cos_p = np.sin(phi_r), np.cos(phi_r)
        s_v = sol["sigma"] / sc.F
        u_v = sol["tau"] / sc.TAU
        th_v = sol["theta"]

        lin_x = (sin_p * s_v + s_r * cos_p * (th_v[:N] - th_r)
                 + K * cos_p * (u_v - u_r))
        lin_z = (cos_p * s_v - s_r * sin_p * (th_v[:N] - th_r)
                 - K * sin_p * (u_v - u_r))

        d_v = np.arcsin(np.clip(u_v * sin_dmax / np.maximum(s_v, 1e-9),
                                -1.0, 1.0))
        true_x = s_v * np.sin(th_v[:N] + d_v)
        true_z = s_v * np.cos(th_v[:N] + d_v)
        return float(np.max(np.hypot(lin_x - true_x, lin_z - true_z)))

    # The reference itself is the first incumbent, so iteration 1 has a real
    # rho rather than a special case.
    ref_sol = {k: ref[k].copy() for k in
               ("x", "z", "vx", "vz", "theta", "omega", "m", "sigma", "tau")}
    prev_mass, prev_defect = penalised_cost(ref_sol, aero_scale)

    history = {"fuel": [], "vc_norm": [], "defect": [], "thrust_defect": [],
               "eta": [], "step": [], "rho": [], "status": [],
               "accepted": [], "w_vc": [], "aero_scale": []}
    best = None
    last = None
    vc_prev = float("inf")
    restarts = 0

    if verbose:
        print(f"  {'It':>3}  {'status':>10}  {'fuel [kg]':>10}  {'|nu|':>10}  "
              f"{'defect':>10}  {'thrust_d':>9}  {'eta':>7}  {'step':>8}  "
              f"{'rho':>7}  {'aero':>5}  {'':>3}")
        print("  " + "-" * 98)

    # ------------------------------------------------------------------
    # SCvx loop
    # ------------------------------------------------------------------
    for it in range(1, params.max_iter + 1):
        push_params()
        status = solve_once()

        if status not in ("optimal", "optimal_inaccurate"):
            # Should not happen: with slack on every dynamics row the reference
            # itself is feasible. If it does, the conditioning has failed, not
            # the geometry -- shrink and retry.
            eta = max(eta * params.alpha_shrink, params.eta_min)
            history["fuel"].append(np.nan)
            history["vc_norm"].append(np.nan)
            history["defect"].append(np.nan)
            history["thrust_defect"].append(np.nan)
            history["eta"].append(eta)
            history["step"].append(np.nan)
            history["rho"].append(np.nan)
            history["status"].append(status)
            history["accepted"].append(False)
            history["w_vc"].append(w_vc)
            history["aero_scale"].append(aero_scale)
            if verbose:
                print(f"  {it:>3}  {status:>10}  {'---':>10}  {'---':>10}  "
                      f"{'---':>10}  {eta:>7.4f}  {'---':>8}  {'---':>7}")
            if eta <= params.eta_min:
                break
            continue

        L_new = float(problem.value)                     # predicted cost
        sol = unpack()
        vc_norm = float(sum(np.abs(v.value).sum() for v in nu.values()))
        new_mass, defect = penalised_cost(sol, aero_scale)
        # Both priced at the weight this subproblem was actually solved with.
        J_prev = prev_mass + w_vc * prev_defect
        J_new = new_mass + w_vc * defect
        # Measured against the reference this step was linearised about, so it
        # must be taken before the reference advances.
        thrust_defect = thrust_linearisation_defect(sol)

        # Step size: the largest scaled state excursion from the reference.
        step = float(max(
            np.max(np.abs(sol["x"] - ref["x"])) / sc.L,
            np.max(np.abs(sol["z"] - ref["z"])) / sc.L,
            np.max(np.abs(sol["vx"] - ref["vx"])) / sc.V,
            np.max(np.abs(sol["vz"] - ref["vz"])) / sc.V,
            np.max(np.abs(sol["theta"] - ref["theta"])),
        ))

        # rho = actual improvement / predicted improvement.
        predicted = J_prev - L_new
        actual = J_prev - J_new
        if abs(predicted) < 1e-12:
            # The subproblem promised nothing; nothing to validate.
            rho = 1.0 if actual >= -1e-12 else 0.0
        else:
            rho = actual / predicted

        fuel = float(vehicle.m_wet - sol["m"][-1])
        accept = rho > params.rho_ok

        history["fuel"].append(fuel)
        history["vc_norm"].append(vc_norm)
        history["defect"].append(defect)
        history["thrust_defect"].append(thrust_defect)
        history["eta"].append(eta)
        history["step"].append(step)
        history["rho"].append(rho)
        history["status"].append(status)
        history["accepted"].append(bool(accept))
        history["w_vc"].append(w_vc)
        history["aero_scale"].append(aero_scale)

        if verbose:
            print(f"  {it:>3}  {status:>10}  {fuel:>10,.0f}  {vc_norm:>10.2e}  "
                  f"{defect:>10.2e}  {thrust_defect:>9.2e}  {eta:>7.4f}  "
                  f"{step:>8.2e}  {rho:>7.3f}  {aero_scale:>5.2f}  "
                  f"{'ok' if accept else 'rej':>3}")

        # Keep the honest cheapest iterate, not whichever solve happened last.
        # A trajectory the linear model does not describe is worth less than a
        # more expensive one it does -- so an iterate only qualifies if its true
        # dynamics residual is small *and* the linearised thrust direction
        # agrees with the real one.
        # Both halves are needed and they check different things. Small ||nu||
        # says the linear model satisfied the dynamics it wrote down; a small
        # thrust defect says those were the dynamics of an actual rocket. The
        # Day 7 guide's model passes the first test and fails the second.
        honest = (vc_norm < params.dyn_tol
                  and thrust_defect < params.defect_tol)
        if honest and (best is None or fuel < best["fuel"]):
            best = dict(sol, fuel=fuel, vc_norm=vc_norm, defect=defect,
                        thrust_defect=thrust_defect, iteration=it)
        last = dict(sol, fuel=fuel, vc_norm=vc_norm, defect=defect,
                    thrust_defect=thrust_defect, iteration=it)

        if accept:
            for k in ("x", "z", "vx", "vz", "theta", "omega", "m",
                      "sigma", "tau"):
                ref[k] = sol[k].copy()
            ref["z"] = np.maximum(ref["z"], 0.0)
            prev_mass, prev_defect = new_mass, defect

        # Trust-region update.
        if rho > params.rho_good:
            eta = min(eta * params.alpha_expand, params.eta_max)
            eta_u = min(eta_u * params.alpha_expand, 2.0)
        elif not accept:
            eta = max(eta * params.alpha_shrink, params.eta_min)
            eta_u = max(eta_u * params.alpha_shrink, params.eta_u_min)

        # Raise the price of slack only when raising it might help -- that is,
        # when the optimiser is still buying just as much as last time. Growing
        # it every iteration regardless drives the weight to its ceiling within
        # nine steps, and a 1e7 coefficient sitting next to an O(1) objective
        # ruins the conditioning: both solvers then return `unbounded` on a
        # problem that is plainly bounded below by -1.
        if vc_norm > params.vc_tol and vc_norm > 0.9 * vc_prev:
            w_vc = min(w_vc * params.w_vc_grow, params.w_vc_max)
        vc_prev = vc_norm

        # Advance the aero continuation, if one was asked for.
        # Advance the aero continuation, gated on the thrust defect rather than
        # on the dynamics residual: while the forcing is being walked in, that
        # residual is dominated by the aero term itself and never falls below
        # its own gate, so the ramp stalls at step zero and silently solves the
        # aero-free problem instead.
        ramping = aero_on and aero_scale < 1.0
        if ramping and accept and thrust_defect < params.defect_tol * 4.0:
            aero_scale = min(1.0, aero_scale + 1.0 / ramp_steps)
            eta = max(eta, 0.05)
            # The forcing term just changed, so the incumbent cost is stale.
            prev_mass, prev_defect = penalised_cost(sol, aero_scale)

        # Convergence: the slack is gone and the iterate has stopped moving.
        if it >= params.min_iter and not ramping:
            recent = [f for f in history["fuel"][-4:] if not np.isnan(f)]
            settled = (len(recent) == 4
                       and max(recent) - min(recent) < params.fuel_tol_kg)
            if vc_norm < params.vc_tol and (step < params.step_tol or settled):
                if verbose:
                    print(f"\n  Converged after {it} iterations "
                          f"(|nu| = {vc_norm:.2e}, step = {step:.2e}, "
                          f"defect = {defect:.2e})")
                break
            if eta <= params.eta_min:
                # A collapsed region with the dynamics still unsatisfied is a
                # stall, not an answer. Shrinking further cannot help: the
                # linearisation is already accurate -- what is wrong is where
                # the reference sits. Re-expand, make slack more expensive, and
                # let it move.
                if (vc_norm > params.dyn_tol
                        and restarts < params.max_restarts):
                    restarts += 1
                    eta = params.eta_0
                    eta_u = params.eta_u_0
                    w_vc = min(w_vc * params.w_vc_grow, params.w_vc_max)
                    prev_mass, prev_defect = penalised_cost(sol, aero_scale)
                    if verbose:
                        print(f"       trust region collapsed with |nu| = "
                              f"{vc_norm:.2e}; restart {restarts} at "
                              f"eta = {eta}, w_vc = {w_vc:.0e}")
                    continue
                if verbose:
                    print(f"\n  Trust region collapsed at iteration {it}.")
                break

    # ------------------------------------------------------------------
    # Package
    # ------------------------------------------------------------------
    elapsed = timer.time() - t_start
    result = best or last
    if result is None:
        if verbose:
            print(f"\n  NO SOLUTION ({elapsed:.1f}s)")
        return {"status": "failed", "history": history, "elapsed": elapsed}

    sigma = result["sigma"]
    tau = result["tau"]
    delta = np.arcsin(np.clip(
        tau / np.maximum(sigma, 1e-6) / vehicle.L_engine, -1.0, 1.0))

    result["t"] = t_grid
    result["delta"] = delta
    result["Tx"] = sigma * np.sin(result["theta"][:N] + delta)
    result["Tz"] = sigma * np.cos(result["theta"][:N] + delta)
    result["status"] = (
        "converged"
        if (result["vc_norm"] < params.dyn_tol
            and result["thrust_defect"] < params.defect_tol)
        else "unconverged"
    )
    result["history"] = history
    result["iterations"] = len(history["fuel"])
    result["elapsed"] = elapsed
    result["t_burn"] = t_burn
    result["N"] = N
    result["theta0_deg"] = theta0_deg
    result["gamma_gs_deg"] = gamma_gs_deg

    result["q"] = np.asarray(dynamic_pressure(result["vx"], result["vz"],
                                              np.maximum(result["z"], 0.0)))
    Fdx, Fdz = aero_force(result["vx"][:N], result["vz"][:N],
                          np.maximum(result["z"][:N], 0.0),
                          result["theta"][:N], aero or AeroConfig())
    result["drag_mag"] = np.hypot(np.asarray(Fdx), np.asarray(Fdz))

    # The check the Day 7 guide's model cannot pass: is the thrust vector
    # actually reachable with a 15 degree gimbal?
    result["max_gimbal_deg"] = float(np.degrees(np.abs(delta).max()))

    if verbose:
        v_f = float(np.hypot(result["vx"][-1], result["vz"][-1]))
        print(f"\n  SOLUTION ({elapsed:.1f}s, {result['iterations']} iterations, "
              f"best from iteration {result['iteration']})")
        print(f"  Propellant        : {result['fuel']:,.0f} kg "
              f"({100 * result['fuel'] / vehicle.m_prop_initial:.1f}% of load)")
        print(f"  Virtual control   : {result['vc_norm']:.2e}")
        print(f"  True defect       : {result['defect']:.2e}")
        print(f"  Touchdown         : ({result['x'][-1]:.2f}, "
              f"{result['z'][-1]:.2f}) m at {v_f:.2f} m/s")
        print(f"  Final pitch       : "
              f"{np.degrees(result['theta'][-1]):.2f} deg, "
              f"{np.degrees(result['omega'][-1]):.2f} deg/s")
        print(f"  Peak gimbal       : {result['max_gimbal_deg']:.2f} deg "
              f"(limit {vehicle.delta_max_deg:.0f})")
        print(f"  Peak q            : {np.max(result['q']) / 1000:.1f} kPa")
    return result


# ======================================================================
# Visualization
# ======================================================================
def plot_scvx_trajectory(result, save_path=None, vehicle=None):
    """Ten panels: the trajectory, the states, and the control that flew it."""
    if result.get("status") == "failed":
        print("Cannot plot - no solution.")
        return
    vehicle = vehicle or Vehicle6DoF()
    save_path = save_path or os.path.join(RESULTS, "day7_scvx.png")

    t = result["t"]
    N = len(t) - 1
    tc = t[:-1]

    fig, ax = plt.subplots(2, 5, figsize=(28, 10))
    fig.suptitle("Day 7: SCvx trajectory optimisation (6-DoF, coupled thrust, "
                 "aerodynamics)", fontsize=14, y=1.02)

    a = ax[0, 0]
    a.plot(result["x"], result["z"], "b-", lw=2)
    a.plot(result["x"][0], result["z"][0], "go", ms=10, label="entry")
    a.plot(result["x"][-1], result["z"][-1], "r^", ms=12, label="pad")
    scale = 0.06 * max(result["z"].max(), 1.0)
    for i in np.linspace(0, N, 12, dtype=int):
        th = result["theta"][i]
        dx, dz = scale * np.sin(th), scale * np.cos(th)
        a.plot([result["x"][i] - dx / 2, result["x"][i] + dx / 2],
               [result["z"][i] - dz / 2, result["z"][i] + dz / 2],
               "k-", lw=2, alpha=0.6)
    a.set_xlabel("Downrange [m]"); a.set_ylabel("Altitude [m]")
    a.set_title("Trajectory and attitude"); a.legend(fontsize=8)
    a.grid(alpha=0.3); a.set_aspect("equal", adjustable="datalim")

    a = ax[0, 1]
    a.plot(t, result["z"], lw=2, label="altitude")
    a.plot(t, result["x"], lw=2, label="downrange")
    a.axhline(0, color="k", lw=0.5)
    a.set_xlabel("Time [s]"); a.set_ylabel("[m]")
    a.set_title("Position"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 2]
    a.plot(t, np.hypot(result["vx"], result["vz"]), "k-", lw=2, label="|v|")
    a.plot(t, result["vx"], lw=1.5, alpha=0.7, label="vx")
    a.plot(t, result["vz"], lw=1.5, alpha=0.7, label="vz")
    a.set_xlabel("Time [s]"); a.set_ylabel("[m/s]")
    a.set_title("Velocity"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 3]
    a.plot(t, np.degrees(result["theta"]), lw=2, color="tab:purple")
    a.axhline(0, color="k", lw=0.5)
    a.set_xlabel("Time [s]"); a.set_ylabel("[deg]")
    a.set_title("Pitch from vertical"); a.grid(alpha=0.3)

    a = ax[0, 4]
    a.plot(t, result["q"] / 1000, lw=2, color="tab:red")
    a.set_xlabel("Time [s]"); a.set_ylabel("[kPa]")
    a.set_title("Dynamic pressure"); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.plot(t, np.degrees(result["omega"]), lw=2, color="tab:orange")
    a.axhline(np.degrees(vehicle.omega_max), color="r", ls=":", alpha=0.6,
              label="rate limit")
    a.axhline(-np.degrees(vehicle.omega_max), color="r", ls=":", alpha=0.6)
    a.set_xlabel("Time [s]"); a.set_ylabel("[deg/s]")
    a.set_title("Pitch rate"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(tc, result["sigma"] / 1e6, lw=2)
    a.axhline(vehicle.T_max / 1e6, color="r", ls=":", alpha=0.6, label="T_max")
    a.axhline(vehicle.T_min / 1e6, color="orange", ls=":", alpha=0.6,
              label="T_min")
    a.set_xlabel("Time [s]"); a.set_ylabel("[MN]")
    a.set_title("Thrust"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 2]
    a.plot(tc, result["drag_mag"] / 1000, lw=2, color="tab:green")
    a.set_xlabel("Time [s]"); a.set_ylabel("[kN]")
    a.set_title("Aerodynamic drag"); a.grid(alpha=0.3)

    a = ax[1, 3]
    a.plot(tc, np.degrees(result["delta"]), lw=2, color="tab:cyan")
    a.axhline(vehicle.delta_max_deg, color="r", ls=":", alpha=0.6,
              label="gimbal limit")
    a.axhline(-vehicle.delta_max_deg, color="r", ls=":", alpha=0.6)
    a.set_xlabel("Time [s]"); a.set_ylabel("[deg]")
    a.set_title("Gimbal angle (torque source)"); a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 4]
    a.plot(t, result["m"] / 1000, lw=2, color="tab:purple")
    a.axhline(vehicle.m_dry / 1000, color="r", ls=":", alpha=0.6, label="dry")
    a.set_xlabel("Time [s]"); a.set_ylabel("[tonnes]")
    a.set_title("Mass"); a.legend(fontsize=8); a.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nTrajectory plot -> {save_path}")
    plt.close()


def plot_scvx_convergence(result, save_path=None, params=None):
    """
    Four panels: what the algorithm did to itself while converging.

    The second panel is the one that matters. It plots the virtual-control norm
    against the true nonlinear defect: the slack the linear model *thinks* it
    needs, against the error it actually has. When both fall together, the
    linear model has become an honest description of the dynamics. When only
    the first falls, the solver has stopped paying for a problem it still has.
    """
    hist = result.get("history")
    if not hist or not hist["fuel"]:
        print("No convergence history.")
        return
    params = params or SCvxParams()
    save_path = save_path or os.path.join(RESULTS, "day7_scvx_convergence.png")

    n = len(hist["fuel"])
    its = np.arange(1, n + 1)
    fuel = np.array(hist["fuel"], dtype=float)
    vc = np.array(hist["vc_norm"], dtype=float)
    dfc = np.array(hist["defect"], dtype=float)
    eta = np.array(hist["eta"], dtype=float)
    step = np.array(hist["step"], dtype=float)
    rho = np.array(hist["rho"], dtype=float)
    ok = np.array(hist["accepted"], dtype=bool)

    fig, ax = plt.subplots(1, 4, figsize=(23, 5))
    fig.suptitle("SCvx convergence history", fontsize=13)

    a = ax[0]
    v = ~np.isnan(fuel)
    a.plot(its[v], fuel[v], "o-", lw=2, color="tab:blue")
    a.set_xlabel("Iteration"); a.set_ylabel("Propellant [kg]")
    a.set_title("Objective"); a.grid(alpha=0.3)

    a = ax[1]
    v = ~np.isnan(vc)
    a.semilogy(its[v], np.maximum(vc[v], 1e-16), "o-", lw=2, color="tab:red",
               label="virtual control $\\|\\nu\\|_1$")
    v = ~np.isnan(dfc)
    a.semilogy(its[v], np.maximum(dfc[v], 1e-16), "s--", lw=2,
               color="tab:brown", label="true nonlinear defect")
    a.axhline(params.vc_tol, color="green", ls=":", alpha=0.7, label="tolerance")
    a.set_xlabel("Iteration"); a.set_ylabel("L1 norm, scaled (log)")
    a.set_title("Slack vs. reality"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[2]
    a.semilogy(its, eta, "o-", lw=2, color="tab:green")
    rej = ~ok
    if rej.any():
        a.semilogy(its[rej], eta[rej], "x", ms=10, color="tab:red",
                   label="step rejected")
        a.legend(fontsize=8)
    a.set_xlabel("Iteration"); a.set_ylabel("$\\eta$ (log)")
    a.set_title("Trust-region radius"); a.grid(alpha=0.3)

    a = ax[3]
    v = ~np.isnan(step)
    a.semilogy(its[v], np.maximum(step[v], 1e-16), "o-", lw=2,
               color="tab:orange", label="step size")
    a.axhline(params.step_tol, color="green", ls=":", alpha=0.7,
              label="tolerance")
    a.set_xlabel("Iteration"); a.set_ylabel("scaled step (log)")
    a.set_title("Iteration step"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a2 = a.twinx()
    v = ~np.isnan(rho)
    a2.plot(its[v], np.clip(rho[v], -0.5, 2.0), ".-", lw=1, alpha=0.5,
            color="tab:gray")
    a2.set_ylabel("step quality $\\rho$", color="tab:gray")
    a2.axhline(1.0, color="tab:gray", ls=":", alpha=0.4)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Convergence plot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    print()
    res = solve_scvx(aero=AeroConfig(), verbose=True)
    if res.get("status") != "failed":
        plot_scvx_trajectory(res)
        plot_scvx_convergence(res)
    print()
