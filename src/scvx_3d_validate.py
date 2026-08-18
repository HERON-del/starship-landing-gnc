"""
Day 17 -- validating the 3-D stack, with a second and independent solver.

Day 16 built a 3-D SCvx solver and it does not converge: its virtual control
stalls around 0.4 against a 1e-6 tolerance, and the plan misses by ~250 m when
flown. That leaves an open question this day is the right shape to answer. Is
the *physics* from Days 13 to 15 wrong, or is it only Day 16's particular
convex formulation that fails?

The way to find out is a second formulation that shares the physics and shares
nothing else. This file is that: same `Vehicle3D`, same quaternion library,
same Euler equations and the same aerodynamics, imported live -- but a
different discretisation (state-transition matrix rather than explicit Euler),
different Jacobians (central differences rather than the hand-derived
analytics), a different reference initialisation, a soft trust-region penalty
alongside the hard one, and a geometric trust schedule instead of Day 16's
accept/reject controller.

Where the two agree, the physics is confirmed. Where they disagree, the
disagreement is located in the formulation rather than the model.

A note on the guide's framing. It proposes rebuilding the vehicle, the
quaternion algebra and the dynamics standalone, on the grounds that earlier
days "live only in markdown, not as a live module". That is not true of this
repository -- every day since Day 1 is an importable module with its own test
suite. Re-typing the physics here would validate a fresh copy of it and
nothing else, which is the opposite of what a validation day is for. The
algorithmic differences are kept; the physics is imported.

Frames follow Days 13 to 16, not the guide: inertial +z is up, the vehicle's
long axis is body +x, and the state is

    s = [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]

with the control the body-frame thrust force vector F. The guide puts mass
first and calls +x up; mixing the two would be a bug factory.
"""

import os
import sys
import time as timer

import cvxpy as cp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.dynamics_3d import (                                  # noqa: E402
    Vehicle3D, G0, G_EARTH, dynamics_3d_derivative,
    attitude_from_pitch, tilt_from_vertical, make_initial_state_3d,
    IDX_POS, IDX_VEL, IDX_QUAT, IDX_OMEGA, IDX_MASS,
)
from src.aero_3d import (                                      # noqa: E402
    AeroConfig3D, aero_force_and_moment_body, propagate_3d_with_aero,
)
from src.quaternion import quat_normalize, quat_angle_between  # noqa: E402
from src.scvx_3d import force_to_gimbal                        # noqa: E402

N_STATE = 14


# ======================================================================
# Dynamics, in the thrust-vector parameterisation
# ======================================================================
def dynamics_force_vector(s, F_body, vehicle, aero_cfg=None,
                          wind_inertial=(0.0, 0.0, 0.0)):
    """
    The 14-state derivative with the body-frame thrust *vector* as control.

    Everything physical here comes from Day 14's derivative through its
    `extra_body_wrench` hook -- gravity, the body-to-inertial rotation, Euler's
    equations with gyroscopic coupling. Only two things are supplied locally:
    the thrust wrench, because the solver's control is a force rather than an
    angle pair, and the mass flow, because Day 14's is written in terms of a
    commanded magnitude.
    """
    s = np.asarray(s, dtype=float)
    F_body = np.asarray(F_body, dtype=float)
    tau = np.cross(np.array([-vehicle.L_engine, 0.0, 0.0]), F_body)

    if aero_cfg is not None and aero_cfg.enabled:
        F_a, tau_a = aero_force_and_moment_body(
            s[IDX_VEL], wind_inertial, s[IDX_QUAT], s[2], aero_cfg)
    else:
        F_a = np.zeros(3)
        tau_a = np.zeros(3)

    ds = dynamics_3d_derivative(
        s, 0.0, 0.0, 0.0, vehicle,
        extra_body_wrench=lambda _s: (F_body + F_a, tau + tau_a))
    ds[IDX_MASS] = -float(np.linalg.norm(F_body)) / (vehicle.isp * G0)
    return ds


def linearize_at(s_ref, F_ref, vehicle, aero_cfg=None,
                 wind_inertial=(0.0, 0.0, 0.0), eps=1e-6):
    """
    Central-difference Jacobians A = df/ds, B = df/dF, and the affine offset.

    Day 16 derived these analytically. Differencing them here is deliberately
    the *other* way of getting the same object, so that agreement between the
    two solvers is not agreement between one derivation and itself.
    """
    def f(s, F):
        return dynamics_force_vector(s, F, vehicle, aero_cfg, wind_inertial)

    A = np.zeros((N_STATE, N_STATE))
    for i in range(N_STATE):
        d = np.zeros(N_STATE)
        d[i] = eps
        A[:, i] = (f(s_ref + d, F_ref) - f(s_ref - d, F_ref)) / (2 * eps)
    B = np.zeros((N_STATE, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        B[:, i] = (f(s_ref, F_ref + d) - f(s_ref, F_ref - d)) / (2 * eps)
    return A, B, f(s_ref, F_ref) - A @ s_ref - B @ F_ref


def discretize(s_ref, F_ref, vehicle, dt, aero_cfg=None,
               wind_inertial=(0.0, 0.0, 0.0), nsub=4):
    """
    Discretise by integrating the state-transition matrix.

    Phi' = A Phi with Phi(0) = I, stepped with RK4, then the control and offset
    carried through it. This is the piece Day 16 did with a bare Euler step,
    and it is the main formulation difference between the two solvers.
    """
    A, B, z = linearize_at(s_ref, F_ref, vehicle, aero_cfg, wind_inertial)
    Phi = np.eye(N_STATE)
    h = dt / nsub
    for _ in range(nsub):
        k1 = A @ Phi
        k2 = A @ (Phi + 0.5 * h * k1)
        k3 = A @ (Phi + 0.5 * h * k2)
        k4 = A @ (Phi + h * k3)
        Phi = Phi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return Phi, Phi @ B * dt, Phi @ z * dt


# ======================================================================
# Reference
# ======================================================================
def initialize_reference(s0, vehicle, K, q_final=None):
    """
    Straight-line states, mid-band thrust along the body axis.

    The thrust level matters more than it looks. Hover for this vehicle is
    about 1.3 MN and T_min is 2.76 MN, so a hover-thrust reference sits below
    a floor the engine is never allowed to come down to -- the first
    linearisation would be built around a command the vehicle cannot issue.
    The guide hit exactly this and records it as its second bug. Mid-band is
    the fix, and it is also simply inside the feasible set.
    """
    q_final = attitude_from_pitch(0.0) if q_final is None else q_final
    ref_s = np.zeros((K, N_STATE))
    for k in range(K):
        a = (K - 1 - k) / (K - 1)
        ref_s[k, IDX_POS] = a * s0[IDX_POS]
        ref_s[k, IDX_VEL] = a * s0[IDX_VEL]
        ref_s[k, IDX_QUAT] = quat_normalize(
            a * s0[IDX_QUAT] + (1 - a) * q_final)
        ref_s[k, IDX_OMEGA] = a * s0[IDX_OMEGA]
        ref_s[k, IDX_MASS] = (a * s0[IDX_MASS]
                              + (1 - a) * (vehicle.m_dry + 1000.0))
    ref_F = np.zeros((K, 3))
    ref_F[:, 0] = 0.5 * (vehicle.T_min + vehicle.T_max)
    return ref_s, ref_F


# ======================================================================
# Convex sub-problem
# ======================================================================
def solve_subproblem(ref_s, ref_F, s0, vehicle, K, tf, trust, q_final,
                     aero_cfg, wind_inertial, gs_half_angle_deg, omega_max,
                     w_nu, w_trust):
    dt = tf / (K - 1)
    s = cp.Variable((K, N_STATE))
    F = cp.Variable((K, 3))
    nu = cp.Variable((K - 1, N_STATE))

    tan_gs = float(np.tan(np.radians(gs_half_angle_deg)))
    cos_delta = float(np.cos(vehicle.delta_max))

    cons = [
        s[0] == s0,
        s[K - 1, 0:3] == np.zeros(3),
        s[K - 1, 3:6] == np.zeros(3),
        s[K - 1, 6:10] == q_final,
        s[K - 1, 10:13] == np.zeros(3),
    ]

    for k in range(K):
        cons += [
            s[k, IDX_MASS] >= vehicle.m_dry,
            s[k, 2] >= 0.0,
            # Glideslope, measured from the vertical. The guide's first bug was
            # this angle being taken from the horizontal instead, which made
            # its own initial condition violate the constraint at k = 0 before
            # a single iteration ran.
            cp.norm(cp.hstack([s[k, 0], s[k, 1]])) <= s[k, 2] * tan_gs + 1e-3,
            cp.norm(s[k, 10:13]) <= omega_max,
            cp.norm(F[k]) <= vehicle.T_max,
            cos_delta * cp.norm(F[k]) <= F[k, 0],
            # Lower thrust bound, linearised about the reference direction --
            # ||F|| >= T_min is not convex, its projection onto a fixed
            # direction is.
            ref_F[k] @ F[k] / max(float(np.linalg.norm(ref_F[k])), 1e-6)
            >= vehicle.T_min,
            cp.norm(s[k] - ref_s[k])
            <= trust * (1.0 + float(np.linalg.norm(ref_s[k]))),
            cp.norm(F[k] - ref_F[k])
            <= trust * (1.0 + float(np.linalg.norm(ref_F[k]))),
        ]

    for k in range(K - 1):
        Ad, Bd, zd = discretize(ref_s[k], ref_F[k], vehicle, dt, aero_cfg,
                                wind_inertial)
        cons += [s[k + 1] == Ad @ s[k] + Bd @ F[k] + zd + nu[k]]

    cost = (-s[K - 1, IDX_MASS] / vehicle.m_wet
            + w_nu * cp.sum(cp.abs(nu))
            + w_trust * (cp.sum_squares((s - ref_s) / vehicle.m_wet)
                         + cp.sum_squares((F - ref_F) / vehicle.T_max)))
    prob = cp.Problem(cp.Minimize(cost), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:                                           # noqa: BLE001
        try:
            prob.solve(solver=cp.SCS)
        except Exception:                                       # noqa: BLE001
            return prob, None, None, None
    return prob, s, F, nu


# ======================================================================
# Outer loop
# ======================================================================
def solve_scvx_validate(s0, vehicle=None, aero_cfg=None, K=30, tf=18.0,
                        max_iter=15, trust0=5.0, shrink=0.75, trust_min=0.03,
                        q_final=None, wind_inertial=(0.0, 0.0, 0.0),
                        gs_half_angle_deg=60.0, omega_max=0.35,
                        w_nu=1e3, w_trust=1e-2, verbose=True):
    """
    Run the loop, and return the last iterate that actually solved.

    The `iterations` count and `ever_solved` flag matter here. The guide's
    version returns its reference array whatever happens, so a run whose very
    first sub-problem was infeasible returns the straight-line initial guess --
    which lands at the origin, upright, at rest, with zero gimbal, because that
    is how the guess was built. Every one of those is a boundary condition a
    test would happily accept. This one says so instead.
    """
    vehicle = vehicle or Vehicle3D()
    q_final = attitude_from_pitch(0.0) if q_final is None else q_final
    s0 = np.asarray(s0, dtype=float)
    ref_s, ref_F = initialize_reference(s0, vehicle, K, q_final)
    trust = trust0
    history = []
    ever = False
    t0 = timer.time()

    if verbose:
        print(f"  {'iter':>4} {'status':>20} {'m_f':>10} {'|nu|_1':>11} "
              f"{'trust':>10} {'radius':>7}")

    for i in range(1, max_iter + 1):
        prob, s, F, nu = solve_subproblem(
            ref_s, ref_F, s0, vehicle, K, tf, trust, q_final, aero_cfg,
            wind_inertial, gs_half_angle_deg, omega_max, w_nu, w_trust)
        if s is None or prob.status not in ("optimal", "optimal_inaccurate"):
            if verbose:
                print(f"  {i:>4} {str(prob.status):>20}")
            history.append({"iter": i, "status": str(prob.status)})
            break

        s_v, F_v, nu_v = s.value, F.value, nu.value
        step = float(np.sum((s_v - ref_s) ** 2) + np.sum((F_v - ref_F) ** 2))
        nu_norm = float(np.sum(np.abs(nu_v)))
        history.append({"iter": i, "status": prob.status,
                        "m_f": float(s_v[-1, IDX_MASS]), "step": step,
                        "nu": nu_norm, "trust": trust})
        if verbose:
            print(f"  {i:>4} {prob.status:>20} {s_v[-1, IDX_MASS]:>10,.0f} "
                  f"{nu_norm:>11.3e} {step:>10.3e} {trust:>7.3f}")

        ref_s, ref_F = s_v, F_v
        ever = True
        trust = max(trust * shrink, trust_min)
        if step < 1e-1 and nu_norm < 1e-1:
            break

    ref_s = ref_s.copy()
    for k in range(len(ref_s)):
        ref_s[k, IDX_QUAT] = quat_normalize(ref_s[k, IDX_QUAT])

    return {
        "s": ref_s, "F": ref_F, "history": history,
        "ever_solved": ever,
        "iterations": sum(1 for h in history if "nu" in h),
        # The last history row is often the infeasible one that ended the
        # loop, and it carries no defect. Report the last row that actually
        # solved, which is the iterate being returned.
        "nu": next((h["nu"] for h in reversed(history) if "nu" in h),
                   float("nan")),
        "nu_best": min((h["nu"] for h in history if "nu" in h),
                       default=float("nan")),
        "tf": tf, "K": K, "elapsed": timer.time() - t0,
        "fuel": float(vehicle.m_wet - ref_s[-1, IDX_MASS]),
        "is_initial_guess": not ever,
    }


# ======================================================================
# Replay -- the plan handed to the real model
# ======================================================================
def replay(result, vehicle=None, aero_cfg=None,
           wind_inertial=(0.0, 0.0, 0.0), substeps=20):
    """Fly the solved control through Day 15's model, as Day 16 does."""
    vehicle = vehicle or Vehicle3D()
    aero_cfg = aero_cfg or AeroConfig3D()
    s_plan, F = result["s"], result["F"]
    dt = result["tf"] / (result["K"] - 1)

    s = make_initial_state_3d(pos=s_plan[0, 0:3], vel=s_plan[0, 3:6],
                              quat=s_plan[0, 6:10], omega=s_plan[0, 10:13],
                              m=s_plan[0, IDX_MASS], vehicle=vehicle)
    hist = [s.copy()]
    for k in range(len(s_plan) - 1):
        T, dy, dz = force_to_gimbal(F[k])
        _, seg = propagate_3d_with_aero(
            s, lambda t, st: (T, dy, dz), (0.0, dt), dt / substeps,
            vehicle, aero_cfg, wind_inertial=wind_inertial)
        s = seg[-1]
        hist.append(s.copy())
    hist = np.array(hist)
    return {
        "hist": hist,
        "miss_m": float(np.linalg.norm(hist[-1, 0:3])),
        "speed_ms": float(np.linalg.norm(hist[-1, 3:6])),
        "tilt_deg": float(np.degrees(tilt_from_vertical(hist[-1, 6:10]))),
        "att_err_deg": float(np.degrees(
            quat_angle_between(hist[-1, 6:10], s_plan[-1, 6:10]))),
        "fuel_kg": float(vehicle.m_wet - hist[-1, IDX_MASS]),
    }


# ======================================================================
# The two validation cases
# ======================================================================
def planar_ic(vehicle=None, theta0_deg=25.0):
    """Downrange and altitude only -- nothing out of plane, no roll."""
    vehicle = vehicle or Vehicle3D()
    return make_initial_state_3d(
        pos=(250.0, 0.0, 600.0), vel=(-45.0, 0.0, -80.0),
        quat=attitude_from_pitch(np.radians(theta0_deg)),
        omega=(0.0, 0.0, 0.0), m=vehicle.m_wet, vehicle=vehicle)


def threed_ic(vehicle=None, theta0_deg=25.0, cross_range=180.0):
    """The same, with real cross-range, out-of-plane velocity and roll."""
    vehicle = vehicle or Vehicle3D()
    q = quat_normalize(np.array([np.cos(np.radians(4.0)),
                                 np.sin(np.radians(4.0)), 0.0, 0.0]))
    q0 = attitude_from_pitch(np.radians(theta0_deg))
    from src.quaternion import quat_multiply
    return make_initial_state_3d(
        pos=(250.0, cross_range, 600.0), vel=(-45.0, -20.0, -80.0),
        quat=quat_multiply(q0, q), omega=(0.02, 0.0, 0.0),
        m=vehicle.m_wet, vehicle=vehicle)


def out_of_plane_extremes(result):
    """Everything that must stay at zero for a planar initial condition."""
    s, F = result["s"], result["F"]
    return {
        "y": float(np.abs(s[:, 1]).max()),
        "vy": float(np.abs(s[:, 4]).max()),
        "roll_rate": float(np.abs(s[:, 10]).max()),
        "yaw_rate": float(np.abs(s[:, 12]).max()),
        "Fy": float(np.abs(F[:, 1]).max()),
    }


def gimbal_angles_deg(result):
    """Commanded deflection off the body axis at every node."""
    F = result["F"]
    mag = np.linalg.norm(F, axis=1)
    return np.degrees(np.arccos(np.clip(
        F[:, 0] / np.maximum(mag, 1e-9), -1.0, 1.0)))


if __name__ == "__main__":
    v = Vehicle3D()
    for label, s0 in (("PLANAR", planar_ic(v)), ("3-D", threed_ic(v))):
        print("=" * 72)
        print(f"{label} CASE")
        print("=" * 72)
        r = solve_scvx_validate(s0, v, K=30, tf=18.0, verbose=True)
        print(f"  ever solved      : {r['ever_solved']}")
        print(f"  fuel             : {r['fuel']:,.1f} kg")
        print(f"  out of plane     : {out_of_plane_extremes(r)}")
        print(f"  peak gimbal      : {gimbal_angles_deg(r).max():.4f} deg")
        print()
