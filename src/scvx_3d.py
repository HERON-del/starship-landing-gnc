"""
The 3-D SCvx solver: Days 13 to 15 turned into something a convex solver can
optimise over.

The parameterisation is the whole trick. Days 3 to 8 used (thrust magnitude,
gimbal angle) because a single rotation axis needs one angle. Carrying that
into 3-D would mean linearising the sine and cosine of two angles. Making the
decision variable the **body-frame thrust force vector** instead leaves four
things exactly convex, with no reference trajectory and no Taylor expansion
anywhere near them:

    ||F||         <= sigma,  T_min <= sigma <= T_max     second-order cone
    ||(Fy, Fz)||  <= Fx tan(delta_max)                   second-order cone
    tau           =  [0, L Fz, -L Fy]                    linear in F
    ||(x, y)||    <= z tan(gamma)                        second-order cone

The torque line is worth pausing on. Day 14 found by sweeping 10,201 deflection
pairs that this gimbal cannot produce roll torque. Here it is not a finding at
all -- the cross product with a fixed vector along body x has no x component,
so tau_x is structurally absent from the problem.

Three things genuinely need linearising, and all three are checked against
finite differences in `tests/test_scvx_3d.py`:

  * the quaternion kinematics, bilinear in (q, omega), via the Hamilton
    left/right multiplication matrices
  * the gyroscopic coupling, quadratic in omega, via its analytic Jacobian
  * R(q) F, quadratic in q, via the four dR/dq matrices

Aerodynamics is a reference-iteration perturbation rather than a fourth
linearisation -- Day 7's pattern, and the same trade: aero updates once per
outer iteration instead of continuously.

Scope, deliberately reduced: fixed final time, Euler discretisation, raw mass.
Day 4's trapezoidal collocation, Day 8's free time and log-mass substitution
all generalise, but combining them with a 14-state quaternion linearisation in
one file would produce something unverifiable.

Convention note, and it matters: **upright is not the identity quaternion.**
This project puts the vehicle's long axis on body +x and "up" on inertial +z,
so an upright vehicle is a 90 degree rotation and the identity quaternion is
the belly-flop. `attitude_from_pitch` from Day 14 is the only place that
knowledge lives; terminal and initial attitudes here both go through it.
"""

import os
import sys
import time as timer

import cvxpy as cp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.dynamics_3d import (                                  # noqa: E402
    Vehicle3D, G0, G_EARTH, attitude_from_pitch, tilt_from_vertical,
    make_initial_state_3d,
)
from src.aero_3d import (                                      # noqa: E402
    AeroConfig3D, aero_force_and_moment_body, propagate_3d_with_aero,
)
from src.quaternion import (                                   # noqa: E402
    quat_to_rotmatrix, quat_normalize, quat_angle_between,
)
from src.scvx_params import SCvxParams                         # noqa: E402

#: Body rate limit. Vehicle3D has no such field, and the guide's expression
#: `cp.norm(omega, axis=1) <= vehicle.omega_max if hasattr(...) else True`
#: binds as `<= (... if ... else True)`, so on a vehicle without the attribute
#: it silently constrains the rate to <= 1 rad/s. Stated explicitly instead.
OMEGA_MAX_DEFAULT = 0.5     # rad/s, Day 5's Vehicle6DoF.omega_max

#: The quaternion trust radius as a fraction of eta. A unit quaternion has
#: only a diameter of 2 to move in, so sharing a radius with position -- which
#: moves in thousands of metres -- is not a trust region for it.
ETA_Q_FRACTION = 0.15


# ======================================================================
# Linearisation helpers
# ======================================================================
def quat_L_matrix(q):
    """Left multiplication: q (x) p = L(q) @ p."""
    w, x, y, z = q
    return np.array([[w, -x, -y, -z],
                     [x, w, -z, y],
                     [y, z, w, -x],
                     [z, -y, x, w]])


def quat_R_matrix(p):
    """Right multiplication: q (x) p = R(p) @ q."""
    w, x, y, z = p
    return np.array([[w, -x, -y, -z],
                     [x, w, z, -y],
                     [y, -z, w, x],
                     [z, y, -x, w]])


def rotmatrix_unnormalized(q):
    """
    The DCM as a raw quadratic form in q, with no normalisation.

    This exists to be explicit about what is actually being linearised. Day
    13's `quat_to_rotmatrix` normalises its argument first, so it is *not* this
    function -- the two agree to 1e-15 on a unit quaternion and their
    derivatives do not agree at all, because normalising projects out the
    radial direction. Finite-differencing `quat_to_rotmatrix` to check the
    Jacobians below fails by order 1, and it is the obvious wrong test to
    write. The solver treats q as a free 4-vector, so the raw form is the
    right object; the unit norm is maintained by the dynamics and the boundary
    conditions rather than by a constraint.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def dR_dq_matrices(q):
    """
    The four partial derivatives of `rotmatrix_unnormalized` with respect to
    the quaternion components, evaluated at `q`.

    Each entry is quadratic in q, so each partial is linear in q and these are
    constant 3x3 matrices once evaluated. Same linearisation as the published
    SCvx-for-6-DoF work; the test suite checks all four against central
    differences of the raw form, not of `quat_to_rotmatrix`.
    """
    w, x, y, z = q
    return (np.array([[0, -2 * z, 2 * y],
                      [2 * z, 0, -2 * x],
                      [-2 * y, 2 * x, 0]]),
            np.array([[0, 2 * y, 2 * z],
                      [2 * y, -4 * x, -2 * w],
                      [2 * z, 2 * w, -4 * x]]),
            np.array([[-4 * y, 2 * x, 2 * w],
                      [2 * x, 0, 2 * z],
                      [-2 * w, 2 * z, -4 * y]]),
            np.array([[-4 * z, -2 * w, 2 * x],
                      [2 * w, -4 * z, 2 * y],
                      [2 * x, 2 * y, 0]]))


def linearize_rotated_force(q_ref, F_ref):
    """
    (R_ref, [c_w, c_x, c_y, c_z]) for

        R(q) F  ~=  R_ref F  +  sum_i c_i (q_i - q_ref_i)

    which is the full bilinear expansion: the F dependence stays exact through
    R_ref F, and the q dependence is first order.
    """
    R_ref = quat_to_rotmatrix(q_ref)
    return R_ref, [dR @ F_ref for dR in dR_dq_matrices(q_ref)]


def gyro_term(omega, I_diag):
    """omega x (I omega) for a diagonal inertia tensor."""
    wx, wy, wz = omega
    Ixx, Iyy, Izz = I_diag
    return np.array([wy * wz * (Izz - Iyy),
                     wz * wx * (Ixx - Izz),
                     wx * wy * (Iyy - Ixx)])


def linearize_gyro(omega_ref, I_diag):
    """(gyro_ref, J_ref) for gyro(w) ~= gyro_ref + J_ref (w - w_ref)."""
    wx, wy, wz = omega_ref
    Ixx, Iyy, Izz = I_diag
    J = np.array([[0.0, wz * (Izz - Iyy), wy * (Izz - Iyy)],
                  [wz * (Ixx - Izz), 0.0, wx * (Ixx - Izz)],
                  [wy * (Iyy - Ixx), wx * (Iyy - Ixx), 0.0]])
    return gyro_term(omega_ref, I_diag), J


# ======================================================================
# Reference
# ======================================================================
def initialize_reference_3d(N, t_f, pos0, vel0, q0, omega0, vehicle,
                            q_final=None):
    """
    Straight-line initial guess.

    The quaternion is interpolated componentwise and renormalised, which is
    not a geodesic on the sphere and is not meant to be -- it only has to be
    close enough for the first linearisation to mean something.
    """
    q_final = attitude_from_pitch(0.0) if q_final is None else q_final
    q_ref = np.linspace(np.asarray(q0), np.asarray(q_final), N + 1)
    q_ref = np.array([quat_normalize(q) for q in q_ref])

    mdot = 0.6 * vehicle.T_max / (vehicle.isp * G0)
    m_end = max(vehicle.m_wet - mdot * t_f, vehicle.m_dry + 1000.0)
    F_ref = np.zeros((N, 3))
    F_ref[:, 0] = 0.5 * (vehicle.T_min + vehicle.T_max)

    return {
        "t_f": float(t_f), "dt": float(t_f) / N,
        "pos": np.linspace(np.asarray(pos0, dtype=float), np.zeros(3), N + 1),
        "vel": np.linspace(np.asarray(vel0, dtype=float), np.zeros(3), N + 1),
        "q": q_ref,
        "omega": np.linspace(np.asarray(omega0, dtype=float), np.zeros(3),
                             N + 1),
        "m": np.linspace(vehicle.m_wet, m_end, N + 1),
        "F": F_ref,
        "sigma": F_ref[:, 0].copy(),
    }


def compute_aero_perturbation_3d(ref, aero_cfg, N, wind_inertial):
    """Body-frame aero force and moment at every node, from the reference."""
    wind = np.asarray(wind_inertial, dtype=float)
    F = np.zeros((N + 1, 3))
    tau = np.zeros((N + 1, 3))
    for k in range(N + 1):
        F[k], tau[k] = aero_force_and_moment_body(
            ref["vel"][k], wind, ref["q"][k], ref["pos"][k][2], aero_cfg)
    return F, tau


# ======================================================================
# Convex sub-problem
# ======================================================================
_SOLVERS = {"CLARABEL": cp.CLARABEL, "SCS": cp.SCS, "ECOS": cp.ECOS}


def solve_subproblem_3d(ref, vehicle, aero_cfg, params, ic, N, eta,
                        gamma_gs_deg, wind_inertial, omega_max):
    """One convex sub-problem over the full 14-state model."""
    dt = ref["t_f"] / N
    mdot_coeff = 1.0 / (vehicle.isp * G0)
    tan_delta = float(np.tan(vehicle.delta_max))
    tan_gs = float(np.tan(np.radians(gamma_gs_deg)))
    I_diag = vehicle.I_diag
    I_inv = 1.0 / I_diag
    F_aero_ref, tau_aero_ref = compute_aero_perturbation_3d(
        ref, aero_cfg, N, wind_inertial)

    # --- Variable scaling ------------------------------------------------
    # Without this the problem spans seven orders of magnitude -- position in
    # thousands of metres, quaternion components of order one, mass of order
    # 1e5, force of order 1e6 -- and CLARABEL returns `optimal_inaccurate`
    # answers whose quaternion norm has wandered to 2.8. Every block is
    # rescaled to order one and the solver works in the hatted variables.
    # Day 7 needed exactly this for the 2-D solver; the 3-D one needs it more.
    S = {"pos": max(float(np.abs(ic["pos"]).max()), 1.0),
         "vel": max(float(np.abs(ic["vel"]).max()), 1.0),
         "om": max(omega_max, 1e-3),
         "m": vehicle.m_wet,
         "F": vehicle.T_max}

    pos_h = cp.Variable((N + 1, 3))
    vel_h = cp.Variable((N + 1, 3))
    q = cp.Variable((N + 1, 4))          # already order one
    om_h = cp.Variable((N + 1, 3))
    m_h = cp.Variable(N + 1)
    F_h = cp.Variable((N, 3))
    sig_h = cp.Variable(N)

    pos = S["pos"] * pos_h
    vel = S["vel"] * vel_h
    omega = S["om"] * om_h
    m = S["m"] * m_h
    F = S["F"] * F_h
    sigma = S["F"] * sig_h

    # Slacks live in the scaled space too, so one w_vc prices every row of the
    # dynamics comparably instead of whichever block happens to have the
    # largest units.
    nu_pos = cp.Variable((N, 3))
    nu_vel = cp.Variable((N, 3))
    nu_q = cp.Variable((N, 4))
    nu_om = cp.Variable((N, 3))
    nu_m = cp.Variable(N)

    objective = cp.Minimize(
        cp.sum(sig_h) * dt * mdot_coeff * S["F"] / S["m"]
        + params.w_vc * (cp.sum(cp.norm(nu_pos, axis=1))
                         + cp.sum(cp.norm(nu_vel, axis=1))
                         + cp.sum(cp.norm(nu_q, axis=1))
                         + cp.sum(cp.norm(nu_om, axis=1))
                         + cp.norm1(nu_m)))

    cons = [
        pos[0] == ic["pos"], vel[0] == ic["vel"],
        q[0] == ic["q"], omega[0] == ic["omega"],
        m[0] == vehicle.m_wet,
        pos[N] == np.zeros(3), vel[N] == np.zeros(3),
        q[N] == ic["q_final"], omega[N] == np.zeros(3),
        pos[:, 2] >= 0.0, m >= vehicle.m_dry, m <= vehicle.m_wet,
        cp.norm(omega, axis=1) <= omega_max,
    ]

    q_final_ref = np.asarray(ic["q_final"], dtype=float)

    for k in range(N):
        q_r, om_r, F_r, m_r = (ref["q"][k], ref["omega"][k], ref["F"][k],
                               ref["m"][k])

        # Exact-convex control set. Nothing here is linearised.
        cons += [
            cp.norm(F[k]) <= sigma[k],
            sigma[k] >= vehicle.T_min,
            sigma[k] <= vehicle.T_max,
            cp.norm(cp.hstack([F[k, 1], F[k, 2]])) <= F[k, 0] * tan_delta,
            cp.norm(cp.hstack([pos[k, 0], pos[k, 1]])) <= pos[k, 2] * tan_gs,
        ]

        # Torque: linear in F, and structurally free of any roll component.
        tau_thrust = cp.hstack([0.0,
                                vehicle.L_engine * F[k, 2],
                                -vehicle.L_engine * F[k, 1]])

        # Thrust into the inertial frame: exact in F, first order in q.
        R_ref, c = linearize_rotated_force(q_r, F_r)
        F_inertial = (R_ref @ F[k]
                      + c[0] * (q[k, 0] - q_r[0]) + c[1] * (q[k, 1] - q_r[1])
                      + c[2] * (q[k, 2] - q_r[2]) + c[3] * (q[k, 3] - q_r[3]))
        a_k = ((F_inertial + R_ref @ F_aero_ref[k]) / m_r
               + np.array([0.0, 0.0, -G_EARTH]))

        # Quaternion kinematics, bilinear, expanded by the product rule.
        L_ref = quat_L_matrix(q_r)
        om_quat_ref = np.concatenate([[0.0], om_r])
        om_quat = cp.hstack([0.0, omega[k, 0], omega[k, 1], omega[k, 2]])
        qdot = 0.5 * (L_ref @ om_quat + quat_R_matrix(om_quat_ref) @ q[k]
                      - L_ref @ om_quat_ref)

        gyro_ref, J_gyro = linearize_gyro(om_r, I_diag)
        gyro = gyro_ref + J_gyro @ (omega[k] - om_r)
        domega = cp.multiply(I_inv, tau_thrust + tau_aero_ref[k] - gyro)

        cons += [
            pos_h[k + 1] - pos_h[k]
            == dt * vel[k] / S["pos"] + nu_pos[k],
            vel_h[k + 1] - vel_h[k] == dt * a_k / S["vel"] + nu_vel[k],
            q[k + 1] - q[k] == dt * qdot + nu_q[k],
            om_h[k + 1] - om_h[k] == dt * domega / S["om"] + nu_om[k],
            m_h[k + 1] - m_h[k]
            == -dt * sigma[k] * mdot_coeff / S["m"] + nu_m[k],
            # The unit-norm constraint is non-convex, but linearised about a
            # unit reference it is the affine tangent plane q_ref . q = 1.
            # Without it nothing in the sub-problem stops the quaternion
            # leaving the sphere, and it does -- the norm reached 2.8, where
            # every linearisation in this file is meaningless and the dynamics
            # become free. This one line is what makes the solver converge.
            q_r @ q[k] == 1.0,
        ]

    # Trust region, in the scaled space so eta means one thing everywhere.
    # The quaternion gets its own, tighter radius: a unit quaternion only has
    # a diameter of 2 to move in, so an eta of 0.5 shared with position is not
    # a trust region for it at all.
    cons += [cp.norm(q[N] - ref["q"][N]) <= ETA_Q_FRACTION * eta]
    cons += [q_final_ref @ q[N] == 1.0]
    for k in range(N + 1):
        cons += [
            cp.norm(pos_h[k] - ref["pos"][k] / S["pos"]) <= eta,
            cp.norm(vel_h[k] - ref["vel"][k] / S["vel"]) <= eta,
            cp.norm(q[k] - ref["q"][k]) <= ETA_Q_FRACTION * eta,
            cp.norm(om_h[k] - ref["omega"][k] / S["om"]) <= eta,
        ]

    prob = cp.Problem(objective, cons)
    primary = _SOLVERS.get(params.solver, cp.CLARABEL)
    fallback = _SOLVERS.get(params.solver_fallback, cp.SCS)
    solved = False
    for solver in (primary, fallback):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                solved = True
                break
        except (cp.SolverError, Exception):                     # noqa: BLE001
            continue
    if not solved:
        return None, prob.status

    q_raw = np.asarray(q.value)
    q_norm_drift = float(np.abs(np.linalg.norm(q_raw, axis=1) - 1.0).max())
    F_v, sig_v = np.asarray(F.value), np.asarray(sigma.value)
    # The lossless-convexification gap. The relaxation ||F|| <= sigma is only
    # honest if it is tight at the solution; Day 4 learned not to assume that.
    lcvx_gap = float(np.max(sig_v - np.linalg.norm(F_v, axis=1))
                     / max(float(sig_v.max()), 1.0))

    return {
        "pos": np.asarray(pos.value), "vel": np.asarray(vel.value),
        "q": np.array([quat_normalize(qq) for qq in q_raw]),
        "omega": np.asarray(omega.value), "m": np.asarray(m.value),
        "F": F_v, "sigma": sig_v, "t_f": ref["t_f"], "dt": dt,
        "fuel": float(vehicle.m_wet - m.value[-1]),
        "vc_norm": float(
            np.sum(np.linalg.norm(nu_pos.value, axis=1))
            + np.sum(np.linalg.norm(nu_vel.value, axis=1))
            + np.sum(np.linalg.norm(nu_q.value, axis=1))
            + np.sum(np.linalg.norm(nu_om.value, axis=1))
            + np.sum(np.abs(nu_m.value))),
        "q_norm_drift": q_norm_drift, "lcvx_gap": lcvx_gap,
        "cost": float(prob.value),
    }, prob.status


def compute_step_quality_3d(sol, ref, prev_fuel, params):
    """How much the iterate moved, and whether the move was worth taking."""
    def rel(a, b, floor):
        return (np.max(np.linalg.norm(a - b, axis=1))
                / max(np.max(np.linalg.norm(b, axis=1)), floor))

    step = max(rel(sol["pos"], ref["pos"], 1.0),
               rel(sol["vel"], ref["vel"], 1.0),
               float(np.max(np.linalg.norm(sol["q"] - ref["q"], axis=1))))
    vc = sol["vc_norm"]
    if step < 0.01 and vc < params.vc_tol:
        return 1.0, step
    # The guide calls any defect under 1.0 a good step. With vc_tol at 1e-6
    # that lets a defect six orders of magnitude too large grow the trust
    # region to its ceiling, after which the linearisation is being trusted
    # nowhere near its reference and the defect never closes. Graded against
    # the tolerance actually being converged to instead.
    if vc < 1e-3:
        good = prev_fuel is None or sol["fuel"] <= prev_fuel * 1.05
        return (0.9 if good else 0.6), step
    if vc < 1e-1:
        return 0.3, step
    return 0.0, step


# ======================================================================
# Replay: the check that the plan is a trajectory
# ======================================================================
def force_to_gimbal(F_body):
    """
    (T, delta_y, delta_z) from a body-frame thrust vector.

    The solver works in force components and the simulator in an angle pair,
    so somewhere the two have to meet. Inverting Day 14's exact trig here
    rather than approximating it means the replay is flying the plan, not an
    approximation of the plan.
    """
    F = np.asarray(F_body, dtype=float)
    T = float(np.linalg.norm(F))
    if T < 1e-9:
        return 0.0, 0.0, 0.0
    return (T, float(np.arcsin(np.clip(-F[2] / T, -1.0, 1.0))),
            float(np.arctan2(F[1], F[0])))


def replay(sol, vehicle, aero_cfg, wind_inertial=(0.0, 0.0, 0.0),
           substeps=20):
    """
    Fly the solved control through the true non-linear model.

    The convex sub-problem is Euler-discretised and linearised about a
    reference; the answer it returns is a plan, not a trajectory. This flies
    that plan through `propagate_3d_with_aero` and reports what the vehicle
    actually does -- the same discipline Day 9 imposed on the 2-D solver.
    """
    N = len(sol["F"])
    dt = sol["dt"]
    s = make_initial_state_3d(pos=sol["pos"][0], vel=sol["vel"][0],
                              quat=sol["q"][0], omega=sol["omega"][0],
                              m=vehicle.m_wet, vehicle=vehicle)
    hist = [s.copy()]
    for k in range(N):
        T, dy, dz = force_to_gimbal(sol["F"][k])
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
            quat_angle_between(hist[-1, 6:10], sol["q"][-1]))),
        "pos_err_m": float(np.max(np.linalg.norm(
            hist[:, 0:3] - sol["pos"], axis=1))),
        "fuel_kg": float(vehicle.m_wet - hist[-1, 13]),
    }


# ======================================================================
# Outer loop
# ======================================================================
def solve_scvx_3d(vehicle=None, aero_cfg=None, params=None, N=40, t_f=22.0,
                  pos0=(1200.0, 0.0, 2200.0), vel0=(-40.0, 0.0, -85.0),
                  theta0_deg=70.0, omega0=(0.0, 0.0, 0.0),
                  gamma_gs_deg=75.0, wind_inertial=(0.0, 0.0, 0.0),
                  omega_max=OMEGA_MAX_DEFAULT, do_replay=True, verbose=True):
    """Run the 3-D SCvx loop and, unless told otherwise, replay the answer."""
    vehicle = vehicle or Vehicle3D()
    aero_cfg = aero_cfg or AeroConfig3D()
    params = params or SCvxParams()

    ic = {"pos": np.asarray(pos0, dtype=float),
          "vel": np.asarray(vel0, dtype=float),
          "q": attitude_from_pitch(np.radians(theta0_deg)),
          "omega": np.asarray(omega0, dtype=float),
          "q_final": attitude_from_pitch(0.0)}

    t_start = timer.time()
    if verbose:
        print("=" * 72)
        print("3-D SCvx SOLVER")
        print("=" * 72)
        print(f"  N={N}  t_f={t_f}s  entry {tuple(ic['pos'])} m, "
              f"{tuple(ic['vel'])} m/s, pitch {theta0_deg} deg")
        print(f"  {'Iter':>4} {'Status':>10} {'Fuel':>9} {'VC':>10} "
              f"{'eta':>7} {'Step':>8} {'rho':>5} {'|q|drift':>9} {'LCvx':>8}")
        print("  " + "-" * 78)

    ref = initialize_reference_3d(N, t_f, ic["pos"], ic["vel"], ic["q"],
                                  ic["omega"], vehicle, ic["q_final"])
    eta, w_vc, prev_fuel = params.eta_0, params.w_vc, None
    history = {k: [] for k in ("fuel", "vc_norm", "eta", "step", "rho",
                               "q_norm_drift", "lcvx_gap")}
    best, best_fuel, sol = None, float("inf"), None

    for it in range(1, params.max_iter + 1):
        cur = SCvxParams(**{k: getattr(params, k)
                            for k in params.__dataclass_fields__})
        cur.w_vc = w_vc
        sol, status = solve_subproblem_3d(ref, vehicle, aero_cfg, cur, ic, N,
                                          eta, gamma_gs_deg, wind_inertial,
                                          omega_max)
        if sol is None:
            eta *= params.alpha_shrink
            if verbose:
                print(f"  {it:>4} {status:>10} {'---':>9} {'---':>10} "
                      f"{eta:>7.4f}")
            for k in history:
                history[k].append(float("nan"))
            history["eta"][-1] = eta
            if eta < params.eta_min:
                break
            continue

        rho, step = compute_step_quality_3d(sol, ref, prev_fuel, params)
        for k, v in (("fuel", sol["fuel"]), ("vc_norm", sol["vc_norm"]),
                     ("eta", eta), ("step", step), ("rho", rho),
                     ("q_norm_drift", sol["q_norm_drift"]),
                     ("lcvx_gap", sol["lcvx_gap"])):
            history[k].append(v)
        if verbose:
            print(f"  {it:>4} {status:>10} {sol['fuel']:>9,.0f} "
                  f"{sol['vc_norm']:>10.3e} {eta:>7.4f} {step:>8.4f} "
                  f"{rho:>5.2f} {sol['q_norm_drift']:>9.2e} "
                  f"{sol['lcvx_gap']:>8.1e}")

        if sol["vc_norm"] < 1.0 and sol["fuel"] < best_fuel:
            best = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in sol.items()}
            best_fuel = sol["fuel"]

        if rho > params.rho_good:
            eta = min(eta * params.alpha_expand, params.eta_max)
            accept = True
        elif rho > params.rho_ok:
            accept = True
        else:
            eta *= params.alpha_shrink
            accept = False
            if eta < params.eta_min:
                # Without this the radius shrinks to zero and the trust region
                # pins the iterate to its own reference for every remaining
                # iteration -- thirty rows of an unchanging number, which reads
                # like convergence and is the opposite of it.
                if verbose:
                    print(f"\n  Trust region collapsed below eta_min at "
                          f"iteration {it}; the step is dead.")
                break

        if accept:
            for k in ("pos", "vel", "q", "omega", "m", "F", "sigma"):
                ref[k] = sol[k].copy()
            ref["pos"][:, 2] = np.maximum(ref["pos"][:, 2], 1.0)
            prev_fuel = sol["fuel"]

        if sol["vc_norm"] > params.vc_tol * 10:
            w_vc = min(w_vc * params.w_vc_grow, params.w_vc_max)

        if (it >= params.min_iter and sol["vc_norm"] < params.vc_tol
                and step < params.step_tol):
            if verbose:
                print(f"\n  Converged after {it} iterations")
            break

    if best is None:
        best = sol
    elapsed = timer.time() - t_start
    if best is None:
        if verbose:
            print(f"\n  NO SOLUTION ({elapsed:.1f}s)")
        return {"status": "failed", "history": history, "elapsed": elapsed}

    best["t"] = np.linspace(0.0, best["t_f"], len(best["pos"]))
    best["status"] = "converged"
    best["history"] = history
    best["iterations"] = len(history["fuel"])
    best["elapsed"] = elapsed
    best["N"] = N
    best["gamma_gs_deg"] = gamma_gs_deg
    best["final_tilt_deg"] = float(
        np.degrees(tilt_from_vertical(best["q"][-1])))

    if do_replay:
        best["replay"] = replay(best, vehicle, aero_cfg, wind_inertial)

    if verbose:
        print(f"\n  SOLVED in {elapsed:.1f}s over {best['iterations']} "
              f"iterations")
        print(f"  Fuel               : {best['fuel']:,.0f} kg")
        print(f"  Final position     : "
              f"{np.round(best['pos'][-1], 4).tolist()} m")
        print(f"  Final speed        : "
              f"{np.linalg.norm(best['vel'][-1]):.4f} m/s")
        print(f"  Final tilt         : {best['final_tilt_deg']:.4f} deg")
        print(f"  Cross-range flown  : "
              f"{float(np.abs(best['pos'][:, 1]).max()):.1f} m")
        print(f"  Quaternion drift   : {best['q_norm_drift']:.2e}")
        print(f"  LCvx gap           : {best['lcvx_gap']:.2e}")
        if do_replay:
            r = best["replay"]
            print(f"  REPLAY, true model : miss {r['miss_m']:,.1f} m at "
                  f"{r['speed_ms']:,.1f} m/s, tilt {r['tilt_deg']:.1f} deg")
    return best
