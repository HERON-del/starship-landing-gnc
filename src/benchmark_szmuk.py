"""
Day 18 -- replicating a published benchmark.

    Szmuk, M. and Acikmese, B., "Successive Convexification for 6-DoF Mars
    Rocket Powered Landing with Free-Final-Time", AIAA SciTech 2018-0617,
    arXiv:1802.03827.

Seventeen days checked this project against tests written alongside the code
it tests. That is a closed loop. This is the first check against numbers chosen
by someone who has never seen the codebase.

Conventions are the paper's, not this project's, and deliberately so
--------------------------------------------------------------------
* **e1 is up.** `g_I = -e1`. Everywhere else in this repository +z is up; here
  the vertical axis is x, because that is what the paper uses and a
  replication that silently re-axes the problem is not a replication.
* **State is [m, r(3), v(3), q(4), w(3)]** -- mass first. Days 13 to 17 put
  mass last. Same fourteen numbers, different order, and mixing them up is the
  obvious way to produce a confident wrong answer.
* **Non-dimensional units** UL, UT, UM. The paper states its numbers are
  "notional... not intended to match a real-world system" and gives no SI
  scale, so converting would mean inventing one.
* **Minimum time, not minimum fuel**, with the final time free through the
  paper's time-dilation variable sigma. Day 16 scoped free final time out of
  the 3-D solver; this is where it comes back.

The physics is imported from Days 13 to 15 wherever the paper's model and this
project's agree, and reimplemented only where they genuinely differ -- the
frame convention and state ordering above, and the paper's own rigid-body
model, which has no aerodynamics at all. Cross-checks against the project's
`quat_to_rotmatrix` and `gyroscopic_term` are in the test suite, which is the
point: two independently written expressions of the same rotation ought to
agree, and if they do not, one of them is wrong.

What is transcribed and what is not
-----------------------------------
Table 1, Table 2 and the boundary conditions are transcribed from the paper.
Two numbers are **not** in the paper and are flagged in the code rather than
filled in:

* `alpha_m`, the mass-depletion constant. The paper never gives it -- its
  results do not depend on a fuel number, since final mass is free. The Day 18
  guide sets it to 1.0 and calls it a placeholder. It is not a harmless one:
  at 1.0 the vehicle exhausts its entire 1 UM of propellant almost
  immediately, the mass floor binds, and the mass row of the dynamics becomes
  unsatisfiable. See `ALPHA_M_NOTE` and the sweep in the test suite.
* The 3-D case's initial velocity. The paper plots the case but never prints
  `v_I,i` for it. The north component here is this project's choice.

The result, and a correction to an earlier one
----------------------------------------------
With `alpha_m` sized so the propellant lasts **and** the paper's own Eq. 22
first-order-hold integrals implemented, **the paper's central claim
reproduces**: ten time-of-flight guesses from 1 to 10 UT all land within
**0.00183 UT** of each other -- inside the paper's own stated bar of 0.01 --
with virtual control between 1e-15 and 1e-17. That is with Algorithm 1 exactly
as printed: **no hard trust region and no quaternion renormalisation**, both of
which the Day 18 guide calls necessary additions.

This corrects a claim published earlier in this project. With a single-endpoint
approximation to the discretisation in place of the paper's integrals, the
sweep spread 21.7 UT and the flight time tracked its own initial guess, and
that was written up as the paper's claim failing to reproduce, with the cost
weights (`w_nu = 1e5` against a sigma coefficient of 1) offered as the reason.
The cost weights are fine. With an accurate discretisation the optimiser drives
the virtual control to machine precision without needing to buy flight time,
and sigma settles at 3.282 from any starting guess. The failure was in this
implementation, not in the paper.

`discretize` (single-endpoint) is kept alongside `discretize_exact_foh` so the
difference is measurable rather than described -- that comparison is what
settles which layer a residual lives in, and it is the whole content of the
test suite's Tests 4 to 6.
"""

import os
import sys
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

N_STATE = 14
IDX_M = 0
IDX_R = slice(1, 4)
IDX_V = slice(4, 7)
IDX_Q = slice(7, 11)
IDX_W = slice(11, 14)

ALPHA_M_NOTE = (
    "alpha_m is not in the paper. Its results never depend on a fuel number "
    "because the final mass is free, so the authors had no reason to print "
    "it. Setting it to 1.0 is not neutral: hover thrust here is m*g ~ 2, so "
    "mdot ~ -2 UM/UT against 1 UM of usable propellant, and the mass floor "
    "binds within half a time unit. After that the mass row of the dynamics "
    "cannot be satisfied by any control and the virtual control absorbs it at "
    "every node -- which is a residual the discretisation is then blamed for."
)


# ======================================================================
# Table 1 -- simulation parameters
# ======================================================================
@dataclass
class PaperVehicle:
    """Table 1, transcribed. `g_I = -e1` because e1 is up in the paper."""

    g_I: np.ndarray = field(
        default_factory=lambda: np.array([-1.0, 0.0, 0.0]))
    m_wet: float = 2.00
    m_dry: float = 1.00
    T_min: float = 0.30
    T_max: float = 5.00
    delta_max_deg: float = 20.0
    theta_max_deg: float = 90.0
    gamma_gs_deg: float = 20.0
    omega_max_deg: float = 60.0
    J_B: np.ndarray = field(default_factory=lambda: 1e-2 * np.eye(3))
    r_TB: np.ndarray = field(
        default_factory=lambda: np.array([-1e-2, 0.0, 0.0]))

    #: NOT from the paper. See ALPHA_M_NOTE. The default is chosen so the
    #: vehicle spends a sensible fraction of its propellant over a trajectory
    #: of a few UT rather than exhausting it in the first half-unit; the test
    #: suite sweeps it and reports what it is worth.
    alpha_m: float = 0.06

    @property
    def hover_thrust(self) -> float:
        """Thrust that exactly cancels gravity at wet mass."""
        return self.m_wet * float(np.linalg.norm(self.g_I))

    def summary(self) -> str:
        g = float(np.linalg.norm(self.g_I))
        return "\n".join([
            "Szmuk & Acikmese (2018), Table 1 -- non-dimensional",
            f"  m_wet / m_dry     : {self.m_wet:.2f} / {self.m_dry:.2f} UM",
            f"  T_min / T_max     : {self.T_min:.2f} / {self.T_max:.2f}",
            f"  hover thrust      : {self.hover_thrust:.2f} "
            f"(T_min is {self.T_min / self.hover_thrust:.2f}x it)",
            f"  min / max accel   : {self.T_min / self.m_wet:.3f} / "
            f"{self.T_max / self.m_wet:.3f} against g = {g:.2f}",
            f"  delta / theta / gs: {self.delta_max_deg:.0f} / "
            f"{self.theta_max_deg:.0f} / {self.gamma_gs_deg:.0f} deg",
            f"  omega_max         : {self.omega_max_deg:.0f} deg/UT",
            f"  alpha_m           : {self.alpha_m:.3f}  (NOT from the paper)",
        ])


@dataclass
class PaperAlgorithm:
    """Table 2, transcribed."""
    w_nu: float = 1e5
    w_delta: float = 1e-3
    w_delta_sigma: float = 1e-1
    nu_tol: float = 1e-10
    delta_tol: float = 1e-3
    N_iter_max: int = 15
    K: int = 50


@dataclass
class PaperBCs:
    r_I_i: np.ndarray = field(
        default_factory=lambda: np.array([4.0, 4.0, 0.0]))
    v_I_f: np.ndarray = field(
        default_factory=lambda: np.array([-0.1, 0.0, 0.0]))
    omega_B_i: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_B_I_f: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    v_I_i: np.ndarray = None
    #: True when v_I_i came from the paper, False when this project chose it.
    v_I_i_is_paper: bool = True


def two_d_case():
    """Section IV.A. Four UL/UT to the west, staying in the Up-East plane."""
    bc = PaperBCs()
    bc.v_I_i = np.array([0.0, -4.0, 0.0])
    return bc


def three_d_case(north=2.0):
    """
    Section IV.B. The paper plots this case and never prints its v_I,i, so
    the north component is this project's choice, not a transcription.
    """
    bc = PaperBCs()
    bc.v_I_i = np.array([0.0, -4.0, float(north)])
    bc.v_I_i_is_paper = False
    return bc


# ======================================================================
# Paper's model
# ======================================================================
def dcm_body_from_inertial(q):
    """C_{B/I}. The paper's convention; its transpose is body to inertial."""
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 + q0 * q3),
         2 * (q1 * q3 - q0 * q2)],
        [2 * (q1 * q2 - q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2),
         2 * (q2 * q3 + q0 * q1)],
        [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1),
         1 - 2 * (q1 ** 2 + q2 ** 2)],
    ])


def omega_matrix(omega):
    """The paper's Omega(w) in its quaternion kinematics."""
    wx, wy, wz = omega
    return np.array([[0, -wx, -wy, -wz],
                     [wx, 0, wz, -wy],
                     [wy, -wz, 0, wx],
                     [wz, wy, -wx, 0]])


def nonlinear_dynamics(x, u, sigma, veh: PaperVehicle):
    """
    The paper's f(x, u), scaled by the time-dilation sigma.

    Free final time enters as a change of independent variable: with tau
    running over [0, 1] and t = sigma * tau, every derivative picks up a
    factor of sigma, and sigma becomes a decision variable the optimiser
    minimises. That is the whole of the free-final-time extension Day 16
    scoped out.
    """
    x = np.asarray(x, dtype=float)
    m, r, v, q, w = (float(x[IDX_M]), x[IDX_R], x[IDX_V], x[IDX_Q], x[IDX_W])
    F_I = dcm_body_from_inertial(q).T @ u
    return sigma * np.concatenate([
        [-veh.alpha_m * float(np.linalg.norm(u))],
        v,
        F_I / m + veh.g_I,
        0.5 * omega_matrix(w) @ q,
        np.linalg.solve(veh.J_B,
                        np.cross(veh.r_TB, u) - np.cross(w, veh.J_B @ w)),
    ])


def linearize_at(x_ref, u_ref, sigma_ref, veh, eps=1e-6):
    """Central-difference A, B, Sigma and the affine remainder z."""
    A = np.zeros((N_STATE, N_STATE))
    for i in range(N_STATE):
        d = np.zeros(N_STATE)
        d[i] = eps
        A[:, i] = (nonlinear_dynamics(x_ref + d, u_ref, sigma_ref, veh)
                   - nonlinear_dynamics(x_ref - d, u_ref, sigma_ref, veh)
                   ) / (2 * eps)
    B = np.zeros((N_STATE, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        B[:, i] = (nonlinear_dynamics(x_ref, u_ref + d, sigma_ref, veh)
                   - nonlinear_dynamics(x_ref, u_ref - d, sigma_ref, veh)
                   ) / (2 * eps)
    Sig = nonlinear_dynamics(x_ref, u_ref, 1.0, veh)
    f = nonlinear_dynamics(x_ref, u_ref, sigma_ref, veh)
    return A, B, Sig, f - A @ x_ref - B @ u_ref - Sig * sigma_ref


def discretize(x_ref, u_ref, sigma_ref, veh, dtau, nsub=5):
    """
    First-order hold, with the state-transition matrix RK4-integrated.

    Simpler than the paper's exact double integrals, which weight u_k and
    u_{k+1} separately across the interval. The guide blames its residual on
    exactly this; the test suite checks that claim by decomposing the residual
    by state block rather than accepting it.
    """
    A, B, Sig, z = linearize_at(x_ref, u_ref, sigma_ref, veh)
    Phi = np.eye(N_STATE)
    h = dtau / nsub
    for _ in range(nsub):
        k1 = A @ Phi
        k2 = A @ (Phi + 0.5 * h * k1)
        k3 = A @ (Phi + 0.5 * h * k2)
        k4 = A @ (Phi + h * k3)
        Phi = Phi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return Phi, Phi @ B * dtau, Phi @ Sig * dtau, Phi @ z * dtau


def discretize_exact_foh(x_ref, u_ref, sigma_ref, veh, dtau, nsub=20):
    """
    The paper's Eq. 22 first-order hold, with the integrals actually taken.

    `discretize` above evaluates the input matrices once, at the interval's
    left endpoint, and multiplies by dtau. The paper instead integrates them
    across the interval against the two hold weights,

        lam_minus(s) = (dtau - s) / dtau,   lam_plus(s) = s / dtau

    which gives a separate matrix for u_k and u_{k+1} rather than one for a
    control held constant. Both are returned here so the two can be compared
    directly, which is the point: Day 18 measured that the guide's residual was
    its omitted alpha_m rather than this, and the honest way to finish that
    argument is to implement the thing it blamed and see what is left.

    Integration is RK4 over the augmented system, with A frozen at the
    reference over the interval -- the same linearisation both versions use, so
    the only difference between them is the quadrature.
    """
    A, B, Sig, z = linearize_at(x_ref, u_ref, sigma_ref, veh)
    n = N_STATE
    Phi = np.eye(n)
    Pm = np.zeros((n, 3))
    Pp = np.zeros((n, 3))
    Ps = np.zeros(n)
    Pz = np.zeros(n)
    h = dtau / nsub

    def deriv(s, Phi_s, _Pm, _Pp, _Ps, _Pz):
        # Phi(s) propagates forward; the integrands carry Phi(s)^-1, which for
        # a frozen A is the backward propagator.
        Phi_inv = np.linalg.solve(Phi_s, np.eye(n))
        lam_m = (dtau - s) / dtau
        lam_p = s / dtau
        return (A @ Phi_s,
                Phi_inv @ B * lam_m,
                Phi_inv @ B * lam_p,
                Phi_inv @ Sig,
                Phi_inv @ z)

    s = 0.0
    for _ in range(nsub):
        k1 = deriv(s, Phi, Pm, Pp, Ps, Pz)
        k2 = deriv(s + h / 2, Phi + h / 2 * k1[0], None, None, None, None)
        k3 = deriv(s + h / 2, Phi + h / 2 * k2[0], None, None, None, None)
        k4 = deriv(s + h, Phi + h * k3[0], None, None, None, None)
        Phi = Phi + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Pm = Pm + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        Pp = Pp + (h / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        Ps = Ps + (h / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        Pz = Pz + (h / 6) * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
        s += h

    return Phi, Phi @ Pm, Phi @ Pp, Phi @ Ps, Phi @ Pz


def initialize_reference(bc, veh, K, sigma_guess):
    """The paper's Algorithm 1 initialisation: straight lines, hover thrust."""
    ref_x = np.zeros((K, N_STATE))
    ref_u = np.zeros((K, 3))
    for k in range(K):
        a2 = k / (K - 1)
        a1 = 1.0 - a2
        m_k = a1 * veh.m_wet + a2 * veh.m_dry
        ref_x[k] = np.concatenate([[m_k], a1 * bc.r_I_i,
                                   a1 * bc.v_I_i + a2 * bc.v_I_f,
                                   [1.0, 0.0, 0.0, 0.0], np.zeros(3)])
        ref_u[k] = -m_k * veh.g_I
    return ref_x, ref_u, float(sigma_guess)


# ======================================================================
# Convex sub-problem -- the paper's Problem 2
# ======================================================================
def solve_subproblem(ref_x, ref_u, ref_sigma, bc, veh, alg, trust_radius,
                     hard_trust=True, exact_foh=True):
    """
    One sub-problem. `hard_trust=False` is the paper's Algorithm 1 literally --
    soft L2 penalty only -- which is worth being able to run, because it is
    the thing the guide claims does not converge.
    """
    K = alg.K
    dtau = 1.0 / (K - 1)
    x = cp.Variable((K, N_STATE))
    u = cp.Variable((K, 3))
    sigma = cp.Variable(nonneg=True)
    nu = cp.Variable((K - 1, N_STATE))

    cons = [
        x[0, IDX_M] == veh.m_wet,
        x[0, IDX_R] == bc.r_I_i,
        x[0, IDX_V] == bc.v_I_i,
        x[0, IDX_W] == bc.omega_B_i,
        x[K - 1, IDX_R] == np.zeros(3),
        x[K - 1, IDX_V] == bc.v_I_f,
        x[K - 1, IDX_Q] == bc.q_B_I_f,
        x[K - 1, IDX_W] == np.zeros(3),
        u[K - 1, 1] == 0, u[K - 1, 2] == 0,
    ]
    # The initial attitude is deliberately unconstrained -- the paper leaves it
    # free, and so does this.

    tan_gs = float(np.tan(np.radians(veh.gamma_gs_deg)))
    cos_theta = float(np.cos(np.radians(veh.theta_max_deg)))
    omega_max = float(np.radians(veh.omega_max_deg))
    cos_delta = float(np.cos(np.radians(veh.delta_max_deg)))

    for k in range(K):
        cons += [
            x[k, IDX_M] >= veh.m_dry,
            cp.norm(x[k, 2:4]) <= x[k, 1] / tan_gs,
            cos_theta <= 1 - 2 * cp.sum_squares(cp.hstack([x[k, 9], x[k, 10]])),
            cp.norm(x[k, IDX_W]) <= omega_max,
            cp.norm(u[k]) <= veh.T_max,
            cos_delta * cp.norm(u[k]) <= u[k, 0],
            (ref_u[k] / max(float(np.linalg.norm(ref_u[k])), 1e-8)) @ u[k]
            >= veh.T_min,
        ]
        if hard_trust:
            scale_x = 1.0 + float(np.linalg.norm(ref_x[k]))
            scale_u = 1.0 + float(np.linalg.norm(ref_u[k]))
            cons += [
                cp.norm(x[k] - ref_x[k]) <= trust_radius * scale_x,
                cp.norm(u[k] - ref_u[k]) <= trust_radius * scale_u,
            ]
    if hard_trust:
        cons += [cp.abs(sigma - ref_sigma)
                 <= trust_radius * max(ref_sigma, 1.0)]

    for k in range(K - 1):
        if exact_foh:
            # The paper's Eq. 22: u_k and u_{k+1} carry separate matrices,
            # because a first-order hold interpolates between them across the
            # interval rather than holding one of them constant.
            Ad, Bm, Bp, Sd, zd = discretize_exact_foh(
                ref_x[k], ref_u[k], ref_sigma, veh, dtau)
            cons += [x[k + 1] == Ad @ x[k] + Bm @ u[k] + Bp @ u[k + 1]
                     + Sd * sigma + zd + nu[k]]
        else:
            Ad, Bd, Sd, zd = discretize(ref_x[k], ref_u[k], ref_sigma, veh,
                                        dtau)
            cons += [x[k + 1] == Ad @ x[k] + Bd @ u[k] + Sd * sigma + zd
                     + nu[k]]

    cost = (sigma
            + alg.w_nu * cp.sum(cp.abs(nu))
            + alg.w_delta * (cp.sum_squares(x - ref_x)
                             + cp.sum_squares(u - ref_u))
            + alg.w_delta_sigma * cp.abs(sigma - ref_sigma))

    prob = cp.Problem(cp.Minimize(cost), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        return prob, None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return prob, None
    return prob, (np.asarray(x.value), np.asarray(u.value),
                  float(sigma.value), np.asarray(nu.value))


def residual_by_block(nu):
    """
    Where the dynamics defect actually lives.

    A scalar |nu| says a solver has not converged. It does not say which row
    of the dynamics it could not satisfy, and that is the diagnostic that
    turns 'it does not converge' into something actionable.
    """
    a = np.abs(np.asarray(nu))
    total = float(a.sum())
    blocks = {"mass": float(a[:, IDX_M].sum()),
              "position": float(a[:, IDX_R].sum()),
              "velocity": float(a[:, IDX_V].sum()),
              "quaternion": float(a[:, IDX_Q].sum()),
              "rate": float(a[:, IDX_W].sum())}
    frac = {k: (v / total if total > 0 else 0.0) for k, v in blocks.items()}
    return blocks, frac, total


# ======================================================================
# Outer loop -- the paper's Algorithm 1
# ======================================================================
def solve_benchmark(bc, veh=None, alg=None, sigma_guess=3.0, trust0=0.6,
                    trust_shrink=0.85, trust_min=0.05, hard_trust=False,
                    renormalize=False, exact_foh=True, verbose=True):
    """
    Algorithm 1, with two documented additions.

    The hard trust region and the reference quaternion renormalisation are not
    in the paper. Both are switchable so the literal recipe can be run and its
    behaviour measured rather than described.
    """
    veh = veh or PaperVehicle()
    alg = alg or PaperAlgorithm()
    ref_x, ref_u, ref_sigma = initialize_reference(bc, veh, alg.K, sigma_guess)
    trust_radius = trust0
    hist = {k: [] for k in ("sigma", "delta", "nu", "radius", "status")}
    converged_at = None
    ever_solved = False

    if verbose:
        print(f"  {'it':>3} {'tf (sigma)':>11} {'delta':>11} {'|nu|_1':>11} "
              f"{'radius':>8}  status")
    for i in range(1, alg.N_iter_max + 1):
        prob, out = solve_subproblem(ref_x, ref_u, ref_sigma, bc, veh, alg,
                                     trust_radius, hard_trust=hard_trust,
                                     exact_foh=exact_foh)
        hist["status"].append(prob.status)
        if out is None:
            if verbose:
                print(f"  {i:>3} {'--':>11} {'--':>11} {'--':>11} "
                      f"{trust_radius:>8.3f}  {prob.status}")
            trust_radius = max(trust_radius * trust_shrink, trust_min)
            continue
        ever_solved = True
        x_val, u_val, sigma_val, nu_val = out

        if renormalize:
            qb = x_val[:, IDX_Q]
            n = np.linalg.norm(qb, axis=1, keepdims=True)
            n[n < 1e-8] = 1.0
            x_val[:, IDX_Q] = qb / n

        delta = float(np.sum((x_val - ref_x) ** 2)
                      + np.sum((u_val - ref_u) ** 2))
        nu_norm = float(np.abs(nu_val).sum())
        for k, v in (("sigma", sigma_val), ("delta", delta), ("nu", nu_norm),
                     ("radius", trust_radius)):
            hist[k].append(v)
        if verbose:
            print(f"  {i:>3} {sigma_val:>11.4f} {delta:>11.3e} "
                  f"{nu_norm:>11.3e} {trust_radius:>8.3f}  {prob.status}")

        ref_x, ref_u, ref_sigma = x_val, u_val, sigma_val
        trust_radius = max(trust_radius * trust_shrink, trust_min)
        if delta <= alg.delta_tol and nu_norm <= alg.nu_tol:
            converged_at = i
            break

    blocks, frac, total = (residual_by_block(nu_val)
                           if ever_solved else ({}, {}, float("nan")))
    return {
        "x": ref_x, "u": ref_u, "sigma": ref_sigma, "history": hist,
        "converged_at": converged_at, "iterations_run": len(hist["sigma"]),
        "ever_solved": ever_solved,
        "nu_total": total, "nu_blocks": blocks, "nu_fraction": frac,
        "vehicle": veh, "bcs": bc,
    }
