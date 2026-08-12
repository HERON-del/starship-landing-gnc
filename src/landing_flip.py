"""
6-DoF flip-and-land trajectory optimization.

The vehicle enters in a belly-flop attitude and must rotate to vertical while
simultaneously decelerating, translating to the pad, and arriving upright with
no residual rotation.

State:   [x, z, vx, vz, theta, omega, m]  at N+1 nodes
Control: [sigma, tau]                     at N nodes

    sigma  thrust magnitude [N]
    tau    torque about the pitch axis [N m]

Why this is harder than Day 4
-----------------------------
Days 1-4 treated the thrust vector as free: the optimiser could point it
anywhere and pay only a pointing-cone penalty. That is false. The engine is
bolted to the vehicle, so the thrust direction *is* the attitude, plus at most
a 15 degree gimbal:

    Tx  = sigma sin(theta + delta)
    Tz  = sigma cos(theta + delta)
    tau = sigma L sin(delta)

Torque and thrust tilt come from the same deflection. Gimbaling to rotate the
vehicle simultaneously tilts the thrust that is decelerating it; there is no
way to buy one without paying for the other. That coupling is the flip.

Writing `tau` as an independent variable bounded by +/- sigma L sin(delta_max)
- which is the obvious move - quietly discards it. The optimiser then gets free
torque with no effect on thrust direction, and will happily rotate the vehicle
while thrusting in an unrelated direction. It solves, and it lies.

What is done here instead
-------------------------
The coupling is kept as an equality and linearised about a reference attitude
and throttle. Writing delta ~ tau / (sigma_ref L) for small deflections and
expanding to first order about (sigma_ref, theta_ref, delta = 0):

    Tx ~ sin(th_ref) sigma + sigma_ref cos(th_ref) (theta - th_ref)
         + cos(th_ref) tau / L
    Tz ~ cos(th_ref) sigma - sigma_ref sin(th_ref) (theta - th_ref)
         - sin(th_ref) tau / L

Affine in (sigma, theta, tau), so the subproblem stays convex, and the torque
term carries its own thrust-tilt cost as it should.

A linearisation is only trustworthy near its reference, so this adds the piece
Day 4 flagged as missing: **trust regions**, adapted between iterations. After
each solve the true non-linear thrust direction is recomputed and compared with
what the linear model predicted; that defect grows the region when the model
held and shrinks it when it did not.

Three details of the loop were found the hard way and are load-bearing:

1. *Every solved subproblem advances the reference.* Rejecting a step and
   re-solving around the same reference with a smaller region is a deadlock -
   the region tightens around a point the solution is far from, and the next
   solve is strictly harder. The defect sizes the next region; it does not veto
   this one.

2. *The torque needs its own trust region.* The gimbal expansion point is the
   previous torque solution, and that solution is bang-bang. Left unbounded it
   flips sign between iterations, moving the linearisation point further than
   the step it was supposed to validate. Constraining theta alone left the
   defect oscillating around 0.12; adding a torque region drove it to 0.0003.

3. *Expand about the previous gimbal angle, not about zero.* Expanding about
   delta = 0 leaves a neglected 0.5 sin(theta) delta^2 term, which at the 15
   degree limit is 0.034 of maximum thrust - a floor no amount of iterating can
   clear.

Note also that lossless convexification is gone, and does not need replacing.
Once the thrust direction is pinned to the attitude there is no free vector to
relax: sigma is simply the throttle, bounded directly. The non-convexity moved
from the magnitude to the direction.

References
----------
[1] Szmuk, M. et al., "Successive Convexification for 6-DoF Mars Rocket
    Powered Landing with Free-Final-Time," AIAA, 2018.
"""

import os
import sys

import cvxpy as cp
import matplotlib
import numpy as np

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
RESULTS = os.path.join(REPO_ROOT, "results")

from src.dynamics_6dof import Vehicle6DoF, G0, G_EARTH     # noqa: E402

SOLVER_CHAIN = ("CLARABEL", "SCS")


def feasible_entry_state(vehicle=None, t_burn=15.0, theta0_deg=60.0,
                         t_flip=3.0, margin=1.25):
    """
    Entry altitude and vertical speed a `t_burn` burn can actually null.

    Same argument as the 3-DoF case, with one addition: while the vehicle is
    still tilted, only cos(theta) of its thrust points up, so the early part of
    the burn decelerates less. The attitude profile is assumed to be the same
    fast flip used to seed the optimiser.

    Returns
    -------
    (z0, vz0) : tuple of float
    """
    vehicle = vehicle or Vehicle6DoF()
    n = 800
    t = np.linspace(0.0, t_burn, n)
    theta = np.radians(theta0_deg) * np.clip(1.0 - t / t_flip, 0.0, 1.0)

    mdot_min = vehicle.T_min / (vehicle.isp * G0)
    m_t = np.maximum(vehicle.m_wet - mdot_min * t, vehicle.m_dry)
    a_min = vehicle.T_min * np.cos(theta) / m_t - G_EARTH

    v_required = float(np.trapezoid(a_min, t))
    if v_required <= 0:
        raise ValueError("Minimum thrust cannot arrest a descent at this attitude.")

    vz0 = -v_required * margin
    z0 = abs(vz0) * t_burn / 2.0
    return float(z0), float(vz0)


def max_entry_pitch_note(vehicle=None):
    """
    Why the entry pitch cannot simply be 90 degrees in this model.

    The engine is lit for the whole trajectory - minimum throttle is 40% and
    there is no coast - so while the vehicle is tilted, sin(theta) of a very
    large thrust is pushing it sideways whether it wants that or not. The pitch
    rate is capped at omega_max, so the flip takes at least theta0/omega_max
    seconds, and the sideways excursion accumulated in that window has to fit
    inside the glideslope corridor and still be nulled by touchdown.

    Measured at N = 80, t_burn = 15 s, with the entry state re-sized for each
    attitude:

        nominal, glideslope 75 deg   60 deg feasible, 65 not
        glideslope loosened to 45    65 deg          (+5)
        omega_max 28.6 -> 51.6 deg/s 75 deg          (+15)

    Relaxing *either* constraint alone moves the ceiling, which is what shows
    the two bind together rather than one being the culprit. The pitch rate is
    much the stronger lever - consistent with the rotation being rate-limited
    rather than torque-limited, since peak torque stays well under maximum.

    A real Starship flips before the landing burn, unpowered, on aerodynamic
    surfaces. That is precisely the freedom this model does not have.
    """
    vehicle = vehicle or Vehicle6DoF()
    return (f"engine lit throughout; flip takes >= theta0/{np.degrees(vehicle.omega_max):.0f} deg/s, "
            f"forced lateral accel up to {vehicle.T_min / vehicle.m_wet:.0f} m/s^2")


def solve_flip_landing(
    vehicle: Vehicle6DoF = None,
    N: int = 80,
    t_burn: float = 15.0,
    x0: float = None,
    z0: float = None,
    vx0: float = 0.0,
    vz0: float = None,
    theta0_deg: float = 60.0,
    omega0: float = 0.0,
    gamma_gs_deg: float = 75.0,
    t_flip: float = None,
    trust0_deg: float = 40.0,
    trust_min_deg: float = 0.02,
    trust_max_deg: float = 60.0,
    max_iters: int = 40,
    tol_deg: float = 0.05,
    defect_tol: float = 0.01,
    fuel_tol_kg: float = 2.0,
    verbose: bool = True,
):
    """
    Solve the flip-and-land trajectory optimization.

    Parameters
    ----------
    vehicle : Vehicle6DoF, optional
    N : int
        Number of time intervals.
    t_burn : float
        Fixed burn time [s].
    x0, z0, vx0, vz0 : float
        Initial translational state [m], [m/s].
    theta0_deg : float
        Initial pitch from vertical [deg]. 90 is a full belly-flop.
    omega0 : float
        Initial pitch rate [rad/s].
    gamma_gs_deg : float
        Glideslope angle from horizontal [deg].
    trust0_deg, trust_min_deg, trust_max_deg : float
        Initial and bounding sizes of the attitude trust region.
    max_iters : int
        Maximum SCvx iterations.
    tol_deg : float
        Convergence tolerance on the attitude step.

    Returns
    -------
    dict with trajectory data and solver status.
    """
    vehicle = vehicle or Vehicle6DoF()
    theta0 = np.radians(theta0_deg)

    # The flip is rate-limited, so it takes at least theta0/omega_max seconds.
    # Seeding the reference with a *linear* sweep across the whole burn implies
    # a far slower rotation than the vehicle would ever fly, and linearising
    # about that gives an infeasible subproblem even where the true problem is
    # fine. Seed it with a fast flip instead.
    if t_flip is None:
        t_flip = float(np.clip(1.4 * theta0 / vehicle.omega_max, 1.5, 0.6 * t_burn))

    if z0 is None or vz0 is None:
        z_auto, vz_auto = feasible_entry_state(vehicle, t_burn, theta0_deg, t_flip)
        z0 = z_auto if z0 is None else z0
        vz0 = vz_auto if vz0 is None else vz0
    if x0 is None:
        x0 = 0.0

    dt = t_burn / N
    t_grid = np.linspace(0.0, t_burn, N + 1)
    sin_dmax = float(np.sin(vehicle.delta_max))

    if verbose:
        print("=" * 70)
        print("FLIP-AND-LAND TRAJECTORY OPTIMIZATION (6-DoF)")
        print("=" * 70)
        print(vehicle.summary())
        print(f"\n  Nodes             : {N + 1}")
        print(f"  Burn time         : {t_burn:.1f} s   (dt = {dt:.4f} s)")
        print(f"  Initial pitch     : {theta0_deg:.0f} deg from vertical")
        print(f"  Seed flip time    : {t_flip:.2f} s "
              f"(rate limit allows no faster than "
              f"{theta0 / vehicle.omega_max:.2f} s)")
        print(f"  Glideslope        : {gamma_gs_deg:.0f} deg")
        print(f"  Initial state     : ({x0:,.0f}, {z0:,.0f}) m, "
              f"({vx0:.1f}, {vz0:.1f}) m/s")
        print()

    # ------------------------------------------------------------------
    # Non-dimensional scales. Same reasoning as Day 3: in SI these
    # coefficients span a dozen orders of magnitude and Clarabel fails on the
    # conditioning rather than the physics.
    # ------------------------------------------------------------------
    L = max(abs(z0), abs(x0), 1.0)
    V = L / t_burn
    M = vehicle.m_wet
    F = vehicle.T_max
    TAU = vehicle.tau_max
    W = vehicle.omega_max

    c_pos = dt * V / L
    c_vel = dt * F / (M * V)
    c_grav = dt * G_EARTH / V
    c_mass = dt * F / (M * vehicle.isp * G0)
    c_th = dt * W
    c_w = dt * (TAU / vehicle.I_pitch) / W

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    x = cp.Variable(N + 1, name="x")
    z = cp.Variable(N + 1, name="z")
    vx = cp.Variable(N + 1, name="vx")
    vz = cp.Variable(N + 1, name="vz")
    th = cp.Variable(N + 1, name="theta")     # radians, unscaled
    w = cp.Variable(N + 1, name="omega")      # omega / omega_max
    m = cp.Variable(N + 1, name="m")
    s = cp.Variable(N, name="sigma")          # thrust / T_max
    u = cp.Variable(N, name="tau")            # torque / tau_max

    # ------------------------------------------------------------------
    # Parameters. Every product of references is pre-multiplied in numpy so
    # each parameter enters the problem linearly and CVXPY can cache the
    # compilation across all SCvx iterations (DPP).
    # ------------------------------------------------------------------
    q_sin = cp.Parameter(N, name="q_sin")     # coefficient on sigma  -> vx
    q_cos = cp.Parameter(N, name="q_cos")     # coefficient on sigma  -> vz
    q_a = cp.Parameter(N, name="q_a")         # coefficient on theta  -> vx
    q_b = cp.Parameter(N, name="q_b")         # coefficient on theta  -> vz
    q_c = cp.Parameter(N, name="q_c")         # constant offset       -> vx
    q_d = cp.Parameter(N, name="q_d")         # constant offset       -> vz
    qu_x = cp.Parameter(N, name="qu_x")       # coefficient on tau    -> vx
    qu_z = cp.Parameter(N, name="qu_z")       # coefficient on tau    -> vz
    p_th = cp.Parameter(N + 1, name="theta_ref")
    p_trust = cp.Parameter(nonneg=True, name="trust")
    p_u = cp.Parameter(N, name="tau_ref")
    p_trust_u = cp.Parameter(nonneg=True, name="trust_tau")

    cons = [
        x[0] == x0 / L, z[0] == z0 / L,
        vx[0] == vx0 / V, vz[0] == vz0 / V,
        th[0] == theta0, w[0] == omega0 / W,
        m[0] == 1.0,
        # Terminal: on the pad, at rest, upright, not rotating.
        x[N] == 0.0, z[N] == 0.0,
        vx[N] == 0.0, vz[N] == 0.0,
        th[N] == 0.0, w[N] == 0.0,
    ]

    # --- dynamics -----------------------------------------------------
    cons += [
        x[1:] == x[:-1] + c_pos * vx[:-1],
        z[1:] == z[:-1] + c_pos * vz[:-1],
        th[1:] == th[:-1] + c_th * w[:-1],
        w[1:] == w[:-1] + c_w * u,
        m[1:] == m[:-1] - c_mass * s,
        # Linearised thrust-attitude coupling folded straight into the
        # velocity update, with the mass reference already divided in.
        vx[1:] == vx[:-1] + cp.multiply(q_sin, s) + cp.multiply(q_a, th[:-1])
        - q_c + cp.multiply(qu_x, u),
        vz[1:] == vz[:-1] + cp.multiply(q_cos, s) - cp.multiply(q_b, th[:-1])
        + q_d - cp.multiply(qu_z, u) - c_grav,
    ]

    # --- bounds and path constraints ----------------------------------
    tan_gs = float(np.tan(np.radians(gamma_gs_deg)))
    cons += [
        z >= 0.0,
        m >= vehicle.m_dry / M, m <= 1.0,
        s >= vehicle.T_min / F, s <= 1.0,
        w >= -1.0, w <= 1.0,
        # |tau| <= sigma * L_engine * sin(delta_max) is exactly |u| <= s once
        # both are normalised by their maxima. Linear, and exact.
        u <= s, u >= -s,
        # Glideslope, both sides.
        x * tan_gs <= z, -x * tan_gs <= z,
        # Trust region: the linearisation is only valid near its reference.
        th - p_th <= p_trust, p_th - th <= p_trust,
        # ...and the torque needs one too. The expansion point for the gimbal
        # angle is the previous torque solution, and that solution is bang-bang:
        # left unbounded it flips sign between iterations, moving the
        # linearisation point further than the step it was meant to validate.
        u - p_u <= p_trust_u, p_u - u <= p_trust_u,
    ]

    problem = cp.Problem(cp.Minimize(-m[N]), cons)

    # ------------------------------------------------------------------
    # SCvx iteration
    # ------------------------------------------------------------------
    theta_ref = theta0 * np.clip(1.0 - t_grid / t_flip, 0.0, 1.0)
    s_ref = np.full(N, 0.7)
    u_ref = np.zeros(N)          # torque reference (normalised)
    mdot_est = 0.7 * F / (vehicle.isp * G0)
    m_ref = np.linspace(vehicle.m_wet,
                        max(vehicle.m_wet - mdot_est * t_burn,
                            vehicle.m_dry + 1000.0), N + 1) / M
    trust = np.radians(trust0_deg)
    trust_u = 2.0            # torque is normalised to [-1, 1]; start unbounded

    def _linear_coeffs():
        """
        First-order expansion of the thrust direction about the current
        reference.

        Expanding about delta = 0 rather than about the *previous* gimbal
        solution leaves an irreducible second-order error: the neglected term is
        0.5 sin(theta) delta^2, which at the 15 degree limit is 0.034 of maximum
        thrust and puts a floor under the defect no amount of iterating can
        clear. Expanding about delta_ref instead lets that floor collapse as the
        reference converges.
        """
        th_r = theta_ref[:N]
        s_r = np.maximum(s_ref, 1e-6)
        # delta_ref from the reference torque: tau = sigma L sin(delta).
        sin_d_ref = np.clip(u_ref * sin_dmax / s_r, -1.0, 1.0)
        d_ref = np.arcsin(sin_d_ref)
        phi_r = th_r + d_ref                       # actual thrust direction
        # d(delta)/d(tau) at the reference, in normalised units
        K = sin_dmax / np.maximum(np.cos(d_ref), 1e-6)
        return th_r, s_r, phi_r, K

    def push_params():
        th_r, s_r, phi_r, K = _linear_coeffs()
        inv_m = c_vel / np.maximum(m_ref[:N], vehicle.m_dry / M)
        sin_p, cos_p = np.sin(phi_r), np.cos(phi_r)

        q_sin.value = inv_m * sin_p
        q_cos.value = inv_m * cos_p
        q_a.value = inv_m * s_r * cos_p
        q_b.value = inv_m * s_r * sin_p
        # Constant offsets carry both the attitude and torque reference terms.
        q_c.value = inv_m * (s_r * cos_p * th_r + K * cos_p * u_ref)
        q_d.value = inv_m * (s_r * sin_p * th_r + K * sin_p * u_ref)
        qu_x.value = inv_m * K * cos_p
        qu_z.value = inv_m * K * sin_p
        p_th.value = theta_ref
        p_trust.value = float(trust)
        p_u.value = u_ref
        p_trust_u.value = float(trust_u)

    def solve_once():
        for name in SOLVER_CHAIN:
            try:
                problem.solve(solver=getattr(cp, name), verbose=False)
                if problem.status is not None:
                    return problem.status
            except Exception:      # noqa: BLE001 - try the next solver
                continue
        return problem.status or "solver_error"

    def linearisation_defect(s_v, th_v, u_v):
        """
        How far the linear model strayed from the true coupling.

        Recomputes the real thrust direction from the solved throttle, attitude
        and torque, and returns the largest disagreement with what the linear
        model predicted, as a fraction of maximum thrust.
        """
        th_r, s_r, phi_r, K = _linear_coeffs()
        sin_p, cos_p = np.sin(phi_r), np.cos(phi_r)

        lin_x = (sin_p * s_v + s_r * cos_p * (th_v[:N] - th_r)
                 + K * cos_p * (u_v - u_ref))
        lin_z = (cos_p * s_v - s_r * sin_p * (th_v[:N] - th_r)
                 - K * sin_p * (u_v - u_ref))

        # True nonlinear values. delta comes from tau = sigma L sin(delta).
        s_safe = np.maximum(s_v, 1e-6)
        sin_delta = np.clip(u_v * sin_dmax / s_safe, -1.0, 1.0)
        delta = np.arcsin(sin_delta)
        true_x = s_v * np.sin(th_v[:N] + delta)
        true_z = s_v * np.cos(th_v[:N] + delta)

        return float(np.max(np.hypot(lin_x - true_x, lin_z - true_z)))

    if verbose:
        print("Running SCvx iterations (attitude linearisation + trust region)...")

    history = []
    status = None
    accepted = 0
    # Keep the best iterate rather than trusting whichever solve happened to be
    # last. A single infeasible subproblem at the end of the run - easy to hit
    # when the trust region has just grown - would otherwise discard a perfectly
    # good converged answer.
    best = None

    for it in range(1, max_iters + 1):
        push_params()
        status = solve_once()

        if status not in ("optimal", "optimal_inaccurate"):
            trust *= 0.5
            if verbose:
                print(f"  iter {it:2d}: {status:<12} "
                      f"trust -> {np.degrees(trust):5.2f} deg")
            if trust < np.radians(trust_min_deg):
                break
            continue

        th_v = np.asarray(th.value)
        s_v = np.asarray(s.value)
        u_v = np.asarray(u.value)
        m_v = np.asarray(m.value)

        step = float(np.max(np.abs(th_v - theta_ref)))
        defect = linearisation_defect(s_v, th_v, u_v)
        fuel = float((1.0 - m_v[-1]) * M)
        history.append((it, np.degrees(step), defect, fuel))

        # Prefer a low defect; break ties on fuel. A cheap trajectory whose
        # linear model does not describe it is worth less than an honest one.
        snapshot = dict(
            x=np.asarray(x.value), z=np.asarray(z.value),
            vx=np.asarray(vx.value), vz=np.asarray(vz.value),
            th=th_v.copy(), w=np.asarray(w.value), m=m_v.copy(),
            s=s_v.copy(), u=u_v.copy(),
            defect=defect, fuel=fuel, status=status, iteration=it,
        )
        if best is None or (defect, fuel) < (best["defect"], best["fuel"]):
            best = snapshot

        # A solved subproblem always advances the reference. Rejecting it and
        # re-solving around the *same* reference with a smaller region is a
        # deadlock: the region only ever gets tighter around a point the
        # solution is far from, and the next solve is strictly harder. The
        # defect sizes the next trust region instead of vetoing this step.
        theta_ref = th_v.copy()
        s_ref = s_v.copy()
        m_ref = m_v.copy()
        u_ref = u_v.copy()
        accepted += 1

        tight = defect < defect_tol
        if tight:
            trust = min(trust * 1.6, np.radians(trust_max_deg))
            trust_u = min(trust_u * 1.6, 2.0)
        else:
            trust = max(trust * 0.6, np.radians(trust_min_deg))
            trust_u = max(trust_u * 0.6, 0.02)

        if verbose:
            print(f"  iter {it:2d}: fuel {fuel:8,.0f} kg   "
                  f"step {np.degrees(step):6.2f} deg   "
                  f"defect {defect:7.4f}   "
                  f"trust -> {np.degrees(trust):5.2f} deg")

        # Converged when the linear model agrees with the true dynamics and the
        # objective has stopped moving. Testing the attitude step alone is not
        # enough: the trust region is often still binding at the optimum, so the
        # step tracks the region size rather than the distance to the solution.
        recent = [h[3] for h in history[-4:]]
        settled = len(recent) == 4 and (max(recent) - min(recent)) < fuel_tol_kg
        if tight and (settled or step < np.radians(tol_deg)):
            if verbose:
                print(f"  Converged after {it} iterations "
                      f"(defect {defect:.5f}, fuel stable to "
                      f"{max(recent) - min(recent):.1f} kg).")
            break

    # ------------------------------------------------------------------
    # Package
    # ------------------------------------------------------------------
    if best is None:
        if verbose:
            print(f"\n  NO SOLUTION - status: {status}")
        return {"status": status or "infeasible"}

    s_v = best["s"]
    u_v = best["u"]
    th_v = best["th"]

    sigma = s_v * F
    tau = u_v * TAU
    sin_delta = np.clip(tau / np.maximum(sigma, 1e-6) / vehicle.L_engine, -1.0, 1.0)
    delta = np.arcsin(sin_delta)
    Tx = sigma * np.sin(th_v[:N] + delta)
    Tz = sigma * np.cos(th_v[:N] + delta)

    result = {
        "t": t_grid,
        "x": best["x"] * L,
        "z": best["z"] * L,
        "vx": best["vx"] * V,
        "vz": best["vz"] * V,
        "theta": th_v,
        "omega": best["w"] * W,
        "m": best["m"] * M,
        "sigma": sigma,
        "tau": tau,
        "delta": delta,
        "Tx": Tx,
        "Tz": Tz,
        "status": best["status"],
        "fuel": float(vehicle.m_wet - best["m"][-1] * M),
        "iterations": len(history),
        "accepted": accepted,
        "best_iteration": best["iteration"],
        "final_defect": best["defect"],
        "history": history,
        "t_burn": t_burn,
        "gamma_gs_deg": gamma_gs_deg,
        "theta0_deg": theta0_deg,
    }

    if verbose:
        print(f"\n  SOLUTION FOUND")
        print(f"  Fuel consumed     : {result['fuel']:,.0f} kg "
              f"({100 * result['fuel'] / vehicle.m_prop_initial:.1f}% of load)")
        print(f"  Pitch             : {theta0_deg:.0f} deg -> "
              f"{np.degrees(th_v[-1]):.3f} deg")
        print(f"  Peak pitch rate   : "
              f"{np.degrees(np.max(np.abs(result['omega']))):.1f} deg/s "
              f"(limit {np.degrees(vehicle.omega_max):.1f})")
        print(f"  Peak gimbal       : {np.degrees(np.max(np.abs(delta))):.2f} deg "
              f"(limit {vehicle.delta_max_deg:.0f})")
        print(f"  Peak torque       : {np.max(np.abs(tau)):,.0f} N m "
              f"({100 * np.max(np.abs(tau)) / vehicle.tau_max:.0f}% of max)")
        print(f"  Throttle range    : {sigma.min() / 1e6:.2f} - "
              f"{sigma.max() / 1e6:.2f} MN")
        print(f"  Linearisation defect: {result['final_defect']:.5f} "
              f"(fraction of T_max)")
    return result


# ======================================================================
# Visualization
# ======================================================================
def plot_flip_landing(result, save_path=None, vehicle=None):
    """Eight-panel view of the flip-and-land trajectory."""
    if result["status"] not in ("optimal", "optimal_inaccurate"):
        print("Cannot plot - not solved.")
        return

    vehicle = vehicle or Vehicle6DoF()
    save_path = save_path or os.path.join(RESULTS, "day5_flip_landing.png")
    t = result["t"]
    N = len(t) - 1
    t_ctrl = t[:-1]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Day 5: Flip-and-Land Trajectory (planar 6-DoF)",
                 fontsize=14, y=1.01)

    # 1: trajectory with attitude ticks
    ax = axes[0, 0]
    ax.plot(result["x"], result["z"], "b-", linewidth=1.5, zorder=2)
    span = max(np.ptp(result["x"]), np.ptp(result["z"]), 1.0)
    scale = 0.09 * span
    for i in np.linspace(0, N, 14, dtype=int):
        xi, zi, thi = result["x"][i], result["z"][i], result["theta"][i]
        dx, dz = scale * np.sin(thi), scale * np.cos(thi)
        ax.plot([xi - dx / 2, xi + dx / 2], [zi - dz / 2, zi + dz / 2],
                "k-", linewidth=2.2, alpha=0.75, zorder=3)
        ax.plot(xi - dx / 2, zi - dz / 2, "o", color="tab:red",
                markersize=4, zorder=4)
    ax.plot(result["x"][0], result["z"][0], "go", markersize=9, label="entry")
    ax.plot(0, 0, "r^", markersize=12, label="pad")
    ax.set_xlabel("Downrange [m]"); ax.set_ylabel("Altitude [m]")
    ax.set_title("Trajectory and attitude (red = engine end)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_aspect("equal")

    # 2: position
    ax = axes[0, 1]
    ax.plot(t, result["z"], linewidth=2, label="altitude")
    ax.plot(t, result["x"], linewidth=2, label="downrange")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Position [m]")
    ax.set_title("Position"); ax.legend(); ax.grid(True, alpha=0.3)

    # 3: velocity
    ax = axes[0, 2]
    ax.plot(t, result["vx"], linewidth=2, label="vx")
    ax.plot(t, result["vz"], linewidth=2, label="vz")
    ax.plot(t, np.hypot(result["vx"], result["vz"]), "k--", linewidth=1,
            alpha=0.6, label="|v|")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity"); ax.legend(); ax.grid(True, alpha=0.3)

    # 4: pitch
    ax = axes[0, 3]
    ax.plot(t, np.degrees(result["theta"]), linewidth=2, color="tab:purple")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(90, color="gray", linestyle=":", alpha=0.6, label="horizontal")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Pitch from vertical [deg]")
    ax.set_title("The flip"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5: pitch rate
    ax = axes[1, 0]
    ax.plot(t, np.degrees(result["omega"]), linewidth=2, color="tab:orange")
    for sgn in (1, -1):
        ax.axhline(sgn * np.degrees(vehicle.omega_max), color="r",
                   linestyle=":", alpha=0.6)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Pitch rate [deg/s]")
    ax.set_title("Pitch rate (dotted = limit)"); ax.grid(True, alpha=0.3)

    # 6: thrust
    ax = axes[1, 1]
    ax.plot(t_ctrl, result["sigma"] / 1e6, linewidth=2, label="thrust")
    ax.axhline(vehicle.T_max / 1e6, color="r", ls=":", alpha=0.6, label="T_max")
    ax.axhline(vehicle.T_min / 1e6, color="orange", ls=":", alpha=0.6, label="T_min")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Thrust [MN]")
    ax.set_title("Throttle"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 7: gimbal and torque
    ax = axes[1, 2]
    ax.plot(t_ctrl, np.degrees(result["delta"]), linewidth=2,
            color="tab:green", label="gimbal")
    for sgn in (1, -1):
        ax.axhline(sgn * vehicle.delta_max_deg, color="r", ls=":", alpha=0.6)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Gimbal angle [deg]")
    ax.set_title("Gimbal (dotted = limit)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 8: SCvx convergence
    ax = axes[1, 3]
    if result.get("history"):
        its = [h[0] for h in result["history"]]
        steps = [h[1] for h in result["history"]]
        defects = [h[2] for h in result["history"]]
        ax.semilogy(its, np.maximum(steps, 1e-6), "o-", linewidth=2,
                    label="attitude step [deg]")
        ax.semilogy(its, np.maximum(defects, 1e-9), "s-", linewidth=2,
                    label="linearisation defect")
        ax.axhline(0.01, color="r", ls=":", alpha=0.6, label="accept threshold")
    ax.set_xlabel("SCvx iteration"); ax.set_ylabel("magnitude")
    ax.set_title("Convergence"); ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nTrajectory plot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    print()
    res = solve_flip_landing()
    if res["status"] in ("optimal", "optimal_inaccurate"):
        plot_flip_landing(res)
    print()


