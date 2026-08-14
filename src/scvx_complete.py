"""
The complete SCvx solver: trapezoidal collocation, free final time, log-mass.

Day 7 produced a principled SCvx loop -- trust regions, virtual control, a step
quality metric measured against the true dynamics. Three numerical weaknesses
survived it, and this module removes them without touching the algorithm around
them.

**Trapezoidal collocation.** Forward Euler is O(dt). Day 5's test suite measured
what that costs: replaying the commanded control through the verified nonlinear
integrator missed the pad by a few percent of the descent, and the note in that
test named trapezoidal collocation as the outstanding fix. Averaging the
dynamics across each interval makes it O(dt^2) for the same node count.

**Free final time.** Days 5-7 fixed the burn duration and optimised inside it.
But the entry state does not care what was guessed: for a given altitude and
speed there is a duration that costs least, and it is not the one typed into the
call. Making `t_f` a decision variable finds it.

The guide's version of this does not work, and the reason is worth stating
because it is easy to miss. It declares `t_f` a variable, bounds it, gives it a
trust region and a `0.1 * t_f` term in the objective -- but computes `dt` from
the *reference* `t_f`, so the variable never enters a single dynamics
constraint. Nothing in the problem resists the penalty, so `t_f` is driven to
its lower bound on every iteration and the reference follows it down. Measured:
it pegs at `t_f_min` regardless of the initial conditions, which is a constant
dressed as an optimisation.

Making it real means confronting the term the guide avoided. Writing `kt` for
`t_f / t_nom`, every dynamics row carries `kt * f(x, u)` -- a product of two
decision quantities, which is not convex. It is linearised the same way every
other product in this project is, about the reference:

    kt * f  ~  kt_ref * f  +  f_ref * (kt - kt_ref)

affine in both, exact at the reference, and bounded by the trust region that
already exists. That is the standard free-final-time treatment from [1].

**Log-mass.** With `zm = ln(m / m_wet)` the mass row becomes
`dzm/dt = -sigma exp(-zm) / (Isp g0)`, and the fuel objective becomes linear:
minimising propellant is exactly maximising `zm[N]`. The `exp(-zm)` is
linearised about the reference, which the trust region keeps close.

The upgrade this buys is easy to overlook: Day 7 divided the thrust term by a
mass frozen from the reference, so mass had no influence on the velocity rows
within an iteration. Here `exp(-zm)` appears in those rows too, linearised, so
the subproblem knows that burning propellant makes the vehicle lighter.

Everything else is Day 7's: the thrust direction is the attitude plus the
gimbal rather than a free vector, every variable is non-dimensional, and the
trust region adapts on a step quality measured against forward-propagated true
dynamics.

State:    [x, z, vx, vz, theta, omega, zm]   at N+1 nodes
Control:  [sigma, tau]                        at N intervals
Scalar:   kt = t_f / t_nom
Virtual:  nu on all seven dynamics rows

References
----------
[1] Szmuk, Acikmese, "Successive Convexification for 6-DoF Powered Descent
    Guidance with Free-Final-Time," AIAA, 2018.
[2] Mao, Szmuk, Acikmese, "Successive Convexification," 2020.
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
from src.dynamics_aero import dynamics_full                   # noqa: E402
from src.integrators import propagate                         # noqa: E402
from src.scvx_params import SCvxParams                        # noqa: E402
from src.scvx import Scales                                   # noqa: E402


# ======================================================================
# Reference
# ======================================================================
def initialize_reference(N, t_nom, x0, z0, vx0, vz0, theta0, omega0,
                         vehicle, seed="flip"):
    """
    Initial reference, in log-mass, with a time-scale factor of 1.

    `kt` starts at 1 by construction: the nominal duration *is* the guess, so
    the solver begins where it was told and moves from there.
    """
    t_grid = np.linspace(0.0, t_nom, N + 1)

    if seed == "flip":
        t_flip = float(np.clip(1.4 * abs(theta0) / vehicle.omega_max,
                               1.5, 0.6 * t_nom))
        theta_ref = theta0 * np.clip(1.0 - t_grid / t_flip, 0.0, 1.0)
    elif seed == "linear":
        theta_ref = np.linspace(theta0, 0.0, N + 1)
    else:
        raise ValueError(f"unknown seed {seed!r}")

    mdot_est = 0.7 * vehicle.T_max / (vehicle.isp * G0)
    m_ref = np.linspace(vehicle.m_wet,
                        max(vehicle.m_wet - mdot_est * t_nom,
                            vehicle.m_dry + 1000.0), N + 1)

    return {
        "x": np.linspace(x0, 0.0, N + 1),
        "z": np.linspace(z0, 0.0, N + 1),
        "vx": np.linspace(vx0, 0.0, N + 1),
        "vz": np.linspace(vz0, 0.0, N + 1),
        "theta": theta_ref,
        "omega": np.linspace(omega0, 0.0, N + 1),
        "zm": np.log(m_ref / vehicle.m_wet),
        "sigma": np.full(N, 0.7 * vehicle.T_max),
        "tau": np.zeros(N),
        "kt": 1.0,
    }


# ======================================================================
# True dynamics residual, trapezoidal
# ======================================================================
def nonlinear_defect(sol, vehicle, aero, sc, c, N):
    """
    Residual of the *exact* dynamics under the same trapezoidal rule.

    Same discretisation as the subproblem, deliberately, so this measures
    linearisation error rather than integration error -- including the error in
    the two products the subproblem had to linearise, `kt * f` and
    `sigma exp(-zm)`.
    """
    kt = sol["kt"]
    x, z = sol["x"] / sc.L, sol["z"] / sc.L
    vx, vz = sol["vx"] / sc.V, sol["vz"] / sc.V
    th, w = sol["theta"], sol["omega"] / sc.W
    zm = sol["zm"]
    s = sol["sigma"] / sc.F
    u = sol["tau"] / sc.TAU

    sin_dmax = float(np.sin(vehicle.delta_max))
    delta = np.arcsin(np.clip(u * sin_dmax / np.maximum(s, 1e-9), -1.0, 1.0))

    # Exact thrust acceleration at both endpoints of every interval.
    def thrust_accel(idx):
        phi = th[idx] + delta
        e = np.exp(-zm[idx])
        return (c["vel"] * s * np.sin(phi) * e,
                c["vel"] * s * np.cos(phi) * e)

    if aero is not None and aero.enabled:
        ax, az = aero_acceleration(sol["vx"], sol["vz"],
                                   np.maximum(sol["z"], 0.0), th,
                                   sol["m"], aero)
        a_x = c["dt"] * np.asarray(ax) / sc.V
        a_z = c["dt"] * np.asarray(az) / sc.V
    else:
        a_x = np.zeros(N + 1)
        a_z = np.zeros(N + 1)

    L = np.arange(N)
    R = np.arange(1, N + 1)
    tx_L, tz_L = thrust_accel(L)
    tx_R, tz_R = thrust_accel(R)

    f_vx = 0.5 * ((tx_L + a_x[L]) + (tx_R + a_x[R]))
    f_vz = 0.5 * ((tz_L + a_z[L] - c["grav"]) + (tz_R + a_z[R] - c["grav"]))
    f_zm = -0.5 * c["mass"] * (s * np.exp(-zm[L]) + s * np.exp(-zm[R]))

    rows = {
        "x": np.abs(x[R] - x[L] - kt * 0.5 * c["pos"] * (vx[L] + vx[R])),
        "z": np.abs(z[R] - z[L] - kt * 0.5 * c["pos"] * (vz[L] + vz[R])),
        "vx": np.abs(vx[R] - vx[L] - kt * f_vx),
        "vz": np.abs(vz[R] - vz[L] - kt * f_vz),
        "theta": np.abs(th[R] - th[L] - kt * 0.5 * c["th"] * (w[L] + w[R])),
        "omega": np.abs(w[R] - w[L] - kt * c["w"] * u),
        "zm": np.abs(zm[R] - zm[L] - kt * f_zm),
    }
    return rows, float(sum(r.sum() for r in rows.values()))


# ======================================================================
# Solver
# ======================================================================
def solve_scvx_complete(
    vehicle: Vehicle6DoF = None,
    aero: AeroConfig = None,
    params: SCvxParams = None,
    N: int = 80,
    t_burn_guess: float = 8.0,
    t_f_min: float = None,
    t_f_max: float = None,
    x0: float = 0.0,
    z0: float = None,
    vx0: float = 0.0,
    vz0: float = None,
    theta0_deg: float = 30.0,
    omega0: float = 0.0,
    gamma_gs_deg: float = 75.0,
    w_time: float = 0.0,
    verbose: bool = True,
):
    """
    Solve the 6-DoF powered descent with trapz collocation and free final time.

    Parameters
    ----------
    t_burn_guess : float
        Nominal burn duration [s]. Sets the time scaling and, unless `z0`/`vz0`
        are given, the entry state. The solver is free to move the *duration*
        away from this; the entry state stays where the guess put it, which is
        what makes the free time meaningful rather than circular.
    t_f_min, t_f_max : float
        Bounds on burn duration [s]. Default to half and double the guess.
    w_time : float
        Weight on a linear penalty against burn duration. Zero by default:
        minimising propellant already prefers short burns, because the 40%
        throttle floor makes fuel nearly proportional to time. Non-zero traces
        out the fuel/duration Pareto front.
    """
    vehicle = vehicle or Vehicle6DoF()
    params = params or SCvxParams()
    theta0 = np.radians(theta0_deg)
    t_nom = float(t_burn_guess)
    t_f_min = 0.5 * t_nom if t_f_min is None else float(t_f_min)
    t_f_max = 2.0 * t_nom if t_f_max is None else float(t_f_max)
    t_start = timer.time()

    if z0 is None or vz0 is None:
        t_flip = float(np.clip(1.4 * theta0 / vehicle.omega_max, 1.5,
                               0.6 * t_nom))
        z_auto, vz_auto = feasible_entry_state(vehicle, t_nom, theta0_deg,
                                               t_flip)
        z0 = z_auto if z0 is None else z0
        vz0 = vz_auto if vz0 is None else vz0

    sc = Scales(vehicle, t_nom, x0, z0)
    dt_nom = t_nom / N
    sin_dmax = float(np.sin(vehicle.delta_max))

    # Scaled coefficients at the nominal duration. Every dynamics row is these
    # times `kt`, which is why one scalar carries the whole free-time effect.
    c = {
        "dt": dt_nom,
        "pos": dt_nom * sc.V / sc.L,
        "vel": dt_nom * sc.F / (sc.M * sc.V),
        "grav": dt_nom * G_EARTH / sc.V,
        "mass": dt_nom * sc.F / (sc.M * vehicle.isp * G0),
        "th": dt_nom * sc.W,
        "w": dt_nom * (sc.TAU / vehicle.I_pitch) / sc.W,
    }

    if verbose:
        print("=" * 70)
        print("COMPLETE SCvx SOLVER (trapz + free final time + log-mass)")
        print("=" * 70)
        print(vehicle.summary())
        print()
        if aero is not None:
            print(aero.summary())
            print()
        print(params.summary())
        print(f"\n  Nodes             : {N + 1}")
        print(f"  Nominal duration  : {t_nom:.2f} s  "
              f"(free within [{t_f_min:.2f}, {t_f_max:.2f}] s)")
        print(f"  Entry state       : ({x0:,.0f}, {z0:,.0f}) m, "
              f"({vx0:.1f}, {vz0:.1f}) m/s")
        print(f"  Entry pitch       : {theta0_deg:.0f} deg from vertical")
        print(f"  Time penalty      : {w_time:g}")
        print()

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    x = cp.Variable(N + 1)
    z = cp.Variable(N + 1)
    vx = cp.Variable(N + 1)
    vz = cp.Variable(N + 1)
    th = cp.Variable(N + 1)
    w = cp.Variable(N + 1)
    zm = cp.Variable(N + 1)          # ln(m / m_wet), so zm[0] = 0
    s = cp.Variable(N)               # sigma / T_max
    u = cp.Variable(N)               # tau / tau_max
    kt = cp.Variable()               # t_f / t_nom
    nu = {k: cp.Variable(N) for k in
          ("x", "z", "vx", "vz", "theta", "omega", "zm")}

    # ------------------------------------------------------------------
    # Parameters. Every reference product is pre-multiplied in numpy so the
    # problem stays DPP and CVXPY compiles it once for the whole run.
    # ------------------------------------------------------------------
    P = {
        "ktr": cp.Parameter(nonneg=True),          # reference time factor
        # position / rotation rows: reference RHS, for the kt linearisation
        "Ax": cp.Parameter(N), "Axc": cp.Parameter(N),
        "Az": cp.Parameter(N), "Azc": cp.Parameter(N),
        "Ath": cp.Parameter(N), "Athc": cp.Parameter(N),
        "Aw": cp.Parameter(N), "Awc": cp.Parameter(N),
        # velocity rows: coefficients of the affine expansion
        "vx_s": cp.Parameter(N), "vx_thL": cp.Parameter(N),
        "vx_thR": cp.Parameter(N), "vx_u": cp.Parameter(N),
        "vx_zmL": cp.Parameter(N), "vx_zmR": cp.Parameter(N),
        "vx_k": cp.Parameter(N), "vx_ref": cp.Parameter(N),
        "vx_c": cp.Parameter(N),
        "vz_s": cp.Parameter(N), "vz_thL": cp.Parameter(N),
        "vz_thR": cp.Parameter(N), "vz_u": cp.Parameter(N),
        "vz_zmL": cp.Parameter(N), "vz_zmR": cp.Parameter(N),
        "vz_k": cp.Parameter(N), "vz_ref": cp.Parameter(N),
        "vz_c": cp.Parameter(N),
        # log-mass row
        "zm_s": cp.Parameter(N), "zm_zmL": cp.Parameter(N),
        "zm_zmR": cp.Parameter(N), "zm_k": cp.Parameter(N),
        "zm_ref": cp.Parameter(N), "zm_c": cp.Parameter(N),
        # references for the trust region
        "xr": cp.Parameter(N + 1), "zr": cp.Parameter(N + 1),
        "vxr": cp.Parameter(N + 1), "vzr": cp.Parameter(N + 1),
        "thr": cp.Parameter(N + 1), "zmr": cp.Parameter(N + 1),
        "ur": cp.Parameter(N),
        "trust": cp.Parameter(nonneg=True),
        "trust_u": cp.Parameter(nonneg=True),
        "trust_kt": cp.Parameter(nonneg=True),
        "wvc": cp.Parameter(nonneg=True),
    }

    zm_dry = float(np.log(vehicle.m_dry / vehicle.m_wet))
    tan_gs = float(np.tan(np.radians(gamma_gs_deg)))

    cons = [
        x[0] == x0 / sc.L, z[0] == z0 / sc.L,
        vx[0] == vx0 / sc.V, vz[0] == vz0 / sc.V,
        th[0] == theta0, w[0] == omega0 / sc.W,
        zm[0] == 0.0,
        x[N] == 0.0, z[N] == 0.0,
        vx[N] == 0.0, vz[N] == 0.0,
        th[N] == 0.0, w[N] == 0.0,
    ]

    # --- dynamics: trapezoidal, free time, log-mass -------------------
    # Each row is  d(state) = kt * f,  with the product linearised as
    #     kt * f  ~  kt_ref * f  +  f_ref * (kt - kt_ref)
    # and the reference cross-term folded into a constant parameter.
    cons += [
        x[1:] - x[:-1] == P["ktr"] * (0.5 * c["pos"]) * (vx[:-1] + vx[1:])
        + P["Ax"] * kt - P["Axc"] + nu["x"],
        z[1:] - z[:-1] == P["ktr"] * (0.5 * c["pos"]) * (vz[:-1] + vz[1:])
        + P["Az"] * kt - P["Azc"] + nu["z"],
        th[1:] - th[:-1] == P["ktr"] * (0.5 * c["th"]) * (w[:-1] + w[1:])
        + P["Ath"] * kt - P["Athc"] + nu["theta"],
        w[1:] - w[:-1] == P["ktr"] * c["w"] * u
        + P["Aw"] * kt - P["Awc"] + nu["omega"],
        vx[1:] - vx[:-1] == P["ktr"] * (
            cp.multiply(P["vx_s"], s) + cp.multiply(P["vx_thL"], th[:-1])
            + cp.multiply(P["vx_thR"], th[1:]) + cp.multiply(P["vx_u"], u)
            + cp.multiply(P["vx_zmL"], zm[:-1])
            + cp.multiply(P["vx_zmR"], zm[1:]) + P["vx_k"]
        ) + P["vx_ref"] * kt - P["vx_c"] + nu["vx"],
        vz[1:] - vz[:-1] == P["ktr"] * (
            cp.multiply(P["vz_s"], s) + cp.multiply(P["vz_thL"], th[:-1])
            + cp.multiply(P["vz_thR"], th[1:]) + cp.multiply(P["vz_u"], u)
            + cp.multiply(P["vz_zmL"], zm[:-1])
            + cp.multiply(P["vz_zmR"], zm[1:]) + P["vz_k"]
        ) + P["vz_ref"] * kt - P["vz_c"] + nu["vz"],
        zm[1:] - zm[:-1] == P["ktr"] * (
            cp.multiply(P["zm_s"], s) + cp.multiply(P["zm_zmL"], zm[:-1])
            + cp.multiply(P["zm_zmR"], zm[1:]) + P["zm_k"]
        ) + P["zm_ref"] * kt - P["zm_c"] + nu["zm"],
    ]

    # --- bounds and path constraints ----------------------------------
    cons += [
        z >= 0.0,
        zm >= zm_dry, zm <= 0.0,
        s >= vehicle.T_min / sc.F, s <= 1.0,
        w >= -1.0, w <= 1.0,
        u <= s, u >= -s,
        x * tan_gs <= z, -x * tan_gs <= z,
        kt >= t_f_min / t_nom, kt <= t_f_max / t_nom,
        cp.abs(x - P["xr"]) <= P["trust"],
        cp.abs(z - P["zr"]) <= P["trust"],
        cp.abs(vx - P["vxr"]) <= P["trust"],
        cp.abs(vz - P["vzr"]) <= P["trust"],
        cp.abs(th - P["thr"]) <= P["trust"],
        cp.abs(zm - P["zmr"]) <= P["trust"],
        cp.abs(u - P["ur"]) <= P["trust_u"],
        # The time factor gets its own, tighter region. It multiplies every
        # dynamics row at once, so a step in it moves the whole trajectory --
        # exactly the situation a trust region exists to bound.
        cp.abs(kt - P["ktr"]) <= P["trust_kt"],
    ]

    # Minimising propellant is exactly maximising the final log-mass, which is
    # linear. That is the third thing log-mass buys.
    objective = (-zm[N] + w_time * kt
                 + P["wvc"] * sum(cp.norm1(v) for v in nu.values()))
    problem = cp.Problem(cp.Minimize(objective), cons)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    ref = initialize_reference(N, t_nom, x0, z0, vx0, vz0, theta0, omega0,
                               vehicle, seed=params.seed)
    eta = params.eta_0
    eta_u = params.eta_u_0
    eta_kt = 0.25
    w_vc = params.w_vc
    aero_on = aero is not None and aero.enabled
    Lidx, Ridx = np.arange(N), np.arange(1, N + 1)

    def coeffs():
        """Every reference-dependent coefficient, as numpy arrays."""
        thr = ref["theta"]
        zmr = ref["zm"]
        sr = np.maximum(ref["sigma"] / sc.F, 1e-6)
        ur = ref["tau"] / sc.TAU
        ktr = float(ref["kt"])

        d_ref = np.arcsin(np.clip(ur * sin_dmax / sr, -1.0, 1.0))
        K = sin_dmax / np.maximum(np.cos(d_ref), 1e-6)
        phiL = thr[Lidx] + d_ref
        phiR = thr[Ridx] + d_ref
        EL = np.exp(-zmr[Lidx])
        ER = np.exp(-zmr[Ridx])

        if aero_on:
            m_phys = vehicle.m_wet * np.exp(zmr)
            ax, az = aero_acceleration(ref["vx"], ref["vz"],
                                       np.maximum(ref["z"], 0.0), thr,
                                       m_phys, aero)
            aax = c["dt"] * np.asarray(ax) / sc.V
            aaz = c["dt"] * np.asarray(az) / sc.V
        else:
            aax = np.zeros(N + 1)
            aaz = np.zeros(N + 1)

        out = {"ktr": ktr}

        # -- position / rotation rows: reference RHS -------------------
        out["Ax"] = 0.5 * c["pos"] * (ref["vx"][Lidx] + ref["vx"][Ridx]) / sc.V
        out["Az"] = 0.5 * c["pos"] * (ref["vz"][Lidx] + ref["vz"][Ridx]) / sc.V
        wr = ref["omega"] / sc.W
        out["Ath"] = 0.5 * c["th"] * (wr[Lidx] + wr[Ridx])
        out["Aw"] = c["w"] * ur

        # -- velocity rows ---------------------------------------------
        for tag, trig_a, trig_b, aero_term in (
            ("vx", np.sin, np.cos, (aax, +1.0)),
            ("vz", np.cos, np.sin, (aaz, -1.0)),
        ):
            aa, grav_sign = aero_term
            sgn = 1.0 if tag == "vx" else -1.0   # sign on the theta/u terms
            tL, tR = trig_a(phiL), trig_a(phiR)
            oL, oR = trig_b(phiL), trig_b(phiR)

            cs_L = c["vel"] * EL * tL
            cs_R = c["vel"] * ER * tR
            cth_L = sgn * c["vel"] * EL * sr * oL
            cth_R = sgn * c["vel"] * ER * sr * oR
            cu_L = sgn * c["vel"] * EL * K * oL
            cu_R = sgn * c["vel"] * ER * K * oR
            czm_L = -c["vel"] * EL * sr * tL
            czm_R = -c["vel"] * ER * sr * tR

            grav = c["grav"] if tag == "vz" else 0.0
            kL = (-cth_L * thr[Lidx] - cu_L * ur - czm_L * zmr[Lidx]
                  + aa[Lidx] - grav)
            kR = (-cth_R * thr[Ridx] - cu_R * ur - czm_R * zmr[Ridx]
                  + aa[Ridx] - grav)

            out[f"{tag}_s"] = 0.5 * (cs_L + cs_R)
            out[f"{tag}_thL"] = 0.5 * cth_L
            out[f"{tag}_thR"] = 0.5 * cth_R
            out[f"{tag}_u"] = 0.5 * (cu_L + cu_R)
            out[f"{tag}_zmL"] = 0.5 * czm_L
            out[f"{tag}_zmR"] = 0.5 * czm_R
            out[f"{tag}_k"] = 0.5 * (kL + kR)
            # Value of that expression at the reference, for the kt term.
            out[f"{tag}_ref"] = 0.5 * (
                c["vel"] * EL * sr * tL + aa[Lidx] - grav
                + c["vel"] * ER * sr * tR + aa[Ridx] - grav
            )

        # -- log-mass row ----------------------------------------------
        out["zm_s"] = -0.5 * c["mass"] * (EL + ER)
        out["zm_zmL"] = 0.5 * c["mass"] * sr * EL
        out["zm_zmR"] = 0.5 * c["mass"] * sr * ER
        out["zm_k"] = -(out["zm_zmL"] * zmr[Lidx] + out["zm_zmR"] * zmr[Ridx])
        out["zm_ref"] = -0.5 * c["mass"] * sr * (EL + ER)

        # -- fold the reference cross-terms of the kt linearisation ----
        for tag in ("Ax", "Az", "Ath", "Aw"):
            out[tag + "c"] = out[tag] * ktr
        for tag in ("vx", "vz", "zm"):
            out[f"{tag}_c"] = out[f"{tag}_ref"] * ktr

        out["xr"] = ref["x"] / sc.L
        out["zr"] = ref["z"] / sc.L
        out["vxr"] = ref["vx"] / sc.V
        out["vzr"] = ref["vz"] / sc.V
        out["thr"] = thr
        out["zmr"] = zmr
        out["ur"] = ur
        return out

    def push():
        cf = coeffs()
        for key, val in cf.items():
            if key in P:
                P[key].value = val
        P["trust"].value = float(eta)
        P["trust_u"].value = float(eta_u)
        P["trust_kt"].value = float(eta_kt)
        P["wvc"].value = float(w_vc)

    def solve_once():
        for name in (params.solver, params.solver_fallback):
            try:
                problem.solve(solver=getattr(cp, name),
                              verbose=params.solver_verbose)
                if problem.status in ("optimal", "optimal_inaccurate"):
                    return problem.status
            except Exception:      # noqa: BLE001
                continue
        return problem.status or "solver_error"

    def unpack():
        kt_v = float(kt.value)
        zm_v = np.asarray(zm.value)
        return {
            "x": np.asarray(x.value) * sc.L,
            "z": np.asarray(z.value) * sc.L,
            "vx": np.asarray(vx.value) * sc.V,
            "vz": np.asarray(vz.value) * sc.V,
            "theta": np.asarray(th.value),
            "omega": np.asarray(w.value) * sc.W,
            "zm": zm_v,
            "m": vehicle.m_wet * np.exp(zm_v),
            "sigma": np.asarray(s.value) * sc.F,
            "tau": np.asarray(u.value) * sc.TAU,
            "kt": kt_v,
            "t_f": kt_v * t_nom,
        }

    def thrust_defect(sol):
        """Day 5's honesty check: linear thrust direction vs the true one."""
        thr = ref["theta"]
        sr = np.maximum(ref["sigma"] / sc.F, 1e-6)
        ur = ref["tau"] / sc.TAU
        d_ref = np.arcsin(np.clip(ur * sin_dmax / sr, -1.0, 1.0))
        K = sin_dmax / np.maximum(np.cos(d_ref), 1e-6)
        phi = thr[Lidx] + d_ref
        sp, cpz = np.sin(phi), np.cos(phi)

        s_v = sol["sigma"] / sc.F
        u_v = sol["tau"] / sc.TAU
        th_v = sol["theta"]
        lin_x = (sp * s_v + sr * cpz * (th_v[Lidx] - thr[Lidx])
                 + K * cpz * (u_v - ur))
        lin_z = (cpz * s_v - sr * sp * (th_v[Lidx] - thr[Lidx])
                 - K * sp * (u_v - ur))
        d_v = np.arcsin(np.clip(u_v * sin_dmax / np.maximum(s_v, 1e-9),
                                -1.0, 1.0))
        return float(np.max(np.hypot(
            lin_x - s_v * np.sin(th_v[Lidx] + d_v),
            lin_z - s_v * np.cos(th_v[Lidx] + d_v))))

    history = {"fuel": [], "vc_norm": [], "defect": [], "thrust_defect": [],
               "eta": [], "step": [], "rho": [], "t_f": [], "status": [],
               "accepted": [], "w_vc": []}
    best = None
    last = None
    converged_sol = None
    vc_prev = float("inf")

    ref_sol = {"x": ref["x"], "z": ref["z"], "vx": ref["vx"], "vz": ref["vz"],
               "theta": ref["theta"], "omega": ref["omega"], "zm": ref["zm"],
               "m": vehicle.m_wet * np.exp(ref["zm"]),
               "sigma": ref["sigma"], "tau": ref["tau"], "kt": ref["kt"]}
    _, prev_defect = nonlinear_defect(ref_sol, vehicle, aero, sc, c, N)
    prev_mass = -ref["zm"][-1]

    if verbose:
        print(f"  {'It':>3}  {'status':>10}  {'fuel [kg]':>10}  {'t_f [s]':>8}  "
              f"{'|nu|':>10}  {'thrust_d':>9}  {'eta':>7}  {'step':>8}  "
              f"{'rho':>7}  {'':>3}")
        print("  " + "-" * 96)

    for it in range(1, params.max_iter + 1):
        push()
        status = solve_once()

        if status not in ("optimal", "optimal_inaccurate"):
            eta = max(eta * params.alpha_shrink, params.eta_min)
            for key, val in (("fuel", np.nan), ("vc_norm", np.nan),
                             ("defect", np.nan), ("thrust_defect", np.nan),
                             ("step", np.nan), ("rho", np.nan),
                             ("t_f", np.nan)):
                history[key].append(val)
            history["eta"].append(eta)
            history["status"].append(status)
            history["accepted"].append(False)
            history["w_vc"].append(w_vc)
            if verbose:
                print(f"  {it:>3}  {status:>10}  {'---':>10}  {'---':>8}  "
                      f"{'---':>10}  {'---':>9}  {eta:>7.4f}")
            if eta <= params.eta_min:
                break
            continue

        L_new = float(problem.value)
        sol = unpack()
        vc_norm = float(sum(np.abs(v.value).sum() for v in nu.values()))
        _, defect = nonlinear_defect(sol, vehicle, aero, sc, c, N)
        td = thrust_defect(sol)
        new_mass = -sol["zm"][-1]
        J_prev = prev_mass + w_vc * prev_defect
        J_new = new_mass + w_vc * defect

        step = float(max(
            np.max(np.abs(sol["x"] - ref["x"])) / sc.L,
            np.max(np.abs(sol["z"] - ref["z"])) / sc.L,
            np.max(np.abs(sol["vx"] - ref["vx"])) / sc.V,
            np.max(np.abs(sol["vz"] - ref["vz"])) / sc.V,
            np.max(np.abs(sol["theta"] - ref["theta"])),
            np.max(np.abs(sol["zm"] - ref["zm"])),
            abs(sol["kt"] - ref["kt"]),
        ))

        predicted = J_prev - L_new
        actual = J_prev - J_new
        rho = (1.0 if actual >= -1e-12 else 0.0) if abs(predicted) < 1e-12 \
            else actual / predicted

        fuel = float(vehicle.m_wet - sol["m"][-1])
        accept = rho > params.rho_ok

        history["fuel"].append(fuel)
        history["vc_norm"].append(vc_norm)
        history["defect"].append(defect)
        history["thrust_defect"].append(td)
        history["eta"].append(eta)
        history["step"].append(step)
        history["rho"].append(rho)
        history["t_f"].append(sol["t_f"])
        history["status"].append(status)
        history["accepted"].append(bool(accept))
        history["w_vc"].append(w_vc)

        if verbose:
            print(f"  {it:>3}  {status:>10}  {fuel:>10,.0f}  "
                  f"{sol['t_f']:>8.3f}  {vc_norm:>10.2e}  {td:>9.2e}  "
                  f"{eta:>7.4f}  {step:>8.2e}  {rho:>7.3f}  "
                  f"{'ok' if accept else 'rej':>3}")

        honest = vc_norm < params.dyn_tol and td < params.defect_tol
        snapshot = dict(sol, fuel=fuel, vc_norm=vc_norm, defect=defect,
                        thrust_defect=td, iteration=it)
        if honest and (best is None or fuel < best["fuel"]):
            best = snapshot
        last = snapshot

        if accept:
            for k in ("x", "z", "vx", "vz", "theta", "omega", "zm",
                      "sigma", "tau"):
                ref[k] = sol[k].copy()
            ref["z"] = np.maximum(ref["z"], 0.0)
            ref["kt"] = sol["kt"]
            prev_mass, prev_defect = new_mass, defect

        if rho > params.rho_good:
            eta = min(eta * params.alpha_expand, params.eta_max)
            eta_u = min(eta_u * params.alpha_expand, 2.0)
            eta_kt = min(eta_kt * params.alpha_expand, 0.5)
        elif not accept:
            eta = max(eta * params.alpha_shrink, params.eta_min)
            eta_u = max(eta_u * params.alpha_shrink, params.eta_u_min)
            eta_kt = max(eta_kt * params.alpha_shrink, 1e-4)

        if vc_norm > params.vc_tol and vc_norm > 0.9 * vc_prev:
            w_vc = min(w_vc * params.w_vc_grow, params.w_vc_max)
        vc_prev = vc_norm

        if it >= params.min_iter:
            recent = [f for f in history["fuel"][-4:] if not np.isnan(f)]
            settled = (len(recent) == 4
                       and max(recent) - min(recent) < params.fuel_tol_kg)
            if vc_norm < params.vc_tol and (step < params.step_tol or settled):
                # The converged iterate *is* the answer. The cheapest-honest
                # bookkeeping below it is a safety net for runs that never get
                # here -- preferring it once SCvx has actually converged buys a
                # few kilograms of linearisation error and calls it a saving.
                if honest:
                    converged_sol = snapshot
                if verbose:
                    print(f"\n  Converged after {it} iterations "
                          f"(|nu| = {vc_norm:.2e}, step = {step:.2e})")
                break
            if eta <= params.eta_min:
                if verbose:
                    print(f"\n  Trust region collapsed at iteration {it}.")
                break

    # ------------------------------------------------------------------
    # Package
    # ------------------------------------------------------------------
    elapsed = timer.time() - t_start
    result = converged_sol or best or last
    if result is None:
        if verbose:
            print(f"\n  NO SOLUTION ({elapsed:.1f}s)")
        return {"status": "failed", "history": history, "elapsed": elapsed}

    sigma, tau = result["sigma"], result["tau"]
    delta = np.arcsin(np.clip(
        tau / np.maximum(sigma, 1e-6) / vehicle.L_engine, -1.0, 1.0))
    result["delta"] = delta
    result["t"] = np.linspace(0.0, result["t_f"], N + 1)
    result["Tx"] = sigma * np.sin(result["theta"][:N] + delta)
    result["Tz"] = sigma * np.cos(result["theta"][:N] + delta)
    result["status"] = ("converged"
                        if (result["vc_norm"] < params.dyn_tol
                            and result["thrust_defect"] < params.defect_tol)
                        else "unconverged")
    result["history"] = history
    result["iterations"] = len(history["fuel"])
    result["elapsed"] = elapsed
    result["N"] = N
    result["t_nom"] = t_nom
    result["t_f_bounds"] = (t_f_min, t_f_max)
    result["theta0_deg"] = theta0_deg
    result["gamma_gs_deg"] = gamma_gs_deg
    result["max_gimbal_deg"] = float(np.degrees(np.abs(delta).max()))
    result["q"] = np.asarray(dynamic_pressure(result["vx"], result["vz"],
                                              np.maximum(result["z"], 0.0)))
    Fdx, Fdz = aero_force(result["vx"][:N], result["vz"][:N],
                          np.maximum(result["z"][:N], 0.0),
                          result["theta"][:N], aero or AeroConfig())
    result["drag_mag"] = np.hypot(np.asarray(Fdx), np.asarray(Fdz))
    # How far exp(zm) and the mass it claims to represent have drifted apart.
    result["log_mass_error"] = float(np.max(np.abs(
        vehicle.m_wet * np.exp(result["zm"]) - result["m"])))

    if verbose:
        v_f = float(np.hypot(result["vx"][-1], result["vz"][-1]))
        print(f"\n  SOLUTION ({elapsed:.1f}s, {result['iterations']} "
              f"iterations, best from iteration {result['iteration']})")
        print(f"  Burn duration     : {result['t_f']:.3f} s "
              f"(guess was {t_nom:.2f} s)")
        print(f"  Propellant        : {result['fuel']:,.0f} kg "
              f"({100 * result['fuel'] / vehicle.m_prop_initial:.1f}% of load)")
        print(f"  Virtual control   : {result['vc_norm']:.2e}")
        print(f"  True defect       : {result['defect']:.2e}")
        print(f"  Thrust defect     : {result['thrust_defect']:.2e}")
        print(f"  Touchdown         : ({result['x'][-1]:.3f}, "
              f"{result['z'][-1]:.3f}) m at {v_f:.3f} m/s")
        print(f"  Peak gimbal       : {result['max_gimbal_deg']:.2f} deg")
        print(f"  Peak q            : {np.max(result['q']) / 1000:.1f} kPa")
    return result


# ======================================================================
# Plots
# ======================================================================
def plot_complete(result, save_path=None, vehicle=None):
    """Twelve panels: the Day 7 set plus log-mass and the free-time history."""
    if result.get("status") == "failed":
        print("Cannot plot - no solution.")
        return
    vehicle = vehicle or Vehicle6DoF()
    save_path = save_path or os.path.join(RESULTS, "day8_complete.png")
    t, N = result["t"], result["N"]
    tc = t[:-1]

    fig, ax = plt.subplots(3, 4, figsize=(24, 15))
    fig.suptitle("Day 8: complete SCvx (trapezoidal, free final time, "
                 "log-mass)", fontsize=15, y=1.01)

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

    for col, (series, lab, ttl) in enumerate((
        (("z", "x"), ("altitude", "downrange"), "Position"),
        (None, None, "Velocity"),
        (("theta",), None, "Pitch from vertical"),
    ), start=1):
        a = ax[0, col]
        if ttl == "Position":
            a.plot(t, result["z"], lw=2, label="altitude")
            a.plot(t, result["x"], lw=2, label="downrange")
            a.axhline(0, color="k", lw=0.5); a.legend(fontsize=8)
            a.set_ylabel("[m]")
        elif ttl == "Velocity":
            a.plot(t, np.hypot(result["vx"], result["vz"]), "k-", lw=2,
                   label="|v|")
            a.plot(t, result["vx"], lw=1.5, alpha=0.7, label="vx")
            a.plot(t, result["vz"], lw=1.5, alpha=0.7, label="vz")
            a.legend(fontsize=8); a.set_ylabel("[m/s]")
        else:
            a.plot(t, np.degrees(result["theta"]), lw=2, color="tab:purple")
            a.axhline(0, color="k", lw=0.5); a.set_ylabel("[deg]")
        a.set_xlabel("Time [s]"); a.set_title(ttl); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.plot(t, np.degrees(result["omega"]), lw=2, color="tab:orange")
    a.axhline(np.degrees(vehicle.omega_max), color="r", ls=":", alpha=0.6)
    a.axhline(-np.degrees(vehicle.omega_max), color="r", ls=":", alpha=0.6)
    a.set_xlabel("Time [s]"); a.set_ylabel("[deg/s]")
    a.set_title("Pitch rate"); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(tc, result["sigma"] / 1e6, lw=2)
    a.axhline(vehicle.T_max / 1e6, color="r", ls=":", alpha=0.6, label="T_max")
    a.axhline(vehicle.T_min / 1e6, color="orange", ls=":", alpha=0.6,
              label="T_min")
    a.set_xlabel("Time [s]"); a.set_ylabel("[MN]")
    a.set_title("Thrust"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 2]
    a.plot(tc, np.degrees(result["delta"]), lw=2, color="tab:cyan")
    a.axhline(vehicle.delta_max_deg, color="r", ls=":", alpha=0.6)
    a.axhline(-vehicle.delta_max_deg, color="r", ls=":", alpha=0.6)
    a.set_xlabel("Time [s]"); a.set_ylabel("[deg]")
    a.set_title("Gimbal angle"); a.grid(alpha=0.3)

    a = ax[1, 3]
    a.plot(tc, result["drag_mag"] / 1000, lw=2, color="tab:green")
    a.set_xlabel("Time [s]"); a.set_ylabel("[kN]")
    a.set_title("Aerodynamic drag"); a.grid(alpha=0.3)

    a = ax[2, 0]
    a.plot(t, result["q"] / 1000, lw=2, color="tab:red")
    a.set_xlabel("Time [s]"); a.set_ylabel("[kPa]")
    a.set_title("Dynamic pressure"); a.grid(alpha=0.3)

    a = ax[2, 1]
    a.plot(t, result["m"] / 1000, lw=2, color="tab:purple")
    a.axhline(vehicle.m_dry / 1000, color="r", ls=":", alpha=0.6, label="dry")
    a.set_xlabel("Time [s]"); a.set_ylabel("[tonnes]")
    a.set_title("Mass"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[2, 2]
    a.plot(t, result["zm"], lw=2, color="tab:brown")
    a.set_xlabel("Time [s]"); a.set_ylabel("$z_m = \\ln(m/m_{wet})$")
    a.set_title(f"Log-mass (exp(zm) vs m: "
                f"{result['log_mass_error']:.1e} kg)")
    a.grid(alpha=0.3)

    a = ax[2, 3]
    hist = result["history"]
    tf = np.array(hist["t_f"], dtype=float)
    v = ~np.isnan(tf)
    a.plot(np.arange(1, len(tf) + 1)[v], tf[v], "o-", lw=2, color="tab:blue")
    a.axhline(result["t_nom"], color="gray", ls="--", alpha=0.7, label="guess")
    for b in result["t_f_bounds"]:
        a.axhline(b, color="r", ls=":", alpha=0.5)
    a.set_xlabel("Iteration"); a.set_ylabel("$t_f$ [s]")
    a.set_title("Free final time converging"); a.legend(fontsize=8)
    a.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nTrajectory plot -> {save_path}")
    plt.close()


def plot_convergence(result, save_path=None, params=None):
    """Five panels: Day 7's four plus the burn duration."""
    hist = result.get("history")
    if not hist or not hist["fuel"]:
        print("No convergence history.")
        return
    params = params or SCvxParams()
    save_path = save_path or os.path.join(RESULTS, "day8_convergence.png")
    n = len(hist["fuel"])
    its = np.arange(1, n + 1)

    def arr(k):
        return np.array(hist[k], dtype=float)

    fig, ax = plt.subplots(1, 5, figsize=(28, 5))
    fig.suptitle("Day 8 convergence history", fontsize=13)

    for a, (key, lab, ttl, logy) in zip(ax, (
        ("fuel", "Propellant [kg]", "Objective", False),
        (None, None, "Slack vs. reality", True),
        ("eta", "$\\eta$ (log)", "Trust-region radius", True),
        ("step", "scaled step (log)", "Iteration step", True),
        ("t_f", "$t_f$ [s]", "Burn duration", False),
    )):
        if ttl == "Slack vs. reality":
            for k, style, col, lb in (("vc_norm", "o-", "tab:red",
                                       "virtual control"),
                                      ("defect", "s--", "tab:brown",
                                       "true defect")):
                y = arr(k)
                m = ~np.isnan(y)
                a.semilogy(its[m], np.maximum(y[m], 1e-16), style, lw=2,
                           color=col, label=lb)
            a.axhline(params.vc_tol, color="green", ls=":", alpha=0.7)
            a.set_ylabel("L1 norm, scaled (log)"); a.legend(fontsize=8)
        else:
            y = arr(key)
            m = ~np.isnan(y)
            if logy:
                a.semilogy(its[m], np.maximum(y[m], 1e-16), "o-", lw=2)
            else:
                a.plot(its[m], y[m], "o-", lw=2)
            a.set_ylabel(lab)
        if ttl == "Burn duration":
            a.axhline(result["t_nom"], color="gray", ls="--", alpha=0.7,
                      label="guess")
            a.legend(fontsize=8)
        a.set_xlabel("Iteration"); a.set_title(ttl); a.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Convergence plot -> {save_path}")
    plt.close()


def plot_comparison(d8, d7, save_path=None):
    """Four panels: what the three upgrades actually changed."""
    save_path = save_path or os.path.join(RESULTS, "day8_comparison.png")
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Day 7 (Euler, fixed time) vs Day 8 (trapezoidal, free time, "
                 "log-mass)", fontsize=13)

    a = ax[0]
    a.plot(d7["x"], d7["z"], lw=2, label=f"Day 7 ({d7['fuel']:,.0f} kg)")
    a.plot(d8["x"], d8["z"], lw=2, label=f"Day 8 ({d8['fuel']:,.0f} kg)")
    a.plot(0, 0, "r^", ms=12)
    a.set_xlabel("Downrange [m]"); a.set_ylabel("Altitude [m]")
    a.set_title("Trajectory"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1]
    a.plot(d7["t"], np.hypot(d7["vx"], d7["vz"]), lw=2, label="Day 7")
    a.plot(d8["t"], np.hypot(d8["vx"], d8["vz"]), lw=2, label="Day 8")
    a.set_xlabel("Time [s]"); a.set_ylabel("Speed [m/s]")
    a.set_title(f"Duration: {d7['t_burn']:.2f} s fixed vs "
                f"{d8['t_f']:.2f} s chosen")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[2]
    labels = ["Day 7\nEuler", "Day 8\ntrapz"]
    vals = [d7["fuel"], d8["fuel"]]
    bars = a.bar(labels, vals, color=["tab:gray", "tab:blue"])
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:,.0f}",
               ha="center", va="bottom", fontsize=10)
    a.set_ylabel("Propellant [kg]")
    saved = 100 * (d7["fuel"] - d8["fuel"]) / d7["fuel"]
    a.set_title(f"Propellant ({saved:.1f}% saved)" if saved >= 0
                else f"Propellant ({-saved:.1f}% worse)")
    a.grid(alpha=0.3, axis="y")

    # The accuracy result that matters is not the linearisation defect -- that
    # barely moves -- but what happens when the commanded control is flown
    # through the verified nonlinear simulator. Euler's miss is the number Day
    # 5's test suite said trapezoidal collocation would be judged against.
    a = ax[3]
    veh = Vehicle6DoF()
    errs, descent = [], d8["z"][0]
    for r in (d7, d8):
        sigma, delta_c = r["sigma"], r["delta"]
        t_f = r.get("t_f", r.get("t_burn"))
        dtc = t_f / len(sigma)

        def control(t, state, vehicle, _s=sigma, _d=delta_c, _dt=dtc):
            k = min(int(t / _dt), len(_s) - 1)
            return _s[k], _d[k]

        y0 = np.array([r["x"][0], r["z"][0], r["vx"][0], r["vz"][0],
                       r["theta"][0], r["omega"][0], veh.m_wet])
        _, y = propagate(
            lambda t, yy, *a_: dynamics_full(t, yy, control, veh, AeroConfig()),
            y0, (0.0, t_f), t_f / 4000, method="rk4")
        errs.append(float(np.hypot(y[-1, 0], y[-1, 1])))

    bars = a.bar(["Day 7\nEuler", "Day 8\ntrapz"], errs,
                 color=["tab:gray", "tab:blue"])
    for b, e in zip(bars, errs):
        a.text(b.get_x() + b.get_width() / 2, b.get_height(),
               f"{e:.2f} m\n({100 * e / descent:.2f}%)",
               ha="center", va="bottom", fontsize=9)
    a.set_ylabel("miss distance [m]")
    a.set_ylim(0, max(errs) * 1.35)
    a.set_title(f"Replay through the 6-DoF simulator "
                f"({errs[0] / max(errs[1], 1e-9):.1f}x better)")
    a.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    print()
    res = solve_scvx_complete(aero=AeroConfig(), verbose=True)
    if res.get("status") != "failed":
        plot_complete(res)
        plot_convergence(res)
        from src.scvx import solve_scvx
        print("\n" + "=" * 70)
        print("COMPARISON WITH DAY 7")
        print("=" * 70)
        d7 = solve_scvx(aero=AeroConfig(), verbose=False)
        if d7.get("status") != "failed":
            plot_comparison(res, d7)
            print(f"  Day 7 : {d7['fuel']:>8,.0f} kg  "
                  f"t_f = {d7['t_burn']:.2f} s (fixed)   "
                  f"thrust defect {d7['thrust_defect']:.2e}")
            print(f"  Day 8 : {res['fuel']:>8,.0f} kg  "
                  f"t_f = {res['t_f']:.2f} s (chosen)  "
                  f"thrust defect {res['thrust_defect']:.2e}")
            saving = d7["fuel"] - res["fuel"]
            print(f"  Delta : {saving:>8,.0f} kg "
                  f"({100 * saving / d7['fuel']:+.1f}%)")
    print()
