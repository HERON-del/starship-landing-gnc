"""
Constrained minimum-fuel powered-descent landing problem.

This replaces the toy hello_convex.py from Day 1 with a properly
constrained formulation using:

    - Real 2-D translational dynamics (position + velocity)
    - Variable mass with Tsiolkovsky flow
    - Glideslope constraint (approach angle)
    - Thrust magnitude bounds via lossless convexification
    - Thrust pointing constraint (gimbal/tilt limit)

The dynamics are Euler-discretized. This is an approximation —
acceptable for the direct transcription approach, and replaced by
RK4-based SCvx in Week 2.

State:   [x, z, vx, vz, m]     at each of N+1 nodes
Control: [Tx, Tz, sigma]       at each of N nodes (sigma = ||T|| slack)

Objective: minimize fuel = sum(sigma) * dt / (Isp * g0)

A note on the defaults, because they are not arbitrary. Minimum throttle is
40% of three Raptors, so thrust-to-weight at minimum throttle is 2.16 at wet
mass. Once the engines are lit the vehicle can only decelerate — vertical
acceleration is at least +8.6 m/s² even with the thrust tilted to the pointing
limit. A fixed-duration burn therefore only closes if the vehicle arrives fast
enough that `t_burn` seconds of that deceleration exactly nulls the velocity,
and high enough that the altitude works out. Those two conditions pin z0 and
vz0 together; see `feasible_entry_state`.

References
----------
[1] Açıkmeşe, B. and Ploen, S., JGCD 2007.
[2] Szmuk, M. and Açıkmeşe, B., JGCD 2020.
"""

import os
import sys

import cvxpy as cp
import matplotlib
import numpy as np

if __name__ == "__main__":          # keep plotting headless-safe when run directly
    matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402

# Allow imports from project root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
RESULTS = os.path.join(REPO_ROOT, "results")

from src.dynamics import Vehicle, G0, G_EARTH            # noqa: E402
from src.constraints import (                            # noqa: E402
    glideslope_constraint,
    thrust_magnitude_constraint,
    pointing_constraint,
    mass_dynamics_linear,
)

# Solver order. ECOS is absent on Python 3.13 and unmaintained; Clarabel is its
# successor and handles the second-order cones here natively.
SOLVER_CHAIN = ("CLARABEL", "SCS")


def min_arrestable_speed(vehicle=None, t_burn=20.0, theta_max_deg=30.0):
    """
    Smallest entry speed a `t_burn` burn can bring to zero, and the altitude
    band that speed is compatible with.

    The engines cannot be throttled below T_min, so once lit the vertical
    acceleration never drops below

        a_min(t) = T_min * cos(theta_max) / m(t) - g

    with `m(t)` falling at the minimum mass flow. Vertical velocity is therefore
    monotonically increasing with slope at least a_min, and nulling `vz0` over
    exactly `t_burn` seconds demands

        |vz0| >= integral of a_min over the burn.

    Given `|vz0|`, altitude loss is bounded too. Decelerating as early as
    possible gives the smallest drop, as late as possible the largest:

        drop_min = integral of (integral a_min)          (linear vz)
        drop_max = |vz0| * t_burn - drop_min

    Returns
    -------
    (v_required, drop_min, drop_max) : tuple of float
    """
    vehicle = vehicle or Vehicle()
    cos_theta = np.cos(np.radians(theta_max_deg))

    n = 2000
    t = np.linspace(0.0, t_burn, n)
    mdot_min = vehicle.T_min / (vehicle.isp * G0)
    m_t = np.maximum(vehicle.m_wet - mdot_min * t, vehicle.m_dry)
    a_min = vehicle.T_min * cos_theta / m_t - G_EARTH

    if np.any(a_min <= 0):
        # A wide pointing cone lets the thrust tilt far enough that its vertical
        # component no longer beats gravity, so the vehicle can still descend
        # under power. There is no minimum entry speed in that regime.
        return 0.0, 0.0, float("inf")

    v_required = float(np.trapezoid(a_min, t))
    # vz(t) under exactly a_min, starting from -v_required
    A = np.concatenate([[0.0], np.cumsum(
        0.5 * (a_min[1:] + a_min[:-1]) * np.diff(t))])
    vz = -v_required + A
    drop_min = float(-np.trapezoid(vz, t))
    # Deferring the excess deceleration to the end keeps the vehicle fast for
    # longer, so the largest drop is |vz0| * t_burn - integral(A). Reported at
    # the minimum entry speed, where the band collapses to a single point.
    integral_A = float(np.trapezoid(A, t))
    drop_max = v_required * t_burn - integral_A
    return v_required, drop_min, drop_max


def altitude_band(v_entry, vehicle=None, t_burn=20.0, theta_max_deg=30.0):
    """Altitude drop range compatible with a given entry speed."""
    v_required, drop_min, _ = min_arrestable_speed(vehicle, t_burn, theta_max_deg)
    if v_required == 0.0:
        return 0.0, float("inf")
    integral_A = v_required * t_burn - drop_min
    return drop_min, abs(v_entry) * t_burn - integral_A


def feasible_entry_state(vehicle=None, t_burn=20.0, theta_max_deg=30.0,
                         margin=1.42):
    """
    Suggest an entry altitude and vertical speed a fixed `t_burn` burn can null.

    Takes the minimum arrestable speed, adds `margin` so the optimiser has room
    to shape the profile rather than being pinned to a single trajectory, and
    places the altitude in the middle of the compatible band.

    The default margin is not cosmetic. Lossless convexification is only
    lossless when the minimum-thrust bound is *not* binding across the whole
    arc. Arrive too slowly and the optimiser parks sigma on T_min, lets ||T||
    drift below it, and returns a trajectory that burns minimum-throttle
    propellant while producing less than minimum-throttle force — not flyable.
    Measured relaxation gap, max(sigma - ||T||) as a fraction of T_min:

        margin 1.05 -> 12.55 %     min||T|| = 0.874 T_min
        margin 1.15 ->  8.60 %     min||T|| = 0.914 T_min
        margin 1.30 ->  3.14 %     min||T|| = 0.969 T_min
        margin 1.42 ->  0.00 %     min||T|| = 1.01  T_min   <- tight
        margin 1.48 -> infeasible at t_burn = 20 s

    1.42 sits inside the tight band and clear of the infeasible edge.
    `tests/test_landing.py` asserts tightness so a regression is visible.

    Returns
    -------
    (z0, vz0) : tuple of float
    """
    v_required, _, _ = min_arrestable_speed(vehicle, t_burn, theta_max_deg)
    if v_required == 0.0:
        return 1500.0, -80.0            # pointing cone is wide; anything works
    vz0 = -v_required * margin
    lo, hi = altitude_band(vz0, vehicle, t_burn, theta_max_deg)
    z0 = 0.5 * (lo + hi)
    return float(z0), float(vz0)


def max_downrange(z0, gamma_gs_deg):
    """Largest |x0| the glideslope cone permits at altitude z0."""
    return z0 / np.tan(np.radians(gamma_gs_deg))


def solve_landing(
    vehicle: Vehicle = None,
    N: int = 60,
    t_burn: float = 20.0,
    x0: float = None,
    z0: float = None,
    vx0: float = -40.0,
    vz0: float = None,
    gamma_gs_deg: float = 80.0,
    theta_max_deg: float = 30.0,
    max_iters: int = 12,
    damping: float = 0.5,
    verbose: bool = True,
):
    """
    Solve the constrained minimum-fuel landing problem.

    Parameters
    ----------
    vehicle : Vehicle, optional
        Vehicle parameters. Uses defaults if None.
    N : int
        Number of time intervals (nodes = N+1).
    t_burn : float
        Total burn time [s]. Fixed for now; free final time comes Week 2.
    x0, z0 : float, optional
        Initial position [m]. x0 is downrange, z0 is altitude. When left as
        None they are derived from `feasible_entry_state` and the glideslope
        cone, so the default problem is solvable by construction.
    vx0, vz0 : float
        Initial velocity [m/s]. vz0 < 0 means descending. vz0 defaults with z0.
    gamma_gs_deg : float
        Glideslope angle from horizontal [deg]. Larger = steeper = tighter.
    theta_max_deg : float
        Maximum thrust angle from vertical [deg].
    max_iters : int
        Mass-reference iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys t, x, z, vx, vz, m, Tx, Tz, sigma, status, fuel
    """
    if vehicle is None:
        vehicle = Vehicle()

    # Size the entry state to the burn if the caller did not pin it down.
    if z0 is None or vz0 is None:
        z_auto, vz_auto = feasible_entry_state(vehicle, t_burn, theta_max_deg)
        z0 = z_auto if z0 is None else z0
        vz0 = vz_auto if vz0 is None else vz0
    if x0 is None:
        x0 = 0.75 * max_downrange(z0, gamma_gs_deg)

    dt = t_burn / N
    t_grid = np.linspace(0, t_burn, N + 1)

    if verbose:
        print("=" * 70)
        print("CONSTRAINED MINIMUM-FUEL LANDING PROBLEM")
        print("=" * 70)
        print(vehicle.summary())
        print(f"\n  Time steps        : {N}")
        print(f"  Burn time         : {t_burn:.1f} s")
        print(f"  dt                : {dt:.4f} s")
        print(f"  Glideslope        : {gamma_gs_deg:.0f} deg "
              f"(max |x| = {max_downrange(z0, gamma_gs_deg):,.0f} m at entry)")
        print(f"  Max thrust angle  : {theta_max_deg:.0f} deg")
        print(f"  Initial position  : ({x0:,.0f}, {z0:,.0f}) m")
        print(f"  Initial velocity  : ({vx0:.1f}, {vz0:.1f}) m/s")
        print()

    # ------------------------------------------------------------------
    # Non-dimensionalisation
    # ------------------------------------------------------------------
    # In SI the problem is atrociously scaled: thrust is ~3e6 N while the
    # velocity-update coefficient dt/m is ~3e-6, so the constraint matrix spans
    # twelve orders of magnitude. Clarabel does not merely mis-solve that, it
    # raises SolverError outright on some instances — which is easy to misread
    # as physical infeasibility. Dividing each quantity by a characteristic
    # scale brings every coefficient to order 1 and the failures disappear.
    L = max(abs(z0), abs(x0), 1.0)                  # length  [m]
    V = L / t_burn                                  # velocity[m/s]
    M = vehicle.m_wet                               # mass    [kg]
    F = vehicle.T_max                               # force   [N]

    c_pos = dt * V / L                              # position update
    c_vel = dt * F / (M * V)                        # thrust -> velocity
    c_grav = dt * G_EARTH / V                       # gravity term
    c_mass = dt * F / (M * vehicle.isp * G0)        # mass flow

    # ------------------------------------------------------------------
    # Decision variables (all non-dimensional)
    # ------------------------------------------------------------------
    x = cp.Variable(N + 1, name="x")       # horizontal position / L
    z = cp.Variable(N + 1, name="z")       # altitude / L
    vx = cp.Variable(N + 1, name="vx")     # horizontal velocity / V
    vz = cp.Variable(N + 1, name="vz")     # vertical velocity / V
    m = cp.Variable(N + 1, name="m")       # mass / M

    Tx = cp.Variable(N, name="Tx")         # horizontal thrust / F
    Tz = cp.Variable(N, name="Tz")         # vertical thrust / F
    sigma = cp.Variable(N, name="sigma")   # thrust magnitude slack / F

    # ------------------------------------------------------------------
    # Objective: minimize fuel consumption
    # ------------------------------------------------------------------
    # Scaled fuel; multiplied back to kilograms when the result is packaged.
    fuel_scaled = cp.sum(sigma) * c_mass
    objective = cp.Minimize(fuel_scaled)

    # ------------------------------------------------------------------
    # Constraints that do not depend on the mass reference
    # ------------------------------------------------------------------
    constraints = [
        x[0] == x0 / L, z[0] == z0 / L,
        vx[0] == vx0 / V, vz[0] == vz0 / V,
        m[0] == 1.0,
        # Terminal: land on the pad, at rest
        x[N] == 0.0, z[N] == 0.0,
        vx[N] == 0.0, vz[N] == 0.0,
    ]

    for k in range(N):
        # Position kinematics (Euler)
        constraints += [
            x[k + 1] == x[k] + c_pos * vx[k],
            z[k + 1] == z[k] + c_pos * vz[k],
        ]
        constraints += glideslope_constraint(x[k], z[k], gamma_gs_deg)
        constraints += thrust_magnitude_constraint(
            Tx[k], Tz[k], sigma[k], vehicle.T_min / F, vehicle.T_max / F
        )
        constraints += pointing_constraint(
            Tx[k], Tz[k], sigma[k], theta_max_deg
        )
        constraints += mass_dynamics_linear(
            m[k], m[k + 1], sigma[k], dt, vehicle.isp, coeff=c_mass
        )

    constraints += [z >= 0.0]                      # never underground
    constraints += [m >= vehicle.m_dry / M]        # cannot burn past dry mass
    constraints += [m <= 1.0]

    # ------------------------------------------------------------------
    # Velocity dynamics: fixed-mass reference, iterated to convergence
    # ------------------------------------------------------------------
    # T/m is bilinear (both are variables), so mass is held at a reference
    # profile inside the velocity update. Solve, take the resulting mass
    # profile as the new reference, repeat. This is a stripped-down SCvx:
    # successive linearisation without the trust region.
    def solve_with_mass_reference(m_ref, iteration):
        vel_constraints = []
        for k in range(N):
            m_k_ref = float(m_ref[k]) / M   # scalar, NOT a cp.Variable
            vel_constraints += [
                vx[k + 1] == vx[k] + c_vel * Tx[k] / m_k_ref,
                vz[k + 1] == vz[k] + c_vel * Tz[k] / m_k_ref - c_grav,
            ]

        prob = cp.Problem(objective, constraints + vel_constraints)
        last_error = None
        for name in SOLVER_CHAIN:
            try:
                prob.solve(solver=getattr(cp, name), verbose=False)
                if prob.status is not None:
                    break
            except Exception as exc:       # noqa: BLE001 - try the next solver
                last_error = exc
                continue
        if prob.status is None and last_error is not None:
            raise last_error

        if verbose:
            if prob.status in ("optimal", "optimal_inaccurate"):
                print(f"  Iteration {iteration}: status = {prob.status}, "
                      f"fuel = {fuel_scaled.value * M:,.1f} kg")
            else:
                print(f"  Iteration {iteration}: status = {prob.status}")
        return prob

    # Initial mass reference: the *minimum-thrust* mass profile.
    #
    # This matters more than it looks. The velocity update divides by m_ref, so
    # the reference sets the effective acceleration floor
    # a_min = T_min*cos(theta)/m_ref - g. Guessing a reference that burns too
    # much propellant makes m_ref too small, inflates a_min, and the vehicle
    # over-decelerates into an overshoot the terminal constraint cannot absorb —
    # the problem reports infeasible even though it is solvable. Starting from
    # the minimum burn gives the largest masses and the loosest floor, and a
    # minimum-fuel solution converges toward it anyway.
    mdot_min = vehicle.T_min / (vehicle.isp * G0)
    m_ref = np.maximum(vehicle.m_wet - mdot_min * t_grid, vehicle.m_dry)

    if verbose:
        print("Solving with mass-reference iterations...")
        print(f"  Initial reference: min-thrust burn, m_final ~ "
              f"{m_ref[-1]:,.0f} kg")

    prob = None
    fuel_prev = None
    for iteration in range(1, max_iters + 1):
        prob = solve_with_mass_reference(m_ref, iteration)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            if verbose:
                print(f"\n  *** SOLVER RETURNED: {prob.status} ***")
                print("  See the diagnostic checklist in docs/infeasibility.md")
            break

        if m.value is None:
            break
        m_new = m.value * M            # back to kilograms
        fuel_now = float(vehicle.m_wet - m_new[-1])

        mass_change = float(np.max(np.abs(m_new - m_ref)))
        fuel_change = abs(fuel_now - fuel_prev) if fuel_prev is not None else np.inf
        fuel_prev = fuel_now

        # Damped update. Taking m_new outright makes the iteration oscillate:
        # the minimum-fuel solution is bang-bang, so a small shift in the mass
        # reference flips a switching time and throws the profile back the other
        # way. Averaging with the previous reference is a crude trust region —
        # SCvx replaces it with a real one in Week 2.
        m_ref = damping * m_new + (1.0 - damping) * m_ref

        if verbose:
            print(f"           max mass-ref change = {mass_change:,.1f} kg, "
                  f"fuel change = {fuel_change:,.1f} kg")

        # Converge on the objective, not on the mass profile. sum(sigma) is
        # *linear*, so distinct bang-bang switching structures can tie on total
        # fuel while producing mid-flight mass histories that differ by several
        # hundred kg — the same degeneracy the Day 1 problem had. Fuel is the
        # invariant; `mass_change` is printed as a diagnostic only.
        if fuel_change < 1.0:
            if verbose:
                print(f"  Converged after {iteration} iterations.")
            break

    # ------------------------------------------------------------------
    # Package results
    # ------------------------------------------------------------------
    if prob is not None and prob.status in ("optimal", "optimal_inaccurate"):
        # Undo the non-dimensionalisation before handing anything back, so the
        # rest of the project only ever sees SI units.
        result = {
            "t": t_grid,
            "x": x.value * L, "z": z.value * L,
            "vx": vx.value * V, "vz": vz.value * V,
            "m": m.value * M,
            "Tx": Tx.value * F, "Tz": Tz.value * F, "sigma": sigma.value * F,
            "status": prob.status,
            "fuel": float(vehicle.m_wet - m.value[-1] * M),
            "gamma_gs_deg": gamma_gs_deg,
            "theta_max_deg": theta_max_deg,
            "t_burn": t_burn,
        }
        if verbose:
            speed0 = np.hypot(vx0, vz0)
            print(f"\n  SOLUTION FOUND")
            print(f"  Entry speed       : {speed0:,.1f} m/s")
            print(f"  Fuel consumed     : {result['fuel']:,.0f} kg "
                  f"({100 * result['fuel'] / vehicle.m_prop_initial:.1f}% of load)")
            print(f"  Final mass        : {result['m'][-1]:,.0f} kg")
            print(f"  Final velocity    : ({result['vx'][-1]:.3f}, "
                  f"{result['vz'][-1]:.3f}) m/s")
            print(f"  Final position    : ({result['x'][-1]:.3f}, "
                  f"{result['z'][-1]:.3f}) m")
            print(f"  Max thrust used   : {np.max(result['sigma']):,.0f} N "
                  f"({100 * np.max(result['sigma']) / vehicle.T_max:.0f}% of max)")
            print(f"  Min thrust used   : {np.min(result['sigma']):,.0f} N "
                  f"({100 * np.min(result['sigma']) / vehicle.T_max:.0f}% of max)")
        return result

    status = prob.status if prob is not None else "not solved"
    if verbose:
        print(f"\n  NO SOLUTION - status: {status}")
    return {"status": status, "gamma_gs_deg": gamma_gs_deg,
            "theta_max_deg": theta_max_deg, "t_burn": t_burn}


# ======================================================================
# Visualization
# ======================================================================
def plot_landing(result, save_path=None, vehicle=None):
    """Generate a 6-panel trajectory plot."""
    if result["status"] not in ("optimal", "optimal_inaccurate"):
        print("Cannot plot - problem was not solved.")
        return

    vehicle = vehicle or Vehicle()
    save_path = save_path or os.path.join(RESULTS, "day3_landing.png")
    t = result["t"]
    t_ctrl = t[:-1]
    gamma = result.get("gamma_gs_deg", 80.0)
    theta_max = result.get("theta_max_deg", 30.0)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Day 3: Constrained Minimum-Fuel Landing", fontsize=14, y=1.02)

    # Panel 1: trajectory in the x-z plane with the glideslope cone
    ax = axes[0, 0]
    ax.plot(result["x"], result["z"], "b-", linewidth=2, label="trajectory")
    ax.plot(result["x"][0], result["z"][0], "go", markersize=10, label="start")
    ax.plot(result["x"][-1], result["z"][-1], "r^", markersize=12, label="landing")

    z_cone = np.linspace(0, max(result["z"]) * 1.1, 50)
    x_cone = z_cone / np.tan(np.radians(gamma))
    ax.plot(x_cone, z_cone, "k--", alpha=0.4, label=f"glideslope {gamma:.0f} deg")
    ax.plot(-x_cone, z_cone, "k--", alpha=0.4)
    ax.fill_betweenx(z_cone, -x_cone, x_cone, alpha=0.06, color="green")
    ax.set_xlabel("Downrange [m]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Trajectory (x-z plane)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Panel 2: position vs time
    ax = axes[0, 1]
    ax.plot(t, result["z"], linewidth=2, label="altitude")
    ax.plot(t, result["x"], linewidth=2, label="downrange")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title("Position vs. time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: velocity
    ax = axes[0, 2]
    speed = np.sqrt(result["vx"] ** 2 + result["vz"] ** 2)
    ax.plot(t, result["vx"], linewidth=2, label="vx")
    ax.plot(t, result["vz"], linewidth=2, label="vz")
    ax.plot(t, speed, "k--", linewidth=1.5, alpha=0.5, label="|v|")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity vs. time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: thrust magnitude against the throttle box
    ax = axes[1, 0]
    thrust_actual = np.sqrt(result["Tx"] ** 2 + result["Tz"] ** 2)
    ax.plot(t_ctrl, result["sigma"] / 1e6, linewidth=2, label="sigma (slack)")
    ax.plot(t_ctrl, thrust_actual / 1e6, "--", linewidth=1.2, alpha=0.8,
            label="||T|| actual")
    ax.axhline(vehicle.T_max / 1e6, color="r", linestyle=":", alpha=0.6,
               label="T_max")
    ax.axhline(vehicle.T_min / 1e6, color="orange", linestyle=":", alpha=0.6,
               label="T_min")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thrust [MN]")
    ax.set_title("Thrust magnitude (sigma overlays ||T|| if lossless)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 5: pointing angle
    ax = axes[1, 1]
    thrust_angle = np.degrees(np.arctan2(np.abs(result["Tx"]),
                                         np.maximum(result["Tz"], 1.0)))
    ax.plot(t_ctrl, thrust_angle, linewidth=2)
    ax.axhline(theta_max, color="r", linestyle=":", alpha=0.6,
               label=f"theta_max ({theta_max:.0f} deg)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle from vertical [deg]")
    ax.set_title("Thrust pointing angle")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 6: mass
    ax = axes[1, 2]
    ax.plot(t, result["m"] / 1000, linewidth=2, color="tab:purple")
    ax.axhline(vehicle.m_dry / 1000, color="r", linestyle=":", alpha=0.6,
               label="dry mass")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mass [tonnes]")
    ax.set_title("Vehicle mass")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nTrajectory plot -> {save_path}")
    plt.close()


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    print()
    res = solve_landing()
    if res["status"] in ("optimal", "optimal_inaccurate"):
        plot_landing(res)
    print()
