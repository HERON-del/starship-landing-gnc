"""
Free-final-time minimum-fuel landing with trapezoidal discretization.

Advances beyond Day 3 by:

    1. Making burn duration an optimisation variable rather than a guess
    2. Using trapezoidal collocation instead of forward Euler
    3. Reporting the whole fuel-vs-duration curve, not just its minimum

How free final time is actually done here
-----------------------------------------
The tempting formulation — declare `t_f = cp.Variable()` and let the solver
pick it — does not work, and it is worth being precise about why. Time enters
the dynamics multiplicatively: every update looks like `x[k+1] = x[k] + dt·v[k]`
with `dt = t_f/N`. So `t_f` multiplies the state and control variables, and
`t_f·v` is a product of two unknowns. That is bilinear, therefore non-convex,
and CVXPY will reject it.

The usual workaround is to hold `dt` at a reference value and let `t_f` float
in the objective. That compiles, but it silently decouples `t_f` from the
trajectory: the dynamics use the reference `dt`, and the only term touching
`t_f` is whatever penalty was added. The solver then drives `t_f` to whichever
bound the penalty prefers, and the reported "optimal burn time" is just the
lower bound in disguise.

What is true is that for a **fixed** `t_f` the problem is convex and solves in
milliseconds, and fuel-versus-duration is a smooth unimodal curve: too short and
there is not enough impulse (infeasible), too long and gravity losses dominate.
So the honest formulation is an outer one-dimensional search over `t_f` wrapping
the convex solve — bracket the feasible interval, then golden-section down to
the minimum. This is what Açıkmeşe & Ploen do, and it retains the guarantee that
matters: every point evaluated is a global optimum of its own subproblem.

State:   [x, z, vx, vz, m]  at N+1 nodes
Control: [Tx, Tz, sigma]    at N nodes (Euler) or N+1 nodes (trapezoidal)

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

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
RESULTS = os.path.join(REPO_ROOT, "results")

from src.dynamics import Vehicle, G0, G_EARTH                  # noqa: E402
from src.constraints import (                                  # noqa: E402
    glideslope_constraint,
    thrust_magnitude_constraint,
    pointing_constraint,
)
from src.discretization import (                               # noqa: E402
    Coeffs,
    dynamics_constraints,
    is_trapz,
    n_control_nodes,
)
from src.landing_problem import feasible_entry_state, max_downrange  # noqa: E402

SOLVER_CHAIN = ("CLARABEL", "SCS")
GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0     # 0.618...


class FixedTimeProblem:
    """
    A convex landing problem whose burn duration is a parameter.

    Built once, then re-solved at many durations. Everything that depends on
    `t_f` lives in `cp.Parameter`s, so CVXPY compiles the problem a single time
    and the line search costs only solve time. Each coefficient appears
    linearly, which keeps the problem DPP-compliant — in particular the mass
    reference is folded into the velocity coefficient rather than dividing by a
    separate parameter, since parameter-over-parameter is not affine.
    """

    def __init__(self, vehicle, N, method, x0, z0, vx0, vz0,
                 gamma_gs_deg, theta_max_deg, t_scale):
        self.vehicle = vehicle
        self.N = N
        self.method = method
        self.trapz = is_trapz(method)
        self.n_ctrl = n_control_nodes(method, N)

        # --- non-dimensional scales (fixed across the whole search) --------
        self.L = max(abs(z0), abs(x0), 1.0)
        self.V = self.L / t_scale
        self.M = vehicle.m_wet
        self.F = vehicle.T_max

        # --- variables ----------------------------------------------------
        self.x = cp.Variable(N + 1, name="x")
        self.z = cp.Variable(N + 1, name="z")
        self.vx = cp.Variable(N + 1, name="vx")
        self.vz = cp.Variable(N + 1, name="vz")
        self.m = cp.Variable(N + 1, name="m")
        self.Tx = cp.Variable(self.n_ctrl, name="Tx")
        self.Tz = cp.Variable(self.n_ctrl, name="Tz")
        self.sigma = cp.Variable(self.n_ctrl, name="sigma")

        # --- duration-dependent coefficients ------------------------------
        self.p_pos = cp.Parameter(nonneg=True, name="c_pos")
        self.p_grav = cp.Parameter(nonneg=True, name="c_grav")
        self.p_mass = cp.Parameter(nonneg=True, name="c_mass")
        self.p_vel = cp.Parameter(self.n_ctrl, nonneg=True, name="c_vel")

        coeffs = Coeffs(pos=self.p_pos, vel=self.p_vel,
                        grav=self.p_grav, mass=self.p_mass)

        # --- constraints ---------------------------------------------------
        cons = [
            self.x[0] == x0 / self.L, self.z[0] == z0 / self.L,
            self.vx[0] == vx0 / self.V, self.vz[0] == vz0 / self.V,
            self.m[0] == 1.0,
            self.x[N] == 0.0, self.z[N] == 0.0,
            self.vx[N] == 0.0, self.vz[N] == 0.0,
        ]
        cons += dynamics_constraints(
            method, self.x, self.z, self.vx, self.vz, self.m,
            self.Tx, self.Tz, self.sigma, coeffs,
        )
        cons += glideslope_constraint(self.x, self.z, gamma_gs_deg)
        cons += thrust_magnitude_constraint(
            self.Tx, self.Tz, self.sigma,
            vehicle.T_min / self.F, vehicle.T_max / self.F,
        )
        cons += pointing_constraint(self.Tx, self.Tz, self.sigma, theta_max_deg)
        cons += [self.z >= 0.0]
        cons += [self.m >= vehicle.m_dry / self.M, self.m <= 1.0]

        # Fuel is exactly the mass drop, and m[0] is pinned, so maximising the
        # final mass *is* minimising fuel. No parameter needed in the objective.
        self.problem = cp.Problem(cp.Minimize(-self.m[N]), cons)

    # ------------------------------------------------------------------
    def _set_coeffs(self, t_f, m_ref_nd):
        dt = t_f / self.N
        self.p_pos.value = dt * self.V / self.L
        self.p_grav.value = dt * G_EARTH / self.V
        self.p_mass.value = dt * self.F / (self.M * self.vehicle.isp * G0)
        floor = self.vehicle.m_dry / self.M
        m_safe = np.maximum(np.asarray(m_ref_nd)[:self.n_ctrl], floor)
        self.p_vel.value = dt * self.F / (self.M * self.V * m_safe)

    def _solve_once(self):
        last = None
        for name in SOLVER_CHAIN:
            try:
                self.problem.solve(solver=getattr(cp, name), verbose=False)
                if self.problem.status is not None:
                    return self.problem.status
            except Exception as exc:          # noqa: BLE001
                last = exc
        if last is not None:
            return "solver_error"
        return self.problem.status

    def solve_at(self, t_f, max_iters=5, tol_kg=10.0, damping=0.5):
        """
        Solve at a fixed burn duration, iterating the mass reference.

        Returns a result dict, or None if no feasible solution was found.
        """
        veh = self.vehicle
        # Initial mass reference: linear from wet mass to a rough final mass.
        mdot = 0.7 * veh.T_max / (veh.isp * G0)
        m_final = max(veh.m_wet - mdot * t_f, veh.m_dry + 500.0)
        m_ref = np.linspace(veh.m_wet, m_final, self.n_ctrl) / self.M

        status = None
        for _ in range(max_iters):
            self._set_coeffs(t_f, m_ref)
            status = self._solve_once()
            if status not in ("optimal", "optimal_inaccurate"):
                return None
            m_new = np.asarray(self.m.value)[:self.n_ctrl]
            change = float(np.max(np.abs(m_new - m_ref)) * self.M)
            m_ref = (1.0 - damping) * m_ref + damping * m_new
            if change < tol_kg:
                break

        if status not in ("optimal", "optimal_inaccurate"):
            return None
        return self._package(t_f, status)

    def _package(self, t_f, status):
        L, V, M, F = self.L, self.V, self.M, self.F
        N = self.N
        veh = self.vehicle

        # Is the lossless relaxation actually lossless here?
        #
        # sigma stands in for ||T|| so that the non-convex floor ||T|| >= T_min
        # can be written convexly. The substitution is only honest while
        # sigma == ||T|| at the optimum. When it is not, the trajectory burns
        # propellant at the sigma rate while producing less than that much
        # force: it is cheaper on paper and unflyable in fact.
        #
        # Empirically the gap only opens when the *pointing* constraint is
        # active — every slack case observed has the tilt at its limit.
        # Saturation is necessary but not sufficient: there are durations where
        # the tilt sits at 30 deg and the relaxation is still exactly tight.
        # Acikmese & Ploen's magnitude-only proof does not cover an active
        # pointing constraint, so this is the boundary of the theorem showing
        # up as a number — which is why it is measured here rather than assumed.
        T_mag = np.hypot(np.asarray(self.Tx.value), np.asarray(self.Tz.value)) * F
        sig = np.asarray(self.sigma.value) * F
        gap = float(np.max(sig - T_mag))
        tilt = np.degrees(np.arctan2(np.abs(np.asarray(self.Tx.value)),
                                     np.maximum(np.asarray(self.Tz.value), 1e-9)))

        return {
            "relaxation_gap": gap,
            "lossless": bool(gap <= 0.01 * veh.T_min),
            "min_thrust_over_Tmin": float(T_mag.min() / veh.T_min),
            "max_tilt_deg": float(np.max(tilt)),
            "t": np.linspace(0.0, t_f, N + 1),
            "x": np.asarray(self.x.value) * L,
            "z": np.asarray(self.z.value) * L,
            "vx": np.asarray(self.vx.value) * V,
            "vz": np.asarray(self.vz.value) * V,
            "m": np.asarray(self.m.value) * M,
            "Tx": np.asarray(self.Tx.value) * F,
            "Tz": np.asarray(self.Tz.value) * F,
            "sigma": np.asarray(self.sigma.value) * F,
            "t_f": float(t_f),
            "status": status,
            "fuel": float(veh.m_wet - self.m.value[N] * M),
            "method": self.method,
        }


# ======================================================================
# Free-time driver
# ======================================================================
def solve_landing_free_time(
    vehicle: Vehicle = None,
    N: int = 50,
    t_f_min: float = 8.0,
    t_f_max: float = 34.0,
    x0: float = None,
    z0: float = None,
    vx0: float = -40.0,
    vz0: float = None,
    gamma_gs_deg: float = 80.0,
    theta_max_deg: float = 30.0,
    method: str = "trapz",
    t_nominal: float = 20.0,
    n_scan: int = 11,
    n_refine: int = 12,
    require_lossless: bool = True,
    verbose: bool = True,
):
    """
    Minimise fuel over both the trajectory and the burn duration.

    Scans `t_f` coarsely to bracket the feasible interval and locate the basin,
    then golden-sections to the minimum. Each evaluation is a full convex solve,
    so the returned trajectory is a genuine optimum of its own subproblem.

    Returns
    -------
    dict with the usual trajectory keys plus `t_f`, `sweep` (the evaluated
    duration/fuel pairs) and `n_solves`.
    """
    vehicle = vehicle or Vehicle()

    # Entry state, sized to the nominal duration so the search has something
    # feasible to find. Held fixed while t_f varies — otherwise the search
    # would be moving the problem, not solving it.
    if z0 is None or vz0 is None:
        z_auto, vz_auto = feasible_entry_state(vehicle, t_nominal, theta_max_deg)
        z0 = z_auto if z0 is None else z0
        vz0 = vz_auto if vz0 is None else vz0
    if x0 is None:
        x0 = 0.75 * max_downrange(z0, gamma_gs_deg)

    if verbose:
        print("=" * 70)
        print(f"FREE-TIME LANDING - {method.upper()} DISCRETIZATION")
        print("=" * 70)
        print(f"  Nodes             : {N + 1}")
        print(f"  t_f bounds        : [{t_f_min:.1f}, {t_f_max:.1f}] s")
        print(f"  Discretization    : {method}")
        print(f"  Glideslope        : {gamma_gs_deg:.0f} deg")
        print(f"  Max thrust angle  : {theta_max_deg:.0f} deg")
        print(f"  Initial state     : ({x0:,.0f}, {z0:,.0f}) m, "
              f"({vx0:.1f}, {vz0:.1f}) m/s")
        print()

    prob = FixedTimeProblem(vehicle, N, method, x0, z0, vx0, vz0,
                            gamma_gs_deg, theta_max_deg, t_scale=t_nominal)

    sweep = []
    n_solves = 0
    n_rejected = 0

    def evaluate(t_f):
        """
        Solve at one duration. Durations whose relaxation has gone slack are
        scored as infeasible: they are cheaper on paper but command less thrust
        than the engines can produce, so they are not candidate answers.
        """
        nonlocal n_solves, n_rejected
        n_solves += 1
        r = prob.solve_at(t_f)
        if r is None:
            sweep.append((float(t_f), np.inf, "infeasible"))
            return None, np.inf
        if require_lossless and not r["lossless"]:
            n_rejected += 1
            sweep.append((float(t_f), float(r["fuel"]), "slack"))
            return None, np.inf
        sweep.append((float(t_f), float(r["fuel"]), "ok"))
        return r, r["fuel"]

    # --- coarse scan to bracket the basin --------------------------------
    if verbose:
        print("Scanning burn durations...")
    grid = np.linspace(t_f_min, t_f_max, n_scan)
    best_r, best_fuel, best_i = None, np.inf, None
    for i, t_f in enumerate(grid):
        r, fuel = evaluate(t_f)
        if verbose:
            kind = sweep[-1][2]
            if kind == "ok":
                tag = f"{fuel:,.0f} kg"
            elif kind == "slack":
                tag = (f"rejected - relaxation slack "
                       f"({sweep[-1][1]:,.0f} kg on paper)")
            else:
                tag = "infeasible"
            print(f"  t_f = {t_f:5.1f} s  ->  {tag}")
        if fuel < best_fuel:
            best_r, best_fuel, best_i = r, fuel, i

    if best_r is None:
        if verbose:
            print("\n  NO FLYABLE SOLUTION at any duration in the range.")
            if n_rejected:
                print(f"  ({n_rejected} durations solved but were rejected for "
                      f"a slack relaxation.)")
        return {"status": "infeasible", "method": method,
                "sweep": sweep, "n_solves": n_solves,
                "n_rejected_slack": n_rejected}

    # --- golden-section refine inside the bracket ------------------------
    lo = grid[max(best_i - 1, 0)]
    hi = grid[min(best_i + 1, len(grid) - 1)]
    if verbose:
        print(f"\nRefining in [{lo:.1f}, {hi:.1f}] s...")

    a, b = lo, hi
    c = b - GOLDEN * (b - a)
    d = a + GOLDEN * (b - a)
    r_c, f_c = evaluate(c)
    r_d, f_d = evaluate(d)
    for _ in range(n_refine):
        if b - a < 0.05:
            break
        if f_c <= f_d:
            b, d, r_d, f_d = d, c, r_c, f_c
            c = b - GOLDEN * (b - a)
            r_c, f_c = evaluate(c)
        else:
            a, c, r_c, f_c = c, d, r_d, f_d
            d = a + GOLDEN * (b - a)
            r_d, f_d = evaluate(d)

    for r, f in ((r_c, f_c), (r_d, f_d)):
        if r is not None and f < best_fuel:
            best_r, best_fuel = r, f

    best_r["sweep"] = sorted(sweep)
    best_r["n_solves"] = n_solves
    best_r["n_rejected_slack"] = n_rejected

    if verbose:
        print(f"\n  SOLUTION FOUND  ({n_solves} convex solves"
              + (f", {n_rejected} rejected for slack relaxation)" if n_rejected
                 else ")"))
        print(f"  Burn time         : {best_r['t_f']:.2f} s")
        print(f"  Fuel consumed     : {best_r['fuel']:,.0f} kg "
              f"({100 * best_r['fuel'] / vehicle.m_prop_initial:.1f}% of load)")
        print(f"  Final mass        : {best_r['m'][-1]:,.0f} kg")
        print(f"  Final velocity    : ({best_r['vx'][-1]:.3f}, "
              f"{best_r['vz'][-1]:.3f}) m/s")
        s = best_r["sigma"]
        print(f"  Thrust range      : {s.min() / 1e6:.2f} - {s.max() / 1e6:.2f} MN "
              f"(limits {vehicle.T_min / 1e6:.2f} - {vehicle.T_max / 1e6:.2f})")
        print(f"  Relaxation gap    : {best_r['relaxation_gap']:,.0f} N "
              f"(lossless), peak tilt {best_r['max_tilt_deg']:.1f} deg")
    return best_r


# ======================================================================
# Comparison
# ======================================================================
def run_comparison(verbose=False, save_path=None, N=50):
    """
    Compare fixed-time against free-time, Euler against trapezoidal.

    All four variants share one entry state so the numbers are commensurable.
    """
    from src.landing_problem import solve_landing

    save_path = save_path or os.path.join(RESULTS, "day4_comparison.png")
    vehicle = Vehicle()
    t_nominal = 20.0
    z0, vz0 = feasible_entry_state(vehicle, t_nominal, 30.0)
    x0 = 0.75 * max_downrange(z0, 80.0)
    entry = dict(x0=x0, z0=z0, vx0=-40.0, vz0=vz0)

    print("\n" + "=" * 70)
    print("COMPARISON: FIXED vs FREE TIME, EULER vs TRAPEZOIDAL")
    print("=" * 70)
    print(f"Shared entry state: ({x0:,.0f}, {z0:,.0f}) m, "
          f"(-40.0, {vz0:.1f}) m/s\n")

    results = {}

    r0 = solve_landing(N=N, t_burn=t_nominal, verbose=False, **entry)
    if r0["status"].startswith("optimal"):
        r0["t_f"] = t_nominal
        r0["method"] = "euler"
        results["Fixed Euler t=20s"] = r0

    for label, method in (("Free-time Euler", "euler"),
                          ("Free-time Trapz", "trapz")):
        r = solve_landing_free_time(N=N, method=method, verbose=verbose,
                                    t_nominal=t_nominal, **entry)
        if r.get("status", "").startswith("optimal"):
            results[label] = r

    print(f"{'Config':<26} {'t_f [s]':>8} {'Fuel [kg]':>11} {'Method':>8}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<26} {r.get('t_f', 0):>8.2f} {r['fuel']:>11,.0f} "
              f"{r.get('method', '?'):>8}")
    print("=" * 70)

    if "Fixed Euler t=20s" in results and "Free-time Trapz" in results:
        base = results["Fixed Euler t=20s"]["fuel"]
        best = results["Free-time Trapz"]["fuel"]
        print(f"\nFuel change (free-time trapz vs fixed Euler): "
              f"{base - best:+,.0f} kg ({100 * (base - best) / base:+.1f}%)")

    if len(results) >= 2:
        _plot_comparison(results, save_path, vehicle)
    return results


def _plot_comparison(results, save_path, vehicle):
    colors = {"Fixed Euler t=20s": "tab:gray",
              "Free-time Euler": "tab:blue",
              "Free-time Trapz": "tab:red"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Day 4: Fixed vs. Free Time, Euler vs. Trapezoidal",
                 fontsize=14, y=1.01)

    ax = axes[0, 0]
    for name, r in results.items():
        ax.plot(r["x"], r["z"], linewidth=2, color=colors.get(name, "k"),
                label=f"{name} ({r['fuel']:,.0f} kg)")
    ax.plot(0, 0, "r^", markersize=12, zorder=5)
    ax.set_xlabel("Downrange [m]"); ax.set_ylabel("Altitude [m]")
    ax.set_title("Trajectory"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    ax = axes[0, 1]
    for name, r in results.items():
        ax.plot(r["t"], r["z"], linewidth=2, color=colors.get(name, "k"), label=name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Altitude [m]")
    ax.set_title("Altitude vs. time"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for name, r in results.items():
        ax.plot(r["t"], np.hypot(r["vx"], r["vz"]), linewidth=2,
                color=colors.get(name, "k"), label=name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Speed [m/s]")
    ax.set_title("Speed vs. time"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for name, r in results.items():
        n = min(len(r["t"]), len(r["sigma"]))
        ax.plot(r["t"][:n], r["sigma"][:n] / 1e6, linewidth=2,
                color=colors.get(name, "k"), label=name)
    ax.axhline(vehicle.T_min / 1e6, color="orange", ls=":", alpha=0.6, label="T_min")
    ax.axhline(vehicle.T_max / 1e6, color="red", ls=":", alpha=0.6, label="T_max")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Thrust [MN]")
    ax.set_title("Thrust magnitude"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # The gravity-loss curve: this is the thing the search is minimising.
    ax = axes[1, 1]
    for name, r in results.items():
        sweep = r.get("sweep")
        if not sweep:
            continue
        ok = sorted((t, f) for t, f, k in sweep if k == "ok")
        slack = sorted((t, f) for t, f, k in sweep if k == "slack")
        if ok:
            ax.plot([t for t, _ in ok], [f for _, f in ok], "o-", ms=3,
                    linewidth=1.5, color=colors.get(name, "k"), label=name)
        if slack:
            ax.plot([t for t, _ in slack], [f for _, f in slack], "x", ms=7,
                    color=colors.get(name, "k"), alpha=0.7,
                    label=f"{name} (slack, rejected)")
        ax.axvline(r["t_f"], color=colors.get(name, "k"), ls="--", alpha=0.5)
    ax.set_xlabel("Burn duration [s]"); ax.set_ylabel("Fuel [kg]")
    ax.set_title("Fuel vs. duration ('x' = relaxation slack, not flyable)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    names = list(results.keys())
    fuels = [results[n]["fuel"] for n in names]
    bars = ax.bar(range(len(names)), fuels,
                  color=[colors.get(n, "k") for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Fuel consumed [kg]"); ax.set_title("Fuel comparison")
    for bar, fuel in zip(bars, fuels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{fuel:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    print()
    solve_landing_free_time(method="trapz")
    print()
    run_comparison()
    print()
