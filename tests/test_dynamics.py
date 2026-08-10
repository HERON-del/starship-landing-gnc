"""
Verification suite for the 3-DoF dynamics model and integrators.

Each test compares numerical output against an independently known result:
  1. Ballistic free-fall  -> closed-form kinematics
  2. Ideal hover          -> altitude must be exactly conserved
  3. Mass depletion       -> closed-form Tsiolkovsky mass flow
  4. Convergence order    -> measured order must match theoretical order

Run:  python tests/test_dynamics.py
"""

import sys
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")           # no display needed; we only save figures
import matplotlib.pyplot as plt  # noqa: E402

# Allow importing from src/ when running this file directly
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Results land next to the repo, not next to whatever directory you happen
# to have cd'd into.
RESULTS = os.path.join(REPO_ROOT, "results")

from src.dynamics import (  # noqa: E402
    Vehicle,
    dynamics_3dof,
    control_zero,
    control_hover,
    control_constant,
    exact_constant_thrust,
    G0,
    G_EARTH,
)
from src.integrators import propagate  # noqa: E402


def _supports_ansi() -> bool:
    """
    Windows consoles need VT processing switched on before ANSI colour
    works; without this the guide's [PASS] markers render as escape gibberish.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
        except Exception:
            return False
    return True


if _supports_ansi():
    PASS = "\033[92m[PASS]\033[0m"
    FAIL = "\033[91m[FAIL]\033[0m"
else:
    PASS = "[PASS]"
    FAIL = "[FAIL]"


def report(name: str, error: float, tolerance: float) -> bool:
    ok = error < tolerance
    tag = PASS if ok else FAIL
    print(f"  {tag} {name:<38} error = {error:.3e}  (tol {tolerance:.1e})")
    return ok


# ======================================================================
# TEST 1 - Ballistic free-fall against closed-form kinematics
# ======================================================================
def test_freefall():
    """
    With zero thrust the vehicle is a point mass under constant gravity:
        z(t) = z0 + vz0 * t - 0.5 * g * t^2
        vz(t) = vz0 - g * t

    RK4 integrates polynomials up to degree 4 exactly, so the error here
    should sit at machine precision (~1e-10 or better).
    """
    print("\nTEST 1 - Ballistic free-fall vs. closed-form kinematics")
    vehicle = Vehicle()

    z0, vz0 = 5000.0, -100.0
    y0 = np.array([0.0, z0, 0.0, vz0, vehicle.m_wet])
    t_end, dt = 20.0, 0.01

    t, y = propagate(
        dynamics_3dof, y0, (0.0, t_end), dt,
        control_zero, vehicle, method="rk4",
    )

    z_exact = z0 + vz0 * t - 0.5 * G_EARTH * t**2
    vz_exact = vz0 - G_EARTH * t

    err_z = np.max(np.abs(y[:, 1] - z_exact))
    err_vz = np.max(np.abs(y[:, 3] - vz_exact))
    err_m = np.max(np.abs(y[:, 4] - vehicle.m_wet))

    ok = True
    ok &= report("altitude", err_z, 1e-6)
    ok &= report("vertical velocity", err_vz, 1e-9)
    ok &= report("mass conserved (no thrust)", err_m, 1e-12)
    return ok, (t, y, z_exact)


# ======================================================================
# TEST 2 - Ideal hover conserves altitude
# ======================================================================
def test_hover():
    """
    If thrust continuously equals instantaneous weight, net force is zero
    and the vehicle must hold altitude exactly, even as it loses mass.

    This tests that the mass-flow term and the thrust term are correctly
    coupled: a sign error or a factor of g0 mistake shows up immediately.
    """
    print("\nTEST 2 - Ideal hover holds altitude")
    vehicle = Vehicle()

    z0 = 1000.0
    y0 = np.array([0.0, z0, 0.0, 0.0, vehicle.m_wet])

    t, y = propagate(
        dynamics_3dof, y0, (0.0, 30.0), 0.01,
        control_hover, vehicle, method="rk4",
    )

    err_z = np.max(np.abs(y[:, 1] - z0))
    err_vz = np.max(np.abs(y[:, 3]))
    prop_burned = vehicle.m_wet - y[-1, 4]

    ok = True
    ok &= report("altitude held", err_z, 1e-6)
    ok &= report("vertical velocity zero", err_vz, 1e-8)
    print(f"         propellant burned hovering 30 s: {prop_burned:,.0f} kg "
          f"({100 * prop_burned / vehicle.m_prop_initial:.1f}% of landing load)")
    return ok, (t, y)


# ======================================================================
# TEST 3 - Mass depletion against the closed-form solution
# ======================================================================
def test_mass_flow():
    """
    Under constant thrust magnitude T, mass depletes linearly:
        m(t) = m0 - (T / (Isp * g0)) * t

    Verifies the Tsiolkovsky mass-flow implementation and the placement
    of g0 (a very common off-by-9.81 bug).
    """
    print("\nTEST 3 - Mass depletion vs. closed-form Tsiolkovsky flow")
    vehicle = Vehicle()

    T = 0.6 * vehicle.T_max
    y0 = np.array([0.0, 20_000.0, 0.0, -200.0, vehicle.m_wet])

    t, y = propagate(
        dynamics_3dof, y0, (0.0, 10.0), 0.01,
        control_constant(0.0, T), vehicle, method="rk4",
    )

    mdot_exact = T / (vehicle.isp * G0)
    m_exact = vehicle.m_wet - mdot_exact * t
    err_m = np.max(np.abs(y[:, 4] - m_exact))

    ok = report("mass profile", err_m, 1e-6)
    print(f"         mass flow rate: {mdot_exact:,.1f} kg/s "
          f"({mdot_exact / vehicle.n_engines:,.1f} kg/s per engine)")
    return ok, None


# ======================================================================
# TEST 4 - Convergence order study
# ======================================================================
def test_convergence_order():
    """
    Measure the empirical order of accuracy of each integrator.

    Halving the step size should reduce global error by:
        Euler (1st order) : factor of 2   -> slope 1 on a log-log plot
        RK4   (4th order) : factor of 16  -> slope 4 on a log-log plot

    Uses a nonlinear reference case (thrust with variable mass) so the
    result is not artificially exact.

    The reference is the *closed-form* solution, not a finely-stepped
    numerical one. Integrating a reference at dt = 1e-4 takes 100,000 steps
    and accumulates ~3e-9 of round-off — an order of magnitude worse than
    RK4 at dt = 0.125, which would put a floor under the measurement and
    drag the apparent order down to ~1.6. Step sizes are likewise chosen to
    stay in the asymptotic regime and above the float64 noise floor.
    """
    print("\nTEST 4 - Empirical convergence order")
    vehicle = Vehicle()

    T = 0.7 * vehicle.T_max
    Tx, Tz = 0.15 * T, T
    ctrl = control_constant(Tx, Tz)
    y0 = np.array([0.0, 10_000.0, 50.0, -250.0, vehicle.m_wet])
    t_end = 10.0

    z_reference = exact_constant_thrust(
        np.array([t_end]), y0, Tx, Tz, vehicle
    )[-1, 1]

    steps = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    results = {}
    ok = True

    for method in ("euler", "rk4"):
        errors = []
        for dt in steps:
            _, y = propagate(
                dynamics_3dof, y0, (0.0, t_end), dt, ctrl, vehicle, method=method
            )
            errors.append(abs(y[-1, 1] - z_reference))
        errors = np.array(errors)
        results[method] = errors

        # Fit slope on a log-log plot -> empirical order of accuracy
        slope = np.polyfit(np.log(steps), np.log(errors), 1)[0]
        expected = 1.0 if method == "euler" else 4.0
        passed = abs(slope - expected) < 0.35
        ok &= passed
        tag = PASS if passed else FAIL
        print(f"  {tag} {method.upper():<6} measured order = {slope:.2f}  "
              f"(theoretical {expected:.0f})")

    # ---- Plot -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(steps, results["euler"], "o-", linewidth=2, markersize=8,
              label="Euler (measured)")
    ax.loglog(steps, results["rk4"], "s-", linewidth=2, markersize=8,
              label="RK4 (measured)")
    ax.loglog(steps, results["euler"][0] * (steps / steps[0]) ** 1,
              "k--", alpha=0.5, label=r"$O(\Delta t)$ reference")
    ax.loglog(steps, results["rk4"][0] * (steps / steps[0]) ** 4,
              "k:", alpha=0.5, label=r"$O(\Delta t^4)$ reference")
    ax.set_xlabel(r"Time step $\Delta t$ [s]")
    ax.set_ylabel("Absolute error in final altitude [m]")
    ax.set_title("Integrator Convergence Order Verification")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(os.path.join(RESULTS, "day2_convergence_order.png"), dpi=150)
    print("         plot -> results/day2_convergence_order.png")
    plt.close()

    # Report the efficiency headline the guide asks you to read off the plot.
    ratio = results["euler"][0] / results["rk4"][0]
    print(f"         at dt = {steps[0]} s, RK4 is {ratio:,.0f}x more accurate "
          f"than Euler for 4x the work")
    return ok, None


# ======================================================================
# Trajectory figure
# ======================================================================
def plot_trajectories(freefall_data, hover_data):
    (t_ff, y_ff, z_exact) = freefall_data
    (t_hv, y_hv) = hover_data

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(t_ff, y_ff[:, 1] / 1000, linewidth=2, label="RK4")
    axes[0, 0].plot(t_ff, z_exact / 1000, "r--", linewidth=1.5, label="analytic")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Altitude [km]")
    axes[0, 0].set_title("Test 1: ballistic free-fall")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(t_ff[1:], np.abs(y_ff[1:, 1] - z_exact[1:]) + 1e-16,
                        linewidth=2, color="tab:red")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("|error| [m]")
    axes[0, 1].set_title("Test 1: integration error (machine precision)")
    axes[0, 1].grid(True, which="both", alpha=0.3)

    axes[1, 0].plot(t_hv, y_hv[:, 1], linewidth=2, color="tab:green")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Altitude [m]")
    axes[1, 0].set_title("Test 2: ideal hover holds altitude")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t_hv, y_hv[:, 4] / 1000, linewidth=2, color="tab:purple")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Mass [tonnes]")
    axes[1, 1].set_title("Test 2: propellant consumed while hovering")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(os.path.join(RESULTS, "day2_verification.png"), dpi=150)
    print("\nTrajectory figure -> results/day2_verification.png")
    plt.close()


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 2 - DYNAMICS AND INTEGRATOR VERIFICATION")
    print("=" * 70)
    print(Vehicle().summary())

    ok1, ff = test_freefall()
    ok2, hv = test_hover()
    ok3, _ = test_mass_flow()
    ok4, _ = test_convergence_order()

    plot_trajectories(ff, hv)

    all_ok = all([ok1, ok2, ok3, ok4])
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
