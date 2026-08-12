"""
Verification of 6-DoF planar dynamics.

Tests:
    1. Ballistic free-fall with constant spin: closed-form check
    2. Angular impulse from a constant gimbal: delta_omega = tau t / I
    3. Simulated bang-bang flip: rotates horizontal toward vertical
    4. Gimbal symmetry: opposite deflection gives opposite rotation

Run:  python tests/test_6dof.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.dynamics_6dof import (                     # noqa: E402
    Vehicle6DoF,
    dynamics_6dof,
    G_EARTH,
    control_zero_6dof,
    control_constant_gimbal,
    control_flip_bang_bang,
)
from src.integrators import propagate                # noqa: E402
from tests.test_dynamics import PASS, FAIL           # noqa: E402


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name:<48}"
    if detail:
        msg += f" {detail}"
    print(msg)
    return bool(ok)


def _run(control, y0, t_end, dt, vehicle):
    return propagate(
        lambda t, y, *a: dynamics_6dof(t, y, control, vehicle),
        y0, (0.0, t_end), dt, method="rk4",
    )


# ======================================================================
def test_ballistic_rotation():
    """
    No thrust: translation is ballistic and rotation is torque-free, so theta
    advances linearly and omega is conserved. Both are exact for RK4 here —
    the position is quadratic in t and the attitude is linear.
    """
    print("\nTEST 1 - Ballistic free-fall with constant spin")
    veh = Vehicle6DoF()

    omega0, vz0, z0, theta0 = 0.1, -50.0, 1000.0, np.pi / 4
    t_end = 5.0
    y0 = np.array([0.0, z0, 0.0, vz0, theta0, omega0, veh.m_wet])
    _, y = _run(control_zero_6dof, y0, t_end, 0.01, veh)

    theta_exp = theta0 + omega0 * t_end
    z_exp = z0 + vz0 * t_end - 0.5 * G_EARTH * t_end ** 2

    ok = True
    ok &= report("theta advances linearly",
                 abs(y[-1, 4] - theta_exp) < 1e-9,
                 f"expected {np.degrees(theta_exp):.3f} deg, "
                 f"got {np.degrees(y[-1, 4]):.3f} deg")
    ok &= report("altitude matches ballistic kinematics",
                 abs(y[-1, 1] - z_exp) < 1e-6,
                 f"error {abs(y[-1, 1] - z_exp):.2e} m")
    ok &= report("omega conserved (no torque)",
                 abs(y[-1, 5] - omega0) < 1e-12,
                 f"omega_f = {y[-1, 5]:.6f} rad/s")
    ok &= report("mass unchanged (no thrust)",
                 abs(y[-1, 6] - veh.m_wet) < 1e-9)
    return ok


# ======================================================================
def test_torque_impulse():
    """
    Constant thrust and gimbal give constant torque, so omega should grow
    linearly at tau/I. Inertia is constant in this model, so unlike the mass
    flow there is nothing to approximate — this should be exact.
    """
    print("\nTEST 2 - Angular impulse from a constant gimbal")
    veh = Vehicle6DoF()

    T_test = veh.T_min
    delta_test = np.radians(5.0)
    t_end = 2.0

    y0 = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, veh.m_wet])
    _, y = _run(control_constant_gimbal(T_test, delta_test), y0, t_end, 0.001, veh)

    tau = T_test * veh.L_engine * np.sin(delta_test)
    d_omega_exp = tau * t_end / veh.I_pitch
    d_omega = y[-1, 5]
    rel = abs(d_omega - d_omega_exp) / abs(d_omega_exp)

    ok = report("angular impulse matches tau*t/I",
                rel < 1e-9,
                f"expected {np.degrees(d_omega_exp):.4f} deg/s, "
                f"got {np.degrees(d_omega):.4f} deg/s ({rel:.2e} rel)")

    # theta should be the double integral: 0.5 * (tau/I) * t^2
    theta_exp = 0.5 * (tau / veh.I_pitch) * t_end ** 2
    ok &= report("theta is the double integral of torque",
                 abs(y[-1, 4] - theta_exp) < 1e-9,
                 f"expected {np.degrees(theta_exp):.4f} deg, "
                 f"got {np.degrees(y[-1, 4]):.4f} deg")
    return ok


# ======================================================================
def test_simulated_flip():
    """
    Bang-bang gimbal from a belly-flop attitude.

    Timings are derived from the vehicle rather than assumed. Control authority
    here is alpha_max = 94.7 deg/s^2, so a rest-to-rest 90 degree rotation takes
    roughly 2*sqrt(90/alpha) ~ 2 s. Allowing 9 s of gimbal, as one might expect
    from a slower vehicle, spins it through two full revolutions.

    Note what a *symmetric* bang-bang cannot do. Thrust is state-dependent here
    (it tracks m*g/cos(theta)), so the accelerating and decelerating phases run
    at different torque and equal durations do not cancel. Empirically the
    vehicle either arrives near vertical still rotating at ~33 deg/s (t1 = 0.9 s)
    or arrests its rotation having overshot to -91 degrees (t1 = 1.6 s). Landing
    both attitude and rate needs asymmetric timing that depends on the whole
    trajectory — which is the argument for solving rather than scheduling, and
    is what landing_flip.py does.
    """
    print("\nTEST 3 - Simulated bang-bang flip")
    veh = Vehicle6DoF()

    theta0 = np.pi / 2
    y0 = np.array([500.0, 2000.0, -30.0, -60.0, theta0, 0.0, veh.m_wet])

    t_flip_start, t1 = 0.5, 0.9
    t_flip_mid = t_flip_start + t1
    t_end = t_flip_start + 2 * t1

    def ctrl(t, s, v):
        return control_flip_bang_bang(t, s, v, t_flip_start, t_flip_mid)

    _, y = _run(ctrl, y0, t_end, 0.002, veh)
    theta_f, omega_f = y[-1, 4], y[-1, 5]
    omega_peak = float(np.max(np.abs(y[:, 5])))

    ok = True
    ok &= report("rotated toward vertical",
                 abs(theta_f) < 0.25 * abs(theta0),
                 f"{np.degrees(theta0):.0f} deg -> {np.degrees(theta_f):.1f} deg")
    ok &= report("reverse gimbal slows the rotation",
                 abs(omega_f) < 0.6 * omega_peak,
                 f"peak {np.degrees(omega_peak):.1f} -> final "
                 f"{np.degrees(omega_f):.1f} deg/s")
    ok &= report("still airborne", y[-1, 1] > 0,
                 f"altitude {y[-1, 1]:,.0f} m")
    ok &= report("mass decreased", y[-1, 6] < veh.m_wet,
                 f"{veh.m_wet:,.0f} -> {y[-1, 6]:,.0f} kg")
    print(f"         residual rate {np.degrees(omega_f):.1f} deg/s is why an "
          f"open-loop schedule is not enough")
    return ok


# ======================================================================
def test_symmetry():
    """Mirror the gimbal and the rotation should mirror exactly."""
    print("\nTEST 4 - Gimbal symmetry")
    veh = Vehicle6DoF()

    T_test = veh.T_min
    delta_test = np.radians(10.0)
    y0 = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, veh.m_wet])

    _, y_pos = _run(control_constant_gimbal(T_test, delta_test), y0, 1.0, 0.001, veh)
    _, y_neg = _run(control_constant_gimbal(T_test, -delta_test), y0, 1.0, 0.001, veh)

    ok = report("opposite gimbal gives opposite omega",
                abs(y_pos[-1, 5] + y_neg[-1, 5]) < 1e-12 * max(abs(y_pos[-1, 5]), 1.0),
                f"+{np.degrees(y_pos[-1, 5]):.4f} vs "
                f"{np.degrees(y_neg[-1, 5]):.4f} deg/s")
    ok &= report("horizontal motion mirrors too",
                 abs(y_pos[-1, 0] + y_neg[-1, 0]) < 1e-9,
                 f"x: {y_pos[-1, 0]:+.4f} vs {y_neg[-1, 0]:+.4f} m")
    ok &= report("vertical motion is identical",
                 abs(y_pos[-1, 1] - y_neg[-1, 1]) < 1e-9,
                 f"z: {y_pos[-1, 1]:,.4f} m")
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 5 - 6-DoF ROTATIONAL DYNAMICS VERIFICATION")
    print("=" * 70)
    print(Vehicle6DoF().summary())

    results = [
        test_ballistic_rotation(),
        test_torque_impulse(),
        test_simulated_flip(),
        test_symmetry(),
    ]

    print("\n" + "=" * 70)
    ok = all(results)
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
