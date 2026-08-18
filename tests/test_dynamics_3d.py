"""
Verification of the 3-D rigid-body dynamics.

Groups:
    1. Zero gimbal deflection produces zero torque
    2. Torque-free motion conserves angular momentum as a VECTOR in the
       inertial frame -- Poinsot's theorem
    3. Torque-free motion conserves rotational kinetic energy
    4. Gimbal torque matches the hand-derived cross product, and roll torque
       is identically zero at every deflection
    5. Free-fall matches the closed-form projectile equation
    6. Axisymmetric roll decoupling, checked through the real derivative and
       shown to break when axisymmetry is removed
    7. Mass depletion matches the Isp formula
    8. Reduction to the planar case -- against an independent derivation, and
       against Day 5's dynamics_6dof

Run:  python tests/test_dynamics_3d.py
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.dynamics_3d import (                                  # noqa: E402
    Vehicle3D, gimbal_force_and_torque_body, gyroscopic_term,
    dynamics_3d_derivative, propagate_3d_dynamics, make_initial_state_3d,
    angular_momentum_inertial, rotational_kinetic_energy,
    attitude_from_pitch, pitch_from_attitude,
    G_EARTH, G0, IDX_MASS,
)
from src.dynamics_6dof import (                                # noqa: E402
    Vehicle6DoF, dynamics_6dof,
)

PASS = "[PASS]"
FAIL = "[FAIL]"
NOTE = "[NOTE]"


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  {tag} {name:<57}" + (f" {detail}" if detail else ""))
    return ok


def note(name, detail=""):
    print(f"  {NOTE} {name:<57}" + (f" {detail}" if detail else ""))


def zero_thrust(t, s):
    return (0.0, 0.0, 0.0)


# ======================================================================
def test_zero_gimbal_zero_torque():
    """An un-gimbaled engine on the centreline cannot torque the vehicle."""
    print("\nTEST 1 - Zero gimbal => zero torque")
    v = Vehicle3D()
    ok = True
    F, tau = gimbal_force_and_torque_body(v.T_max, 0.0, 0.0, v)

    ok &= report("thrust is pure +x at zero deflection",
                 np.allclose(F, [v.T_max, 0.0, 0.0], atol=1e-9),
                 f"F = {np.round(F, 6).tolist()}")
    ok &= report("torque is exactly zero at zero deflection",
                 float(np.abs(tau).max()) < 1e-9,
                 f"|tau|max = {float(np.abs(tau).max()):.2e}")

    # The force and the moment arm are collinear, so this has to hold at every
    # thrust level -- not just the one that happens to be tested above.
    worst = max(
        float(np.abs(gimbal_force_and_torque_body(T, 0.0, 0.0, v)[1]).max())
        for T in np.linspace(0.0, v.T_max, 25))
    ok &= report("still zero across the whole throttle range",
                 worst < 1e-9, f"worst = {worst:.2e}")
    return ok


def test_angular_momentum_conservation():
    """Torque-free tumbling conserves L as a vector in the inertial frame."""
    print("\nTEST 2 - Angular momentum conservation (Poinsot)")
    v = Vehicle3D()
    ok = True
    s0 = make_initial_state_3d(omega=(0.3, 0.5, 0.1), vehicle=v)
    t, hist = propagate_3d_dynamics(s0, zero_thrust, (0.0, 15.0), 0.01, v)

    L = np.array([angular_momentum_inertial(s, v) for s in hist])
    L0 = L[0]
    mag = float(np.linalg.norm(L0))
    dev = float(np.linalg.norm(L - L0, axis=1).max())

    # Deliberately a *relative* tolerance. np.allclose would apply its default
    # rtol of 1e-5 to a quantity of order 1e7 and quietly accept a drift of a
    # hundred units while an atol of 1e-2 sat in the call looking strict.
    ok &= report("L conserved as a vector, relative drift < 1e-8",
                 dev / mag < 1e-8,
                 f"max|dL| = {dev:.3e} on |L| = {mag:.3e} "
                 f"({dev / mag:.2e} relative)")

    # The direction is the part Poinsot is about: the body-frame rate precesses
    # the whole time, so a test that only checked |L| could pass with the
    # rotation wired wrong.
    body_swing = float(np.degrees(np.arccos(np.clip(np.dot(
        hist[:, 10:13] / np.linalg.norm(hist[:, 10:13], axis=1, keepdims=True),
        hist[0, 10:13] / np.linalg.norm(hist[0, 10:13])), -1.0, 1.0)).max()))
    ok &= report("body-frame omega really does precess (test has teeth)",
                 body_swing > 5.0,
                 f"omega swings {body_swing:.1f} deg in body axes")

    L_dir = float(np.degrees(np.arccos(np.clip(
        (L @ L0) / (np.linalg.norm(L, axis=1) * mag), -1.0, 1.0)).max()))
    ok &= report("L direction fixed in the inertial frame",
                 L_dir < 1e-4, f"max direction change = {L_dir:.2e} deg")
    return ok


def test_rotational_energy_conservation():
    """Torque-free motion does no work on the rotational degrees of freedom."""
    print("\nTEST 3 - Rotational kinetic energy conservation")
    v = Vehicle3D()
    ok = True
    s0 = make_initial_state_3d(omega=(0.2, 0.4, 0.15), vehicle=v)
    t, hist = propagate_3d_dynamics(s0, zero_thrust, (0.0, 15.0), 0.01, v)

    ke = np.array([rotational_kinetic_energy(s, v) for s in hist])
    rel = float(np.abs(ke - ke[0]).max() / ke[0])
    ok &= report("rotational KE conserved to < 1e-9 relative",
                 rel < 1e-9,
                 f"KE0 = {ke[0]:,.1f} J, max relative drift = {rel:.2e}")
    return ok


def test_gimbal_torque_formula():
    """Torque matches r x F worked out by hand, and never has an x component."""
    print("\nTEST 4 - Gimbal torque formula")
    v = Vehicle3D()
    ok = True
    F, tau = gimbal_force_and_torque_body(
        5.0e6, np.radians(5.0), np.radians(3.0), v)
    expected = np.array([0.0, v.L_engine * F[2], -v.L_engine * F[1]])

    ok &= report("torque matches the hand-derived cross product",
                 np.allclose(tau, expected, atol=1e-9),
                 f"|diff| = {float(np.abs(tau - expected).max()):.2e}")

    # Test 4 in the guide checks one deflection pair. The claim being made is
    # about every pair, so sweep the full square -- this is the model's
    # honest limitation and it is worth pinning down rather than sampling.
    grid = np.linspace(-v.delta_max, v.delta_max, 41)
    worst_roll = max(
        abs(gimbal_force_and_torque_body(v.T_max, dy, dz, v)[1][0])
        for dy in grid for dz in grid)
    ok &= report("no roll torque at ANY deflection in the envelope",
                 worst_roll < 1e-9,
                 f"worst |tau_x| over {grid.size**2} pairs = {worst_roll:.2e}")
    note("this is a limitation, not a result",
         "one effective engine cannot roll; real vehicles use several")
    return ok


def test_free_fall():
    """Zero thrust leaves nothing but gravity."""
    print("\nTEST 5 - Free-fall matches the closed form")
    v = Vehicle3D()
    ok = True
    z0 = 1000.0
    s0 = make_initial_state_3d(pos=(0.0, 0.0, z0), vehicle=v)
    t, hist = propagate_3d_dynamics(s0, zero_thrust, (0.0, 5.0), 0.001, v)

    err = abs(hist[-1, 2] - (z0 - 0.5 * G_EARTH * t[-1] ** 2))
    ok &= report("altitude matches z0 - 0.5*g*t^2",
                 err < 1e-9, f"error = {err:.3e} m")
    ok &= report("mass unchanged with the engine off",
                 abs(hist[-1, IDX_MASS] - v.m_wet) < 1e-12)
    ok &= report("no lateral motion appears from nowhere",
                 float(np.abs(hist[:, [0, 1]]).max()) < 1e-12)
    return ok


def test_axisymmetric_roll_decoupling():
    """
    Roll acceleration depends only on roll torque -- and only because the body
    is axisymmetric.

    Checked through `dynamics_3d_derivative` rather than by recomputing the
    formula alongside it. A test that re-derives the thing it is testing
    passes whatever the code does.
    """
    print("\nTEST 6 - Axisymmetric roll decoupling")
    v = Vehicle3D()
    ok = True
    omega = (0.8, 2.0, 1.5)          # large transverse rates, and a real roll
    s = make_initial_state_3d(omega=omega, vehicle=v)

    d = dynamics_3d_derivative(s, v.T_max, np.radians(9.0), np.radians(-6.0), v)
    ok &= report("roll acceleration is zero under any gimbal command",
                 abs(d[10]) < 1e-12, f"domega_x = {d[10]:.3e} rad/s^2")

    gyro = gyroscopic_term(np.array(omega), v.I_body)
    ok &= report("the roll-axis gyroscopic term is exactly zero (I_yy = I_zz)",
                 abs(gyro[0]) < 1e-9, f"gyro_x = {gyro[0]:.2e}")
    ok &= report("but the transverse components are not (test has teeth)",
                 float(np.abs(gyro[1:]).min()) > 1e6,
                 f"gyro_y, gyro_z = {gyro[1]:.3e}, {gyro[2]:.3e}")

    # The decoupling is a property of the inertia tensor, not of the code.
    # Break axisymmetry and it must disappear, otherwise the check above is
    # passing for the wrong reason.
    v_tri = Vehicle3D(I_yaw=1.3 * Vehicle3D().I_pitch_yaw)
    gyro_tri = gyroscopic_term(np.array(omega), v_tri.I_body)
    ok &= report("and it disappears when axisymmetry is removed",
                 abs(gyro_tri[0]) > 1e6, f"gyro_x = {gyro_tri[0]:.3e}")
    return ok


def test_mass_depletion():
    """Constant thrust burns propellant at exactly T/(Isp g0)."""
    print("\nTEST 7 - Mass depletion matches the Isp formula")
    v = Vehicle3D()
    ok = True
    T = v.T_max
    s0 = make_initial_state_3d(vehicle=v)
    t, hist = propagate_3d_dynamics(
        s0, lambda tt, ss: (T, 0.0, 0.0), (0.0, 5.0), 0.001, v)

    mdot = T / (v.isp * G0)
    expected = v.m_wet - mdot * t[-1]
    ok &= report("mass matches m0 - (T/(Isp*g0))*t",
                 abs(hist[-1, IDX_MASS] - expected) < 1e-6,
                 f"numerical = {hist[-1, IDX_MASS]:,.4f} kg, "
                 f"analytical = {expected:,.4f} kg")
    ok &= report("mass never falls below dry",
                 float(hist[:, IDX_MASS].min()) >= v.m_dry - 1e-9)
    note("burn rate", f"{mdot:,.0f} kg/s at full thrust; "
                      f"{v.m_prop_initial / mdot:.1f} s of propellant")
    return ok


# ----------------------------------------------------------------------
def _planar_reference(theta, m, T, delta, L, I):
    """
    A planar rocket derived from scratch, referencing neither model.

    Body long axis b = (sin theta, cos theta) with theta measured from
    vertical. The nozzle deflects the thrust by a further delta in the same
    sense, which is the convention Day 5 fixed with Tx = T sin(theta + delta).
    The engine sits at r = -L*b, below the centre of mass. The torque about
    the axis that increases theta is then (r x F)_y = r_z Fx - r_x Fz, and
    that is the only piece with any freedom left in it once the thrust
    convention is chosen.
    """
    b = np.array([np.sin(theta), np.cos(theta)])
    F = T * np.array([np.sin(theta + delta), np.cos(theta + delta)])
    r = -L * b
    tau = r[1] * F[0] - r[0] * F[1]
    return F / m - np.array([0.0, G_EARTH]), tau / I


def test_planar_reduction():
    """
    The 3-D model reduced to the plane, against two references.

    This is the load-bearing check of the day. Days 1 to 12 all rest on the
    planar model, so a 3-D layer that disagreed with it would put them in
    question -- and, as it turns out, one of them does disagree.
    """
    print("\nTEST 8 - Reduction to the planar case")
    v3, v5 = Vehicle3D(), Vehicle6DoF()
    ok = True
    T = 0.7 * v3.T_max

    # -- 8a: against a derivation written from vectors, nothing shared -----
    worst_a, worst_alpha = 0.0, 0.0
    for th in np.radians([0.0, 15.0, 30.0, 70.0, 90.0, 110.0]):
        for dl in np.radians([-12.0, -5.0, 0.0, 5.0, 12.0]):
            a_ref, alpha_ref = _planar_reference(
                th, v3.m_wet, T, dl, v3.L_engine, v3.I_pitch_yaw)
            s = make_initial_state_3d(
                pos=(0.0, 0.0, 2000.0), vel=(-30.0, 0.0, -80.0),
                quat=attitude_from_pitch(th), omega=(0.0, 0.02, 0.0),
                vehicle=v3)
            d = dynamics_3d_derivative(s, T, dl, 0.0, v3)
            worst_a = max(worst_a, abs(d[3] - a_ref[0]), abs(d[5] - a_ref[1]))
            worst_alpha = max(worst_alpha, abs(d[11] - alpha_ref))

    ok &= report("translational accel matches the independent derivation",
                 worst_a < 1e-9, f"worst error = {worst_a:.2e} m/s^2")
    ok &= report("angular accel matches the independent derivation",
                 worst_alpha < 1e-12,
                 f"worst error = {worst_alpha:.2e} rad/s^2")

    # -- 8b: planar control keeps the motion planar -----------------------
    s0 = make_initial_state_3d(
        pos=(0.0, 0.0, 2000.0), vel=(-30.0, 0.0, -80.0),
        quat=attitude_from_pitch(np.radians(70.0)), vehicle=v3)
    t, hist = propagate_3d_dynamics(
        s0, lambda tt, ss: (T, np.radians(5.0), 0.0), (0.0, 5.0), 0.005, v3)

    ok &= report("out-of-plane position stays at zero",
                 float(np.abs(hist[:, 1]).max()) < 1e-12,
                 f"max |y| = {float(np.abs(hist[:, 1]).max()):.2e} m")
    ok &= report("out-of-plane velocity stays at zero",
                 float(np.abs(hist[:, 4]).max()) < 1e-12)
    ok &= report("roll rate stays at zero",
                 float(np.abs(hist[:, 10]).max()) < 1e-12)
    ok &= report("yaw rate stays at zero",
                 float(np.abs(hist[:, 12]).max()) < 1e-12)

    # -- 8c: against Day 5's dynamics_6dof --------------------------------
    # Same vehicle, same state, same control, one derivative each.
    theta, delta = np.radians(70.0), np.radians(5.0)
    st5 = np.array([0.0, 2000.0, -30.0, -80.0, theta, 0.02, v5.m_wet])
    d5 = dynamics_6dof(0.0, st5, lambda tt, ss, vv: (T, delta), v5)
    s3 = make_initial_state_3d(
        pos=(0.0, 0.0, 2000.0), vel=(-30.0, 0.0, -80.0),
        quat=attitude_from_pitch(theta), omega=(0.0, 0.02, 0.0), vehicle=v3)
    d3 = dynamics_3d_derivative(s3, T, delta, 0.0, v3)

    ok &= report("translational accel agrees with Day 5",
                 abs(d3[3] - d5[2]) < 1e-9 and abs(d3[5] - d5[3]) < 1e-9,
                 f"dvx {d3[3]:.6f} vs {d5[2]:.6f}, "
                 f"dvz {d3[5]:.6f} vs {d5[3]:.6f}")
    ok &= report("mass flow agrees with Day 5",
                 abs(d3[13] - d5[6]) < 1e-9)

    alpha_ref = _planar_reference(
        theta, v3.m_wet, T, delta, v3.L_engine, v3.I_pitch_yaw)[1]
    ok &= report("angular accel: 3-D model agrees with the reference",
                 abs(d3[11] - alpha_ref) < 1e-12,
                 f"{d3[11]:.8f} vs {alpha_ref:.8f} rad/s^2")

    # This one is a measurement, not an assertion. See the note below.
    matches_day5 = abs(d3[11] - d5[5]) < 1e-9
    mirrors_day5 = abs(d3[11] + d5[5]) < 1e-9
    report("angular accel: Day 5 disagrees, by an exact sign flip",
           (not matches_day5) and mirrors_day5,
           f"3-D {d3[11]:+.8f} vs Day 5 {d5[5]:+.8f} rad/s^2")
    note("KNOWN DEFECT IN dynamics_6dof, not in this file",
         "")
    note("  Day 5 uses tau = +T*L*sin(delta) with Tx = T*sin(theta+delta).",
         "")
    note("  Those two conventions are not compatible: given that thrust tilt,",
         "")
    note("  r x F comes out as -T*L*sin(delta). Verified three ways -- the",
         "")
    note("  vector derivation above, this 3-D model, and the physical check",
         "")
    note("  that deflecting the nozzle toward +x must push the tail toward +x",
         "")
    note("  and therefore the nose toward -x. Day 5 has the vehicle tilt the",
         "")
    note("  same way the thrust already points, which removes the",
         "")
    note("  non-minimum-phase behaviour a gimballed rocket actually has.",
         "")
    note("  landing_flip.py carries the same pairing, so the optimiser and the",
         "")
    note("  simulator agree with each other -- which is why twelve days of",
         "")
    note("  tests, every one of them comparing those two, never saw it.",
         "")
    note("  Left unfixed today: both are load-bearing for Days 5-12.",
         "")
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 14 - 3-D RIGID-BODY DYNAMICS VERIFICATION")
    print("=" * 70)
    results = [
        test_zero_gimbal_zero_torque(),
        test_angular_momentum_conservation(),
        test_rotational_energy_conservation(),
        test_gimbal_torque_formula(),
        test_free_fall(),
        test_axisymmetric_roll_decoupling(),
        test_mass_depletion(),
        test_planar_reduction(),
    ]
    print("\n" + "=" * 70)
    ok = all(results)
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
