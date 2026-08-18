"""
Verification of the 3-D aerodynamics.

Groups:
    1. Angle round-trip: (alpha, beta, V) -> (u, v, w) -> (alpha, beta, V)
    2. Low-speed guard, and the disabled switch
    3. Purely axial flow: no side force, no lift, and no moment
    4. Purely broadside flow: exactly Day 6's belly values
    5. Lift curve shape
    6. Side force opposes sideslip, in both directions
    7. Moment matches the hand-derived cross product, and never rolls
    8. Dynamic pressure: doubling the speed quadruples the force, exactly
    9. Composition: the combined model is Day 14 plus this wrench and
       nothing else, and planar motion stays planar
   10. Reduction to Day 6 -- where it holds, and where it does not

Run:  python tests/test_aero_3d.py
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.aero_3d import (                                      # noqa: E402
    AeroConfig3D, aero_angles, angles_to_relative_wind,
    effective_area_and_Cd, lift_coefficient, aero_force_body,
    aero_moment_body, aero_force_and_moment_body, relative_wind_body,
    dynamics_3d_with_aero_derivative, propagate_3d_with_aero,
    angle_history, static_margin_sign,
)
from src.dynamics_3d import (                                  # noqa: E402
    Vehicle3D, dynamics_3d_derivative, make_initial_state_3d,
    attitude_from_pitch,
)
from src.aero import (                                         # noqa: E402
    AeroConfig, aero_force as aero_force_planar, drag_area,
)
from src.quaternion import quat_to_rotmatrix                   # noqa: E402

PASS = "[PASS]"
FAIL = "[FAIL]"
NOTE = "[NOTE]"
ALT = 3000.0


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  {tag} {name:<55}" + (f" {detail}" if detail else ""))
    return ok


def note(text, detail=""):
    print(f"  {NOTE} {text:<55}" + (f" {detail}" if detail else ""))


# ======================================================================
def test_angle_round_trip():
    """The two angles and the speed carry all of the relative wind."""
    print("\nTEST 1 - Angle round-trip")
    ok = True
    worst = 0.0
    for a_deg in (-80.0, -30.0, 0.0, 25.0, 60.0, 89.0):
        for b_deg in (-40.0, -5.0, 0.0, 5.0, 40.0):
            a, b, V = np.radians(a_deg), np.radians(b_deg), 120.0
            uvw = angles_to_relative_wind(a, b, V)
            a2, b2, V2, _ = aero_angles(uvw, v_min=1.0)
            worst = max(worst, abs(a2 - a), abs(b2 - b), abs(V2 - V) / V)
    ok &= report("alpha, beta and V survive the round trip",
                 worst < 1e-12, f"worst error = {worst:.2e}")

    # beta is out of the pitch plane by construction, so planar wind must give
    # exactly zero -- the property every one of Days 1-12 relied on implicitly.
    betas = [abs(aero_angles([u, 0.0, w], 1.0)[1])
             for u in (-100.0, -10.0, 10.0, 100.0)
             for w in (-100.0, 0.0, 100.0)]
    ok &= report("planar wind gives exactly zero sideslip",
                 max(betas) < 1e-15, f"worst |beta| = {max(betas):.2e}")

    # And the total off-axis angle collapses onto |alpha| when beta is zero.
    worst2 = max(abs(aero_angles([u, 0.0, w], 1.0)[3]
                     - abs(aero_angles([u, 0.0, w], 1.0)[0]))
                 for u in (-100.0, -10.0, 10.0, 100.0)
                 for w in (-90.0, 30.0, 90.0))
    ok &= report("angle-off-axis reduces to |alpha| in the plane",
                 worst2 < 1e-12, f"worst error = {worst2:.2e}")
    return ok


def test_low_speed_guard():
    """A near-zero wind has no direction, so it gets no force."""
    print("\nTEST 2 - Low-speed guard and the disable switch")
    cfg = AeroConfig3D()
    ok = True
    F = aero_force_body([0.5, -0.2, 0.3], ALT, cfg)
    ok &= report("force is exactly zero below v_min",
                 float(np.abs(F).max()) == 0.0)
    ok &= report("and finite", bool(np.isfinite(F).all()))
    ok &= report("exactly zero wind does not divide by zero",
                 float(np.abs(aero_force_body([0.0, 0.0, 0.0], ALT,
                                              cfg)).max()) == 0.0)
    off = AeroConfig3D(enabled=False)
    ok &= report("disabled config produces no force at any speed",
                 float(np.abs(aero_force_body([-120.0, 8.0, 40.0], ALT,
                                              off)).max()) == 0.0)
    return ok


def test_purely_axial_flow():
    """Wind straight down the long axis: drag only, and no moment at all."""
    print("\nTEST 3 - Purely axial flow")
    cfg = AeroConfig3D()
    ok = True

    # Tolerances here are relative to the drag, not absolute. Tail-on flow is
    # alpha = pi, where sin(2 alpha) and sin(alpha) are a few times 1e-16
    # rather than zero, and an absolute 1e-12 on a force of order 1e5 N is
    # asking floating point for more than it has.
    for u, label in ((150.0, "nose-on"), (-150.0, "tail-on")):
        F = aero_force_body([u, 0.0, 0.0], ALT, cfg)
        tau = aero_moment_body(F, cfg)
        mag = float(np.linalg.norm(F))
        ok &= report(f"{label}: no side force, no lift",
                     max(abs(F[1]), abs(F[2])) / mag < 1e-15,
                     f"|F_perp| / |F| = {max(abs(F[1]), abs(F[2])) / mag:.2e}")
        ok &= report(f"{label}: drag opposes the wind exactly",
                     F[0] * u < 0.0,
                     f"F = {np.round(F / 1e3, 3).tolist()} kN")
        ok &= report(f"{label}: aero moment is zero",
                     float(np.abs(tau).max()) / (mag * cfg.x_cp) < 1e-15,
                     f"|tau| / (|F| x_cp) = "
                     f"{float(np.abs(tau).max()) / (mag * cfg.x_cp):.2e}")
    note("same geometry as Day 14's zero-gimbal result",
         "a force collinear with the offset arm has no moment about it")

    ok &= report("nose-on and tail-on present the same area",
                 abs(effective_area_and_Cd(np.pi, cfg)[0]
                     - effective_area_and_Cd(0.0, cfg)[0]) < 1e-12)
    return ok


def test_purely_broadside_flow():
    """Broadside is Day 6's belly-flop case, and must reproduce it exactly."""
    print("\nTEST 4 - Purely broadside flow")
    cfg = AeroConfig3D()
    ok = True
    A, Cd = effective_area_and_Cd(np.pi / 2.0, cfg)
    ok &= report("effective area equals A_belly", abs(A - cfg.A_belly) < 1e-12,
                 f"{A:.1f} m^2")
    ok &= report("effective Cd equals Cd_belly", abs(Cd - cfg.Cd_belly) < 1e-12)
    ok &= report("lift is zero at alpha = 90 deg",
                 abs(lift_coefficient(np.pi / 2.0, cfg)) < 1e-15)

    old = AeroConfig()
    ok &= report("matches Day 6's belly values exactly",
                 abs(A - old.A_belly) < 1e-12 and abs(Cd - old.Cd_belly) < 1e-12)
    ok &= report("A_belly / A_nose is the 7.1x this vehicle has",
                 abs(cfg.A_belly / cfg.A_nose - 7.0736) < 1e-3,
                 f"{cfg.A_belly / cfg.A_nose:.4f}x")
    return ok


def test_lift_curve_shape():
    """Zero nose-on, zero broadside, peak halfway between."""
    print("\nTEST 5 - Lift curve shape")
    cfg = AeroConfig3D()
    ok = True
    ok &= report("Cl(0) = 0", abs(lift_coefficient(0.0, cfg)) < 1e-15)
    ok &= report("Cl(90 deg) = 0",
                 abs(lift_coefficient(np.pi / 2.0, cfg)) < 1e-15)
    ok &= report("Cl(45 deg) = Cl_max",
                 abs(lift_coefficient(np.pi / 4.0, cfg) - cfg.Cl_max) < 1e-15)
    ok &= report("Cl is odd in alpha",
                 abs(lift_coefficient(0.4, cfg)
                     + lift_coefficient(-0.4, cfg)) < 1e-15)

    grid = np.radians(np.linspace(0.0, 90.0, 901))
    peak = grid[int(np.argmax([lift_coefficient(a, cfg) for a in grid]))]
    ok &= report("the peak really is at 45 deg",
                 abs(np.degrees(peak) - 45.0) < 0.06,
                 f"peak at {np.degrees(peak):.2f} deg")

    # Lift must be perpendicular to the wind, which is the one thing about it
    # that is derivable rather than chosen.
    worst = 0.0
    for a_deg in (10.0, 30.0, 45.0, 70.0):
        uvw = angles_to_relative_wind(np.radians(a_deg), 0.0, 120.0)
        pure = AeroConfig3D(Cd_nose=0.0, Cd_belly=0.0, Cy_beta=0.0)
        F = aero_force_body(uvw, ALT, pure)
        worst = max(worst, abs(float(F @ uvw)) / (np.linalg.norm(F) * 120.0))
    ok &= report("lift is perpendicular to the relative wind",
                 worst < 1e-12, f"worst |cos| = {worst:.2e}")
    return ok


def test_side_force_sign():
    """Sideslip has to be opposed, or the vehicle has no yaw stiffness."""
    print("\nTEST 6 - Side force opposes sideslip")
    cfg = AeroConfig3D()
    ok = True
    for sign, label in ((+1.0, "positive"), (-1.0, "negative")):
        uvw = angles_to_relative_wind(0.0, sign * np.radians(15.0), 120.0)
        F = aero_force_body(uvw, ALT, cfg)
        ok &= report(f"{label} sideslip produces a restoring side force",
                     F[1] * sign < 0.0, f"F_y = {F[1] / 1e3:+.1f} kN")
    zero = aero_force_body(angles_to_relative_wind(0.3, 0.0, 120.0), ALT, cfg)
    ok &= report("zero sideslip produces exactly zero side force",
                 abs(zero[1]) < 1e-12)
    return ok


def test_moment_formula():
    """The moment is r x F, with everything that implies."""
    print("\nTEST 7 - Moment formula")
    cfg = AeroConfig3D()
    ok = True
    F = np.array([1.2e5, -3.4e4, 8.7e4])
    tau = aero_moment_body(F, cfg)
    expected = np.array([0.0, cfg.x_cp * F[2], -cfg.x_cp * F[1]])
    ok &= report("matches the hand-derived cross product",
                 np.allclose(tau, expected, atol=1e-9),
                 f"|diff| = {float(np.abs(tau - expected).max()):.2e}")

    # Same limitation as the engine: an offset along the long axis cannot
    # produce a moment about that axis, whatever the force is.
    rng = np.random.default_rng(14)
    worst = max(abs(aero_moment_body(rng.normal(scale=1e5, size=3), cfg)[0])
                for _ in range(500))
    ok &= report("never produces a roll moment, for any force",
                 worst < 1e-9, f"worst |tau_x| over 500 = {worst:.2e}")
    note("so aero cannot roll this vehicle either",
         "neither can the gimbal -- the roll axis has no authority at all")
    return ok


def test_dynamic_pressure_scaling():
    """q ~ V^2, and the coefficients depend on direction only."""
    print("\nTEST 8 - Dynamic-pressure scaling")
    cfg = AeroConfig3D()
    ok = True
    base = np.array([-90.0, 12.0, 35.0])
    F1 = aero_force_body(base, ALT, cfg)
    F2 = aero_force_body(base * 2.0, ALT, cfg)
    ratio = float(np.linalg.norm(F2) / np.linalg.norm(F1))
    ok &= report("doubling the speed quadruples the force, exactly",
                 abs(ratio - 4.0) < 1e-9, f"ratio = {ratio:.12f}")
    ok &= report("and does not rotate it",
                 float(np.abs(F2 / 4.0 - F1).max()) < 1e-6,
                 "direction is a function of the angles alone")

    # Density enters linearly, which is the other half of q.
    r = (np.linalg.norm(aero_force_body(base, 0.0, cfg))
         / np.linalg.norm(aero_force_body(base, 8500.0, cfg)))
    ok &= report("force scales with density, e-fold over one scale height",
                 abs(r - np.e) < 2e-2, f"sea level / 8500 m = {r:.4f}")
    return ok


def test_composition():
    """
    Day 15 is a layer on Day 14, not a fork of it.

    The combined derivative must equal Day 14's derivative plus exactly the
    aerodynamic wrench -- otherwise two copies of gravity, the frame rotation
    and Euler's equations start drifting apart, which is a class of bug this
    project has already paid for once.
    """
    print("\nTEST 9 - Composition onto Day 14")
    v, cfg = Vehicle3D(), AeroConfig3D()
    ok = True
    s = make_initial_state_3d(
        pos=(0.0, 0.0, 3000.0), vel=(-40.0, 6.0, -110.0),
        quat=attitude_from_pitch(np.radians(65.0)), omega=(0.1, 0.2, -0.05),
        vehicle=v)
    T, dy, dz = 0.7 * v.T_max, np.radians(4.0), np.radians(-2.0)

    d14 = dynamics_3d_derivative(s, T, dy, dz, v)
    d15 = dynamics_3d_with_aero_derivative(s, T, dy, dz, v, cfg)
    F, tau = aero_force_and_moment_body(s[3:6], (0.0, 0.0, 0.0), s[6:10],
                                        s[2], cfg)
    R = quat_to_rotmatrix(s[6:10])
    expect = d14.copy()
    expect[3:6] += (R @ F) / s[13]
    expect[10:13] += tau / v.I_diag
    ok &= report("combined = Day 14 + the aero wrench, exactly",
                 float(np.abs(d15 - expect).max()) < 1e-9,
                 f"worst |diff| = {float(np.abs(d15 - expect).max()):.2e}")

    disabled = AeroConfig3D(enabled=False)
    ok &= report("aero disabled reproduces Day 14 bit for bit",
                 float(np.abs(
                     dynamics_3d_with_aero_derivative(
                         s, T, dy, dz, v, disabled) - d14).max()) == 0.0)

    # Planar in, planar out -- with aero now in the loop.
    s0 = make_initial_state_3d(
        pos=(0.0, 0.0, 3000.0), vel=(-30.0, 0.0, -90.0),
        quat=attitude_from_pitch(np.radians(70.0)), vehicle=v)
    _, h = propagate_3d_with_aero(
        s0, lambda t, ss: (0.7 * v.T_max, np.radians(4.0), 0.0),
        (0.0, 8.0), 0.005, v, cfg)
    ok &= report("planar motion stays planar with aero on",
                 float(np.abs(h[:, 1]).max()) < 1e-12
                 and float(np.abs(h[:, 4]).max()) < 1e-12,
                 f"max |y| = {float(np.abs(h[:, 1]).max()):.2e} m")
    ok &= report("and picks up no roll or yaw rate",
                 float(np.abs(h[:, [10, 12]]).max()) < 1e-12)
    return ok


def test_reduction_to_day6():
    """
    Against Day 6, which is the check that matters.

    Days 6 to 12 all rest on `src/aero.py`. Where the two models must agree,
    they are asserted to agree; where they do not, the disagreement is measured
    and reported rather than tuned away.
    """
    print("\nTEST 10 - Reduction to Day 6's planar aero")
    ok = True
    # Drag alone, so the comparison is not confounded by the lift term.
    new = AeroConfig3D(Cl_max=0.0, Cy_beta=0.0)
    old = AeroConfig(Cl_max=0.0)

    def planar_pair(theta, vx, vz, cfg_new, cfg_old):
        R = quat_to_rotmatrix(attitude_from_pitch(theta))
        v_rel = R.T @ np.array([vx, 0.0, vz])
        F_new = R @ aero_force_body(v_rel, ALT, cfg_new)
        fx, fz = aero_force_planar(vx, vz, ALT, theta, cfg_old)
        return F_new, np.array([float(fx), 0.0, float(fz)])

    # -- where it must agree: the wind straight down, any attitude ---------
    worst = 0.0
    for th_deg in (0.0, 15.0, 30.0, 50.0, 70.0, 90.0):
        a, b = planar_pair(np.radians(th_deg), 0.0, -100.0, new, old)
        worst = max(worst, float(np.abs(a - b).max())
                    / max(float(np.linalg.norm(b)), 1.0))
    ok &= report("identical for vertical descent, at every pitch angle",
                 worst < 1e-12, f"worst relative error = {worst:.2e}")

    # -- and the reason: the blend angle coincides there -------------------
    worst_blend = 0.0
    for a in np.radians(np.linspace(0.0, 180.0, 61)):
        A, Cd = effective_area_and_Cd(a, new)
        worst_blend = max(worst_blend, abs(A * Cd - float(drag_area(a, old))))
    ok &= report("the blend formula itself is Day 6's, unchanged",
                 worst_blend < 1e-9, f"worst diff = {worst_blend:.2e}")

    # -- where it does not: any downrange velocity at all ------------------
    print()
    note("Day 6 blends area and Cd by pitch from VERTICAL. The angle that "
         "decides")
    note("how much vehicle the air sees is the angle to the relative WIND. "
         "Those")
    note("agree only when the wind is vertical, which is the case above. "
         "Day 6 is")
    note("inconsistent about it internally too -- it computes the "
         "wind-relative")
    note("angle for its lift term and then uses the vertical one for area "
         "and Cd.")
    print()
    print(f"    {'pitch':>6} {'vx':>6} {'vz':>6} | {'Day 6 CdA':>10} "
          f"{'3-D CdA':>10} {'ratio':>7}")
    flags = []
    for th_deg, vx, vz in ((70.0, 0.0, -100.0), (70.0, -30.0, -80.0),
                           (70.0, -60.0, -60.0), (30.0, -40.0, -90.0),
                           (90.0, -50.0, -70.0)):
        th = np.radians(th_deg)
        R = quat_to_rotmatrix(attitude_from_pitch(th))
        _, _, _, off = aero_angles(R.T @ np.array([vx, 0.0, vz]), new.v_min)
        A, Cd = effective_area_and_Cd(off, new)
        old_CdA = float(drag_area(th, old))
        flags.append(A * Cd / old_CdA)
        print(f"    {th_deg:6.0f} {vx:6.0f} {vz:6.0f} | {old_CdA:10.1f} "
              f"{A * Cd:10.1f} {A * Cd / old_CdA:7.3f}")
    ok &= report("the two disagree once the wind is not vertical",
                 min(flags) < 0.75,
                 f"worst ratio {min(flags):.3f} -- Day 6 overstates CdA by "
                 f"{1 / min(flags):.1f}x")
    note("Not fixed today. aero.py is load-bearing for Days 6-12.", "")

    # -- with lift included, and both terms, in the case that must agree ---
    full_new, full_old = AeroConfig3D(Cy_beta=0.0), AeroConfig()
    worst_full = 0.0
    for th_deg in (0.0, 15.0, 30.0, 50.0, 70.0, 90.0):
        a, b = planar_pair(np.radians(th_deg), 0.0, -100.0, full_new, full_old)
        worst_full = max(worst_full, float(np.abs(a - b).max())
                         / max(float(np.linalg.norm(b)), 1.0))
    ok &= report("drag AND lift together match Day 6 exactly, vertical flow",
                 worst_full < 1e-12,
                 f"worst relative error = {worst_full:.2e}")
    note("This one only passes because the guide's lift direction was wrong.",
         "")
    note("  It specifies [-w, 0, u], which at small angle of attack produces",
         "")
    note("  a lift that overwhelms the drag's own normal component and turns",
         "")
    note("  the vehicle AWAY from the wind -- making a centre of pressure aft",
         "")
    note("  of the centre of mass destabilising, which is backwards. Corrected",
         "")
    note("  to [w, 0, -u] here. Day 6 already had it right; see Test 11.", "")
    return ok


def test_weathervaning():
    """
    Does the assembled force and moment behave like a body in a wind?

    The individual pieces can each be right while the assembly points the
    vehicle the wrong way. This is the end-to-end check: a body with its centre
    of pressure aft of the centre of mass must turn to face the wind, and one
    with it forward must turn away.
    """
    print("\nTEST 11 - Weathervaning, the end-to-end check")
    v = Vehicle3D()
    ok = True

    # The sign of the moment at a perturbed angle of attack decides this, and
    # it needs no integration at all. alpha increases with the body pitch rate
    # (d alpha/dt = omega_y for a wind fixed in inertial space), so a restoring
    # moment is one with tau_y opposite in sign to alpha.
    for x_cp, restoring, label in ((5.0, True, "x_cp = +5 m (aft)"),
                                   (-5.0, False, "x_cp = -5 m (forward)")):
        cfg = AeroConfig3D(x_cp=x_cp)
        signs = []
        for a_deg in (-40.0, -15.0, -5.0, 5.0, 15.0, 40.0):
            F = aero_force_body(
                angles_to_relative_wind(np.radians(a_deg), 0.0, 150.0),
                8000.0, cfg)
            signs.append(aero_moment_body(F, cfg)[1] * np.radians(a_deg) < 0.0)
        ok &= report(f"{label}: {static_margin_sign(cfg).split(' (')[0]}",
                     all(s == restoring for s in signs),
                     f"{'restoring' if restoring else 'diverging'} at all six "
                     f"angles of attack tested")

    # And confirm it plays out that way when actually flown.
    cfg = AeroConfig3D()
    s0 = make_initial_state_3d(
        pos=(0.0, 0.0, 8000.0), vel=(0.0, 0.0, -150.0),
        quat=attitude_from_pitch(np.radians(150.0)), vehicle=v)
    _, h = propagate_3d_with_aero(
        s0, lambda t, s: (0.0, 0.0, 0.0), (0.0, 40.0), 0.005, v, cfg)
    off = np.degrees(angle_history(h, cfg=cfg)[:, 3])
    ok &= report("flown: a stable body turns back toward the wind",
                 off.max() <= off[0] + 1e-6,
                 f"off-axis angle starts at {off[0]:.1f} deg, "
                 f"stays within [{off.min():.1f}, {off.max():.1f}]")

    note("LIMITATION: there is a restoring moment but no aerodynamic damping,",
         "")
    note("  so a disturbed vehicle oscillates and never settles. Real damping",
         "")
    note("  comes from a moment proportional to the body rate, which this",
         "")
    note("  model does not have. Day 16's solver should not read the",
         "")
    note("  oscillation as physical.", "")
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 15 - 3-D AERODYNAMICS VERIFICATION")
    print("=" * 70)
    results = [
        test_angle_round_trip(),
        test_low_speed_guard(),
        test_purely_axial_flow(),
        test_purely_broadside_flow(),
        test_lift_curve_shape(),
        test_side_force_sign(),
        test_moment_formula(),
        test_dynamic_pressure_scaling(),
        test_composition(),
        test_reduction_to_day6(),
        test_weathervaning(),
    ]
    print("\n" + "=" * 70)
    ok = all(results)
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
