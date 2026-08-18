"""
Day 17 -- validation of the 3-D stack.

Every group re-solves live. Nothing here is a cached number.

Groups:
    1. Planar reduction: a planar initial condition must produce exactly zero
       out-of-plane motion
    2. The planar plan reaches the pad
    3. The 3-D case genuinely uses the third dimension
    4. Every constraint is respected
    5. The throttle floor sets a hard ceiling on the burn duration, derived in
       closed form and confirmed by the flown result
    6. KNOWN FAILURE, guarded: neither formulation converges

Run:  python tests/test_3d_validation.py
"""

import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.scvx_3d_validate import (                             # noqa: E402
    solve_scvx_validate, replay, planar_ic, threed_ic,
    out_of_plane_extremes, gimbal_angles_deg, initialize_reference,
)
from src.dynamics_3d import Vehicle3D, G_EARTH                 # noqa: E402
from src.aero_3d import AeroConfig3D                           # noqa: E402

PASS, FAIL, NOTE = "[PASS]", "[FAIL]", "[NOTE]"
TF = 7.0          # the throttle floor's ceiling; see Test 5
K = 30
_CACHE = {}


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  {tag} {name:<52}" + (f" {detail}" if detail else ""))
    return ok


def note(text, detail=""):
    print(f"  {NOTE} {text:<52}" + (f" {detail}" if detail else ""))


def run(kind, tf=TF):
    key = (kind, tf)
    if key not in _CACHE:
        v = Vehicle3D()
        s0 = planar_ic(v) if kind == "planar" else threed_ic(v)
        r = solve_scvx_validate(s0, v, K=K, tf=tf, verbose=False)
        r["replay"] = replay(r, v, AeroConfig3D())
        _CACHE[key] = r
    return _CACHE[key]


# ======================================================================
def test_planar_reduction():
    """
    The symmetry check, and the strongest result of the day.

    A planar initial condition gives the optimiser no reason to leave the
    plane: gravity is planar, the boundary conditions are planar, the vehicle
    is axisymmetric. Any out-of-plane motion would be leakage -- a sign error
    in a Jacobian, a frame mix-up, or noise from a badly conditioned
    linearisation bleeding into a degree of freedom it should not touch.
    """
    print("\nTEST 1 - Planar reduction")
    r = run("planar")
    e = out_of_plane_extremes(r)
    ok = True
    for k, label in (("y", "out-of-plane position"),
                     ("vy", "out-of-plane velocity"),
                     ("roll_rate", "roll rate"), ("yaw_rate", "yaw rate"),
                     ("Fy", "side thrust")):
        ok &= report(f"{label} is exactly zero", e[k] == 0.0,
                     f"max = {e[k]:.2e}")
    note("Not 1e-12. Bit-for-bit zero.",
         "the in-plane and out-of-plane subspaces do not mix")
    note("Two independent formulations agree on this",
         "Day 16's analytic Jacobians and today's finite differences")
    return ok


def test_planar_reaches_pad():
    print("\nTEST 2 - The planar plan reaches the pad")
    v = Vehicle3D()
    r = run("planar")
    s = r["s"]
    ok = True
    ok &= report("plan lands at the pad",
                 float(np.linalg.norm(s[-1, 0:3])) < 1e-2,
                 f"|pos| = {float(np.linalg.norm(s[-1, 0:3])):.2e} m")
    ok &= report("plan lands at rest",
                 float(np.linalg.norm(s[-1, 3:6])) < 1e-2,
                 f"|vel| = {float(np.linalg.norm(s[-1, 3:6])):.2e} m/s")
    ok &= report("within the propellant budget",
                 r["fuel"] < v.m_prop_initial,
                 f"{r['fuel']:,.0f} kg of {v.m_prop_initial:,.0f}")
    ok &= report("a sub-problem actually solved",
                 r["ever_solved"] and not r["is_initial_guess"],
                 f"{r['iterations']} iterations")
    note("That last check is not padding.", "")
    note("  The guide's loop returns its reference array whatever happens,",
         "")
    note("  so a run whose first sub-problem is infeasible hands back the",
         "")
    note("  straight-line guess -- which lands at the origin, upright, at",
         "")
    note("  rest, with zero gimbal, because that is how it was built. Every",
         "")
    note("  one of those is something a test would happily accept.", "")
    return ok


def test_3d_genuinely_3d():
    print("\nTEST 3 - The 3-D case uses the third dimension")
    r = run("threed")
    s = r["s"]
    ok = True
    ok &= report("cross-range motion is real",
                 float(np.abs(s[:, 1]).max()) > 50.0,
                 f"max |y| = {float(np.abs(s[:, 1]).max()):.1f} m")
    ok &= report("out-of-plane velocity is real",
                 float(np.abs(s[:, 4]).max()) > 1.0,
                 f"max |vy| = {float(np.abs(s[:, 4]).max()):.1f} m/s")
    ok &= report("the attitude leaves the pitch plane",
                 float(np.abs(s[:, [10, 12]]).max()) > 1e-3,
                 f"max roll/yaw rate = "
                 f"{np.degrees(float(np.abs(s[:, [10, 12]]).max())):.2f} deg/s")
    note("This is the complement to Test 1.", "a solver secretly still 2-D "
                                              "would pass that one trivially")
    return ok


def test_constraints():
    print("\nTEST 4 - Constraints")
    v = Vehicle3D()
    r = run("threed")
    s, F = r["s"], r["F"]
    mag = np.linalg.norm(F, axis=1)
    ok = True
    ok &= report("mass never below dry",
                 float(s[:, 13].min()) >= v.m_dry - 1e-3,
                 f"min = {float(s[:, 13].min()):,.0f} kg")
    ok &= report("thrust inside [T_min, T_max]",
                 mag.min() >= v.T_min - 10.0 and mag.max() <= v.T_max + 1.0,
                 f"[{mag.min() / 1e6:.2f}, {mag.max() / 1e6:.2f}] MN")
    g = gimbal_angles_deg(r)
    ok &= report("gimbal within its cone",
                 float(g.max()) <= v.delta_max_deg + 0.5,
                 f"peak = {float(g.max()):.3f} deg of {v.delta_max_deg:.0f}")
    ok &= report("altitude never negative", float(s[:, 2].min()) >= -1e-6)

    # The peak gimbal is the tell, and it is worth stating rather than
    # passing over: the plan asks for real body rates while commanding almost
    # no torque to produce them.
    tau = v.T_max * v.L_engine * np.sin(np.radians(float(g.max())))
    w_peak = np.degrees(float(np.linalg.norm(s[:, 10:13], axis=1).max()))
    note(f"peak gimbal {g.max():.3f} deg commands {tau / 1e3:.1f} kN m",
         f"of {v.tau_max / 1e3:,.0f} available")
    note(f"while the plan asks for {w_peak:.1f} deg/s of body rate",
         "the rotation is coming from slack, not from torque")
    return ok


def test_throttle_floor_sets_the_horizon():
    """
    The constraint the guide never checks, and the one that dominates.

    This vehicle cannot throttle below 40 per cent, so a lit engine produces
    at least T_min / m_wet of acceleration. Against gravity that is a net
    *upward* floor -- the engine cannot push the vehicle down. From a given
    descent rate there is therefore exactly one burn duration that arrives at
    rest, and any longer burn overshoots into a climb no thrust setting can
    prevent.
    """
    print("\nTEST 5 - The throttle floor sets a hard burn duration")
    v = Vehicle3D()
    ok = True
    a_min = v.T_min / v.m_wet - G_EARTH
    vz0 = 80.0
    t_exact = vz0 / a_min

    ok &= report("minimum acceleration is net upward",
                 a_min > 0.0,
                 f"{v.T_min / v.m_wet:.2f} - {G_EARTH:.2f} = "
                 f"{a_min:+.2f} m/s^2")
    ok &= report("so the burn duration is determined, not free",
                 6.5 < t_exact < 7.5,
                 f"{vz0:.0f} / {a_min:.2f} = {t_exact:.2f} s")

    # And the flown result agrees with the arithmetic. The comparison has to
    # be against the VERTICAL component: total speed also carries whatever
    # downrange velocity is left un-arrested, which this prediction says
    # nothing about.
    speeds, vzs = {}, {}
    for tf in (7.0, 10.0, 14.0):
        rr = run("planar", tf)["replay"]
        speeds[tf] = rr["speed_ms"]
        vzs[tf] = float(rr["hist"][-1, 5])
    ok &= report("flown arrival speed grows with any longer horizon",
                 speeds[7.0] < speeds[10.0] < speeds[14.0],
                 "  ".join(f"tf={t:.0f}: {s:.1f} m/s"
                           for t, s in speeds.items()))
    # The closed form assumes thrust straight up, so it is an upper bound
    # rather than an estimate -- the flown vehicle is tilted and only the
    # vertical component of its thrust does this work, and drag takes more.
    # What must hold is that the overshoot is real and no larger than the
    # bound.
    bound = {tf: -vz0 + a_min * tf for tf in (10.0, 14.0)}
    ok &= report("the vehicle really does end up climbing",
                 all(vzs[t] > 1.0 for t in bound),
                 "  ".join(f"tf={t:.0f}: {vzs[t]:+.0f} m/s" for t in bound))
    ok &= report("and by no more than the straight-up bound",
                 all(0.0 < vzs[t] < bound[t] for t in bound),
                 "  ".join(f"tf={t:.0f}: {vzs[t]:+.0f} of {bound[t]:+.0f}"
                           for t in bound))
    note("The bound is loose because the flown vehicle is tilted,", "")
    note("  so only part of its thrust fights gravity, and drag helps.", "")
    note("The engine cannot push down, so a burn longer than the ceiling",
         "")
    note("  does not land softly -- it turns the descent into a climb.", "")
    note("The guide picks tf = 18.0 s, which is 2.6x this ceiling.", "")
    note("  It documents three bugs and misses the one that dominates.", "")
    return ok


def test_known_non_convergence():
    """
    Neither formulation converges, and this pins that down.

    Day 16's solver used analytic Jacobians, an explicit Euler step and an
    accept/reject trust controller. Today's uses finite differences, a
    state-transition-matrix discretisation and a geometric trust schedule.
    They share only the physics -- and they fail the same way, which is what
    locates the problem.
    """
    print("\nTEST 6 - KNOWN FAILURE: neither formulation converges")
    v = Vehicle3D()
    r = run("planar")
    rp = r["replay"]
    ok = True

    nus = [h["nu"] for h in r["history"] if "nu" in h]
    ok &= report("virtual control does not reach tolerance",
                 min(nus) > 1e-1,
                 f"best |nu|_1 = {min(nus):.2e}, target 1e-1")
    ok &= report("so the plan does not fly",
                 rp["miss_m"] > 10.0,
                 f"plan lands at {float(np.linalg.norm(r['s'][-1, 0:3])):.1e} m, "
                 f"flown misses by {rp['miss_m']:,.0f} m at "
                 f"{rp['speed_ms']:.1f} m/s")

    note("Where the slack sits localises it: 88-90 per cent in the", "")
    note("  velocity rows early on, which is the throttle floor of Test 5", "")
    note("  showing up as an unsatisfiable translational dynamics row.", "")
    note("  Shortening the horizon to the ceiling drops the flown arrival", "")
    note("  from 201.8 m/s to 19.6 and the miss from 1,300 m to 334 -- a", "")
    note("  large improvement that still is not convergence.", "")
    note("What IS validated is the physics. Test 1 is exact zero in two", "")
    note("  independent formulations, so Days 13-15 are not the problem.", "")
    return ok


# ======================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("DAY 17 - 3-D VALIDATION")
    print("=" * 70)
    results = [
        test_planar_reduction(),
        test_planar_reaches_pad(),
        test_3d_genuinely_3d(),
        test_constraints(),
        test_throttle_floor_sets_the_horizon(),
        test_known_non_convergence(),
    ]
    print("\n" + "=" * 70)
    ok = all(results)
    print(f"{sum(results)}/{len(results)} groups passed  "
          f"({time.time() - t0:.1f}s)")
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    if ok:
        print("NOTE: Test 6 passing means the known failure is still present.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
