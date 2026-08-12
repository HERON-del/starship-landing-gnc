"""
Verification suite for the atmosphere, aero model and two-phase pipeline.

Tests:
    1. Atmosphere against known values and monotonicity
    2. Drag opposes velocity, scales as v^2, and vanishes at rest
    3. Reference area and Cd interpolate correctly with attitude
    4. Unpowered belly-flop reaches terminal velocity
    5. The belly-flop is worth a large amount of free delta-v
    6. Two-phase landing beats the single-phase flip on propellant

Run:  python tests/test_aero.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import atmosphere as atm                              # noqa: E402
from src.aero import (                                         # noqa: E402
    AeroConfig, effective_area, effective_Cd, aero_force,
    dynamic_pressure, terminal_velocity, drag_area,
)
from src.dynamics_aero import simulate_entry                   # noqa: E402
from src.landing_flip import solve_flip_landing                # noqa: E402
from src.dynamics_6dof import Vehicle6DoF                      # noqa: E402
from tests.test_dynamics import PASS, FAIL                     # noqa: E402


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name:<48}"
    if detail:
        msg += f" {detail}"
    print(msg)
    return bool(ok)


# ======================================================================
def test_atmosphere():
    print("\nTEST 1 - Atmosphere")
    ok = True
    ok &= report("sea-level density is 1.225 kg/m^3",
                 abs(float(atm.density(0.0)) - 1.225) < 1e-9,
                 f"{float(atm.density(0.0)):.4f}")
    ok &= report("density falls by 1/e at the scale height",
                 abs(float(atm.density(atm.H_SCALE)) / 1.225
                     - np.exp(-1.0)) < 1e-9)
    z = np.linspace(0.0, 12000.0, 200)
    ok &= report("density decreases monotonically",
                 bool(np.all(np.diff(atm.density(z)) < 0)))
    ok &= report("negative altitude is clamped, not extrapolated",
                 abs(float(atm.density(-500.0)) - 1.225) < 1e-9,
                 "no super-sea-level spike")
    ok &= report("speed of sound at sea level ~340 m/s",
                 abs(float(atm.speed_of_sound(0.0)) - 340.3) < 1.0,
                 f"{float(atm.speed_of_sound(0.0)):.1f} m/s")
    return ok


# ======================================================================
def test_drag():
    print("\nTEST 2 - Drag force")
    cfg = AeroConfig()
    ok = True

    Fx, Fz = aero_force(0.0, -100.0, 1000.0, np.pi / 2, cfg)
    ok &= report("drag opposes a descent (Fz > 0)", float(Fz) > 0,
                 f"Fz = {float(Fz)/1e6:.2f} MN")

    _, Fz1 = aero_force(0.0, -50.0, 1000.0, np.pi / 2, cfg)
    _, Fz2 = aero_force(0.0, -100.0, 1000.0, np.pi / 2, cfg)
    ratio = float(Fz2) / float(Fz1)
    ok &= report("drag scales as v^2", abs(ratio - 4.0) < 0.05,
                 f"doubling speed gives {ratio:.2f}x")

    Fx0, Fz0 = aero_force(0.0, -1.0, 1000.0, np.pi / 2, cfg)
    ok &= report("no force below the cutoff speed",
                 abs(float(Fx0)) + abs(float(Fz0)) == 0.0)

    _, Fz_low = aero_force(0.0, -100.0, 0.0, np.pi / 2, cfg)
    _, Fz_high = aero_force(0.0, -100.0, 8500.0, np.pi / 2, cfg)
    ok &= report("drag falls with altitude",
                 float(Fz_high) < float(Fz_low),
                 f"{float(Fz_low)/1e6:.2f} -> {float(Fz_high)/1e6:.2f} MN")

    q = float(dynamic_pressure(0.0, -100.0, 0.0))
    ok &= report("dynamic pressure matches 0.5 rho v^2",
                 abs(q - 0.5 * 1.225 * 1e4) < 1e-6, f"{q/1000:.2f} kPa")
    return ok


# ======================================================================
def test_area():
    print("\nTEST 3 - Attitude-dependent area")
    cfg = AeroConfig()
    ok = True
    ok &= report("upright uses the base area",
                 abs(float(effective_area(0.0, cfg)) - cfg.A_base) < 1e-9,
                 f"{cfg.A_base:.0f} m^2")
    ok &= report("broadside uses the belly area",
                 abs(float(effective_area(np.pi / 2, cfg)) - cfg.A_belly) < 1e-9,
                 f"{cfg.A_belly:.0f} m^2")
    ok &= report("Cd interpolates the same way",
                 abs(float(effective_Cd(0.0, cfg)) - cfg.Cd_nose) < 1e-9
                 and abs(float(effective_Cd(np.pi/2, cfg)) - cfg.Cd_belly) < 1e-9)
    ratio = float(drag_area(np.pi / 2, cfg)) / float(drag_area(0.0, cfg))
    ok &= report("Cd*A ratio broadside vs base is large",
                 ratio > 20.0, f"{ratio:.1f}x")
    ok &= report("area is symmetric in attitude sign",
                 abs(float(effective_area(0.7, cfg))
                     - float(effective_area(-0.7, cfg))) < 1e-9)
    return ok


# ======================================================================
def test_terminal_velocity():
    print("\nTEST 4 - Unpowered belly-flop reaches terminal velocity")
    veh, cfg = Vehicle6DoF(), AeroConfig()
    r = simulate_entry(veh, cfg, z0=12_000.0, vz0=-120.0, theta0_deg=90.0,
                       z_stop=300.0)
    arrival = r["handoff"]["speed"]
    v_term = r["terminal_velocity"]

    ok = report("arrival speed matches terminal velocity",
                abs(arrival - v_term) < 0.05 * v_term,
                f"{arrival:.1f} vs {v_term:.1f} m/s")
    ok &= report("no propellant used", r["propellant_used"] == 0.0)
    ok &= report("mass unchanged", abs(r["m"][-1] - veh.m_wet) < 1e-6)
    # At terminal velocity drag balances weight, so q is pinned by the vehicle,
    # not by the altitude: q = mg / (Cd A). It does not decay as the vehicle
    # slows - it converges, and rising density is exactly offset by falling
    # speed. (Asserting decay here is wrong, and this model says so.)
    q_expected = veh.m_wet * 9.80665 / float(drag_area(np.pi / 2, cfg))
    ok &= report("terminal dynamic pressure equals mg/(Cd A)",
                 abs(r["q"][-1] - q_expected) < 0.1 * q_expected,
                 f"{r['q'][-1]/1000:.2f} vs {q_expected/1000:.2f} kPa")
    return ok


# ======================================================================
def test_bellyflop_value():
    print("\nTEST 5 - What the belly-flop is worth")
    veh, cfg = Vehicle6DoF(), AeroConfig()
    kw = dict(z0=12_000.0, vz0=-120.0, z_stop=300.0)

    vac = simulate_entry(veh, AeroConfig(enabled=False), theta0_deg=90.0, **kw)
    nose = simulate_entry(veh, cfg, theta0_deg=0.0, **kw)
    belly = simulate_entry(veh, cfg, theta0_deg=90.0, **kw)

    v_vac = vac["handoff"]["speed"]
    v_nose = nose["handoff"]["speed"]
    v_belly = belly["handoff"]["speed"]
    print(f"         vacuum {v_vac:6.1f} | nose-first {v_nose:6.1f} | "
          f"belly-flop {v_belly:6.1f} m/s")

    ok = report("belly-flop is much slower than vacuum fall",
                v_belly < 0.25 * v_vac,
                f"{v_belly:.0f} vs {v_vac:.0f} m/s")
    ok &= report("belly-flop beats nose-first substantially",
                 v_belly < 0.5 * v_nose,
                 f"{v_belly:.0f} vs {v_nose:.0f} m/s")

    # Rocket-equation value of the velocity drag removed for free.
    dv = v_vac - v_belly
    prop = veh.m_wet * (1.0 - np.exp(-dv / (veh.isp * 9.80665)))
    print(f"         {dv:.0f} m/s removed for free ~ {prop:,.0f} kg of "
          f"propellant not spent")
    ok &= report("free delta-v exceeds a typical landing burn",
                 prop > 10_000.0, f"{prop:,.0f} kg equivalent")
    return ok


# ======================================================================
def test_two_phase_beats_single():
    print("\nTEST 6 - Two-phase landing beats the single-phase flip")
    veh, cfg = Vehicle6DoF(), AeroConfig()

    # Single phase: hand over at 60 deg, which forces a long burn.
    single = solve_flip_landing(vehicle=veh, aero=cfg, N=60, t_burn=15.0,
                                theta0_deg=60.0, max_iters=30, verbose=False)
    # Two phase: coast to terminal velocity, hand over near-upright, burn briefly.
    two = solve_flip_landing(vehicle=veh, aero=cfg, N=60, t_burn=4.5,
                             theta0_deg=0.0, max_iters=30, verbose=False)

    if not (single["status"].startswith("optimal")
            and two["status"].startswith("optimal")):
        print("  One of the two solves failed - cannot compare.")
        return False

    saving = single["fuel"] - two["fuel"]
    ok = report("short burn from upright uses far less propellant",
                two["fuel"] < 0.5 * single["fuel"],
                f"{two['fuel']:,.0f} vs {single['fuel']:,.0f} kg "
                f"(saves {saving:,.0f}, {single['fuel']/two['fuel']:.1f}x)")
    ok &= report("both land on the pad",
                 abs(two["z"][-1]) < 1.0 and abs(single["z"][-1]) < 1.0)
    print("         the throttle floor sets the bill, so burn duration is "
          "the cost driver")
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 6 - AERODYNAMICS VERIFICATION")
    print("=" * 70)
    results = [
        test_atmosphere(), test_drag(), test_area(),
        test_terminal_velocity(), test_bellyflop_value(),
        test_two_phase_beats_single(),
    ]
    ok = all(results)
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
