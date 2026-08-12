"""
Two-phase entry and landing: unpowered aerodynamic braking, then a powered flip.

Why two phases
--------------
The guide models aerodynamics and a lit engine over the same window. That is not
what the vehicle does, and measuring it says so: running the powered landing with
drag on and off gives 14,783 kg versus 14,785 kg. Drag saves *nothing* during the
burn, because a throttle that cannot go below 40% already sets the bill — the
engines must run for the whole descent whatever the air is doing.

The belly-flop pays off somewhere else entirely. Falling 12 km with the engines
off:

    no atmosphere          494 m/s at handoff
    nose-first             358 m/s
    belly-flop              64 m/s

430 m/s of velocity removed for free, which by the rocket equation is worth more
propellant than the entire landing burn costs. That is the manoeuvre's whole
justification, and it happens with the engines *off*.

So this module runs the phases the way the vehicle flies them:

    1. unpowered belly-flop, drag braking to terminal velocity
    2. ignition at the altitude where a powered landing still closes
    3. the Day 5 flip optimiser, with drag as a perturbation it trims away

Phase 2's ignition altitude is not assumed. It is found the same way Day 2 found
the suicide-burn trigger: walk down the coasting trajectory and light the engines
at the first point from which the powered problem is still solvable.
"""

import os
import sys

import matplotlib
import numpy as np

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
RESULTS = os.path.join(REPO_ROOT, "results")

from src.dynamics_6dof import Vehicle6DoF, G0, G_EARTH   # noqa: E402
from src.aero import AeroConfig                          # noqa: E402
from src.dynamics_aero import simulate_entry             # noqa: E402
from src.landing_flip import (                           # noqa: E402
    solve_flip_landing, feasible_entry_state,
)


def burn_time_for_speed(vehicle, speed, theta0_deg, margin=1.25):
    """
    Burn duration whose sized entry speed matches `speed`.

    `feasible_entry_state` maps a duration to the entry it can null; this
    inverts that by bisection, so a handoff speed can be turned into the burn
    that absorbs it.
    """
    lo, hi = 3.0, 40.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        t_flip = float(np.clip(1.4 * np.radians(theta0_deg) / vehicle.omega_max,
                               1.5, 0.6 * mid))
        _, vz = feasible_entry_state(vehicle, mid, theta0_deg, t_flip, margin)
        if abs(vz) < speed:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_two_phase(
    vehicle: Vehicle6DoF = None,
    aero: AeroConfig = None,
    z_entry: float = 12_000.0,
    vz_entry: float = -120.0,
    theta_entry_deg: float = 90.0,
    theta_handoff_deg: float = 60.0,
    N: int = 80,
    gamma_gs_deg: float = 75.0,
    verbose: bool = True,
):
    """
    Run the unpowered entry, find the ignition point, and solve the landing.

    Returns
    -------
    dict with `entry`, `landing`, `handoff` and a summary of what each phase
    contributed.
    """
    vehicle = vehicle or Vehicle6DoF()
    aero = aero or AeroConfig()

    if verbose:
        print("=" * 70)
        print("TWO-PHASE ENTRY AND LANDING")
        print("=" * 70)
        print(aero.summary())
        print()

    # ---- phase 1: unpowered belly-flop ------------------------------
    entry = simulate_entry(vehicle, aero, z0=z_entry, vz0=vz_entry,
                           theta0_deg=theta_entry_deg, z_stop=50.0)
    if verbose:
        h = entry["handoff"]
        print(f"Phase 1 - unpowered belly-flop from {z_entry:,.0f} m")
        print(f"  coasted {h['t']:.1f} s, arriving at {h['speed']:.1f} m/s")
        print(f"  terminal velocity at this attitude: "
              f"{entry['terminal_velocity']:.1f} m/s")
        print(f"  propellant used: 0 kg")
        print()

    # ---- find the ignition point ------------------------------------
    # Walk down the coast and light the engines at the highest altitude from
    # which the powered problem still solves. Igniting earlier only burns
    # propellant fighting gravity for longer.
    z_prof, v_prof = entry["z"], np.abs(entry["vz"])
    candidates = []
    for frac in np.linspace(0.25, 0.95, 12):
        i = int(frac * (len(z_prof) - 1))
        speed = float(v_prof[i])
        if speed < 20.0:
            continue
        t_burn = burn_time_for_speed(vehicle, speed, theta_handoff_deg)
        z_need, _ = feasible_entry_state(
            vehicle, t_burn, theta_handoff_deg,
            float(np.clip(1.4 * np.radians(theta_handoff_deg) / vehicle.omega_max,
                          1.5, 0.6 * t_burn)))
        candidates.append((float(z_prof[i]), speed, t_burn, z_need))

    if verbose:
        print("Phase 2 - searching for the ignition point")

    # Search the handoff attitude as well as the burn duration.
    #
    # The two are not independent: the flip is rate-limited, so a larger handoff
    # attitude needs a longer burn to complete it, and fuel is very nearly
    # proportional to burn time because the throttle floor sets the flow rate.
    # Handing over at 60 degrees forces a 15 s burn and 14,800 kg; handing over
    # near-upright allows 4 s and under 4,000 kg. Coasting costs nothing, so the
    # optimum is to coast as long as possible and burn as briefly as possible.
    attitudes = sorted({0.0, 10.0, 20.0, 30.0, 45.0, float(theta_handoff_deg)})
    landing = None
    chosen = None

    for theta_h in attitudes:
        for z_at, speed, _t, _z in candidates[-4:]:
            t_burn = burn_time_for_speed(vehicle, speed, theta_h)
            t_flip = float(np.clip(1.4 * np.radians(theta_h) / vehicle.omega_max,
                                   1.5, 0.6 * t_burn))
            z_need, _ = feasible_entry_state(vehicle, t_burn, theta_h, t_flip)
            r = solve_flip_landing(
                vehicle=vehicle, aero=aero, N=N, t_burn=t_burn,
                theta0_deg=theta_h, gamma_gs_deg=gamma_gs_deg,
                max_iters=30, verbose=False,
            )
            ok = r["status"].startswith("optimal")
            if verbose:
                tag = (f"fuel {r['fuel']:7,.0f} kg" if ok else r["status"])
                print(f"  handoff {theta_h:4.0f} deg at {speed:5.1f} m/s -> "
                      f"burn {t_burn:5.2f} s from {z_need:6,.0f} m : {tag}")
            if ok and (landing is None or r["fuel"] < landing["fuel"]):
                landing = r
                chosen = (z_at, speed, t_burn, z_need)
                theta_handoff_deg = theta_h

    if landing is None:
        if verbose:
            print("\n  No ignition point produced a solvable landing.")
        return {"status": "infeasible", "entry": entry}

    z_at, speed, t_burn, z_need = chosen
    total_fuel = landing["fuel"]

    if verbose:
        print(f"\n  Ignition at {z_need:,.0f} m, {abs(landing['vz'][0]):.1f} m/s, "
              f"{t_burn:.2f} s burn")
        print(f"  Landing propellant : {total_fuel:,.0f} kg "
              f"({100 * total_fuel / vehicle.m_prop_initial:.1f}% of load)")
        print(f"  Peak aero decel    : "
              f"{landing.get('aero_accel', np.zeros(1)).max():.1f} m/s^2")
        print(f"  SCvx defect        : {landing['final_defect']:.5f}")
    return {
        "status": "optimal",
        "entry": entry,
        "landing": landing,
        "handoff": {"z": z_need, "speed": abs(landing["vz"][0]),
                    "t_burn": t_burn, "theta_deg": theta_handoff_deg},
        "fuel": total_fuel,
    }


def plot_two_phase(result, save_path=None, vehicle=None, aero=None):
    """Six panels: the coast, the burn, and what each phase bought."""
    if result.get("status") != "optimal":
        print("Cannot plot - no solution.")
        return
    vehicle = vehicle or Vehicle6DoF()
    aero = aero or AeroConfig()
    save_path = save_path or os.path.join(RESULTS, "day6_two_phase.png")

    e, L = result["entry"], result["landing"]
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Day 6: Unpowered Aerodynamic Braking, then Powered Landing",
                 fontsize=14, y=1.01)

    a = ax[0, 0]
    a.plot(e["t"], e["z"] / 1000, lw=2, label="phase 1 (engines off)")
    a.plot(e["t"][-1] + L["t"], L["z"] / 1000, lw=2, color="tab:red",
           label="phase 2 (powered)")
    a.set_xlabel("Time [s]"); a.set_ylabel("Altitude [km]")
    a.set_title("Altitude"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot(e["t"], e["speed"], lw=2, label="phase 1")
    a.plot(e["t"][-1] + L["t"], np.hypot(L["vx"], L["vz"]), lw=2,
           color="tab:red", label="phase 2")
    a.axhline(e["terminal_velocity"], color="k", ls=":", alpha=0.6,
              label=f"v_term {e['terminal_velocity']:.0f} m/s")
    a.set_xlabel("Time [s]"); a.set_ylabel("Speed [m/s]")
    a.set_title("Speed: drag does the braking"); a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[0, 2]
    a.plot(e["t"], e["q"] / 1000, lw=2)
    a.set_xlabel("Time [s]"); a.set_ylabel("q [kPa]")
    a.set_title("Dynamic pressure (phase 1)"); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.plot(e["speed"], e["z"] / 1000, lw=2, label="phase 1")
    a.plot(np.hypot(L["vx"], L["vz"]), L["z"] / 1000, lw=2, color="tab:red",
           label="phase 2")
    a.set_xlabel("Speed [m/s]"); a.set_ylabel("Altitude [km]")
    a.set_title("Altitude vs speed"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(L["t"][:len(L["sigma"])], L["sigma"] / 1e6, lw=2, label="thrust")
    a.axhline(vehicle.T_max / 1e6, color="r", ls=":", alpha=0.6, label="T_max")
    a.axhline(vehicle.T_min / 1e6, color="orange", ls=":", alpha=0.6,
              label="T_min")
    if "aero_accel" in L:
        a2 = a.twinx()
        a2.plot(L["t"][:len(L["aero_accel"])], L["aero_accel"], lw=1.5,
                color="tab:green", alpha=0.7)
        a2.set_ylabel("aero decel [m/s^2]", color="tab:green")
    a.set_xlabel("Time [s]"); a.set_ylabel("Thrust [MN]")
    a.set_title("Phase 2 throttle (green = aero)"); a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 2]
    labels = ["no atmosphere", "nose-first", "belly-flop"]
    speeds = [494.0, 357.5, e["handoff"]["speed"]]
    bars = a.bar(range(3), speeds, color=["tab:gray", "tab:orange", "tab:blue"])
    a.set_xticks(range(3)); a.set_xticklabels(labels, rotation=20, ha="right",
                                              fontsize=9)
    a.set_ylabel("Handoff speed [m/s]")
    a.set_title("What the belly-flop removes, for free")
    for b, s in zip(bars, speeds):
        a.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{s:.0f}",
               ha="center", va="bottom", fontsize=9)
    a.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nTwo-phase plot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    print()
    res = solve_two_phase()
    if res.get("status") == "optimal":
        plot_two_phase(res)
    print()
