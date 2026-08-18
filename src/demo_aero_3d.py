"""
Day 15 demonstration and exploration.

Three figures and five experiments about what a genuinely 3-D wind does that a
planar one could not: it arrives from out of the plane, and it rotates the
vehicle as well as pushing it.

Run:  python src/demo_aero_3d.py
"""

import os
import sys

import matplotlib
import numpy as np

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
RESULTS = os.path.join(REPO_ROOT, "results")

from src.aero_3d import (                                      # noqa: E402
    AeroConfig3D, effective_area_and_Cd, lift_coefficient, aero_force_body,
    aero_moment_body, aero_angles, angles_to_relative_wind,
    aero_force_and_moment_body,
    relative_wind_body, propagate_3d_with_aero, angle_history,
    static_margin_sign,
)
from src.dynamics_3d import (                                  # noqa: E402
    Vehicle3D, make_initial_state_3d, attitude_from_pitch, tilt_from_vertical,
    gimbal_force_and_torque_body,
)
from src.aero import AeroConfig, drag_area                     # noqa: E402
from src.quaternion import quat_to_rotmatrix                   # noqa: E402

RULE = "-" * 78
ALT = 3000.0


def _coast(t, s):
    return (0.0, 0.0, 0.0)


# ======================================================================
def plot_coefficients(save_path=None):
    """Area, drag coefficient and lift across the full range of orientation."""
    save_path = save_path or os.path.join(RESULTS, "day15_coefficients.png")
    cfg = AeroConfig3D()
    deg = np.linspace(0.0, 180.0, 721)
    rad = np.radians(deg)
    A, Cd = zip(*[effective_area_and_Cd(a, cfg) for a in rad])
    A, Cd = np.array(A), np.array(Cd)
    Cl = np.array([lift_coefficient(a, cfg) for a in rad])

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("Day 15: the coefficients, as functions of where the wind "
                 "comes from", fontsize=13)

    a = ax[0]
    a.plot(deg, A * Cd, lw=2, color="tab:purple")
    a.axhline(cfg.A_nose * cfg.Cd_nose, color="gray", ls=":", alpha=0.7,
              label="nose-on")
    a.axhline(cfg.A_belly * cfg.Cd_belly, color="gray", ls="--", alpha=0.7,
              label="broadside")
    a.set_xlabel("Angle of the wind off the body axis [deg]")
    a.set_ylabel("Cd * A  [m^2]")
    a.set_title(f"Drag area\n(broadside is "
                f"{cfg.A_belly * cfg.Cd_belly / (cfg.A_nose * cfg.Cd_nose):.0f}x "
                f"nose-on)")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1]
    a.plot(deg, A, lw=2, color="tab:blue", label="area [m^2]")
    a.set_ylabel("Effective area [m^2]", color="tab:blue")
    b = a.twinx()
    b.plot(deg, Cd, lw=2, color="tab:red", ls="--", label="Cd")
    b.set_ylabel("Cd_eff", color="tab:red")
    a.set_xlabel("Angle of the wind off the body axis [deg]")
    a.set_title("Both blend on sin, so both are\nsymmetric about broadside")
    a.grid(alpha=0.3)

    a = ax[2]
    a.plot(deg, Cl, lw=2, color="tab:green")
    a.axhline(0.0, color="black", lw=0.6)
    a.axvline(45.0, color="gray", ls=":", alpha=0.7, label="peak, 45 deg")
    a.axvline(135.0, color="gray", ls=":", alpha=0.7)
    a.set_xlabel("Angle of attack alpha [deg]")
    a.set_ylabel("Cl")
    a.set_title("Lift coefficient\n(zero nose-on and broadside)")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Coefficient sweep -> {save_path}")
    plt.close()


# ======================================================================
def _crosswind_run(wind, cfg=None, dz_deg=0.0, t_end=12.0):
    v = Vehicle3D()
    cfg = cfg or AeroConfig3D()
    s0 = make_initial_state_3d(
        pos=(0.0, 0.0, 3000.0), vel=(-20.0, 0.0, -90.0),
        quat=attitude_from_pitch(np.radians(80.0)), vehicle=v)
    return propagate_3d_with_aero(
        s0, lambda t, s: (0.7 * v.T_max, np.radians(4.0),
                          np.radians(dz_deg)),
        (0.0, t_end), 0.005, v, cfg, wind_inertial=wind)


def plot_crosswind(save_path=None):
    """
    A crosswind, and no yaw-gimbal input at all.

    Everything that happens out of the plane here is aerodynamic. This is the
    thing the planar model could not represent: not that the answer changes a
    little, but that a whole degree of freedom starts moving on its own.
    """
    save_path = save_path or os.path.join(RESULTS, "day15_crosswind.png")
    cfg = AeroConfig3D()
    wind = (15.0, 8.0, 0.0)
    ts, hist = _crosswind_run(wind, cfg)
    _, calm = _crosswind_run((0.0, 0.0, 0.0), cfg)

    ang = angle_history(hist, wind, cfg)
    ang_calm = angle_history(calm, (0.0, 0.0, 0.0), cfg)

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("Day 15: a crosswind, with the yaw gimbal held at exactly "
                 "zero", fontsize=13)

    a = ax[0]
    a.plot(hist[:, 0], hist[:, 2], lw=2, color="tab:blue", label="crosswind")
    a.plot(calm[:, 0], calm[:, 2], lw=1.5, color="gray", ls="--",
           label="still air")
    a.set_xlabel("Downrange x [m]"); a.set_ylabel("Altitude z [m]")
    a.set_title("In the plane")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1]
    a.plot(ts, hist[:, 1], lw=2, color="tab:red", label="crosswind")
    a.plot(ts, calm[:, 1], lw=1.5, color="gray", ls="--", label="still air")
    a.set_xlabel("Time [s]"); a.set_ylabel("Out-of-plane position y [m]")
    a.set_title(f"Out of the plane\n(ends {hist[-1, 1]:+.1f} m off, "
                f"with no yaw command)")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[2]
    a.plot(ts, np.degrees(ang[:, 1]), lw=2, color="tab:red", label="sideslip")
    a.plot(ts, np.degrees(ang_calm[:, 1]), lw=1.5, color="gray", ls="--",
           label="sideslip, still air")
    a.plot(ts, np.degrees(ang[:, 0]), lw=1.2, color="tab:blue", alpha=0.7,
           label="angle of attack")
    a.set_xlabel("Time [s]"); a.set_ylabel("[deg]")
    a.set_title("The two aerodynamic angles")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Crosswind descent -> {save_path}")
    plt.close()
    return ts, hist, ang


# ======================================================================
def plot_day6_gap(save_path=None):
    """
    What Day 6's blend angle costs, across the flight envelope.

    Day 6 decides how much vehicle the air sees from the pitch angle relative
    to *vertical*. The quantity that actually decides it is the angle relative
    to the *wind*. They agree along one line -- vertical descent -- and nowhere
    else.
    """
    save_path = save_path or os.path.join(RESULTS, "day15_day6_gap.png")
    new, old = AeroConfig3D(), AeroConfig()

    pitch = np.radians(np.linspace(0.0, 100.0, 121))
    flight = np.radians(np.linspace(-60.0, 60.0, 121))   # velocity off vertical
    ratio = np.zeros((pitch.size, flight.size))
    for i, th in enumerate(pitch):
        R = quat_to_rotmatrix(attitude_from_pitch(th))
        for j, fa in enumerate(flight):
            vel = 100.0 * np.array([np.sin(fa), 0.0, -np.cos(fa)])
            _, _, _, off = aero_angles(R.T @ vel, new.v_min)
            A, Cd = effective_area_and_Cd(off, new)
            ratio[i, j] = (A * Cd) / float(drag_area(th, old))

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("Day 15: what Day 6's blend angle is worth -- true drag area "
                 "divided by Day 6's", fontsize=13)

    a = ax[0]
    im = a.pcolormesh(np.degrees(flight), np.degrees(pitch), ratio,
                      cmap="RdBu_r", vmin=0.0, vmax=2.0, shading="auto")
    a.axvline(0.0, color="k", lw=1.2, ls="--")
    a.plot([-60, 60], [80, 80], color="k", lw=0.8, alpha=0.4)
    a.set_xlabel("Velocity angle off vertical [deg]")
    a.set_ylabel("Vehicle pitch off vertical [deg]")
    a.set_title("Ratio over the envelope\n(1.0 only on the dashed line)")
    fig.colorbar(im, ax=a)

    a = ax[1]
    for th_deg, c in ((30.0, "tab:green"), (70.0, "tab:blue"),
                      (90.0, "tab:red")):
        i = int(np.argmin(np.abs(np.degrees(pitch) - th_deg)))
        a.plot(np.degrees(flight), ratio[i], lw=2, color=c,
               label=f"pitch {th_deg:.0f} deg")
    a.axhline(1.0, color="k", lw=0.8, ls="--")
    a.set_xlabel("Velocity angle off vertical [deg]")
    a.set_ylabel("true CdA / Day 6 CdA")
    a.set_title("Slices. Day 6 is right only where\nthe wind is vertical")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # And what that is worth over a real descent.
    v = Vehicle3D()
    cfg = AeroConfig3D()
    s0 = make_initial_state_3d(
        pos=(0.0, 0.0, 6000.0), vel=(-40.0, 0.0, -120.0),
        quat=attitude_from_pitch(np.radians(80.0)), vehicle=v)
    ts, h = propagate_3d_with_aero(s0, _coast, (0.0, 30.0), 0.005, v, cfg)
    ang = angle_history(h, cfg=cfg)
    true_CdA = np.array([np.prod(effective_area_and_Cd(o, cfg))
                         for o in ang[:, 3]])
    th_hist = np.array([np.arctan2(
        (quat_to_rotmatrix(s[6:10]) @ np.array([1.0, 0.0, 0.0]))[0],
        (quat_to_rotmatrix(s[6:10]) @ np.array([1.0, 0.0, 0.0]))[2])
        for s in h])
    day6_CdA = np.array([float(drag_area(t, old)) for t in th_hist])

    a = ax[2]
    a.plot(ts, true_CdA, lw=2, color="tab:purple", label="wind-relative")
    a.plot(ts, day6_CdA, lw=2, color="tab:orange", ls="--",
           label="Day 6 (vertical-relative)")
    a.set_xlabel("Time [s]"); a.set_ylabel("Cd * A [m^2]")
    a.set_title("Over an unpowered descent\nfrom 6 km")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Day 6 gap -> {save_path}")
    plt.close()
    return float(np.mean(day6_CdA / true_CdA)), float(
        np.max(day6_CdA / true_CdA))


# ======================================================================
def experiment_a():
    """Lift over drag, and where the model puts its best orientation."""
    print(f"\nEXPERIMENT A - Lift-to-drag ratio\n{RULE}")
    cfg = AeroConfig3D()
    grid = np.radians(np.linspace(1.0, 89.0, 441))
    ld = []
    for a in grid:
        A, Cd = effective_area_and_Cd(a, cfg)
        ld.append(abs(lift_coefficient(a, cfg)) / Cd)
    ld = np.array(ld)
    k = int(np.argmax(ld))
    print(f"  peak L/D {ld[k]:.4f} at alpha = {np.degrees(grid[k]):.2f} deg")
    at45 = ld[int(np.argmin(np.abs(np.degrees(grid) - 45.0)))]
    print(f"  L/D at 45 deg (where Cl peaks): {at45:.4f}")
    print("  The peak is NOT at 45 degrees, where the lift coefficient peaks,")
    print("  because Cd is still climbing there. This is a bluff body -- the")
    print("  best L/D it manages is well under 1, which is the honest reading:")
    print("  a falling cylinder is not a wing and cannot glide anywhere.")


def experiment_b():
    """How hard the centre of pressure bites."""
    print(f"\nEXPERIMENT B - Centre-of-pressure sensitivity\n{RULE}")
    v = Vehicle3D()
    print(f"  {'x_cp [m]':>9} {'stability':>12} {'peak |tau| [MN m]':>19} "
          f"{'off-axis range [deg]':>22}")
    for x_cp in (-5.0, 0.0, 1.0, 5.0, 10.0, 15.0):
        cfg = AeroConfig3D(x_cp=x_cp)
        F = aero_force_body(
            angles_to_relative_wind(np.radians(20.0), 0.0, 150.0), 8000.0, cfg)
        tau = float(np.abs(aero_moment_body(F, cfg)).max())
        s0 = make_initial_state_3d(
            pos=(0.0, 0.0, 8000.0), vel=(0.0, 0.0, -150.0),
            quat=attitude_from_pitch(np.radians(150.0)), vehicle=v)
        _, h = propagate_3d_with_aero(s0, _coast, (0.0, 30.0), 0.005, v, cfg)
        off = np.degrees(angle_history(h, cfg=cfg)[:, 3])
        word = static_margin_sign(cfg).split(" (")[0]
        print(f"  {x_cp:9.1f} {word:>12} {tau / 1e6:19.3f} "
              f"{f'{off.min():.1f} to {off.max():.1f}':>22}")
    print("  The moment is exactly linear in x_cp -- it is r x F and nothing")
    print("  else -- so the sensitivity is entirely in how fast the vehicle")
    print("  then rotates. At x_cp = 0 the attitude is frozen: the range shown")
    print("  there is the WIND direction moving as the vehicle accelerates,")
    print("  not the vehicle turning. That row is the control case.")


def experiment_c():
    """Wind out of every plane at once."""
    print(f"\nEXPERIMENT C - Wind with no preferred plane\n{RULE}")
    cfg = AeroConfig3D()
    print(f"  {'wind (x, y, z)':>20} {'max |alpha|':>12} {'max |beta|':>11} "
          f"{'out-of-plane':>13} {'final tilt':>11}")
    for wind in ((0.0, 0.0, 0.0), (15.0, 0.0, 0.0), (0.0, 8.0, 0.0),
                 (15.0, 8.0, 0.0), (15.0, 8.0, -3.0)):
        ts, h = _crosswind_run(wind, cfg)
        ang = angle_history(h, wind, cfg)
        print(f"  {str(tuple(wind)):>20} "
              f"{np.degrees(np.abs(ang[:, 0]).max()):11.2f}d "
              f"{np.degrees(np.abs(ang[:, 1]).max()):10.2f}d "
              f"{np.abs(h[:, 1]).max():12.2f}m "
              f"{np.degrees(tilt_from_vertical(h[-1, 6:10])):10.2f}d")
    print("  A purely downrange wind moves alpha and leaves beta at exactly")
    print("  zero -- it is still a planar problem. Any lateral component at")
    print("  all breaks that, and the vertical component then couples back")
    print("  into alpha as well. Real turbulence is never aligned with a")
    print("  body axis, which is the case the planar model never had to face.")


def experiment_d(mean_ratio, max_ratio):
    """The Day 6 comparison, as a number."""
    print(f"\nEXPERIMENT D - Against Day 6\n{RULE}")
    new, old = AeroConfig3D(), AeroConfig()
    print(f"  {'pitch':>6} {'vel off vert':>13} {'Day 6 CdA':>10} "
          f"{'true CdA':>9} {'ratio':>7}")
    for th_deg, fa_deg in ((70.0, 0.0), (70.0, 20.0), (70.0, 45.0),
                           (30.0, 24.0), (90.0, 35.0)):
        th = np.radians(th_deg)
        R = quat_to_rotmatrix(attitude_from_pitch(th))
        vel = 100.0 * np.array([np.sin(np.radians(fa_deg)), 0.0,
                                -np.cos(np.radians(fa_deg))])
        _, _, _, off = aero_angles(R.T @ vel, new.v_min)
        A, Cd = effective_area_and_Cd(off, new)
        d6 = float(drag_area(th, old))
        print(f"  {th_deg:6.0f} {fa_deg:12.0f}d {d6:10.1f} {A * Cd:9.1f} "
              f"{A * Cd / d6:7.3f}")
    print(f"  Over a full unpowered descent from 6 km, the ratio averages "
          f"{mean_ratio:.2f}x")
    print(f"  and reaches {max_ratio:.2f}x at worst.")
    print("  The error is NOT one-directional, which is the awkward part. In")
    print("  the table above Day 6 gives too much drag at 90 deg of pitch and")
    print("  only half what it should at 30 deg. So it cannot be dismissed as")
    print("  a conservative margin -- it is a bias whose sign depends on where")
    print("  in the envelope the vehicle is, which is exactly the kind of")
    print("  error an optimiser will find and exploit. Not fixed today:")
    print("  aero.py carries Days 6 to 12.")


def experiment_e():
    """What the wind does that no gimbal commanded -- and what did the pushing."""
    print(f"\nEXPERIMENT E - Aerodynamic cross-coupling, unforced\n{RULE}")
    v = Vehicle3D()
    cfg = AeroConfig3D()
    wind = np.array([15.0, 8.0, 0.0])
    ts, h = _crosswind_run(tuple(wind), cfg)
    ang = angle_history(h, wind, cfg)
    _, calm = _crosswind_run((0.0, 0.0, 0.0), cfg)

    F_thrust_body = gimbal_force_and_torque_body(
        0.7 * v.T_max, np.radians(4.0), 0.0, v)[0]
    long_axis = [quat_to_rotmatrix(s[6:10]) @ np.array([1.0, 0.0, 0.0])
                 for s in h]
    yaw = np.degrees([np.arctan2(b[1], np.hypot(b[0], b[2]))
                      for b in long_axis])
    Fy_thrust = np.array([(quat_to_rotmatrix(s[6:10]) @ F_thrust_body)[1]
                          for s in h])
    Fy_aero = np.array([
        (quat_to_rotmatrix(s[6:10])
         @ aero_force_and_moment_body(s[3:6], wind, s[6:10], s[2], cfg)[0])[1]
        for s in h])
    dt = float(ts[1] - ts[0])

    print("  yaw gimbal commanded : 0.000 deg, for the whole flight")
    print(f"  peak sideslip        : "
          f"{np.degrees(np.abs(ang[:, 1]).max()):.3f} deg")
    print(f"  peak yaw rate        : "
          f"{np.degrees(np.abs(h[:, 12]).max()):.3f} deg/s")
    print(f"  body yawed to        : {yaw.min():.2f} deg out of plane")
    print(f"  out-of-plane drift   : {h[-1, 1]:+.1f} m "
          f"(still air: {calm[-1, 1]:+.2e} m)")
    print()
    print("  Out-of-plane impulse, split by what delivered it:")
    print(f"    aerodynamic  {np.trapezoid(Fy_aero, dx=dt) / 1e6:+7.2f} MN s"
          f"   downwind, as expected")
    print(f"    thrust       {np.trapezoid(Fy_thrust, dx=dt) / 1e6:+7.2f} MN s"
          f"   upwind, three times larger")
    print()
    print("  So the interesting part is not that aero pushes the vehicle")
    print("  sideways. It is that aero *turns* it and the engine then does")
    print("  the pushing. The first two seconds drift downwind under the side")
    print("  force alone; by then the aerodynamic yaw moment has swung the")
    print("  body about 20 degrees out of plane, and 4.8 MN of thrust pointed")
    print("  20 degrees wrong overwhelms every aerodynamic force in the model.")
    print("  The vehicle ends up hundreds of metres UPWIND, carried there by")
    print("  its own engine. The planar model cannot express that failure at")
    print("  all, and it is a control problem rather than an aero one.")


if __name__ == "__main__":
    print("=" * 78)
    print("DAY 15 - 3-D AERODYNAMICS DEMONSTRATION")
    print("=" * 78)
    print(AeroConfig3D().summary())
    plot_coefficients()
    plot_crosswind()
    mean_ratio, max_ratio = plot_day6_gap()
    experiment_a()
    experiment_b()
    experiment_c()
    experiment_d(mean_ratio, max_ratio)
    experiment_e()
    print()
