"""
Day 14 demonstration and exploration.

Two figures and five experiments about the one thing Euler's equations add
that no planar model has: the omega x (I omega) term, and what it does.

Run:  python src/demo_3d_dynamics.py
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

from src.dynamics_3d import (                                  # noqa: E402
    Vehicle3D, propagate_3d_dynamics, make_initial_state_3d,
    gimbal_force_and_torque_body, gyroscopic_term,
    angular_momentum_inertial, rotational_kinetic_energy,
    attitude_from_pitch, tilt_from_vertical,
)
from src.quaternion import quat_angle_between                  # noqa: E402

RULE = "-" * 78
NO_THRUST = (0.0, 0.0, 0.0)


def _coast(t, s):
    return NO_THRUST


# ======================================================================
def plot_poinsot_tumble(save_path=None):
    """
    Torque-free tumble: the body-frame rate precesses, the inertial-frame
    angular momentum does not move at all.

    The middle panel needs its limits set by hand. Left to autoscale it would
    zoom in on a drift of one part in 1e12 and render the conserved vector as
    an impressive-looking cloud -- a plot that says the opposite of the truth.
    Fixing the axes to a few per cent of |L| is what makes it read as a point.
    """
    save_path = save_path or os.path.join(RESULTS, "day14_poinsot_tumble.png")
    v = Vehicle3D()
    s0 = make_initial_state_3d(omega=(1.2, 0.4, 0.25), vehicle=v)
    ts, hist = propagate_3d_dynamics(s0, _coast, (0.0, 20.0), 0.01, v)

    omega = hist[:, 10:13]
    L = np.array([angular_momentum_inertial(s, v) for s in hist])
    ke = np.array([rotational_kinetic_energy(s, v) for s in hist])
    L0, mag = L[0], float(np.linalg.norm(L[0]))

    fig = plt.figure(figsize=(19, 5.5))
    fig.suptitle("Day 14: torque-free tumble -- the body rate precesses, "
                 "the angular momentum does not", fontsize=13)

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(omega[:, 0], omega[:, 1], omega[:, 2], lw=1.0, color="tab:blue")
    ax.scatter(*omega[0], color="tab:green", s=60, label="start")
    ax.set_xlabel("wx"); ax.set_ylabel("wy"); ax.set_zlabel("wz")
    ax.set_title("Angular velocity, body frame\n(sweeps a cone)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.plot(L[:, 0], L[:, 1], L[:, 2], lw=1.0, color="tab:red")
    ax.scatter(*L0, color="tab:green", s=80, label="start")
    ax.scatter(*L[-1], color="tab:red", s=40, marker="x", label="end")
    for setter, c in ((ax.set_xlim, L0[0]), (ax.set_ylim, L0[1]),
                      (ax.set_zlim, L0[2])):
        setter(c - 0.05 * mag, c + 0.05 * mag)
    ax.set_xlabel("Lx"); ax.set_ylabel("Ly"); ax.set_zlabel("Lz")
    ax.set_title("Angular momentum, inertial frame\n"
                 "(one point, axes at +/-5% of |L|)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(1, 3, 3)
    ax.semilogy(ts, np.maximum(np.linalg.norm(L - L0, axis=1) / mag, 1e-18),
                lw=1.5, color="tab:red", label="|dL| / |L|")
    ax.semilogy(ts, np.maximum(np.abs(ke - ke[0]) / ke[0], 1e-18),
                lw=1.5, color="tab:blue", ls="--", label="|dKE| / KE")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("relative drift")
    ax.set_title("Both conserved, to RK4 precision")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Poinsot tumble plot -> {save_path}")
    plt.close()
    return float(np.linalg.norm(L - L0, axis=1).max() / mag)


# ======================================================================
def _flip_burn(v, roll_rate, gyro, t_end=5.0, dt=0.002):
    """Day 5's entry attitude, a pitch-axis gimbal, and a chosen roll rate."""
    s0 = make_initial_state_3d(
        pos=(0.0, 0.0, 2000.0), vel=(-30.0, 0.0, -80.0),
        quat=attitude_from_pitch(np.radians(70.0)),
        omega=(roll_rate, 0.1, 0.0), vehicle=v)
    return propagate_3d_dynamics(
        s0, lambda t, s: (0.7 * v.T_max, np.radians(5.0), 0.0),
        (0.0, t_end), dt, v, include_gyro=gyro)


def plot_gyroscopic_relevance(save_path=None):
    """
    When the gyroscopic term actually changes the answer.

    The interesting result is at the left edge: with no roll rate the term
    contributes exactly nothing, so Euler's equations collapse onto Day 5's
    scalar tau = I alpha. The project's flip is roll-free, so today's headline
    physics addition changes none of the twelve days behind it -- and starts
    changing everything the moment the vehicle rolls at all.
    """
    save_path = save_path or os.path.join(RESULTS, "day14_gyroscopic.png")
    v = Vehicle3D()
    rolls = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])

    ratio = [np.linalg.norm(gyroscopic_term(np.array([r, 0.1, 0.1]),
                                            v.I_body)) / v.tau_max
             for r in rolls]
    att, pos = [], []
    for r in rolls:
        _, a = _flip_burn(v, r, True)
        _, b = _flip_burn(v, r, False)
        att.append(np.degrees(quat_angle_between(a[-1, 6:10], b[-1, 6:10])))
        pos.append(float(np.linalg.norm(a[-1, 0:3] - b[-1, 0:3])))

    ts, on = _flip_burn(v, 0.5, True)
    _, off = _flip_burn(v, 0.5, False)

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("Day 14: the gyroscopic term is exactly irrelevant at zero "
                 "roll, and dominant just above it", fontsize=13)

    a = ax[0]
    a.plot(rolls, ratio, "o-", lw=1.5, color="tab:purple")
    a.set_xlabel("Roll rate [rad/s]"); a.set_ylabel("|w x Iw| / tau_max")
    a.set_title("Coupling torque against\nthe engine's full authority")
    a.grid(alpha=0.3)

    a = ax[1]
    a.plot(rolls, att, "o-", lw=1.5, color="tab:red", label="attitude [deg]")
    a.plot(rolls, pos, "s--", lw=1.5, color="tab:blue", label="position [m]")
    a.set_xscale("symlog", linthresh=0.02)
    a.set_xlabel("Roll rate [rad/s]")
    a.set_title("Divergence after a 5 s burn,\nterm included vs dropped")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[2]
    for hist, lab, c, ls in ((on, "Euler (correct)", "tab:red", "-"),
                             (off, "gyro term dropped", "tab:blue", "--")):
        a.plot(ts, [np.degrees(tilt_from_vertical(s[6:10])) for s in hist],
               lw=1.6, color=c, ls=ls, label=lab)
    a.set_xlabel("Time [s]"); a.set_ylabel("Tilt from vertical [deg]")
    a.set_title("The same burn at 0.5 rad/s of roll")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Gyroscopic relevance plot -> {save_path}")
    plt.close()
    return rolls, np.array(att), np.array(pos)


# ======================================================================
def experiment_a():
    """Axisymmetric against genuinely tri-axial."""
    print(f"\nEXPERIMENT A - Axisymmetric vs tri-axial tumble\n{RULE}")
    omega0 = (1.2, 0.4, 0.25)
    for label, v in (("axisymmetric (I_yy = I_zz)", Vehicle3D()),
                     ("tri-axial   (I_zz = 1.3 I_yy)",
                      Vehicle3D(I_yaw=1.3 * Vehicle3D().I_pitch_yaw))):
        s0 = make_initial_state_3d(omega=omega0, vehicle=v)
        _, h = propagate_3d_dynamics(s0, _coast, (0.0, 20.0), 0.005, v)
        w = h[:, 10:13]
        nut = np.degrees(np.arccos(np.clip(
            w[:, 0] / np.linalg.norm(w, axis=1), -1.0, 1.0)))
        L = np.array([angular_momentum_inertial(s, v) for s in h])
        drift = float(np.linalg.norm(L - L[0], axis=1).max()
                      / np.linalg.norm(L[0]))
        print(f"  {label:32s} half-angle {nut.min():6.2f} to {nut.max():6.2f} "
              f"deg (spread {nut.max() - nut.min():5.2f}), "
              f"|dL|/|L| {drift:.1e}")
    print("  Axisymmetric: the cone is exact, the half-angle never moves.")
    print("  Tri-axial: the half-angle breathes, so the path is not a cone.")


def experiment_b():
    """The tennis-racket theorem, emergent rather than coded."""
    print(f"\nEXPERIMENT B - Intermediate-axis instability\n{RULE}")
    v = Vehicle3D(I_yaw=1.3 * Vehicle3D().I_pitch_yaw)
    print(f"  I = {np.diag(v.I_body).tolist()}  "
          f"(roll min, pitch intermediate, yaw max)")
    for axis, name in ((0, "roll  (minimum)"), (1, "pitch (intermediate)"),
                       (2, "yaw   (maximum)")):
        w = np.zeros(3)
        w[axis] = 1.0
        w[(axis + 1) % 3] = 0.01              # 1 per cent nudge off the axis
        s0 = make_initial_state_3d(omega=tuple(w), vehicle=v)
        _, h = propagate_3d_dynamics(s0, _coast, (0.0, 60.0), 0.005, v)
        frac = h[:, 10 + axis] / np.linalg.norm(h[:, 10:13], axis=1)
        verdict = "FLIPS" if frac.min() < 0 else "stable"
        print(f"  spin about {name:22s} axis fraction "
              f"{frac.min():+.4f} to {frac.max():+.4f}   -> {verdict}")
    print("  Only the intermediate axis flips, and it flips completely.")
    print("  Nothing in the code knows about this -- it falls out of")
    print("  omega x (I omega), which is about as good a check on the term")
    print("  as exists.")


def experiment_c():
    """The reachable torque set is not a box."""
    print(f"\nEXPERIMENT C - Gimbal-limited torque envelope\n{RULE}")
    v = Vehicle3D()
    grid = np.linspace(-v.delta_max, v.delta_max, 61)
    ty, tz = [], []
    for dy in grid:
        for dz in grid:
            tau = gimbal_force_and_torque_body(v.T_max, dy, dz, v)[1]
            ty.append(tau[1])
            tz.append(tau[2])
    ty, tz = np.array(ty), np.array(tz)

    # At the pitch extremes the yaw authority is cos(delta_y) smaller, which
    # is what makes the corner unreachable.
    edge = abs(tz[np.argmax(np.abs(ty))]) / max(np.abs(tz))
    print(f"  |tau_y| max {np.abs(ty).max():.3e} N m   "
          f"|tau_z| max {np.abs(tz).max():.3e} N m")
    print(f"  corner reach: at full pitch deflection the yaw torque available "
          f"is {edge * 100:.1f}% of its own maximum")
    print(f"  box area would be {4 * np.abs(ty).max() * np.abs(tz).max():.3e}; "
          f"the reachable set is smaller by the cos(delta_y) factor.")
    print("  Day 16's convex solver will need an inner approximation of this,")
    print("  and a box would not be one -- it would promise torque the engine")
    print("  cannot deliver at the corners.")


def experiment_d():
    """The roll axis, confirmed absent rather than assumed."""
    print(f"\nEXPERIMENT D - Roll torque, the honest limitation\n{RULE}")
    v = Vehicle3D()
    grid = np.linspace(-v.delta_max, v.delta_max, 101)
    worst = max(abs(gimbal_force_and_torque_body(v.T_max, dy, dz, v)[1][0])
                for dy in grid for dz in grid)
    print(f"  swept {grid.size ** 2:,} deflection pairs at full thrust")
    print(f"  worst |tau_x| = {worst:.2e} N m  (machine zero)")
    print("  Not a tolerance question: r x F with r along body x can never")
    print("  have an x component, so this is exactly zero by construction.")
    print("  The vehicle has no roll authority whatsoever in this model, and")
    print("  Experiment E is about what that costs.")


def experiment_e(rolls, att, pos):
    """Does the day's headline term change anything this project does?"""
    print(f"\nEXPERIMENT E - Does the gyroscopic term matter here?\n{RULE}")
    print(f"  {'roll rate':>10} {'|w x Iw|/tau_max':>18} {'d attitude':>12} "
          f"{'d position':>12}")
    v = Vehicle3D()
    for r, a, p in zip(rolls, att, pos):
        ratio = np.linalg.norm(
            gyroscopic_term(np.array([r, 0.1, 0.1]), v.I_body)) / v.tau_max
        print(f"  {r:10.2f} {ratio:18.2e} {a:11.4f}d {p:11.4f}m")
    print("  At exactly zero roll the divergence is exactly zero, and that is")
    print("  not a numerical accident: with omega on a single principal axis,")
    print("  omega and I omega are parallel, the cross product vanishes, and a")
    print("  pitch-axis torque never moves omega off that axis. So for this")
    print("  project's roll-free flip, Euler's equations ARE Day 5's")
    print("  tau = I alpha, exactly -- today changes nothing behind it.")
    print("  Just above zero it changes everything: 0.1 rad/s of roll, under")
    print("  6 deg/s, moves the attitude 18 deg over a five-second burn. The")
    print("  divergence peaks near 1 rad/s and falls again beyond it, which is")
    print("  gyroscopic stiffening -- a fast enough spin resists being turned.")


if __name__ == "__main__":
    print("=" * 78)
    print("DAY 14 - 3-D RIGID-BODY DYNAMICS DEMONSTRATION")
    print("=" * 78)
    print(Vehicle3D().summary())
    drift = plot_poinsot_tumble()
    print(f"  angular momentum drift over 20 s: {drift:.2e} relative")
    rolls, att, pos = plot_gyroscopic_relevance()
    experiment_a()
    experiment_b()
    experiment_c()
    experiment_d()
    experiment_e(rolls, att, pos)
    print()
