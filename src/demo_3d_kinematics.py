"""
Day 13 demonstration and exploration.

Two figures and four experiments, all of them about the same question: what
does a three-dimensional rotation actually do, and where does the Euler-angle
description of it stop working.

Run:  python src/demo_3d_kinematics.py
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

from src.quaternion import (                                   # noqa: E402
    quat_from_axis_angle, quat_multiply, quat_conjugate, quat_to_euler,
    euler_to_quat, quat_rotate_vector, quat_to_axis_angle, quat_normalize,
    quat_angle_between, quats_equal,
)
from src.dynamics_3d_kinematics import (                       # noqa: E402
    make_initial_state, propagate_3d, zero_accel, zero_alpha, norm_history,
    IDX_QUAT, IDX_OMEGA,
)

RULE = "-" * 78


# ======================================================================
def plot_tumble(save_path=None):
    """
    A free tumble about a tilted axis, and what the body axes do.

    Nothing forces this motion -- angular acceleration is zero throughout, so
    the body simply keeps rotating at whatever rate it started with. The point
    of the figure is that the *body* axes trace a perfectly regular cone in
    inertial space while the *Euler angles* describing them do not look
    regular at all.
    """
    save_path = save_path or os.path.join(RESULTS, "day13_tumble.png")
    s0 = make_initial_state(omega=(0.35, 0.20, 0.85))
    ts, hist = propagate_3d(s0, zero_accel, zero_alpha, (0.0, 20.0), 0.005)

    body_axes = np.array([[quat_rotate_vector(s[IDX_QUAT], e)
                           for e in np.eye(3)] for s in hist])
    eul = np.degrees(np.array([quat_to_euler(s[IDX_QUAT]) for s in hist]))
    norms = norm_history(hist)

    fig = plt.figure(figsize=(19, 5))
    fig.suptitle("Day 13: a free tumble, three ways of describing it",
                 fontsize=13)

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    for i, (c, lab) in enumerate(zip(("tab:red", "tab:green", "tab:blue"),
                                     ("body x", "body y", "body z"))):
        ax.plot(body_axes[:, i, 0], body_axes[:, i, 1], body_axes[:, i, 2],
                lw=1, color=c, alpha=0.8, label=lab)
    ax.set_title("Body axes on the unit sphere")
    ax.legend(fontsize=7)
    ax.set_box_aspect((1, 1, 1))

    ax = fig.add_subplot(1, 3, 2)
    for col, lab, c in zip(range(3), ("roll", "pitch", "yaw"),
                           ("tab:red", "tab:green", "tab:blue")):
        ax.plot(ts, eul[:, col], lw=1.2, color=c, label=lab)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("[deg]")
    ax.set_title("The same motion in Euler angles")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(1, 3, 3)
    ax.plot(ts, np.abs(norms - 1.0), lw=1.5, color="tab:purple")
    ax.set_yscale("log")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("| |q| - 1 |")
    ax.set_title("Norm error, renormalised every step")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Tumble plot -> {save_path}")
    plt.close()
    return ts, hist


# ======================================================================
def plot_gimbal_lock(save_path=None):
    """
    A smooth rotation straight through pitch = 90 degrees.

    The body turns at a constant 0.6 rad/s about a single axis -- about as
    benign a motion as exists. The Euler description of it is not benign at
    all, and the middle panel is the reason this project stores quaternions.
    """
    save_path = save_path or os.path.join(RESULTS, "day13_gimbal_lock.png")
    dt = 0.002
    s0 = make_initial_state(quat=euler_to_quat(0.0, np.radians(60.0), 0.0),
                            omega=(0.0, 0.6, 0.0))
    ts, hist = propagate_3d(s0, zero_accel, zero_alpha, (0.0, 2.0), dt)
    eul = np.degrees(np.array([quat_to_euler(s[IDX_QUAT]) for s in hist]))
    rate = np.abs(np.diff(eul, axis=0)).max(axis=1) / dt
    ang = np.degrees([quat_angle_between(hist[0][IDX_QUAT], s[IDX_QUAT])
                      for s in hist])

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("Day 13: gimbal lock is a property of the coordinates, "
                 "not of the motion", fontsize=13)

    a = ax[0]
    for col, lab, c in zip(range(3), ("roll", "pitch", "yaw"),
                           ("tab:red", "tab:green", "tab:blue")):
        a.plot(ts, eul[:, col], lw=1.5, color=c, label=lab)
    a.axhline(90, color="k", ls=":", alpha=0.6, label="pitch = 90")
    a.set_xlabel("Time [s]"); a.set_ylabel("[deg]")
    a.set_title("Euler angles"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1]
    a.semilogy(ts[:-1], np.maximum(np.radians(rate), 1e-6), lw=1.5,
               color="tab:red")
    a.axhline(0.6, color="k", ls="--", alpha=0.7,
              label="actual body rate 0.6 rad/s")
    a.set_xlabel("Time [s]"); a.set_ylabel("[rad/s] (log)")
    a.set_title("Rate the Euler description demands")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[2]
    a.plot(ts, ang, lw=1.5, color="tab:blue")
    a.set_xlabel("Time [s]"); a.set_ylabel("[deg]")
    a.set_title("Rotation from the start, in quaternions")
    a.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Gimbal-lock plot -> {save_path}")
    plt.close()
    return float(np.radians(rate).max())


# ======================================================================
def experiment_a():
    """Composition order matters, and by how much."""
    print(f"\nEXPERIMENT A - Rotations do not commute\n{RULE}")
    q1 = quat_from_axis_angle([1, 0, 0], np.pi / 2)
    q2 = quat_from_axis_angle([0, 1, 0], np.pi / 2)
    ab = quat_multiply(q1, q2)
    ba = quat_multiply(q2, q1)
    sep = np.degrees(quat_angle_between(ab, ba))
    print(f"  x-then-y and y-then-x differ by {sep:.1f} deg")
    v = np.array([0, 0, 1.0])
    print(f"    +z lands at {np.round(quat_rotate_vector(ab, v), 3).tolist()} "
          f"one way")
    print(f"    and         {np.round(quat_rotate_vector(ba, v), 3).tolist()} "
          f"the other")
    same = quat_from_axis_angle([1, 0, 0], 0.4)
    print(f"  about a shared axis they do commute: "
          f"{quats_equal(quat_multiply(q1, same), quat_multiply(same, q1))}")
    return sep


def experiment_b():
    """
    Angular velocity in the wrong frame.

    The kinematics take omega in the body frame. Feeding an inertially-fixed
    rate instead is the single most common three-dimensional mistake, and it
    is invisible for a single-axis spin -- which is exactly why it survives
    into code that only ever gets tested on one.
    """
    print(f"\nEXPERIMENT B - Angular velocity in the wrong frame\n{RULE}")

    def inertial_omega(t, s):
        """Rotate the body rate into the body frame from a fixed inertial one."""
        R = np.array([quat_rotate_vector(quat_conjugate(s[IDX_QUAT]), e)
                      for e in np.eye(3)]).T
        target = R @ np.array([0.0, 0.0, 0.9])
        return (target - s[IDX_OMEGA]) * 50.0     # drive omega toward it

    for label, omega0, alpha in (
            ("single axis, body", (0.0, 0.0, 0.9), zero_alpha),
            ("single axis, inertial", (0.0, 0.0, 0.9), inertial_omega),
            ("tilted axis, body", (0.7, 0.0, 0.9), zero_alpha),
            ("tilted axis, inertial", (0.7, 0.0, 0.9), inertial_omega)):
        s0 = make_initial_state(omega=omega0)
        _, h = propagate_3d(s0, zero_accel, alpha, (0.0, 6.0), 0.002)
        end = np.degrees(quat_angle_between(h[0][IDX_QUAT], h[-1][IDX_QUAT]))
        print(f"  {label:>24}: rotated {end:7.2f} deg from the start")
    print("  Identical for a single axis, different once the axis is tilted -")
    print("  which is why a frame bug hides until the motion stops being planar.")


def experiment_c(peak_rate):
    """Quantify what gimbal lock would do to a controller."""
    print(f"\nEXPERIMENT C - What the singularity costs\n{RULE}")
    print(f"  body rate                    0.600 rad/s, constant")
    print(f"  peak rate Euler angles need  {peak_rate:.1f} rad/s")
    print(f"  ratio                        {peak_rate / 0.6:.0f}x")
    print("  A controller reading Euler angles would have to track that spike")
    print("  to follow a motion the vehicle finds completely unremarkable.")


def experiment_d():
    """The Day 5 flip, expressed in quaternions."""
    print(f"\nEXPERIMENT D - Day 5's flip, in quaternions\n{RULE}")
    theta0 = np.radians(70.0)
    q_start = quat_from_axis_angle([0, 0, 1], theta0)
    q_end = quat_from_axis_angle([0, 0, 1], 0.0)
    rel = quat_multiply(quat_conjugate(q_start), q_end)
    axis, angle = quat_to_axis_angle(rel)
    print(f"  entry attitude   {np.degrees(theta0):.1f} deg from vertical")
    print(f"  relative rotation {np.degrees(angle):.4f} deg about "
          f"{np.round(axis, 6).tolist()}")
    print(f"  matches the planar answer: "
          f"{abs(np.degrees(angle) - 70.0) < 1e-9}")
    # And the same rotation reached by integrating a constant rate, as Day 5's
    # rate-limited flip would fly it.
    t_flip = 2.5
    s0 = make_initial_state(quat=q_start, omega=(0.0, 0.0, -theta0 / t_flip))
    _, h = propagate_3d(s0, zero_accel, zero_alpha, (0.0, t_flip), 0.001)
    err = np.degrees(quat_angle_between(h[-1][IDX_QUAT], q_end))
    print(f"  flown at a constant rate for {t_flip} s, ends "
          f"{err:.2e} deg from vertical")


if __name__ == "__main__":
    print("=" * 78)
    print("DAY 13 - 3-D KINEMATICS DEMONSTRATION")
    print("=" * 78)
    plot_tumble()
    peak = plot_gimbal_lock()
    experiment_a()
    experiment_b()
    experiment_c(peak)
    experiment_d()
    print()
