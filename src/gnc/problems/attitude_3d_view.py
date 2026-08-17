"""
Day 13 — attitude in three dimensions, in the viewer.

Twelve days of this project ran on a single angle. A planar vehicle has one
number for its attitude, that number integrates as `dtheta/dt = omega`, and
nothing about it can ever go wrong in an interesting way. Three dimensions
remove all three of those comforts at once: rotations stop commuting, the
angular velocity has to be *in* a frame, and every three-parameter description
of an orientation has a singularity somewhere.

This entry is the only one in the viewer with no optimiser behind it and no
landing to fly. Nothing here computes a force. Translational acceleration and
body angular acceleration are prescribed from outside, so the vehicle simply
holds station or falls, and turns at whatever rate you set. That is deliberate:
every three-dimensional bug from here on is going to be a frame confusion or a
sign error, and those are far easier to see in a model whose answers can be
checked against closed-form rotations than in one where the forces are also in
question. Day 14's Euler equations are where the physics arrives.

**What to actually look at.** Set the scenario to *through gimbal lock* and
watch the telemetry strip. The vehicle turns at a constant rate about one axis
-- about as benign a motion as exists -- and the tilt trace shows exactly that,
a straight line through 90 degrees with nothing happening at it. The roll,
pitch and yaw traces of the same motion tear themselves apart at that instant,
and the Euler-rate trace spikes to 524x the body rate at the default step -- and
to twice that if you halve the step, because the true rate there is unbounded
and all a finite step can do is put a floor under how badly it reads. The motion
is smooth; the coordinates are not. That difference is the reason this project
stores a quaternion.

**The frame control is the day's other lesson, and it is sharper than the Day
13 write-up made it.** Switch *Rate frame* to `inertial (the bug)` and the same
vector is read as fixed in the world rather than fixed in the vehicle. Building
this entry showed that the two readings are not merely *similar* on easy cases
-- they are bit-for-bit identical on a whole family of them, because a body
rate composes onto the right of the initial attitude and an inertial rate onto
the left, and those agree exactly whenever the two commute. A zero rate, an
upright start, and a rate parallel to the axis the vehicle is already tilted
about are all in that family. So are three of the four scenarios here, and so
is every planar case this project ran for twelve days. Only *tumble from a
tilt* separates them, by 28 deg.

That also corrects the Day 13 write-up. Experiment B in `demo_3d_kinematics`
compares a body rate of (0.7, 0, 0.9) against an inertial rate of (0, 0, 0.9)
and reports 31.96 against 50.61 deg -- but those are two different rate
vectors, so the comparison mixes the frame with the vector and does not isolate
the frame at all. Its single-axis half stands; the tilted half does not. This
entry holds the vector fixed and changes only how it is read.

**Renormalisation is a smaller effect than it is usually sold as.** Turn it off
and the norm trace drifts, but at a fine step it drifts by parts in a trillion,
not visibly. The honest argument for renormalising is not that the solution
falls apart quickly. It is that the error is one-sided: it accumulates in one
direction rather than averaging out, and an un-normalised quaternion silently
stops representing a pure rotation while still looking like one.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.quaternion import (
    quat_multiply, quat_conjugate, quat_normalize, quat_from_axis_angle,
    quat_rotate_vector, quat_to_euler, quat_to_axis_angle, quat_angle_between,
    euler_to_quat,
)
from src.dynamics_3d_kinematics import (
    make_initial_state, propagate_3d, zero_alpha, gravity_accel, zero_accel,
    IDX_POS, IDX_VEL, IDX_QUAT,
)

from ..registry import Problem, register
from ..types import Param, Series, Trajectory

# The body long axis. This model has no vehicle in it -- it rotates a frame,
# not a rocket -- so the choice is a rendering convention, picked to agree with
# the planar days: upright means the long axis is along inertial +z.
BODY_LONG = np.array([0.0, 0.0, 1.0])
UP_SIM = np.array([0.0, 0.0, 1.0])

# Simulation frame is z-up; the renderer is y-up. The change of basis is a
# proper rotation, -90 degrees about +x: (x, y, z) -> (x, z, -y). Using an axis
# *swap* instead would have determinant -1 and would quietly mirror every
# rotation in the scene -- the render would look plausible and turn the wrong
# way. Applying the same rotation to the body basis keeps the long axis on the
# renderer's +y, which is what its vehicle model is built along.
Q_SIM_TO_VIEW = quat_from_axis_angle([1.0, 0.0, 0.0], -np.pi / 2.0)

SCENARIOS = ["through gimbal lock", "tumble from a tilt", "single-axis spin",
             "Day 5 flip (70 deg)", "custom"]


@register
class Attitude3D(Problem):
    slug = "attitude-3d"
    title = "3-D Attitude Kinematics"
    summary = ("Quaternions, and the singularity in the Euler angles that "
               "twelve planar days could not have found.")
    phase = "Day 13"
    scene_scale = 300.0
    enforces_terminal_state = False

    def params(self) -> list[Param]:
        return [
            Param("scenario", "Scenario", "through gimbal lock", kind="choice",
                  choices=SCENARIOS, group="Motion",
                  help="Each preset sets the initial attitude and the body "
                       "rate below; choose custom to drive them yourself."),
            Param("duration", "Duration", 6.0, min=1.0, max=20.0, step=0.5,
                  unit="s", group="Motion"),

            Param("wx", "Body rate, x (roll)", 0.0, min=-120.0, max=120.0,
                  step=1.0, unit="deg/s", group="Custom motion",
                  help="Used only when the scenario is custom. This is the "
                       "rate a strapdown gyro reports: components along the "
                       "vehicle's own axes, not the world's."),
            Param("wy", "Body rate, y (pitch)", 34.38, min=-120.0, max=120.0,
                  step=1.0, unit="deg/s", group="Custom motion"),
            Param("wz", "Body rate, z (yaw)", 0.0, min=-120.0, max=120.0,
                  step=1.0, unit="deg/s", group="Custom motion"),
            Param("pitch0_deg", "Initial pitch", 60.0, min=-89.0, max=89.0,
                  step=1.0, unit="deg", group="Custom motion",
                  help="Pitch is the angle that runs into the singularity at "
                       "+/-90, which for this vehicle is lying flat -- the "
                       "belly-flop attitude."),
            Param("roll0_deg", "Initial roll", 0.0, min=-180.0, max=180.0,
                  step=5.0, unit="deg", group="Custom motion"),
            Param("yaw0_deg", "Initial yaw", 0.0, min=-180.0, max=180.0,
                  step=5.0, unit="deg", group="Custom motion"),

            Param("frame", "Rate frame", "body", kind="choice",
                  choices=["body", "inertial (the bug)"], group="Frame",
                  help="The kinematics take omega in the body frame. Treating "
                       "an inertially-fixed rate as if it were a body rate is "
                       "the classic first three-dimensional bug. It is exactly "
                       "invisible unless the starting attitude fails to "
                       "commute with the rotation -- only 'tumble from a tilt' "
                       "shows it here."),

            Param("dt", "Integrator step", 0.01, min=0.005, max=0.25,
                  step=0.005, unit="s", group="Integration"),
            Param("renormalize", "Renormalise each step", True, kind="bool",
                  group="Integration",
                  help="RK4 knows nothing about the unit-norm constraint. "
                       "Switch this off and the norm trace shows how far it "
                       "wanders -- which is less than the usual telling of "
                       "this suggests, and one-sided, which is the real "
                       "problem."),

            Param("translation", "Translation", "hover", kind="choice",
                  choices=["hover", "ballistic"], group="Translation",
                  help="Prescribed, and completely uncoupled from the "
                       "rotation. Nothing in this model computes a force, so "
                       "the vehicle's path does not depend on where it is "
                       "pointing. Day 14 changes that."),
            Param("altitude", "Start altitude", 150.0, min=20.0, max=400.0,
                  step=10.0, unit="m", group="Translation"),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)
        wrong = str(p["frame"]).startswith("inertial")

        t0 = time.perf_counter()
        try:
            shown = _run(p, wrong=wrong)
            other = _run(p, wrong=not wrong)
        except Exception as exc:                                # noqa: BLE001
            return _failed([f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        return _trajectory(shown, other, p, wrong, elapsed)


# ======================================================================
# Scenario setup
# ======================================================================
def _scenario(p) -> tuple[np.ndarray, np.ndarray]:
    """Initial quaternion and constant rate vector, in rad and rad/s."""
    name = str(p["scenario"])
    if name == "through gimbal lock":
        # Starts 30 degrees short of the singularity and drives straight
        # through it at 0.6 rad/s. The motion is a single-axis rotation; only
        # the description of it misbehaves.
        return euler_to_quat(0.0, np.radians(60.0), 0.0), np.array([0.0, 0.6, 0.0])
    if name == "tumble from a tilt":
        # Starts tilted about an axis the rate is not parallel to, which is
        # the only configuration in which the frame error is visible at all.
        return (euler_to_quat(np.radians(25.0), np.radians(40.0), 0.0),
                np.array([0.35, 0.20, 0.85]))
    if name == "single-axis spin":
        return euler_to_quat(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.9])
    if name == "Day 5 flip (70 deg)":
        # Day 5's entry attitude, flipped back to vertical at a constant rate.
        # A prescribed rate does not stop when it arrives, so past 2.5 s the
        # vehicle keeps going and tips over the far side.
        theta0 = np.radians(70.0)
        return (quat_from_axis_angle([0.0, 1.0, 0.0], theta0),
                np.array([0.0, -theta0 / 2.5, 0.0]))
    return (euler_to_quat(np.radians(float(p["roll0_deg"])),
                          np.radians(float(p["pitch0_deg"])),
                          np.radians(float(p["yaw0_deg"]))),
            np.radians([float(p["wx"]), float(p["wy"]), float(p["wz"])]))


def _run(p, wrong: bool) -> dict[str, Any]:
    """Propagate the attitude, one frame convention or the other."""
    q0, omega = _scenario(p)
    dt = float(p["dt"])
    T = float(p["duration"])
    alt = float(p["altitude"])
    ballistic = str(p["translation"]) == "ballistic"

    s0 = make_initial_state(pos=(0.0, 0.0, alt), quat=q0, omega=omega)
    accel = gravity_accel() if ballistic else zero_accel
    ts, hist = propagate_3d(s0, accel, zero_alpha, (0.0, T), dt,
                            renormalize=bool(p["renormalize"]))
    quats = hist[:, IDX_QUAT].copy()

    # A constant rate has a closed form in both readings, and the difference
    # between them is the whole lesson. Read as a *body* rate the rotation
    # composes on the right of the initial attitude; read as an *inertial* rate
    # it composes on the left. Neither is integrated here, so the comparison
    # carries no integration error -- it is the frame and nothing else.
    mag = float(np.linalg.norm(omega))
    if mag < 1e-12:
        exact_body = np.repeat(q0[None, :], len(ts), axis=0)
        exact_inertial = exact_body
    else:
        axis = omega / mag
        steps = [quat_from_axis_angle(axis, mag * t) for t in ts]
        exact_body = np.array([quat_multiply(q0, e) for e in steps])
        exact_inertial = np.array([quat_multiply(e, q0) for e in steps])
    if wrong:
        quats = exact_inertial

    # The integrator against the closed form for the same body rate. Always
    # measured on the body reading, since that is the only one this file
    # actually integrates.
    int_err = float(np.degrees(max(
        _small_angle(a, b) for a, b in zip(hist[:, IDX_QUAT], exact_body))))

    return {"t": ts, "state": hist, "quats": quats, "q0": q0, "omega": omega,
            "ballistic": ballistic, "wrong": wrong, "int_err": int_err}


def _small_angle(q1, q2) -> float:
    """
    Rotation angle between two nearly-identical orientations [rad].

    `quat_angle_between` goes through arccos of the scalar part, which is fine
    at large separations and useless at small ones: an error of 1e-9 rad puts
    the scalar part within 1e-19 of 1, which rounds to exactly 1, and the
    function dutifully reports zero. Going through the vector part instead is
    well conditioned in exactly the regime an integration error lives in.
    """
    rel = quat_multiply(quat_conjugate(quat_normalize(q1)), quat_normalize(q2))
    if rel[0] < 0.0:
        rel = -rel
    return float(2.0 * np.arcsin(min(float(np.linalg.norm(rel[1:])), 1.0)))


def _degeneracy(q0, omega) -> str | None:
    """
    Why the two frame readings can come out identical.

    They differ only when the initial attitude fails to commute with the
    rotation the rate generates. That rules out three whole families of test
    case -- and the obvious ones to write are all in it.
    """
    mag = float(np.linalg.norm(omega))
    if mag < 1e-12:
        return "the rate is zero"
    axis0, ang0 = quat_to_axis_angle(q0)
    if abs(ang0) < 1e-9:
        return "the vehicle starts upright, so there is nothing for the " \
               "rotation to compose with -- left and right multiplication " \
               "onto the identity are the same operation"
    if abs(abs(float(np.dot(omega / mag, axis0))) - 1.0) < 1e-9:
        return "the rate is parallel to the axis the vehicle is already " \
               "tilted about, and rotations about a shared axis commute"
    return None


# ======================================================================
# Payload
# ======================================================================
def _to_view_quat(q_sim) -> list[float]:
    """Sim-frame scalar-first quaternion to renderer-frame [x, y, z, w]."""
    q = quat_normalize(quat_multiply(
        quat_multiply(Q_SIM_TO_VIEW, quat_normalize(q_sim)),
        quat_conjugate(Q_SIM_TO_VIEW)))
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _to_view_vec(v) -> list[float]:
    return [float(v[0]), float(v[2]), float(-v[1])]


def _euler_rate_deg_s(eul_deg: np.ndarray, dt: float) -> np.ndarray:
    """
    Per-step change in the Euler description, in degrees per second.

    Differences are wrapped to the shortest path so that an ordinary +/-180
    crossing of atan2 does not read as a spike. What survives the wrapping at
    the singularity is real: the description genuinely has to move 180 degrees
    in one step, so this number grows without bound as dt shrinks.
    """
    d = np.diff(eul_deg, axis=0)
    d = (d + 180.0) % 360.0 - 180.0
    return np.abs(d).max(axis=1) / dt


def _trajectory(r, other, p, wrong, elapsed) -> Trajectory:
    ts = r["t"]
    hist = r["state"]
    quats = r["quats"]
    n = len(ts)
    dt = float(p["dt"])

    pos = hist[:, IDX_POS].copy()
    vel = hist[:, IDX_VEL].copy()
    pos[:, 2] = np.maximum(pos[:, 2], 0.0)

    norms = np.linalg.norm(quats, axis=1)
    eul = np.degrees(np.array([quat_to_euler(q) for q in quats]))
    tilt = np.degrees([np.arccos(np.clip(
        float(np.dot(quat_rotate_vector(q, BODY_LONG), UP_SIM)), -1.0, 1.0))
        for q in quats])
    swept = np.degrees([quat_angle_between(r["q0"], q) for q in quats])
    rate = _euler_rate_deg_s(eul, dt)

    # How close the run came to the singularity, and how far the two frame
    # conventions ended up apart.
    lock_margin = float(np.min(np.abs(np.abs(eul[:, 1]) - 90.0)))
    split = float(np.degrees(quat_angle_between(quats[-1], other["quats"][-1])))

    return Trajectory(
        t_state=ts.tolist(),
        t_control=ts[:-1].tolist(),
        position=[_to_view_vec(v) for v in pos],
        velocity=[_to_view_vec(v) for v in vel],
        thrust=np.zeros((n - 1, 3)).tolist(),
        attitude=[_to_view_quat(q) for q in quats],
        series=[
            Series("tilt", "Tilt of long axis from vertical", "deg",
                   [float(v) for v in tilt]),
            Series("swept", "Separation from start (shortest path)", "deg",
                   [float(v) for v in swept]),
            Series("roll", "Roll (ZYX)", "deg", eul[:, 0].tolist()),
            Series("pitch", "Pitch (ZYX)", "deg", eul[:, 1].tolist()),
            Series("yaw", "Yaw (ZYX)", "deg", eul[:, 2].tolist()),
            Series("euler_rate", "Rate the Euler description demands", "deg/s",
                   rate.tolist(), on="control"),
            Series("norm_err", "Quaternion norm error", "-",
                   np.abs(norms - 1.0).tolist()),
            Series("altitude", "Altitude", "m", pos[:, 2].tolist()),
        ],
        status="propagated",
        feasible=True,
        cost=None,
        solve_time_ms=elapsed,
        solver=("kinematics, rate read as inertial" if wrong
                else "kinematics, rate read as body"),
        thrust_max=1.0,
        notes=_notes(r, other, p, wrong, rate, lock_margin, split, norms),
        diagnostics={
            "scenario": str(p["scenario"]),
            "rate_frame": "inertial" if wrong else "body",
            "body_rate_deg_s": [float(v)
                                for v in np.degrees(r["omega"])],
            "final_separation_deg": float(swept[-1]),
            "swept_angle_deg": float(np.degrees(
                np.linalg.norm(r["omega"]) * float(ts[-1]))),
            "final_tilt_deg": float(tilt[-1]),
            "integration_error_deg": float(r["int_err"]),
            "peak_euler_rate_deg_s": float(rate.max()),
            "peak_euler_rate_ratio": float(
                np.radians(rate.max()) / max(float(np.linalg.norm(r["omega"])),
                                             1e-12)),
            "closest_approach_to_lock_deg": lock_margin,
            "max_norm_error": float(np.abs(norms - 1.0).max()),
            "frame_disagreement_deg": split,
        },
    )


def _notes(r, other, p, wrong, rate, lock_margin, split, norms) -> list[str]:
    body_rate = float(np.linalg.norm(r["omega"]))
    ratio = np.radians(float(rate.max())) / max(body_rate, 1e-12)
    dt = float(p["dt"])
    swept_total = np.degrees(body_rate * float(p["duration"]))
    notes = [
        f"Kinematics only -- nothing here computes a force. The body turns at "
        f"a constant {np.degrees(body_rate):.1f} deg/s and the translation is "
        f"prescribed independently, so where the vehicle goes does not depend "
        f"on where it points. Day 14's Euler equations are what couple them."
    ]

    if swept_total > 180.0:
        notes.append(
            f"The separation trace folds back after 180 deg, and that is "
            f"correct rather than a plotting artifact: it is the *shortest* "
            f"rotation between two orientations, and no two orientations are "
            f"ever more than 180 deg apart. This run sweeps "
            f"{swept_total:,.0f} deg of arc and ends "
            f"{np.degrees(quat_angle_between(r['q0'], r['quats'][-1])):.1f} "
            f"deg from where it started."
        )

    if not r["wrong"]:
        notes.append(
            f"The integrated attitude and the closed-form answer for the same "
            f"constant rate agree to {r['int_err']:.2e} deg over this run at "
            f"dt = {dt:g} s. That is the check worth having on this file: a "
            f"rotation at a constant body rate is one of the few cases with an "
            f"exact solution to compare against, and it is the last such case "
            f"before Day 14 makes the rate itself a computed quantity."
        )

    if lock_margin < 5.0:
        notes.append(
            f"This run passes within {lock_margin:.2f} deg of the gimbal-lock "
            f"pitch. The Euler description peaks at {rate.max():,.0f} deg/s, "
            f"{ratio:.0f}x the actual body rate, while the tilt and "
            f"separation traces show nothing at all happening there. "
            f"That ratio is what would break a controller reading angles. "
            f"Read the number with its step size attached: at the crossing the "
            f"description has to move a full 180 deg in a single step, so this "
            f"reading is exactly 180/dt and doubles every time you halve the "
            f"step. That is not a numerical artifact to be tuned away -- it is "
            f"the measurement telling you the true rate there is unbounded."
        )
    else:
        notes.append(
            f"This run stays {lock_margin:.1f} deg clear of the gimbal-lock "
            f"pitch. The Euler description still runs at up to "
            f"{rate.max():,.1f} deg/s, {ratio:.1f}x the body rate -- the "
            f"amplification does not switch on at the singularity, it grows "
            f"as you approach it, and a controller does not have to reach 90 "
            f"deg of pitch to start suffering for it. Switch the scenario to "
            f"'through gimbal lock' to see the limit of that trend."
        )

    why = _degeneracy(r["q0"], r["omega"])
    if why is not None:
        notes.append(
            f"Reading the rate as a body rate and as an inertial rate end "
            f"{split:.3f} deg apart on this motion -- that is, they agree "
            f"exactly, and the bug is not small here but absent. The reason is "
            f"that {why}. Switch to 'tumble from a tilt', where the rate is "
            f"neither zero nor parallel to the starting tilt, and the two "
            f"readings separate immediately."
        )
    else:
        notes.append(
            f"The two frame readings end {split:.1f} deg apart. Neither is "
            f"obviously wrong on its own -- both are smooth, both hold the "
            f"norm, both are perfectly plausible rotations, and only the "
            f"comparison shows it. That is the whole difficulty with a frame "
            f"error: it does not crash, it does not drift, it just quietly "
            f"points somewhere else."
        )
        notes.append(
            "The condition for the two to differ at all is that the initial "
            "attitude fails to commute with the rotation the rate generates. "
            "Three families of test case are therefore blind to it by "
            "construction: a zero rate, an upright start, and a rate parallel "
            "to the axis the vehicle is already tilted about. The obvious "
            "cases to write -- spin from upright, flip about the pitch axis -- "
            "are all in that list, and so is every planar case this project "
            "ran for twelve days."
        )

    drift = float(np.abs(norms - 1.0).max())
    if not bool(p["renormalize"]):
        notes.append(
            f"Renormalisation is off. The norm wandered by {drift:.2e} over "
            f"this run at dt = {dt:g} s -- coarsen the step and watch that "
            f"grow steeply, since it is RK4 truncation and nothing else. Which "
            f"means the usual telling of this, that the solution falls apart "
            f"without renormalising, is overstated at any sane step size. The "
            f"real argument is that the error is one-sided: it accumulates in "
            f"one direction rather than averaging out, and the quaternion "
            f"stops representing a pure rotation while still looking like one."
        )
    if r["wrong"]:
        notes.append(
            "The inertial-frame path is computed in closed form rather than "
            "integrated, so the difference you see between the two frames is "
            "the frame alone and carries no integration error."
        )

    notes.append(
        "The rendered attitude is always normalised for display, whatever the "
        "renormalise control says, because a non-unit quaternion is not a "
        "rotation and there is nothing sensible to draw. The norm trace "
        "carries the raw value."
    )
    notes.append(
        "None of the twelve days before this one could have hit the "
        "singularity shown here. A planar vehicle has a single angle, and one "
        "angle for one degree of freedom has no singularity to hit -- the "
        "earlier work was immune by construction rather than by luck. What it "
        "was not immune to is the frame error, which would have sat in the "
        "code unseen."
    )
    return notes


def _failed(notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status="error", feasible=False,
        solver="3-D kinematics", notes=notes,
    )
