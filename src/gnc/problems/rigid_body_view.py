"""
Day 14 -- Euler's equations, in the viewer.

Day 13 gave the project a way to track any 3-D orientation. This gives it
something to track: a real inertia tensor, a two-axis engine gimbal, and the
`omega x (I omega)` coupling term that has no planar equivalent at all. Thrust
and gravity only -- aerodynamics is Day 15, laid on top of this rather than
built into it.

**The gyroscopic switch is the day's whole argument.** Turn it off and the
vehicle obeys `tau = I alpha` per axis, which is Day 5's model with three of
them instead of one. Turn it on and the axes couple. At **zero roll rate the
two are bit-for-bit identical** -- not close, identical, because omega then
lies on a single principal axis, omega and `I omega` are parallel, and their
cross product is exactly zero. So every one of the twelve planar days behind
this one was already solving Euler's equations correctly without knowing it.

Give the vehicle any roll at all and that stops being true immediately. At
0.1 rad/s -- under 6 deg/s, a rounding error of a disturbance -- the same burn
ends 18 degrees and 17 metres away from where the uncoupled model says. The
divergence peaks near 1 rad/s and *falls* beyond it, which is gyroscopic
stiffening: a fast enough spin resists being turned.

**And the vehicle cannot undo a roll.** The gimbal torque is `r x F` with `r`
along the body long axis, so its roll component is identically zero at every
deflection -- zero by construction, not by tolerance. Roll it and the roll
stays. That pairing is the honest state of the model: an axis it has no
authority over, which changes the answer by tens of degrees when disturbed.
Real vehicles buy roll control by throttling several engines differentially;
this project has modelled the engines as one effective thruster since Day 2.

**Breaking axisymmetry is worth doing once.** Push the yaw inertia off the
pitch inertia and the roll axis stops decoupling. Take the thrust to zero and
spin about the pitch axis, now the intermediate one, and the vehicle flips
itself end over end -- the tennis-racket theorem, which nothing in this project
codes for and which falls straight out of the coupling term.

One thing this entry cannot show, recorded in `LOG.md` instead: the reduction
check against Day 5 found that `dynamics_6dof` has the gimbal torque sign
inconsistent with its own thrust tilt. It is not fixed, because it is
load-bearing for Days 5 to 12. This entry uses the corrected sign, so its
rotation direction for a given gimbal command is the opposite of every earlier
viewer entry's.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.dynamics_3d import (
    Vehicle3D, propagate_3d_dynamics, make_initial_state_3d,
    gimbal_force_and_torque_body, gyroscopic_term,
    angular_momentum_inertial, attitude_from_pitch, tilt_from_vertical,
)
from src.quaternion import (
    quat_to_rotmatrix, rotmatrix_to_quat, quat_normalize, quat_angle_between,
)

from ..registry import Problem, register
from ..types import Param, Series, Trajectory

# Change of basis into the renderer, solved rather than guessed.
#
# The simulation frame is z-up with the vehicle's long axis on body +x. The
# renderer is y-up with its vehicle model built along body +y. Both halves have
# to move: C takes the inertial frame across, D takes the body frame across,
# and the attitude is R_view = C R_sim D^T.
#
# D is not just "put x where y was" -- the transverse axes have to land in the
# right places too, or out-of-plane motion renders rotated about the long axis
# and looks plausible while being wrong. These two are the matrices for which
# `attitude_from_pitch(theta)` renders identically to `quats_from_pitch(theta)`,
# which is how the twelve planar days already draw the same vehicle.
C_SIM_TO_VIEW = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, -1.0, 0.0]])
D_BODY_TO_VIEW = np.array([[0.0, 0.0, -1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])


@register
class RigidBody3D(Problem):
    slug = "rigid-body-3d"
    title = "3-D Rigid-Body Dynamics"
    summary = ("Euler's equations, a two-axis gimbal, and the coupling term "
               "that is exactly zero until the vehicle rolls.")
    phase = "Day 14"
    scene_scale = 1400.0
    enforces_terminal_state = False

    def params(self) -> list[Param]:
        return [
            Param("include_gyro", "Gyroscopic coupling", True, kind="bool",
                  group="Physics",
                  help="The omega x (I omega) term. Off is not a physical "
                       "model -- it is Day 5's tau = I alpha applied per axis, "
                       "kept so the term's contribution can be measured by "
                       "difference rather than argued about."),
            Param("roll_rate", "Entry roll rate", 0.0, min=0.0, max=3.0,
                  step=0.05, unit="rad/s", group="Physics",
                  help="At exactly zero the two physics settings agree "
                       "bit-for-bit, which is why twelve planar days never "
                       "needed this term. The gimbal cannot remove a roll, so "
                       "whatever you set here stays for the whole burn."),
            Param("I_yaw_ratio", "Yaw inertia / pitch inertia", 1.0, min=0.5,
                  max=2.0, step=0.05, group="Physics",
                  help="1.0 is the axisymmetric rocket. Move it and the roll "
                       "axis stops decoupling; combined with zero throttle and "
                       "a pitch-axis spin it produces the tennis-racket flip."),

            Param("throttle", "Throttle", 0.70, min=0.0, max=1.0, step=0.05,
                  group="Control",
                  help="Zero is a torque-free tumble, where angular momentum "
                       "and rotational energy are both conserved exactly and "
                       "the diagnostics below say by how much."),
            Param("delta_y_deg", "Gimbal, pitch axis", 2.0, min=-15.0,
                  max=15.0, step=0.5, unit="deg", group="Control",
                  help="Day 5's single gimbal axis. Held constant for the "
                       "whole burn -- this is a dynamics model, not a "
                       "guidance law, so nothing here closes the loop."),
            Param("delta_z_deg", "Gimbal, yaw axis", 0.0, min=-15.0, max=15.0,
                  step=0.5, unit="deg", group="Control",
                  help="The new axis. Leave it at zero and the motion stays "
                       "exactly planar; move it and the vehicle leaves the "
                       "plane every earlier day was confined to."),
            Param("t_burn", "Duration", 6.0, min=1.0, max=20.0, step=0.5,
                  unit="s", group="Control"),

            Param("z0", "Entry altitude", 800.0, min=200.0, max=2500.0,
                  step=50.0, unit="m", group="Entry state"),
            Param("vx0", "Entry downrange speed", -30.0, min=-120.0, max=0.0,
                  step=5.0, unit="m/s", group="Entry state"),
            Param("vz0", "Entry descent rate", -80.0, min=-160.0, max=0.0,
                  step=5.0, unit="m/s", group="Entry state"),
            Param("theta0_deg", "Entry pitch", 70.0, min=0.0, max=120.0,
                  step=5.0, unit="deg", group="Entry state",
                  help="Measured from vertical, Day 5's convention. 70 is the "
                       "attitude the flip starts from; 90 is the belly-flop."),

            Param("dt", "Integrator step", 0.01, min=0.002, max=0.05,
                  step=0.002, unit="s", group="Integration",
                  help="RK4 on this model is fourth order and converges fast, "
                       "so the default is set for a smooth animation rather "
                       "than for accuracy, which it has to spare."),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)
        gyro = bool(p["include_gyro"])

        t0 = time.perf_counter()
        try:
            shown = _run(p, gyro)
            other = _run(p, not gyro)
        except Exception as exc:                                # noqa: BLE001
            return _failed([f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        return _trajectory(shown, other, p, gyro, elapsed)


# ======================================================================
def _vehicle(p) -> Vehicle3D:
    base = Vehicle3D()
    ratio = float(p["I_yaw_ratio"])
    return Vehicle3D(I_yaw=None if abs(ratio - 1.0) < 1e-12
                     else ratio * base.I_pitch_yaw)


def _run(p, gyro: bool) -> dict[str, Any]:
    """Propagate the burn, stopping at the ground."""
    v = _vehicle(p)
    T = float(p["throttle"]) * v.T_max
    dy = np.radians(float(p["delta_y_deg"]))
    dz = np.radians(float(p["delta_z_deg"]))

    s0 = make_initial_state_3d(
        pos=(0.0, 0.0, float(p["z0"])),
        vel=(float(p["vx0"]), 0.0, float(p["vz0"])),
        quat=attitude_from_pitch(np.radians(float(p["theta0_deg"]))),
        omega=(float(p["roll_rate"]), 0.0, 0.0),
        vehicle=v)

    ts, hist = propagate_3d_dynamics(
        s0, lambda t, s: (T, dy, dz), (0.0, float(p["t_burn"])),
        float(p["dt"]), v, include_gyro=gyro)

    # Stop at the ground rather than flying underneath it. Keep at least two
    # samples so the payload stays a trajectory.
    below = np.flatnonzero(hist[:, 2] < 0.0)
    if below.size:
        cut = max(int(below[0]) + 1, 2)
        ts, hist = ts[:cut], hist[:cut]
        hist[-1, 2] = max(hist[-1, 2], 0.0)

    return {"t": ts, "hist": hist, "v": v, "T": T, "dy": dy, "dz": dz,
            "gyro": gyro, "grounded": bool(below.size)}


# ======================================================================
def _to_view_vec(v3) -> list[float]:
    v = C_SIM_TO_VIEW @ np.asarray(v3, dtype=float)
    return [float(v[0]), float(v[1]), float(v[2])]


def _to_view_quat(q_sim) -> list[float]:
    R = C_SIM_TO_VIEW @ quat_to_rotmatrix(
        quat_normalize(np.asarray(q_sim, dtype=float))) @ D_BODY_TO_VIEW.T
    w, x, y, z = rotmatrix_to_quat(R)
    return [float(x), float(y), float(z), float(w)]


def _trajectory(r, other, p, gyro, elapsed) -> Trajectory:
    ts, hist, v = r["t"], r["hist"], r["v"]
    n = len(ts)
    pos = hist[:, 0:3].copy()
    pos[:, 2] = np.maximum(pos[:, 2], 0.0)
    vel = hist[:, 3:6]
    quats = hist[:, 6:10]
    omega = hist[:, 10:13]
    mass = hist[:, 13]

    # Thrust in the inertial frame, one sample per control interval.
    F_body = gimbal_force_and_torque_body(r["T"], r["dy"], r["dz"], v)[0]
    thrust = np.array([quat_to_rotmatrix(q) @ F_body for q in quats[:-1]])

    tilt = np.degrees([tilt_from_vertical(q) for q in quats])
    gyro_mag = np.array([np.linalg.norm(gyroscopic_term(w, v.I_body))
                         for w in omega]) / v.tau_max
    L = np.array([angular_momentum_inertial(s, v) for s in hist])
    L_drift = (float(np.linalg.norm(L - L[0], axis=1).max()
                     / max(np.linalg.norm(L[0]), 1e-9))
               if np.linalg.norm(L[0]) > 1e-9 else 0.0)

    # The other physics setting, same everything else.
    o = other["hist"]
    k = min(len(o), n) - 1
    d_att = float(np.degrees(quat_angle_between(quats[k], o[k, 6:10])))
    d_pos = float(np.linalg.norm(pos[k] - o[k, 0:3]))

    return Trajectory(
        t_state=ts.tolist(),
        t_control=ts[:-1].tolist(),
        position=[_to_view_vec(x) for x in pos],
        velocity=[_to_view_vec(x) for x in vel],
        thrust=[_to_view_vec(x) for x in thrust],
        attitude=[_to_view_quat(q) for q in quats],
        series=[
            Series("altitude", "Altitude", "m", pos[:, 2].tolist()),
            Series("speed", "Speed", "m/s",
                   np.linalg.norm(vel, axis=1).tolist()),
            Series("tilt", "Tilt from vertical", "deg", tilt.tolist()),
            Series("out_of_plane", "Out-of-plane position", "m",
                   pos[:, 1].tolist()),
            Series("roll_rate", "Roll rate", "deg/s",
                   np.degrees(omega[:, 0]).tolist()),
            Series("pitch_rate", "Pitch rate", "deg/s",
                   np.degrees(omega[:, 1]).tolist()),
            Series("yaw_rate", "Yaw rate", "deg/s",
                   np.degrees(omega[:, 2]).tolist()),
            Series("gyro_frac", "Coupling torque / tau_max", "-",
                   gyro_mag.tolist()),
            Series("mass", "Vehicle mass", "kg", mass.tolist()),
        ],
        status="grounded" if r["grounded"] else "propagated",
        feasible=True,
        cost=float(v.m_wet - mass[-1]),
        solve_time_ms=elapsed,
        solver=("Euler's equations, coupling included" if gyro
                else "per-axis tau = I alpha, coupling dropped"),
        thrust_max=v.T_max,
        notes=_notes(r, other, p, gyro, d_att, d_pos, gyro_mag, L_drift, tilt),
        diagnostics={
            "physics": "full Euler" if gyro else "coupling dropped",
            "axisymmetric": bool(v.is_axisymmetric),
            "final_tilt_deg": float(tilt[-1]),
            "final_altitude_m": float(pos[-1, 2]),
            "out_of_plane_m": float(np.abs(pos[:, 1]).max()),
            "peak_coupling_frac": float(gyro_mag.max()),
            "fuel_kg": float(v.m_wet - mass[-1]),
            "angular_momentum_drift": L_drift,
            "attitude_vs_other_physics_deg": d_att,
            "position_vs_other_physics_m": d_pos,
        },
    )


def _notes(r, other, p, gyro, d_att, d_pos, gyro_mag, L_drift, tilt) -> list:
    v = r["v"]
    roll = float(p["roll_rate"])
    notes = [
        f"Open-loop dynamics, not guidance. The gimbal is held at "
        f"{float(p['delta_y_deg']):.1f} deg pitch and "
        f"{float(p['delta_z_deg']):.1f} deg yaw for the whole burn and nothing "
        f"closes the loop, so the vehicle will happily tumble -- it ends at "
        f"{tilt[-1]:.1f} deg from vertical. What is being shown is the physics, "
        f"not a landing."
    ]

    if roll < 1e-12:
        notes.append(
            f"With no roll rate the two physics settings agree to "
            f"{d_att:.2e} deg and {d_pos:.2e} m -- that is, exactly. The "
            f"coupling term is not small here, it is zero: omega sits on a "
            f"single principal axis, so omega and I omega are parallel and "
            f"their cross product vanishes, and a pitch-axis torque never "
            f"moves omega off that axis. Every planar day before this one was "
            f"already solving Euler's equations correctly by accident. Nudge "
            f"the roll rate and watch that stop being true."
        )
    else:
        notes.append(
            f"At {roll:.2f} rad/s of roll ({np.degrees(roll):.1f} deg/s) the "
            f"two physics settings end {d_att:.2f} deg and {d_pos:.1f} m "
            f"apart, with the coupling torque peaking at "
            f"{gyro_mag.max() * 100:.1f}% of the engine's full torque "
            f"authority. A term worth a few per cent of the control authority "
            f"moves the answer by tens of degrees because it acts for the "
            f"whole burn and nothing here corrects for it."
        )
        notes.append(
            "Note which way that trend runs. The divergence grows with roll "
            "rate up to about 1 rad/s and falls away beyond it -- past that "
            "the spin is fast enough to resist being turned at all, which is "
            "gyroscopic stiffening rather than the term becoming unimportant."
        )

    notes.append(
        f"The gimbal cannot undo this. Torque is r x F with r along the body "
        f"long axis, so the roll component is identically zero at every "
        f"deflection -- zero by construction, not by tolerance, checked across "
        f"10,201 deflection pairs in the test suite. Whatever roll the vehicle "
        f"arrives with, it keeps. Roll authority needs several engines "
        f"throttled differentially, and this project has modelled them as one "
        f"effective thruster since Day 2."
    )

    if abs(float(p["delta_z_deg"])) < 1e-9 and roll < 1e-12:
        notes.append(
            f"Yaw gimbal and roll rate both zero, so this run is exactly "
            f"planar: out-of-plane position stays at "
            f"{float(np.abs(r['hist'][:, 1]).max()):.2e} m. That is the "
            f"reduction the whole day rests on -- Days 1 to 12 live in this "
            f"plane, and a 3-D model that drifted out of it would put them in "
            f"question. Move the yaw gimbal to leave it."
        )

    spinning = float(np.abs(r["hist"][:, 10:13]).max()) > 1e-9
    if float(p["throttle"]) < 1e-9 and spinning:
        notes.append(
            f"Engine off, so this is a torque-free tumble and angular momentum "
            f"is a conserved *vector* in the inertial frame -- Poinsot's "
            f"theorem. Measured drift over this run: {L_drift:.2e} relative. "
            f"The body-frame rates below precess the whole time while that "
            f"stays fixed, which is the part worth watching."
        )
    elif float(p["throttle"]) < 1e-9:
        notes.append(
            "Engine off and nothing rotating, so this is a ballistic arc and "
            "the attitude never changes. Add a roll rate, or a yaw gimbal and "
            "some throttle, to get a tumble worth looking at -- with the "
            "engine off it will conserve angular momentum exactly."
        )

    if not v.is_axisymmetric:
        notes.append(
            f"Axisymmetry is broken (I_yaw / I_pitch = "
            f"{float(p['I_yaw_ratio']):.2f}), so the roll axis no longer "
            f"decouples. Set the throttle to zero and the entry pitch rate "
            f"comes from roll only, and this becomes the tennis-racket "
            f"theorem: spins about the largest and smallest inertia axes hold, "
            f"spins about the intermediate one flip end over end. Nothing in "
            f"this project codes for that -- it falls out of the coupling term."
        )

    notes.append(
        "One caveat about direction. The reduction check against Day 5 found "
        "that dynamics_6dof pairs Tx = T sin(theta + delta) with "
        "tau = +T L sin(delta), and those two are not compatible -- r x F "
        "gives the opposite sign. This entry uses the corrected sign, so for "
        "the same gimbal command it rotates the vehicle the opposite way to "
        "every earlier entry in this viewer. Day 5 is not fixed, because it is "
        "load-bearing for Days 5 to 12; see LOG.md."
    )
    return notes


def _failed(notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status="error", feasible=False,
        solver="3-D rigid body", notes=notes,
    )
