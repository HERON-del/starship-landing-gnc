"""
Day 15 -- what a crosswind actually does, in the viewer.

Day 6 put aerodynamics in this project as a single drag force with the
reference area blended by one pitch angle. That was a complete description
while the vehicle and the wind both lived in one plane. This entry adds the two
angles a 3-D wind needs -- angle of attack and sideslip -- a real
drag/lift/side-force split, and the piece Day 6 never had at all: aerodynamic
**moments**. A body at an angle to the wind does not just get pushed. It gets
turned.

**Set the lateral wind and leave the yaw gimbal at zero.** Everything that then
happens out of the plane was commanded by nobody. In still air the out-of-plane
state holds at machine zero for the whole flight; add a few metres per second
of crosswind and the vehicle develops sideslip, a yaw rate, and hundreds of
metres of drift.

**But read the impulse split before deciding what caused it.** The obvious
story is that the aerodynamic side force pushed the vehicle downwind. The
diagnostics say otherwise: aerodynamics contributes a few MN-seconds downwind,
and the *engine* contributes several times more, upwind. What happens is that
the aerodynamic yaw moment swings the body twenty-odd degrees out of plane, and
4.8 MN of thrust pointed twenty degrees wrong overwhelms every aerodynamic
force in the model. The vehicle ends up upwind, carried there by its own
engine. That is a control problem wearing an aerodynamics costume, and the
planar model could not express it at all.

**The centre of pressure decides which way the vehicle turns.** Positive `x_cp`
puts it aft and the vehicle weathervanes into the wind like an arrow; negative
puts it forward and the vehicle diverges. There is no damping term in this
model, so a stable vehicle oscillates about the wind direction rather than
settling -- a real limitation, not a rendering artefact.

Two things this entry cannot show, both recorded in `LOG.md`. The Day 15 guide
specifies a lift direction that turns the vehicle away from the wind, which
would make an aft centre of pressure destabilising; it is corrected here. And
Day 6's `aero.py` blends its reference area by pitch from vertical rather than
by the angle to the wind, which is wrong by up to a factor of two in either
direction across the envelope -- not fixed, because Days 6 to 12 rest on it.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.aero_3d import (
    AeroConfig3D, propagate_3d_with_aero, angle_history,
    effective_area_and_Cd, aero_force_and_moment_body, static_margin_sign,
)
from src.dynamics_3d import (
    Vehicle3D, make_initial_state_3d, gimbal_force_and_torque_body,
    attitude_from_pitch, tilt_from_vertical,
)
from src.quaternion import quat_to_rotmatrix

from ..registry import Problem, register
from ..types import Param, Series, Trajectory
# One source of truth for the simulation-to-renderer change of basis. Day 14
# solved it and its test suite pins it; importing rather than restating it is
# the same choice this day made about the dynamics itself.
from .rigid_body_view import _to_view_vec, _to_view_quat


@register
class Aero3D(Problem):
    slug = "aero-3d"
    title = "3-D Aerodynamics"
    summary = ("Angle of attack, sideslip, and a crosswind that turns the "
               "vehicle before it pushes it.")
    phase = "Day 15"
    scene_scale = 3000.0
    enforces_terminal_state = False

    def params(self) -> list[Param]:
        return [
            Param("wind_y", "Crosswind, lateral", 8.0, min=-25.0, max=25.0,
                  step=1.0, unit="m/s", group="Wind",
                  help="The only control here that can move the vehicle out "
                       "of the plane. Set it to zero and every out-of-plane "
                       "state stays at machine zero for the whole flight."),
            Param("wind_x", "Wind, downrange", 15.0, min=-30.0, max=30.0,
                  step=1.0, unit="m/s", group="Wind",
                  help="Purely downrange wind changes the angle of attack and "
                       "leaves sideslip at exactly zero -- still a planar "
                       "problem, however strong it is."),
            Param("wind_z", "Wind, vertical", 0.0, min=-15.0, max=15.0,
                  step=1.0, unit="m/s", group="Wind"),

            Param("aero_on", "Aerodynamics", True, kind="bool", group="Aero",
                  help="Off reproduces Day 14 exactly, which is worth doing "
                       "once: the angles are still computed and shown "
                       "but nothing acts on them, and every "
                       "out-of-plane trace goes flat."),
            Param("x_cp", "Centre of pressure offset", 5.0, min=-10.0,
                  max=15.0, step=1.0, unit="m", group="Aero",
                  help="Measured from the centre of mass. Positive is aft, "
                       "which weathervanes the vehicle into the wind; "
                       "negative is forward and it diverges instead. At zero "
                       "there is no aerodynamic moment at all."),
            Param("Cy_beta", "Side-force derivative", 0.8, min=0.0, max=2.0,
                  step=0.1, group="Aero", help="Side force per radian of "
                                               "sideslip. Zero removes the "
                                               "yaw stiffness entirely."),
            Param("Cl_max", "Peak lift coefficient", 0.4, min=0.0, max=1.0,
                  step=0.05, group="Aero"),

            Param("throttle", "Throttle", 0.70, min=0.0, max=1.0, step=0.05,
                  group="Control",
                  help="Zero is the honest way to see the aerodynamics on "
                       "their own -- with the engine off, nothing but air "
                       "touches the vehicle."),
            Param("delta_y_deg", "Gimbal, pitch axis", 4.0, min=-15.0,
                  max=15.0, step=0.5, unit="deg", group="Control"),
            Param("delta_z_deg", "Gimbal, yaw axis", 0.0, min=-15.0, max=15.0,
                  step=0.5, unit="deg", group="Control",
                  help="Leave this at zero. The point of the entry is what "
                       "happens out of the plane when nothing commands it."),
            Param("t_burn", "Duration", 12.0, min=2.0, max=30.0, step=1.0,
                  unit="s", group="Control"),

            Param("z0", "Entry altitude", 3000.0, min=500.0, max=8000.0,
                  step=250.0, unit="m", group="Entry state"),
            Param("vx0", "Entry downrange speed", -20.0, min=-120.0, max=0.0,
                  step=5.0, unit="m/s", group="Entry state"),
            Param("vz0", "Entry descent rate", -90.0, min=-200.0, max=0.0,
                  step=5.0, unit="m/s", group="Entry state"),
            Param("theta0_deg", "Entry pitch", 80.0, min=0.0, max=120.0,
                  step=5.0, unit="deg", group="Entry state",
                  help="From vertical, Day 5's convention. 90 is the "
                       "belly-flop, where the vehicle shows the air seven "
                       "times the area it does nose-on."),

            Param("dt", "Integrator step", 0.02, min=0.005, max=0.05,
                  step=0.005, unit="s", group="Integration",
                  help="The aerodynamic derivative is the expensive "
                       "part of this entry and it runs twice per solve, "
                       "once with the wind and once without. RK4 has "
                       "accuracy to spare at this step."),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)
        t0 = time.perf_counter()
        try:
            shown = _run(p, wind=True)
            calm = _run(p, wind=False)
        except Exception as exc:                                # noqa: BLE001
            return _failed([f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0
        return _trajectory(shown, calm, p, elapsed)


# ======================================================================
def _run(p, wind: bool) -> dict[str, Any]:
    v = Vehicle3D()
    cfg = AeroConfig3D(x_cp=float(p["x_cp"]), Cy_beta=float(p["Cy_beta"]),
                       Cl_max=float(p["Cl_max"]),
                       enabled=bool(p["aero_on"]))
    w = (np.array([float(p["wind_x"]), float(p["wind_y"]), float(p["wind_z"])])
         if wind else np.zeros(3))
    T = float(p["throttle"]) * v.T_max
    dy, dz = (np.radians(float(p["delta_y_deg"])),
              np.radians(float(p["delta_z_deg"])))

    s0 = make_initial_state_3d(
        pos=(0.0, 0.0, float(p["z0"])),
        vel=(float(p["vx0"]), 0.0, float(p["vz0"])),
        quat=attitude_from_pitch(np.radians(float(p["theta0_deg"]))),
        vehicle=v)
    ts, hist = propagate_3d_with_aero(
        s0, lambda t, s: (T, dy, dz), (0.0, float(p["t_burn"])),
        float(p["dt"]), v, cfg, wind_inertial=w)

    below = np.flatnonzero(hist[:, 2] < 0.0)
    grounded = bool(below.size)
    if grounded:
        cut = max(int(below[0]) + 1, 2)
        ts, hist = ts[:cut], hist[:cut]
        hist[-1, 2] = max(hist[-1, 2], 0.0)

    return {"t": ts, "hist": hist, "v": v, "cfg": cfg, "wind": w,
            "T": T, "dy": dy, "dz": dz, "grounded": grounded}


def _impulse_split(r):
    """Out-of-plane impulse, separated into what the air did and what the
    engine did. The whole point of the entry lives in this ratio."""
    v, cfg, hist = r["v"], r["cfg"], r["hist"]
    F_thrust_body = gimbal_force_and_torque_body(
        r["T"], r["dy"], r["dz"], v)[0]
    Fy_t, Fy_a = [], []
    for s in hist:
        R = quat_to_rotmatrix(s[6:10])
        Fy_t.append((R @ F_thrust_body)[1])
        Fy_a.append((R @ aero_force_and_moment_body(
            s[3:6], r["wind"], s[6:10], s[2], cfg)[0])[1])
    dt = float(r["t"][1] - r["t"][0]) if len(r["t"]) > 1 else 0.0
    return (float(np.trapezoid(Fy_a, dx=dt)),
            float(np.trapezoid(Fy_t, dx=dt)))


# ======================================================================
def _trajectory(r, calm, p, elapsed) -> Trajectory:
    ts, hist, v, cfg = r["t"], r["hist"], r["v"], r["cfg"]
    n = len(ts)
    pos = hist[:, 0:3].copy()
    pos[:, 2] = np.maximum(pos[:, 2], 0.0)
    vel, quats, omega, mass = (hist[:, 3:6], hist[:, 6:10],
                               hist[:, 10:13], hist[:, 13])

    F_body = gimbal_force_and_torque_body(r["T"], r["dy"], r["dz"], v)[0]
    thrust = np.array([quat_to_rotmatrix(q) @ F_body for q in quats[:-1]])

    ang = angle_history(hist, r["wind"], cfg)
    CdA = np.array([float(np.prod(effective_area_and_Cd(o, cfg)))
                    for o in ang[:, 3]])
    tilt = np.degrees([tilt_from_vertical(q) for q in quats])
    long_axis = [quat_to_rotmatrix(q) @ np.array([1.0, 0.0, 0.0])
                 for q in quats]
    yaw = np.degrees([np.arctan2(b[1], np.hypot(b[0], b[2]))
                      for b in long_axis])

    imp_aero, imp_thrust = _impulse_split(r)
    k = min(len(calm["hist"]), n) - 1
    drift_vs_calm = float(pos[k, 1] - calm["hist"][k, 1])

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
            Series("out_of_plane", "Out-of-plane position", "m",
                   pos[:, 1].tolist()),
            Series("alpha", "Angle of attack", "deg",
                   np.degrees(ang[:, 0]).tolist()),
            Series("beta", "Sideslip", "deg",
                   np.degrees(ang[:, 1]).tolist()),
            Series("off_axis", "Wind angle off the body axis", "deg",
                   np.degrees(ang[:, 3]).tolist()),
            Series("yaw", "Body yaw out of plane", "deg", yaw.tolist()),
            Series("yaw_rate", "Yaw rate", "deg/s",
                   np.degrees(omega[:, 2]).tolist()),
            Series("CdA", "Drag area Cd*A", "m^2", CdA.tolist()),
            Series("tilt", "Tilt from vertical", "deg", tilt.tolist()),
            Series("mass", "Vehicle mass", "kg", mass.tolist()),
        ],
        status="grounded" if r["grounded"] else "propagated",
        feasible=True,
        cost=float(v.m_wet - mass[-1]),
        solve_time_ms=elapsed,
        solver=("thrust + gravity + 3-D aero" if cfg.enabled
                else "thrust + gravity only (Day 14)"),
        thrust_max=v.T_max,
        notes=_notes(r, calm, p, ang, yaw, imp_aero, imp_thrust,
                     drift_vs_calm, pos),
        diagnostics={
            "aerodynamics": "on" if cfg.enabled else "off",
            "static_margin": static_margin_sign(cfg),
            "out_of_plane_final_m": float(pos[-1, 1]),
            "out_of_plane_vs_still_air_m": drift_vs_calm,
            "peak_sideslip_deg": float(np.degrees(np.abs(ang[:, 1]).max())),
            "peak_alpha_deg": float(np.degrees(np.abs(ang[:, 0]).max())),
            "body_yaw_max_deg": float(np.abs(yaw).max()),
            "aero_impulse_y_MNs": imp_aero / 1e6,
            "thrust_impulse_y_MNs": imp_thrust / 1e6,
            "peak_drag_area_m2": float(CdA.max()),
            "final_tilt_deg": float(tilt[-1]),
            "fuel_kg": float(v.m_wet - mass[-1]),
        },
    )


def _notes(r, calm, p, ang, yaw, imp_aero, imp_thrust, drift, pos) -> list:
    cfg = r["cfg"]
    lateral = abs(float(p["wind_y"])) > 1e-9
    notes = [
        f"Open-loop, not guidance. The gimbal is held at "
        f"{float(p['delta_y_deg']):.1f} deg pitch and "
        f"{float(p['delta_z_deg']):.1f} deg yaw for the whole flight and "
        f"nothing closes the loop, so the vehicle is free to tumble. What is "
        f"being shown is the physics."
    ]

    if not cfg.enabled:
        notes.append(
            "Aerodynamics is off, so this is Day 14 exactly -- thrust and "
            "gravity and nothing else. The angle traces below are still "
            "computed from the state and the wind you set, and they are "
            "still correct; nothing acts on them. The out-of-plane "
            "position stays at zero however hard you blow. Switch it on."
        )
        return notes

    if not lateral:
        notes.append(
            f"No lateral wind, so this run is exactly planar: out-of-plane "
            f"position stays at {float(np.abs(pos[:, 1]).max()):.2e} m. A "
            f"purely downrange or vertical wind moves the angle of attack and "
            f"leaves sideslip at exactly zero -- however strong it is, it is "
            f"still a problem the planar model could have solved. Set the "
            f"lateral wind to something and watch that stop being true."
        )
    else:
        notes.append(
            f"Out-of-plane drift {pos[-1, 1]:+.1f} m, against "
            f"{calm['hist'][-1, 1]:+.2e} m in still air, with the yaw gimbal "
            f"commanded at exactly zero the whole way. Peak sideslip "
            f"{np.degrees(np.abs(ang[:, 1]).max()):.2f} deg. Nobody asked for "
            f"any of it."
        )
        notes.append(
            f"Read the impulse split before deciding what caused that. "
            f"Aerodynamics contributed {imp_aero / 1e6:+.2f} MN s out of "
            f"plane; thrust contributed {imp_thrust / 1e6:+.2f} MN s. The "
            f"aerodynamic side force is not what moved the vehicle -- the "
            f"aerodynamic *moment* swung the body {np.abs(yaw).max():.1f} deg "
            f"out of plane, and after that the engine was pointing the wrong "
            f"way and did the rest. Thrust here is 4.8 MN against side forces "
            f"of a few hundred kN, so it wins easily once it is misaligned."
        )
        if imp_aero * imp_thrust < 0:
            notes.append(
                "Note the two impulses have opposite signs. The air pushes "
                "the vehicle downwind and the engine, once turned, carries it "
                "back upwind past where it started. Watching only the wind "
                "direction would predict the drift backwards."
            )

    notes.append(
        f"Centre of pressure {float(p['x_cp']):+.1f} m from the centre of "
        f"mass: {static_margin_sign(cfg)}. Aft is an arrow, forward is a dart "
        f"thrown backwards. At zero there is no aerodynamic moment at all and "
        f"the attitude only changes if the gimbal changes it."
    )
    notes.append(
        "Limitation worth knowing before reading the oscillation as physical: "
        "there is a restoring moment here but no aerodynamic damping -- no "
        "moment proportional to body rate. So a disturbed vehicle swings about "
        "the wind direction forever instead of settling. Real vehicles damp; "
        "this model does not, and adding a rate term probably belongs before "
        "any controller is tuned against it."
    )
    nose = float(np.prod(effective_area_and_Cd(0.0, cfg)))
    belly = float(np.prod(effective_area_and_Cd(np.pi / 2.0, cfg)))
    notes.append(
        f"The drag area swings between {nose:.0f} and {belly:.0f} m^2 "
        f"depending only on how the wind meets the body -- a factor of "
        f"{belly / nose:.0f}. "
        f"Day 6 decided that from the pitch angle relative to *vertical* "
        f"instead of relative to the wind, which is right only in vertical "
        f"descent and wrong by up to a factor of two either way elsewhere. "
        f"Not fixed -- see LOG.md."
    )
    return notes


def _failed(notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status="error", feasible=False,
        solver="3-D aerodynamics", notes=notes,
    )
