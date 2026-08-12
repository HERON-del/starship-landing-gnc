"""
Day 5 — the flip manoeuvre, in the viewer.

Wraps `src.landing_flip.solve_flip_landing`. This is the first problem whose
attitude the optimiser actually solves for: every earlier entry inferred an
attitude from its thrust vector, which is a rendering convenience. Here `theta`
is a state variable with its own dynamics, its own rate limit, and its own
terminal condition, so the vehicle in the scene is genuinely flipping.

Two things this problem exists to make visible.

**Torque and thrust tilt are the same deflection.** The engine is bolted to the
vehicle, so gimbaling to rotate it also tilts the thrust that is decelerating
it. Watch the trajectory: the vehicle is thrown sideways during the flip and has
to come back, because that lateral push is not optional.

**There is a ceiling on entry pitch.** Push the entry attitude up and the
problem stops solving — not from bad conditioning but from physics. The engine
is lit throughout, the pitch rate is capped, and the lateral excursion built up
during the flip has to fit inside the glideslope corridor and still be nulled by
touchdown. Raise the pitch-rate limit or loosen the corridor and the ceiling
moves; the panel reports which one you relaxed.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.landing_flip import solve_flip_landing, feasible_entry_state
from src.dynamics_6dof import Vehicle6DoF, G_EARTH

from ..registry import Problem, register
from ..types import Param, Series, Trajectory, quats_from_pitch


@register
class FlipLanding(Problem):
    slug = "landing-flip"
    title = "Flip-and-Land (6-DoF)"
    summary = "Rotation is a state, not an inference. SCvx with trust regions."
    phase = "Day 5"
    scene_scale = 1400.0

    def params(self) -> list[Param]:
        return [
            Param("theta0_deg", "Entry pitch", 60.0, min=0.0, max=85.0, step=5.0,
                  unit="deg", group="Entry state",
                  help="From vertical; 90 is a full belly-flop. There is a "
                       "ceiling well below that - the engine cannot be shut "
                       "off during the flip."),
            Param("t_burn", "Burn time", 15.0, min=8.0, max=26.0, step=0.5,
                  unit="s", group="Entry state",
                  help="Entry altitude and speed are sized to this, so the "
                       "burn can actually null them."),
            Param("x0", "Downrange offset", 0.0, min=-400.0, max=400.0,
                  step=10.0, unit="m", group="Entry state"),
            Param("vx0", "Entry horizontal speed", 0.0, min=-60.0, max=60.0,
                  step=2.0, unit="m/s", group="Entry state"),

            Param("gamma_gs_deg", "Glideslope angle", 75.0, min=30.0, max=86.0,
                  step=1.0, unit="deg", group="Constraints",
                  help="Loosen this and the entry-pitch ceiling rises - one "
                       "half of the pair that binds it."),
            Param("omega_max_deg", "Max pitch rate", 28.6, min=10.0, max=90.0,
                  step=1.0, unit="deg/s", group="Constraints",
                  help="The other half. The flip takes at least "
                       "theta0/omega_max seconds, and the lateral excursion "
                       "accumulates for exactly that long."),
            Param("delta_max_deg", "Max gimbal", 15.0, min=3.0, max=30.0,
                  step=1.0, unit="deg", group="Constraints"),

            Param("m_prop", "Landing propellant", 30000.0, min=10000.0,
                  max=60000.0, step=1000.0, unit="kg", group="Vehicle"),
            Param("n_engines", "Engines lit", 3, kind="int", min=1, max=6,
                  step=1, group="Vehicle"),
            Param("I_pitch", "Pitch inertia", 2.7e7, min=1.0e7, max=6.0e7,
                  step=1.0e6, unit="kg m^2", group="Vehicle"),

            Param("N", "Nodes", 80, kind="int", min=40, max=110, step=10,
                  group="Solver",
                  help="Below ~70 the linearisation defect stops converging."),
            Param("trust0_deg", "Initial trust region", 40.0, min=5.0, max=60.0,
                  step=5.0, unit="deg", group="Solver"),
            Param("max_iters", "Max SCvx iterations", 30, kind="int", min=5,
                  max=45, step=5, group="Solver"),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)

        vehicle = Vehicle6DoF(
            m_prop_initial=float(p["m_prop"]),
            n_engines=int(p["n_engines"]),
            I_pitch=float(p["I_pitch"]),
            delta_max_deg=float(p["delta_max_deg"]),
            omega_max=float(np.radians(p["omega_max_deg"])),
        )

        t0 = time.perf_counter()
        try:
            r = solve_flip_landing(
                vehicle=vehicle, N=int(p["N"]), t_burn=float(p["t_burn"]),
                x0=float(p["x0"]), vx0=float(p["vx0"]),
                theta0_deg=float(p["theta0_deg"]),
                gamma_gs_deg=float(p["gamma_gs_deg"]),
                trust0_deg=float(p["trust0_deg"]),
                max_iters=int(p["max_iters"]),
                verbose=False,
            )
        except Exception as exc:      # noqa: BLE001
            return _failed("error", [f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        if not str(r.get("status", "")).startswith("optimal"):
            return _failed(r.get("status", "infeasible"),
                           _diagnose(vehicle, p, r))

        n_int = len(r["t"]) - 1
        pos = np.column_stack([r["x"], r["z"], np.zeros(len(r["t"]))])
        vel = np.column_stack([r["vx"], r["vz"], np.zeros(len(r["t"]))])
        thrust = np.column_stack([r["Tx"], r["Tz"], np.zeros(n_int)])

        theta_deg = np.degrees(r["theta"])
        omega_deg = np.degrees(r["omega"])
        delta_deg = np.degrees(r["delta"])
        speed = np.hypot(r["vx"], r["vz"])
        twr = r["sigma"] / (r["m"][:n_int] * G_EARTH)

        lateral = float(np.max(np.abs(r["x"])))
        notes = [
            f"Flipped {p['theta0_deg']:.0f} deg to upright in {r['t_burn']:.1f} s "
            f"on {r['fuel']:,.0f} kg. SCvx converged in {r['iterations']} "
            f"iterations to a linearisation defect of {r['final_defect']:.5f} "
            f"of maximum thrust.",
            f"Peak pitch rate {np.max(np.abs(omega_deg)):.1f} of "
            f"{p['omega_max_deg']:.1f} deg/s; peak gimbal "
            f"{np.max(np.abs(delta_deg)):.1f} of {p['delta_max_deg']:.0f} deg. "
            + ("Rate-limited: the gimbal has slack, so a stronger one would not "
               "flip it faster."
               if np.max(np.abs(delta_deg)) < 0.9 * p["delta_max_deg"]
               else "Gimbal-limited: the deflection is saturated."),
        ]
        if lateral > 1.0:
            notes.append(
                f"The engine cannot be shut off during the flip, so its "
                f"horizontal component throws the vehicle {lateral:,.0f} m "
                f"sideways before it can be nulled. Push the entry pitch up and "
                f"that excursion eventually will not fit inside the glideslope "
                f"corridor - which is the ceiling."
            )
        if r["final_defect"] > 0.01:
            notes.append(
                f"WARNING: SCvx did not fully converge - the linear model still "
                f"disagrees with the true dynamics by "
                f"{100 * r['final_defect']:.1f}% of maximum thrust. The "
                f"trajectory is indicative rather than trustworthy. Raise the "
                f"iteration budget or the node count."
            )
        notes.append(
            "Attitude here is a solved state, not inferred from the thrust "
            "vector as in the earlier problems. Discretisation is still forward "
            "Euler, so the replayed trajectory misses the pad by a few percent "
            "of the descent even though the attitude tracks well."
        )

        return Trajectory(
            t_state=r["t"].tolist(),
            t_control=r["t"][:-1].tolist(),
            position=pos.tolist(),
            velocity=vel.tolist(),
            thrust=thrust.tolist(),
            # The optimiser's own attitude, not a guess from the thrust vector.
            attitude=quats_from_pitch(r["theta"]).tolist(),
            series=[
                Series("altitude", "Altitude", "m", r["z"].tolist()),
                Series("downrange", "Downrange", "m", r["x"].tolist()),
                Series("speed", "Speed", "m/s", speed.tolist()),
                Series("pitch", "Pitch from vertical", "deg", theta_deg.tolist()),
                Series("rate", "Pitch rate", "deg/s", omega_deg.tolist()),
                Series("mass", "Vehicle mass", "kg", r["m"].tolist()),
                Series("thrust", "Thrust", "N", r["sigma"].tolist(), on="control"),
                Series("gimbal", "Gimbal angle", "deg", delta_deg.tolist(),
                       on="control"),
                Series("twr", "Thrust/weight", "-", twr.tolist(), on="control"),
            ],
            status=r["status"],
            feasible=True,
            cost=float(r["fuel"]),
            solve_time_ms=elapsed,
            solver=f"SCvx, {r['iterations']} iterations",
            thrust_max=vehicle.T_max,
            notes=notes,
            diagnostics={
                "fuel_kg": float(r["fuel"]),
                "scvx_iterations": int(r["iterations"]),
                "linearisation_defect": float(r["final_defect"]),
                "peak_pitch_rate_deg_s": float(np.max(np.abs(omega_deg))),
                "peak_gimbal_deg": float(np.max(np.abs(delta_deg))),
                "peak_torque_frac": float(np.max(np.abs(r["tau"])) / vehicle.tau_max),
                "max_lateral_excursion_m": lateral,
                "entry_altitude_m": float(r["z"][0]),
                "entry_speed_ms": float(np.hypot(r["vx"][0], r["vz"][0])),
                "final_pitch_deg": float(theta_deg[-1]),
                "final_rate_deg_s": float(omega_deg[-1]),
                "final_position_error_m": float(np.hypot(r["x"][-1], r["z"][-1])),
                "glideslope_deg": float(p["gamma_gs_deg"]),
            },
        )


def _failed(status: str, notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status=status, feasible=False,
        solver="SCvx", notes=notes,
    )


def _diagnose(vehicle, p, r) -> list[str]:
    """Name the mechanism rather than just reporting infeasible."""
    theta0 = float(p["theta0_deg"])
    omega_max_deg = float(p["omega_max_deg"])
    flip_time = theta0 / max(omega_max_deg, 1e-6)
    lateral_accel = vehicle.T_min / vehicle.m_wet

    notes = [f"No flip trajectory exists at a {theta0:.0f} deg entry pitch."]

    if theta0 > 5.0:
        notes.append(
            f"The engine is lit throughout - minimum throttle is 40% and there "
            f"is no coast - so while tilted it pushes the vehicle sideways at "
            f"up to {lateral_accel:.0f} m/s^2. The pitch rate limit means the "
            f"flip takes at least {flip_time:.1f} s, and the excursion built in "
            f"that window must fit the {p['gamma_gs_deg']:.0f} deg glideslope "
            f"corridor and still be nulled by touchdown."
        )
        notes.append(
            "Two knobs move this ceiling, and relaxing either one alone is "
            "enough: raise the max pitch rate so the flip ends sooner, or "
            "loosen the glideslope so the excursion fits. That both work is "
            "what shows the pair binds together."
        )
    else:
        notes.append(
            "At a near-upright entry the attitude is not the problem. Check "
            "the burn time - entry altitude and speed are sized to it, and the "
            "minimum-throttle floor sets how much velocity a given burn can "
            "null."
        )
    return notes
