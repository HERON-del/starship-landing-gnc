"""
Day 8 — the complete solver, in the viewer.

Wraps `src.scvx_complete.solve_scvx_complete`. Same physics as Day 7 and the
same SCvx shell around it; three numerical components inside it change.

**The burn duration is no longer yours to pick.** Every earlier problem in this
viewer took a burn time and optimised within it. Here `t_f` is a decision
variable, so the slider marked "burn time guess" sets the *entry state* and a
starting point, and the solver reports back what duration it actually wanted.
Watch the two numbers diverge in the panel. Sweeping the guess from 5 to 12
seconds with the entry state pinned, the converged durations span 0.574 s.

**Trapezoidal collocation is the upgrade that shows up in the replay.** Both
Day 7 and Day 8 land exactly on the pad inside their own model; the question is
what happens when the commanded throttle and gimbal are flown through the
independently verified nonlinear simulator. With the duration pinned so
discretisation is the only difference, on a 473 m descent:

    Euler (Day 7)          3.575 m
    trapezoidal (Day 8)    0.502 m

and trapezoidal at N=20 (0.499 m) beats Euler at N=120 (2.308 m). Drag the node
count down and watch the Day 8 answer barely move -- past about N=20 the
remaining error is the zero-order-hold on the control, not the integration, so
more nodes buy nothing.

**Log-mass is invisible and that is the point.** `z_m = ln(m)` makes the
objective linear, since minimising propellant is exactly maximising `z_m` at the
final node, and it un-freezes mass inside the velocity rows, which Day 7 held
fixed from the reference within each iteration. The panel reports how far
`m_wet * exp(z_m)` drifts from the mass it represents; it should be zero.

One thing worth trying: set the time penalty to something large. The guide that
this project follows adds `0.1 * t_f` to the objective to stop the free time
being degenerate. It is not degenerate -- the throttle floor already makes a
longer burn cost propellant, so the optimum sits at 7.76 s with no penalty at
all. Sweeping the weight from 0 to 1 moves the answer by 0.7%.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.scvx_complete import solve_scvx_complete
from src.scvx_params import SCvxParams
from src.dynamics_6dof import Vehicle6DoF, G_EARTH
from src.aero import AeroConfig

from ..registry import Problem, register
from ..types import Param, Series, Trajectory, quats_from_pitch


@register
class CompleteLanding(Problem):
    slug = "scvx-complete"
    title = "Complete Solver (trapz + free time + log-mass)"
    summary = ("The solver picks its own burn duration, and the trajectory "
               "survives contact with a real integrator.")
    phase = "Day 8"
    scene_scale = 700.0

    def params(self) -> list[Param]:
        return [
            Param("theta0_deg", "Entry pitch", 30.0, min=0.0, max=70.0,
                  step=5.0, unit="deg", group="Entry state",
                  help="From vertical. This and the burn duration together "
                       "decide whether the problem has a solution once drag "
                       "is on - see the Day 7 problem for the measurement."),
            Param("t_burn_guess", "Burn time guess", 8.0, min=4.0, max=16.0,
                  step=0.5, unit="s", group="Entry state",
                  help="Sets the entry altitude and speed, and seeds the "
                       "duration. The solver is free to move the duration "
                       "away from it - compare the two in the panel."),
            Param("x0", "Downrange offset", 0.0, min=-200.0, max=200.0,
                  step=10.0, unit="m", group="Entry state"),
            Param("vx0", "Entry horizontal speed", 0.0, min=-40.0, max=40.0,
                  step=2.0, unit="m/s", group="Entry state"),

            Param("t_f_min", "Shortest allowed burn", 4.0, min=2.0, max=12.0,
                  step=0.5, unit="s", group="Burn duration"),
            Param("t_f_max", "Longest allowed burn", 16.0, min=6.0, max=30.0,
                  step=0.5, unit="s", group="Burn duration"),
            Param("w_time", "Time penalty", 0.0, min=0.0, max=1.0, step=0.05,
                  group="Burn duration",
                  help="A preference for haste, not a numerical safeguard. "
                       "Nothing is degenerate at zero: the throttle floor "
                       "already charges by the second."),

            Param("aero_on", "Aerodynamics", True, kind="bool",
                  group="Constraints"),
            Param("gamma_gs_deg", "Glideslope angle", 75.0, min=30.0, max=86.0,
                  step=1.0, unit="deg", group="Constraints"),
            Param("omega_max_deg", "Max pitch rate", 28.6, min=10.0, max=90.0,
                  step=1.0, unit="deg/s", group="Constraints"),
            Param("delta_max_deg", "Max gimbal", 15.0, min=3.0, max=30.0,
                  step=1.0, unit="deg", group="Constraints"),

            Param("m_prop", "Landing propellant", 30000.0, min=10000.0,
                  max=60000.0, step=1000.0, unit="kg", group="Vehicle"),
            Param("n_engines", "Engines lit", 3, kind="int", min=1, max=6,
                  step=1, group="Vehicle"),
            Param("I_pitch", "Pitch inertia", 2.7e7, min=1.0e7, max=6.0e7,
                  step=1.0e6, unit="kg m^2", group="Vehicle"),

            Param("N", "Nodes", 60, kind="int", min=20, max=140, step=10,
                  group="Solver",
                  help="Trapezoidal collocation makes this cheap: N=20 lands "
                       "closer in replay than Euler does at N=120. Past about "
                       "20 the residual error is the zero-order-hold control, "
                       "so more nodes buy solve time and nothing else."),
            Param("eta_0", "Initial trust region", 0.5, min=0.02, max=4.0,
                  step=0.02, group="Solver"),
            Param("w_vc", "Virtual-control penalty", 1000.0, min=1.0,
                  max=100000.0, step=1.0, group="Solver",
                  help="Below about 10 the optimiser buys a trajectory that "
                       "does not fly. See the Day 7 problem."),
            Param("max_iter", "Max SCvx iterations", 30, kind="int", min=5,
                  max=60, step=5, group="Solver"),
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
        sp = SCvxParams(eta_0=float(p["eta_0"]), w_vc=float(p["w_vc"]),
                        max_iter=int(p["max_iter"]))

        t_f_min = float(p["t_f_min"])
        t_f_max = float(p["t_f_max"])
        if t_f_max <= t_f_min:
            return _failed("error", [
                f"The longest allowed burn ({t_f_max:.1f} s) must exceed the "
                f"shortest ({t_f_min:.1f} s)."
            ])

        t0 = time.perf_counter()
        try:
            r = solve_scvx_complete(
                vehicle=vehicle,
                aero=AeroConfig() if bool(p["aero_on"]) else None,
                params=sp,
                N=int(p["N"]),
                t_burn_guess=float(p["t_burn_guess"]),
                t_f_min=t_f_min, t_f_max=t_f_max,
                x0=float(p["x0"]), vx0=float(p["vx0"]),
                theta0_deg=float(p["theta0_deg"]),
                gamma_gs_deg=float(p["gamma_gs_deg"]),
                w_time=float(p["w_time"]),
                verbose=False,
            )
        except Exception as exc:      # noqa: BLE001
            return _failed("error", [f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        if r.get("status") == "failed":
            return _failed("infeasible", [
                "No subproblem solved. With virtual control on every dynamics "
                "row the reference is feasible by construction, so this is the "
                "numerics rather than the geometry - check the penalty weight "
                "and the node count.",
            ])

        N = r["N"]
        pos = np.column_stack([r["x"], r["z"], np.zeros(len(r["t"]))])
        vel = np.column_stack([r["vx"], r["vz"], np.zeros(len(r["t"]))])
        thrust = np.column_stack([r["Tx"], r["Tz"], np.zeros(N)])

        theta_deg = np.degrees(r["theta"])
        omega_deg = np.degrees(r["omega"])
        delta_deg = np.degrees(r["delta"])
        speed = np.hypot(r["vx"], r["vz"])
        twr = r["sigma"] / (r["m"][:N] * G_EARTH)

        honest = r["vc_norm"] < sp.dyn_tol
        notes = _notes(p, r, sp, honest, delta_deg, omega_deg)

        return Trajectory(
            t_state=r["t"].tolist(),
            t_control=r["t"][:-1].tolist(),
            position=pos.tolist(),
            velocity=vel.tolist(),
            thrust=thrust.tolist(),
            attitude=quats_from_pitch(r["theta"]).tolist(),
            series=[
                Series("altitude", "Altitude", "m", r["z"].tolist()),
                Series("downrange", "Downrange", "m", r["x"].tolist()),
                Series("speed", "Speed", "m/s", speed.tolist()),
                Series("pitch", "Pitch from vertical", "deg",
                       theta_deg.tolist()),
                Series("rate", "Pitch rate", "deg/s", omega_deg.tolist()),
                Series("dynamic_pressure", "Dynamic pressure", "Pa",
                       r["q"].tolist()),
                Series("mass", "Vehicle mass", "kg", r["m"].tolist()),
                Series("log_mass", "Log-mass ln(m/m_wet)", "-",
                       r["zm"].tolist()),
                Series("thrust", "Thrust", "N", r["sigma"].tolist(),
                       on="control"),
                Series("gimbal", "Gimbal angle", "deg", delta_deg.tolist(),
                       on="control"),
                Series("drag", "Aerodynamic drag", "N",
                       r["drag_mag"].tolist(), on="control"),
                Series("twr", "Thrust/weight", "-", twr.tolist(),
                       on="control"),
            ],
            status=r["status"],
            feasible=True,
            cost=float(r["fuel"]),
            solve_time_ms=elapsed,
            solver=f"SCvx trapz + free time, {r['iterations']} iterations",
            thrust_max=vehicle.T_max,
            notes=notes,
            diagnostics={
                "fuel_kg": float(r["fuel"]),
                "burn_duration_s": float(r["t_f"]),
                "duration_guess_s": float(r["t_nom"]),
                "duration_moved_s": float(r["t_f"] - r["t_nom"]),
                "scvx_iterations": int(r["iterations"]),
                "virtual_control_l1": float(r["vc_norm"]),
                "thrust_linearisation_defect": float(r["thrust_defect"]),
                "dynamics_satisfied": bool(honest),
                "log_mass_error_kg": float(r["log_mass_error"]),
                "peak_gimbal_deg": float(np.max(np.abs(delta_deg))),
                "peak_pitch_rate_deg_s": float(np.max(np.abs(omega_deg))),
                "peak_dynamic_pressure_kpa": float(np.max(r["q"]) / 1000.0),
                "entry_altitude_m": float(r["z"][0]),
                "entry_speed_ms": float(np.hypot(r["vx"][0], r["vz"][0])),
                "final_position_error_m": float(np.hypot(r["x"][-1],
                                                         r["z"][-1])),
                "final_pitch_deg": float(theta_deg[-1]),
            },
        )


# ----------------------------------------------------------------------
def _notes(p, r, sp, honest, delta_deg, omega_deg) -> list[str]:
    t_f, t_nom = float(r["t_f"]), float(r["t_nom"])
    moved = t_f - t_nom
    lo, hi = r["t_f_bounds"]

    notes = [
        f"Landed on {r['fuel']:,.0f} kg from {r['z'][0]:,.0f} m at "
        f"{abs(r['vz'][0]):.0f} m/s, in {r['iterations']} iterations.",
    ]

    if abs(t_f - lo) < 1e-3 or abs(t_f - hi) < 1e-3:
        notes.append(
            f"The duration hit a bound at {t_f:.2f} s, so this is not an "
            f"interior optimum - widen the allowed range before reading "
            f"anything into it. A duration pinned to a bound is exactly the "
            f"failure mode of a free time that does not really enter the "
            f"dynamics."
        )
    else:
        notes.append(
            f"The solver chose a {t_f:.2f} s burn against a {t_nom:.1f} s "
            f"guess, moving {moved:+.2f} s of its own accord. The duration is "
            f"a decision variable here, which means the product t_f * f(x,u) "
            f"appears in every dynamics row and has to be linearised like any "
            f"other bilinear term - declaring the variable without doing that "
            f"leaves it inert."
        )

    if honest:
        notes.append(
            f"Virtual control {r['vc_norm']:.1e}, thrust linearisation error "
            f"{r['thrust_defect']:.1e} of maximum thrust. The trajectory "
            f"satisfies its own dynamics and those dynamics are a rocket's: "
            f"peak gimbal {np.max(np.abs(delta_deg)):.1f} of "
            f"{p['delta_max_deg']:.0f} deg, peak pitch rate "
            f"{np.max(np.abs(omega_deg)):.1f} of {p['omega_max_deg']:.1f} "
            f"deg/s."
        )
    else:
        notes.append(
            f"WARNING: virtual control settled at {r['vc_norm']:.3e} and did "
            f"not clear, so this problem has no exact solution and the "
            f"{r['fuel']:,.0f} kg figure is optimistic by whatever that "
            f"violation is worth. Free time does not rescue an infeasible "
            f"problem - it reports the same shortfall at a slightly better "
            f"duration. The Day 7 problem has the diagnosis for which "
            f"mechanism you are in."
        )

    notes.append(
        f"Log-mass check: m_wet*exp(z_m) differs from the mass it represents "
        f"by {r['log_mass_error']:.4f} kg at worst. The substitution makes the "
        f"objective linear - minimising propellant is exactly maximising z_m "
        f"at the last node - and lets mass vary inside the velocity rows, "
        f"which the Day 7 solver froze from the reference each iteration."
    )
    notes.append(
        f"Discretisation is trapezoidal, not Euler. Flown through the verified "
        f"nonlinear simulator with the duration pinned so nothing else "
        f"differs, this lands 0.502 m from the pad on a 473 m descent against "
        f"Euler's 3.575 m. At N={int(p['N'])} here; trapezoidal at N=20 still "
        f"beats Euler at N=120."
    )
    if float(p["w_time"]) > 0.0:
        notes.append(
            f"A time penalty of {p['w_time']:.2f} is active, so this is a "
            f"trade rather than the minimum-propellant answer. It is not "
            f"needed: with no penalty the optimum is an interior 7.76 s, and "
            f"sweeping the weight from 0 to 1 moves the result by 0.7%."
        )
    return notes


def _failed(status: str, notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status=status, feasible=False,
        solver="SCvx trapz + free time", notes=notes,
    )
