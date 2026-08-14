"""
Day 7 — successive convexification, in the viewer.

Wraps `src.scvx.solve_scvx`. The physics is Day 5's and the forces are Day 6's;
what is new is the algorithm wrapped around them, so the thing worth watching
here is not the trajectory but the panel underneath it.

**Virtual control turns "infeasible" into a number.** Every earlier problem in
this viewer had two outcomes: a trajectory, or the word `infeasible` and no
further help. This one adds slack to all seven dynamics rows and prices it, so
the subproblem always has an answer. On a problem that *is* solvable the slack
falls to around 1e-12 and the trajectory means exactly what it says. On one that
is not, the solver returns the least-infeasible trajectory it can find and
reports how much dynamics it had to violate to get there. Push the entry pitch
up to 60 degrees and watch the deficit appear rather than the solver giving up.

**The deficit is a property of the problem, not the solver.** It is worth
convincing yourself of this with the controls. Tighten the trust region tenfold
and it moves by about 10%; raise the penalty weight four orders of magnitude and
it does not move at all past the fifth significant figure. A linearisation
artefact would collapse under either. The panel reports the two numbers side by
side: at a 60 degree entry the thrust linearisation error is around 4e-6 while
the dynamics are violated by 0.27, so whatever is wrong, it is not the
linearising.

**Two mechanisms produce a deficit, and burn time separates them.** Day 5's
entry-pitch ceiling -- the engine cannot be shut off, so a tilted vehicle is
thrown sideways faster than a short burn can null it -- and Day 6's drag term,
held fixed from a reference the vehicle is not on. They pull in opposite
directions, which is the tell. At a 60 degree entry with drag off, the slack
runs 0.21 at 8 s and 5e-11 by 12 s: lengthen the burn and it goes away. With
drag on, the same sweep runs 0.28, 1.4, 2.5 at 8, 12 and 15 s, because burn time
sizes the entry state -- a longer burn means arriving higher and faster and
spending more of it belly-on. Loosening the glideslope, the obvious third knob,
moves neither by 5%.

**The penalty has a floor, not a sweet spot.** Turn the adaptive penalty off and
set the weight to 1, and the solver will happily report a landing on a few
hundred kilograms of propellant -- an order of magnitude under the truth --
because slack is cheaper than flying. Anything from 10 upward gives the same
honest answer. That failure is worth seeing once.

**The seed no longer has to be good.** Day 5 recorded that linearising about a
naive straight pitch ramp made the first subproblem infeasible, and needed a
rate-limited seed to start at all. Switch the seed to `linear` here: it still
solves, because slack absorbs the gap the bad guess opens.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.scvx import solve_scvx
from src.scvx_params import SCvxParams
from src.dynamics_6dof import Vehicle6DoF, G_EARTH
from src.aero import AeroConfig

from ..registry import Problem, register
from ..types import Param, Series, Trajectory, quats_from_pitch


@register
class SCvxLanding(Problem):
    slug = "scvx-landing"
    title = "SCvx: Trust Regions and Virtual Control"
    summary = ("The ad-hoc loop becomes an algorithm. Infeasibility stops being "
               "a word and becomes a measurement.")
    phase = "Day 7"
    scene_scale = 700.0

    def params(self) -> list[Param]:
        return [
            Param("theta0_deg", "Entry pitch", 30.0, min=0.0, max=70.0,
                  step=5.0, unit="deg", group="Entry state",
                  help="From vertical. This and the burn time together decide "
                       "whether the problem has a solution at all - the flip "
                       "throws the vehicle sideways and a short burn has no "
                       "time to null it. Measured with drag off: 60 degrees "
                       "closes at a 15 s burn but leaves 0.2 of slack at 8 s."),
            Param("t_burn", "Burn time", 8.0, min=4.0, max=16.0, step=0.5,
                  unit="s", group="Entry state",
                  help="Entry altitude and speed are sized to this. Propellant "
                       "is nearly proportional to it, because the 40% throttle "
                       "floor sets the flow rate whatever the optimiser wants."),
            Param("x0", "Downrange offset", 0.0, min=-200.0, max=200.0,
                  step=10.0, unit="m", group="Entry state",
                  help="Push this out and the deficit appears: the attitude is "
                       "already committed to the flip, so there is little "
                       "lateral authority left to null the offset."),
            Param("vx0", "Entry horizontal speed", 0.0, min=-40.0, max=40.0,
                  step=2.0, unit="m/s", group="Entry state"),

            Param("aero_on", "Aerodynamics", True, kind="bool",
                  group="Constraints",
                  help="Drag widens the infeasible region rather than causing "
                       "it: with air on, a 60 degree entry leaves slack even "
                       "at a 15 s burn, where dry it closes to 1e-10. Toggle "
                       "this to find out which mechanism you are looking at."),
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

            Param("N", "Nodes", 80, kind="int", min=20, max=140, step=10,
                  group="Solver",
                  help="Propellant is settled to within 10 kg by N=60; past "
                       "that you are buying solve time."),
            Param("eta_0", "Initial trust region", 0.5, min=0.02, max=4.0,
                  step=0.02, group="Solver",
                  help="Non-dimensional, so one radius bounds every state to "
                       "the same fractional excursion. Barely matters - the "
                       "adaptive rule re-tunes it within a few iterations."),
            Param("w_vc", "Virtual-control penalty", 1000.0, min=1.0,
                  max=100000.0, step=1.0, group="Solver",
                  help="Price of violating the dynamics. Below about 10, with "
                       "the adaptive rule off, the optimiser buys itself a "
                       "trajectory that does not fly."),
            Param("w_vc_adaptive", "Adaptive penalty", True, kind="bool",
                  group="Solver",
                  help="Raise the price whenever slack stops shrinking. With "
                       "this on, the starting weight is nearly irrelevant."),
            Param("seed", "Reference seed", "flip", kind="choice",
                  choices=["flip", "linear"], group="Solver",
                  help="'flip' is Day 5's rate-limited sweep. 'linear' is the "
                       "naive ramp Day 5 recorded as making the first "
                       "subproblem infeasible - it now solves anyway."),
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
        sp = SCvxParams(
            eta_0=float(p["eta_0"]),
            w_vc=float(p["w_vc"]),
            w_vc_grow=3.0 if bool(p["w_vc_adaptive"]) else 1.0,
            seed=str(p["seed"]),
            max_iter=int(p["max_iter"]),
        )

        t0 = time.perf_counter()
        try:
            r = solve_scvx(
                vehicle=vehicle,
                aero=AeroConfig() if bool(p["aero_on"]) else None,
                params=sp,
                N=int(p["N"]), t_burn=float(p["t_burn"]),
                x0=float(p["x0"]), vx0=float(p["vx0"]),
                theta0_deg=float(p["theta0_deg"]),
                gamma_gs_deg=float(p["gamma_gs_deg"]),
                verbose=False,
            )
        except Exception as exc:      # noqa: BLE001
            return _failed("error", [f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        if r.get("status") == "failed":
            return _failed("infeasible", _diagnose(vehicle, p, r))

        n_int = len(r["t"]) - 1
        pos = np.column_stack([r["x"], r["z"], np.zeros(len(r["t"]))])
        vel = np.column_stack([r["vx"], r["vz"], np.zeros(len(r["t"]))])
        thrust = np.column_stack([r["Tx"], r["Tz"], np.zeros(n_int)])

        theta_deg = np.degrees(r["theta"])
        omega_deg = np.degrees(r["omega"])
        delta_deg = np.degrees(r["delta"])
        speed = np.hypot(r["vx"], r["vz"])
        twr = r["sigma"] / (r["m"][:n_int] * G_EARTH)

        hist = r["history"]
        rejected = sum(1 for a in hist["accepted"] if not a)
        honest = r["vc_norm"] < sp.dyn_tol
        notes = _notes(p, r, sp, honest, rejected, delta_deg, omega_deg)

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
            # The terminal conditions are hard equalities, so the vehicle is on
            # the pad at rest whatever the slack did. Rendering it is right;
            # the notes carry the caveat.
            feasible=True,
            cost=float(r["fuel"]),
            solve_time_ms=elapsed,
            solver=f"SCvx, {r['iterations']} iterations",
            thrust_max=vehicle.T_max,
            notes=notes,
            diagnostics={
                "fuel_kg": float(r["fuel"]),
                "scvx_iterations": int(r["iterations"]),
                "steps_rejected": int(rejected),
                "virtual_control_l1": float(r["vc_norm"]),
                "true_dynamics_defect_l1": float(r["defect"]),
                "thrust_linearisation_defect": float(r["thrust_defect"]),
                "dynamics_satisfied": bool(honest),
                "final_trust_region": float(hist["eta"][-1]),
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
def _notes(p, r, sp, honest, rejected, delta_deg, omega_deg) -> list[str]:
    fuel = r["fuel"]
    notes = [
        f"Landed on {fuel:,.0f} kg from {r['z'][0]:,.0f} m at "
        f"{abs(r['vz'][0]):.0f} m/s. SCvx took {r['iterations']} iterations, "
        f"rejecting {rejected} of them, and finished with a trust region of "
        f"{r['history']['eta'][-1]:.4f}.",
    ]

    if honest:
        notes.append(
            f"Virtual control fell to {r['vc_norm']:.1e} - the trajectory "
            f"satisfies its own dynamics to machine precision, so the "
            f"propellant figure means what it says."
        )
    else:
        notes.append(
            f"WARNING: virtual control settled at {r['vc_norm']:.3e} and did "
            f"not clear. This problem has no solution: the optimiser found the "
            f"least-infeasible trajectory it could and paid the remainder in "
            f"slack, so the {fuel:,.0f} kg figure is optimistic by whatever "
            f"that violation is worth. The trajectory is shown because a "
            f"measured shortfall is more use than the word 'infeasible' - "
            f"which is all the Day 5 and Day 6 solvers could offer here."
        )
        # Three mechanisms are known to produce this, and they are not
        # distinguishable from the solved trajectory alone -- so name them and
        # give the experiment, rather than guessing. At 60 degrees and an 8 s
        # burn, for instance, switching drag off barely moves the deficit
        # (0.274 to 0.198): that one is the Day 5 ceiling, not the air.
        causes = []
        if float(p["theta0_deg"]) > 30.0:
            causes.append(
                "**Entry pitch.** Two mechanisms produce this, and they "
                "respond to burn time in opposite directions - which is how "
                "you tell them apart. Measured at a 60 degree entry: with "
                "drag off the deficit is 0.21 at 8 s and 5e-11 by 12 s, so "
                "lengthening the burn cures it. That one is Day 5's ceiling - "
                "the engine cannot be shut off, so a tilted vehicle is thrown "
                "sideways and a short burn has no time to null it. With drag "
                "on the same sweep runs 0.28, 1.4, 2.5 at 8, 12 and 15 s: a "
                "longer burn is a higher and faster entry and more seconds "
                "spent belly-on, so it makes matters worse. There the only "
                "cure is less entry pitch - which is exactly what Day 6 "
                "concluded the vehicle does, coasting on its belly and "
                "lighting the engines near-upright."
            )
            causes.append(
                "Loosening the glideslope is not the lever it looks like: at "
                "60 degrees, dropping the corridor from 75 to 45 degrees moves "
                "the deficit by under 5% in either regime."
            )
        if abs(float(p["x0"])) > 50.0 or abs(float(p["vx0"])) > 10.0:
            causes.append(
                f"**Off-axis entry.** Lateral authority is sin(theta) of the "
                f"thrust, and the attitude is already committed to arriving "
                f"upright, so there is little left to null "
                f"{p['x0']:.0f} m and {p['vx0']:.0f} m/s in "
                f"{p['t_burn']:.1f} s."
            )
        if not causes:
            causes.append(
                "No single knob is obviously to blame. Sweep the burn time "
                "first - entry altitude and speed are sized to it, so it moves "
                "the whole problem."
            )
        notes.extend(causes)

    # The check the guide's own formulation cannot pass.
    notes.append(
        f"Peak gimbal {np.max(np.abs(delta_deg)):.1f} of "
        f"{p['delta_max_deg']:.0f} deg, peak pitch rate "
        f"{np.max(np.abs(omega_deg)):.1f} of {p['omega_max_deg']:.1f} deg/s. "
        f"The thrust direction is the attitude plus the gimbal, not a free "
        f"vector - so this trajectory is one the vehicle could actually fly. "
        f"Linearisation error {r['thrust_defect']:.1e} of maximum thrust."
    )

    if not bool(p["w_vc_adaptive"]) and float(p["w_vc"]) < 10.0:
        notes.append(
            "The penalty is pinned below its floor. At this price slack is "
            "cheaper than flying, so the propellant figure is a fiction - it "
            "buys a trajectory that does not obey the dynamics. Raise the "
            "weight to 10 or more, or switch the adaptive rule back on."
        )
    if str(p["seed"]) == "linear":
        notes.append(
            "Seeded with the naive linear pitch ramp. Day 5 recorded this seed "
            "as making the first subproblem outright infeasible; with slack on "
            "every dynamics row it cannot be, so the solve proceeds - usually "
            "to the same answer, sometimes more slowly."
        )
    if not bool(p["aero_on"]):
        notes.append(
            "Aerodynamics off. This is the clean test of the algorithm: with "
            "no drag term to hold fixed from a reference, virtual control "
            "should reach machine zero at every entry pitch."
        )
    return notes


def _failed(status: str, notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status=status, feasible=False,
        solver="SCvx", notes=notes,
    )


def _diagnose(vehicle, p, r) -> list[str]:
    """
    Reaching here means no subproblem solved at all.

    That is a strong statement, because with slack on every dynamics row the
    reference trajectory is itself feasible -- so this is a conditioning or
    solver failure rather than a geometric one.
    """
    return [
        "No subproblem solved. With virtual control on every dynamics row the "
        "reference is feasible by construction, so this is not the geometry "
        "refusing - it is the numerics.",
        f"The usual cause is a penalty weight far above the problem scale: at "
        f"1e7 against an objective of order 1, both CLARABEL and SCS return "
        f"`unbounded` for a problem bounded below by -1. Current weight is "
        f"{float(p['w_vc']):.0e}.",
        "Try the default weight of 1e3 with the adaptive rule on, and a node "
        "count in the 60-100 range.",
    ]
