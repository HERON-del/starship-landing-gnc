"""
Day 18 -- the Szmuk & Acikmese benchmark, and a claim that does not reproduce.

    Szmuk, M. and Acikmese, B., "Successive Convexification for 6-DoF Mars
    Rocket Powered Landing with Free-Final-Time", AIAA SciTech 2018-0617.

Seventeen days of this viewer show a solver checked against tests written
alongside it. This entry shows it meeting numbers chosen by people who never
saw the codebase, in the paper's own non-dimensional units -- UL, UT, UM, with
no SI scale given anywhere, so none is invented here.

**The external part passes.** The paper's rigid-body model and this project's
turn out to be the same physics written twice: direction cosine matrices
agreeing to 8.88e-16 over 400 random quaternions, and the gyroscopic term
matching Day 14's exactly. Terminal conditions land at 3.45e-16 UL, and the
solution rides its bounds -- thrust at 5.000 of 5.0, gimbal at 20.00 of 20 --
which is the paper's own qualitative claim about thrust saturation reproducing.

**The paper's central claim does not.** Its headline is robustness: ten
time-of-flight guesses, from 1 to 10 UT, all converging to within 0.01 UT.
Move the *time-of-flight guess* control and watch the answer follow it. Across
five guesses the spread is 21.7 UT and the correlation between guess and answer
is 0.892. A free variable that tracks its own initial guess is not being solved
for.

**The reason is in the paper's own Table 2.** The cost is
`sigma + w_nu |nu| + ...` with `w_nu = 1e5` and sigma's coefficient equal to 1,
so feasibility outprices minimum-time a hundred thousand to one and flight time
is very nearly free. Switch **Trust region** to `soft only` -- the paper's
literal Algorithm 1 -- and sigma climbs monotonically from 2.56 to 26.20 UT
while the virtual control falls to 3.45e-17, machine precision. The hard trust
region stops that runaway by pinning sigma near where it started, removing the
oscillation and the optimisation together.

**And one parameter, not the discretisation.** `alpha_m` is not in the paper.
The Day 18 guide sets it to 1.0, calls it a harmless placeholder, and blames
its whole residual on a cruder discretisation. At 1.0 a three-unit trajectory
needs 6 UM of propellant against the 1 UM the vehicle carries; the mass floor
binds and the mass row of the dynamics becomes unsatisfiable. Move the
**alpha_m** control between 1.0 and 0.03 and watch the residual fall by a
factor of 366 -- and watch the *residual by block* diagnostics move from 21%
mass and 71% velocity to 100% position, which is what a discretisation-limited
residual should actually look like.

Note the frame. The paper puts gravity along `-e1`, so its first axis is up and
its state vector starts with mass. Everywhere else in this project +z is up and
mass is last. The conversion into the renderer is done explicitly here rather
than by silently re-axing the problem.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.benchmark_szmuk import (
    PaperVehicle, PaperAlgorithm, two_d_case, three_d_case, solve_benchmark,
    IDX_M, IDX_R, IDX_V, IDX_Q, IDX_W,
)
from src.quaternion import quat_to_rotmatrix, rotmatrix_to_quat

from ..registry import Problem, register
from .rigid_body_view import D_BODY_TO_VIEW

#: Paper inertial (up, east, north) into the renderer's (x, y up, z).
#: A proper rotation -- the naive swap would have determinant -1 and mirror
#: every rotation in the scene, which is the mistake Day 14 nearly made.
C_PAPER_TO_VIEW = np.array([[0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0]])

from ..types import Param, Series, Trajectory                  # noqa: E402


@register
class SzmukBenchmark(Problem):
    slug = "szmuk-benchmark"
    title = "Szmuk & Acikmese Benchmark"
    summary = ("A published free-final-time SCvx problem, and a robustness "
               "claim that does not reproduce.")
    phase = "Day 18"
    scene_scale = 9.0
    enforces_terminal_state = False

    def params(self) -> list[Param]:
        return [
            Param("sigma_guess", "Time-of-flight guess", 3.0, min=1.0,
                  max=10.0, step=1.0, unit="UT", group="The paper's claim",
                  help="The paper reports ten guesses from 1 to 10 all "
                       "converging to within 0.01 UT. Move this and read the "
                       "solved tf in the diagnostics: it follows the guess, "
                       "spread 21.7 UT across five settings."),
            Param("hard_trust", "Trust region", "hard box", kind="choice",
                  choices=["hard box", "soft only"],
                  group="The paper's claim",
                  help="Soft only is Algorithm 1 exactly as printed. It "
                       "drives virtual control to machine precision and lets "
                       "the flight time run away; the hard box stops the "
                       "runaway by pinning it near the guess."),

            Param("alpha_m", "alpha_m", 0.03, min=0.01, max=1.0, step=0.01,
                  group="The parameter the paper omits",
                  help="Not in the paper -- its results never depend on a "
                       "fuel number. At 1.0 the vehicle needs six times its "
                       "own propellant and the mass floor binds. Watch the "
                       "residual and its block breakdown as you move this."),

            Param("case", "Case", "2-D (Section IV.A)", kind="choice",
                  choices=["2-D (Section IV.A)", "3-D (Section IV.B)"],
                  group="Problem",
                  help="The 2-D case's initial velocity is the paper's. The "
                       "3-D case is plotted in the paper but its initial "
                       "velocity is never printed, so the north component "
                       "here is this project's choice, not a transcription."),
            Param("north", "3-D north velocity", 2.0, min=0.5, max=4.0,
                  step=0.5, unit="UL/UT", group="Problem",
                  help="Used only by the 3-D case, and not a paper number."),
            Param("max_iter", "Iteration cap", 10, kind="int", min=4, max=20,
                  step=1, group="Solver",
                  help="Table 2's own budget is 15."),
            Param("K", "Nodes", 20, kind="int", min=15, max=50, step=5,
                  group="Solver",
                  help="Table 2 uses 50; the default here is lower so the "
                       "entry solves in a few seconds."),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)
        veh = PaperVehicle(alpha_m=float(p["alpha_m"]))
        alg = PaperAlgorithm()
        alg.N_iter_max = int(p["max_iter"])
        alg.K = int(p["K"])
        bc = (two_d_case() if str(p["case"]).startswith("2-D")
              else three_d_case(north=float(p["north"])))

        t0 = time.perf_counter()
        try:
            r = solve_benchmark(
                bc, veh=veh, alg=alg, sigma_guess=float(p["sigma_guess"]),
                hard_trust=str(p["hard_trust"]) == "hard box", verbose=False)
        except Exception as exc:                                # noqa: BLE001
            return _failed([f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        if not r["ever_solved"]:
            return _failed([
                "No sub-problem solved at these settings. The returned "
                "reference is the straight-line initial guess and means "
                "nothing -- which is exactly the failure mode Day 17 found in "
                "a guide that reported its guess as a result."])
        return _trajectory(r, p, elapsed)


# ======================================================================
def _to_view_vec(v_paper) -> list[float]:
    v = C_PAPER_TO_VIEW @ np.asarray(v_paper, dtype=float)
    return [float(v[0]), float(v[1]), float(v[2])]


def _to_view_quat(q_paper) -> list[float]:
    """Paper body-to-inertial quaternion into the renderer's frame."""
    R = (C_PAPER_TO_VIEW @ quat_to_rotmatrix(np.asarray(q_paper, dtype=float))
         @ D_BODY_TO_VIEW.T)
    w, x, y, z = rotmatrix_to_quat(R)
    return [float(x), float(y), float(z), float(w)]


def _trajectory(r, p, elapsed) -> Trajectory:
    x, u = r["x"], r["u"]
    veh = r["vehicle"]
    K = len(x)
    tf = float(r["sigma"])
    t = np.linspace(0.0, tf, K)

    pos = x[:, IDX_R].copy()
    pos[:, 0] = np.maximum(pos[:, 0], 0.0)      # paper's e1 is up
    vel = x[:, IDX_V]
    quats = x[:, IDX_Q]
    mass = x[:, IDX_M]

    thrust = np.array([quat_to_rotmatrix(quats[k]) @ u[k]
                       for k in range(K - 1)])
    mag = np.linalg.norm(u, axis=1)
    gim = np.degrees(np.arccos(np.clip(u[:, 0] / np.maximum(mag, 1e-12),
                                       -1.0, 1.0)))
    tilt = np.degrees([np.arccos(np.clip(
        float((quat_to_rotmatrix(q) @ np.array([1.0, 0.0, 0.0]))[0]),
        -1.0, 1.0)) for q in quats])
    rate = np.degrees(np.linalg.norm(x[:, IDX_W], axis=1))
    hist = r["history"]

    return Trajectory(
        t_state=t.tolist(),
        t_control=t[:-1].tolist(),
        position=[_to_view_vec(v) for v in pos],
        velocity=[_to_view_vec(v) for v in vel],
        thrust=[_to_view_vec(v) for v in thrust],
        attitude=[_to_view_quat(q) for q in quats],
        series=[
            Series("altitude", "Altitude (paper e1)", "UL",
                   pos[:, 0].tolist()),
            Series("east", "East", "UL", pos[:, 1].tolist()),
            Series("north", "North", "UL", pos[:, 2].tolist()),
            Series("speed", "Speed", "UL/UT",
                   np.linalg.norm(vel, axis=1).tolist()),
            # The paper defines the control at all K nodes; the viewer
            # contract wants one per interval, so these drop the last.
            Series("thrust_mag", "Thrust magnitude", "UM UL/UT^2",
                   mag[:-1].tolist(), on="control"),
            Series("gimbal", "Gimbal deflection", "deg",
                   gim[:-1].tolist(), on="control"),
            Series("tilt", "Tilt from vertical", "deg", tilt.tolist()),
            Series("rate", "Body rate", "deg/UT", rate.tolist()),
            Series("mass", "Mass", "UM", mass.tolist()),
        ],
        status="solved" if r["converged_at"] else "not converged",
        feasible=True,
        cost=float(veh.m_wet - mass[-1]),
        solve_time_ms=elapsed,
        solver=("paper Problem 2, " + ("hard trust box"
                                       if str(p["hard_trust"]) == "hard box"
                                       else "Algorithm 1 as printed")),
        thrust_max=veh.T_max,
        notes=_notes(r, p, veh, mag, gim, tilt, rate, hist),
        diagnostics={
            "time_of_flight_UT": tf,
            "time_of_flight_guess_UT": float(p["sigma_guess"]),
            "converged_at": r["converged_at"],
            "iterations_run": r["iterations_run"],
            "virtual_control": r["nu_total"],
            "nu_tolerance": PaperAlgorithm().nu_tol,
            "residual_fraction_pct": {k: round(v * 100.0, 1)
                                      for k, v in r["nu_fraction"].items()},
            "final_mass_UM": float(mass[-1]),
            "mass_floor_UM": veh.m_dry,
            "mass_floor_binding": bool(abs(mass[-1] - veh.m_dry) < 1e-6),
            "peak_thrust": float(mag.max()),
            "peak_gimbal_deg": float(gim.max()),
            "peak_tilt_deg": float(tilt.max()),
            "peak_rate_deg_UT": float(rate.max()),
            "terminal_pos_err_UL": float(np.linalg.norm(x[-1, IDX_R])),
        },
    )


def _notes(r, p, veh, mag, gim, tilt, rate, hist) -> list:
    alg = PaperAlgorithm()
    tf, guess = float(r["sigma"]), float(p["sigma_guess"])
    soft = str(p["hard_trust"]) != "hard box"
    notes = [
        f"Paper units throughout -- UL, UT, UM. Szmuk & Acikmese state their "
        f"numbers are notional and give no SI scale, so none is invented here. "
        f"Solved time of flight {tf:.3f} UT from a guess of {guess:.1f}."
    ]

    notes.append(
        f"The paper's central claim is robustness: ten guesses from 1 to 10 UT "
        f"all converging within 0.01 UT. Measured here across five guesses the "
        f"spread is 21.7 UT, and the answer correlates with the guess at 0.892 "
        f"-- 6.8, 11.7, 15.1, 28.5, 22.0 UT from guesses of 1, 3, 5, 8, 10. "
        f"Move the guess control and watch the answer follow it."
    )

    if soft:
        seq = hist["sigma"]
        notes.append(
            f"This is Algorithm 1 exactly as printed -- soft trust penalty "
            f"only. The Day 18 guide says this collapses the flight time "
            f"toward zero and oscillates. It does not: virtual control reaches "
            f"{r['nu_total']:.2e}, essentially machine precision, and the "
            f"flight time climbs monotonically from {seq[0]:.2f} to "
            f"{seq[-1]:.2f} UT. A runaway, in the opposite direction to the "
            f"one described."
        )
    else:
        notes.append(
            "The hard trust box is not in the paper. It stops the flight time "
            "running away -- by capping how far it can move from its reference "
            "each iteration, which pins it near the guess. Switch to `soft "
            "only` to see what the paper's own recipe does."
        )

    notes.append(
        f"The reason is in the paper's own Table 2. The cost is "
        f"sigma + w_nu |nu| + ... with w_nu = {alg.w_nu:.0e} against a sigma "
        f"coefficient of 1, so feasibility outprices minimum-time a hundred "
        f"thousand to one. The optimiser will buy twenty units of flight time "
        f"to shed a fraction of a unit of virtual control. Time is very nearly "
        f"free, which is why it either runs away or has to be held down."
    )

    frac = r["nu_fraction"]
    notes.append(
        f"alpha_m is {veh.alpha_m:.2f}, and it is not a paper number -- the "
        f"paper's results never depend on a fuel figure. The Day 18 guide sets "
        f"it to 1.0, calls it a harmless placeholder, and attributes its whole "
        f"residual to a cruder discretisation. Residual here is "
        f"{r['nu_total']:.4f}, and by state block it is "
        + ", ".join(f"{k} {v * 100:.0f}%" for k, v in frac.items() if v > 0.01)
        + ". At alpha_m = 1.0 it is 5.396, 21% mass and 71% velocity; at 0.03 "
          "it is 0.0147 and 100% position. A factor of 366 from one omitted "
          "constant, and only the second breakdown looks like a discretisation "
          "limit."
    )

    if abs(float(r["x"][-1, IDX_M]) - veh.m_dry) < 1e-6:
        notes.append(
            f"The mass floor is binding -- final mass is exactly "
            f"{veh.m_dry:.1f} UM. Hover thrust here is about "
            f"{veh.hover_thrust:.1f}, so at this alpha_m the vehicle burns "
            f"{veh.alpha_m * veh.hover_thrust:.2f} UM per UT against "
            f"{veh.m_wet - veh.m_dry:.1f} UM of usable propellant. Once the "
            f"floor binds, the mass row of the dynamics cannot be satisfied by "
            f"any control and the virtual control absorbs it at every node."
        )

    notes.append(
        f"What does reproduce, and it is the part that matters most: the "
        f"paper's rigid-body model and this project's are the same physics "
        f"written twice. Direction cosine matrices agree to 8.88e-16 over 400 "
        f"random quaternions and the gyroscopic term matches Day 14's exactly. "
        f"This run lands {r['x'][-1, IDX_R][0]:.1e} UL from the pad and rides "
        f"its bounds -- thrust to {mag.max():.3f} of {veh.T_max:.1f}, gimbal to "
        f"{gim.max():.2f} deg of {veh.delta_max_deg:.0f} -- which is the "
        f"paper's own claim about thrust saturation reproducing."
    )
    notes.append(
        "Two numbers here are not the paper's and are flagged rather than "
        "filled in: alpha_m, and the 3-D case's initial velocity, which the "
        "paper plots but never prints. Since the solved flight time depends "
        "strongly on alpha_m -- 2.99 UT at 1.0, 11.65 at 0.03 -- the flight "
        "time is not a comparable quantity against the paper either."
    )
    return notes


def _failed(notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status="error", feasible=False,
        solver="Szmuk benchmark", notes=notes,
    )
