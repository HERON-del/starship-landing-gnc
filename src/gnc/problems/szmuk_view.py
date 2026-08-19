"""
Day 18 -- the Szmuk & Acikmese benchmark, and what it took to reproduce it.

    Szmuk, M. and Acikmese, B., "Successive Convexification for 6-DoF Mars
    Rocket Powered Landing with Free-Final-Time", AIAA SciTech 2018-0617.

Seventeen days of this viewer show a solver checked against tests written
alongside it. This entry shows it meeting numbers chosen by people who never
saw the codebase, in the paper's own non-dimensional units -- UL, UT, UM, with
no SI scale given anywhere, so none is invented here.

**The paper's central claim reproduces.** Its headline is robustness: ten
time-of-flight guesses from 1 to 10 UT all converging within 0.01 UT. Measured
here across all ten, the spread is **0.00183 UT** -- comfortably inside the
paper's own bar -- with the virtual control between 1e-15 and 1e-17. That is
with Algorithm 1 exactly as printed: **no hard trust region and no quaternion
renormalisation**, both of which the Day 18 guide calls necessary additions.
Move the *time-of-flight guess* control and watch the answer stay at 3.282.

**Two things had to be right for that, and the entry lets you break either.**

*Discretisation.* Switch **Quadrature** to `single endpoint` -- evaluating the
input matrices once at the interval's left edge instead of integrating them
across it, as the guide does. The state transition matrix is unchanged and the
input matrix moves by 4.6%, and that is the difference between a residual of
2e-16 and one of order 1. It also destroys the robustness: the flight time
starts tracking its own initial guess.

*The mass constant.* `alpha_m` is not in the paper -- its results never depend
on a fuel figure, so the authors had no reason to print one. The guide sets it
to 1.0 and calls it a harmless placeholder. At 1.0 a three-unit trajectory
needs 6 UM of propellant against the 1 UM the vehicle carries; the mass floor
binds and the mass row of the dynamics becomes unsatisfiable by any control.
Move the **alpha_m** control up and watch the *residual by block* diagnostics
fill up with mass and velocity -- and note that switching the quadrature back
to the paper's does **not** rescue it, because no quadrature fixes a constraint
that is simply binding.

**This corrects a claim published earlier in this project.** With the
single-endpoint approximation in place, the sweep spread 21.7 UT and the flight
time followed its guess, and that was written up here as the paper's claim
failing to reproduce -- with the cost weights offered as the reason. The cost
weights are fine. The failure was in this implementation.

Note the frame. The paper puts gravity along `-e1`, so its first axis is up and
its state vector starts with mass. Everywhere else in this project +z is up and
mass is last. The conversion into the renderer is explicit here rather than a
silent re-axing of the problem.
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
from ..types import Param, Series, Trajectory
from .rigid_body_view import D_BODY_TO_VIEW

#: Paper inertial (up, east, north) into the renderer's (x, y up, z).
#: A proper rotation -- the naive swap has determinant -1 and would mirror
#: every rotation in the scene, which is the mistake Day 14 nearly made.
C_PAPER_TO_VIEW = np.array([[0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0]])


@register
class SzmukBenchmark(Problem):
    slug = "szmuk-benchmark"
    title = "Szmuk & Acikmese Benchmark"
    summary = ("A published free-final-time SCvx problem, reproduced -- and "
               "the two things that had to be right.")
    phase = "Day 18"
    scene_scale = 9.0
    enforces_terminal_state = False

    def params(self) -> list[Param]:
        return [
            Param("sigma_guess", "Time-of-flight guess", 3.0, min=1.0,
                  max=10.0, step=1.0, unit="UT", group="The paper's claim",
                  help="The paper reports ten guesses from 1 to 10 all "
                       "converging within 0.01 UT. Measured across all ten "
                       "here the spread is 0.00183. Move this and watch the "
                       "solved flight time stay at 3.282."),
            Param("exact_foh", "Quadrature", "paper Eq. 22", kind="choice",
                  choices=["paper Eq. 22", "single endpoint"],
                  group="The paper's claim",
                  help="Eq. 22 integrates the input matrices across each "
                       "interval against the hold weights. Single endpoint "
                       "evaluates them once at the left edge -- a 4.6% "
                       "difference on one matrix, and the difference between "
                       "a residual of 2e-16 and one of order 1."),
            Param("hard_trust", "Trust region", "soft only (the paper)",
                  kind="choice",
                  choices=["soft only (the paper)", "hard box"],
                  group="The paper's claim",
                  help="Soft only is Algorithm 1 as printed, and it works. "
                       "The hard box is the guide's addition; with the "
                       "paper's quadrature it is unnecessary and measurably "
                       "worse -- 0.025 UT of sweep spread against 0.002."),

            Param("alpha_m", "alpha_m", 0.03, min=0.01, max=1.0, step=0.01,
                  group="The parameter the paper omits",
                  help="Not in the paper -- its results never depend on a "
                       "fuel number. At 1.0 the vehicle needs six times its "
                       "own propellant, the mass floor binds, and no "
                       "quadrature can rescue it. Watch the residual and its "
                       "block breakdown as you move this."),

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
            Param("max_iter", "Iteration cap", 12, kind="int", min=4, max=20,
                  step=1, group="Solver", help="Table 2's own budget is 15."),
            Param("K", "Nodes", 25, kind="int", min=15, max=50, step=5,
                  group="Solver",
                  help="Table 2 uses 50; lower here so the entry solves in a "
                       "few seconds."),
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
                hard_trust=str(p["hard_trust"]) == "hard box",
                exact_foh=str(p["exact_foh"]) == "paper Eq. 22",
                renormalize=False, verbose=False)
        except Exception as exc:                                # noqa: BLE001
            return _failed([f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        if not r["ever_solved"]:
            return _failed([
                "No sub-problem solved at these settings. The returned "
                "reference is the straight-line initial guess and means "
                "nothing -- which is the failure mode Day 17 found in a guide "
                "that reported its own guess as a result."])
        return _trajectory(r, p, elapsed)


# ======================================================================
def _to_view_vec(v_paper) -> list[float]:
    v = C_PAPER_TO_VIEW @ np.asarray(v_paper, dtype=float)
    return [float(v[0]), float(v[1]), float(v[2])]


def _to_view_quat(q_paper) -> list[float]:
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
    pos[:, 0] = np.maximum(pos[:, 0], 0.0)      # the paper's e1 is up
    vel, quats, mass = x[:, IDX_V], x[:, IDX_Q], x[:, IDX_M]

    thrust = np.array([quat_to_rotmatrix(quats[k]) @ u[k]
                       for k in range(K - 1)])
    mag = np.linalg.norm(u, axis=1)
    gim = np.degrees(np.arccos(np.clip(u[:, 0] / np.maximum(mag, 1e-12),
                                       -1.0, 1.0)))
    tilt = np.degrees([np.arccos(np.clip(
        float((quat_to_rotmatrix(q) @ np.array([1.0, 0.0, 0.0]))[0]),
        -1.0, 1.0)) for q in quats])
    rate = np.degrees(np.linalg.norm(x[:, IDX_W], axis=1))
    converged = r["nu_total"] < PaperAlgorithm().nu_tol

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
            # The paper defines the control at all K nodes; the viewer wants
            # one per interval, so these drop the last.
            Series("thrust_mag", "Thrust magnitude", "UM UL/UT^2",
                   mag[:-1].tolist(), on="control"),
            Series("gimbal", "Gimbal deflection", "deg",
                   gim[:-1].tolist(), on="control"),
            Series("tilt", "Tilt from vertical", "deg", tilt.tolist()),
            Series("rate", "Body rate", "deg/UT", rate.tolist()),
            Series("mass", "Mass", "UM", mass.tolist()),
        ],
        status="converged" if converged else "residual remains",
        feasible=True,
        cost=float(veh.m_wet - mass[-1]),
        solve_time_ms=elapsed,
        solver=("paper Problem 2, "
                + ("Eq. 22 quadrature" if str(p["exact_foh"]) == "paper Eq. 22"
                   else "single-endpoint quadrature")),
        thrust_max=veh.T_max,
        notes=_notes(r, p, veh, mag, gim, converged),
        diagnostics={
            "time_of_flight_UT": tf,
            "time_of_flight_guess_UT": float(p["sigma_guess"]),
            "virtual_control": r["nu_total"],
            "nu_tolerance": PaperAlgorithm().nu_tol,
            "converged": converged,
            "iterations_run": r["iterations_run"],
            "residual_fraction_pct": {k: round(v * 100.0, 1)
                                      for k, v in r["nu_fraction"].items()},
            "final_mass_UM": float(mass[-1]),
            "mass_floor_binding": bool(abs(mass[-1] - veh.m_dry) < 1e-6),
            "peak_thrust": float(mag.max()),
            "peak_gimbal_deg": float(gim.max()),
            "peak_tilt_deg": float(tilt.max()),
            "terminal_pos_err_UL": float(np.linalg.norm(x[-1, IDX_R])),
        },
    )


def _notes(r, p, veh, mag, gim, converged) -> list:
    tf, guess = float(r["sigma"]), float(p["sigma_guess"])
    exact = str(p["exact_foh"]) == "paper Eq. 22"
    hard = str(p["hard_trust"]) == "hard box"
    notes = [
        f"Paper units throughout -- UL, UT, UM. Szmuk & Acikmese state their "
        f"numbers are notional and give no SI scale, so none is invented. "
        f"Solved flight time {tf:.4f} UT from a guess of {guess:.1f}, with a "
        f"dynamics residual of {r['nu_total']:.2e} against a tolerance of "
        f"{PaperAlgorithm().nu_tol:.0e}."
    ]

    if exact and not hard:
        notes.append(
            "This is Algorithm 1 exactly as printed -- the paper's own "
            "quadrature, soft trust penalty only, no quaternion "
            "renormalisation -- and the paper's central claim reproduces: ten "
            "guesses from 1 to 10 UT land within 0.00183 UT of each other, "
            "inside the paper's own stated bar of 0.01, with the virtual "
            "control between 1e-15 and 1e-17. Move the guess and watch the "
            "answer stay put."
        )
    if not exact:
        notes.append(
            f"Quadrature is the single-endpoint approximation -- the input "
            f"matrices evaluated once at each interval's left edge rather than "
            f"integrated across it. The residual here is {r['nu_total']:.2e}. "
            f"The state transition matrix is identical either way and the "
            f"input matrix differs by 4.6%; that is the whole cause. Switch "
            f"back to Eq. 22 and it collapses to machine precision -- and the "
            f"robustness returns with it."
        )
    if hard:
        notes.append(
            "The hard trust box is the Day 18 guide's addition, not the "
            "paper's. With the paper's quadrature it is unnecessary, and it "
            "measurably degrades the result -- 0.025 UT of sweep spread "
            "against 0.002 without it, because capping how far the flight "
            "time may move each iteration pins it near its guess."
        )

    frac = r["nu_fraction"]
    live = ", ".join(f"{k} {v * 100:.0f}%" for k, v in frac.items() if v > 0.01)
    notes.append(
        f"alpha_m is {veh.alpha_m:.2f}, and it is not a paper number -- the "
        f"paper's results never depend on a fuel figure. The guide sets it to "
        f"1.0 and calls it harmless. Residual by state block here: {live}. At "
        f"alpha_m = 1.0 it is dominated by mass and velocity, because hover "
        f"thrust is about {veh.hover_thrust:.1f} and the vehicle would need "
        f"six times its propellant load -- the mass floor binds and no "
        f"quadrature rescues a constraint that is simply active."
    )

    if abs(float(r["x"][-1, IDX_M]) - veh.m_dry) < 1e-6:
        notes.append(
            f"The mass floor is binding: final mass is exactly "
            f"{veh.m_dry:.1f} UM. At this alpha_m the vehicle burns "
            f"{veh.alpha_m * veh.hover_thrust:.2f} UM per UT against "
            f"{veh.m_wet - veh.m_dry:.1f} UM usable. Lower alpha_m to get a "
            f"trajectory the propellant can actually pay for."
        )

    notes.append(
        f"What was external and passed regardless: the paper's rigid-body "
        f"model and this project's are the same physics written twice. "
        f"Direction cosine matrices agree to 8.88e-16 over 400 random "
        f"quaternions, and the gyroscopic term matches Day 14's exactly. This "
        f"run lands {float(np.linalg.norm(r['x'][-1, IDX_R])):.1e} UL from the "
        f"pad and rides its bounds -- thrust to {mag.max():.3f} of "
        f"{veh.T_max:.1f}, gimbal to {gim.max():.2f} deg of "
        f"{veh.delta_max_deg:.0f}."
    )
    notes.append(
        "An earlier version of this entry reported that the paper's claim did "
        "not reproduce, and blamed the paper's cost weights. That was wrong: "
        "the single-endpoint quadrature was this implementation's shortcut, "
        "not the paper's, and it was what broke the sweep. The correction is "
        "recorded in LOG.md rather than quietly swapped in."
    )
    return notes


def _failed(notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status="error", feasible=False,
        solver="Szmuk benchmark", notes=notes,
    )
