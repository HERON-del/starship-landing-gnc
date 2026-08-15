"""
Day 9 — the flown trajectory, in the viewer.

Every other problem here renders what the optimiser *planned*. This one renders
what the vehicle actually *did*: a small dispersed fleet is run, and the
trajectory drawn in the scene is one of them flown open-loop through the
independently verified nonlinear simulator, against a vehicle whose mass, Isp
and drag differ from what the planner assumed, in wind the planner was never
told about.

That distinction is the whole of Day 9. The planned trajectory ends at the pad
by construction — `x[N] == 0` and `z[N] == 0` are hard equality constraints, so
the solver's own terminal error never exceeds about 5e-08 m. Reading accuracy
off that number measures whether the constraint was enforced, not whether the
vehicle landed. Flown, the same plans have a CEP of 3.74 m over 250 samples.

**Set the sample selector to `worst speed` and watch what breaks.** It is not
position and it is not propellant — there are 22 tonnes of margin nobody spends.
It is the speed at contact. A minimum-fuel trajectory is bang-bang and brings
the vehicle to rest exactly at the pad with no slack anywhere, so any error in
net deceleration puts it on one side or the other, and open-loop there is
nothing to restore it. Measured on a single plan against a swept true propellant
load: at nominal it stops 0.10 m up at 0.03 m/s, and 200 kg heavier — 0.67% of
the load — it reaches the pad at 6.54 m/s.

**The dispersion sliders say which error matters.** Take the entry dispersion to
zero and the miss barely moves: over the sweeps, CEP goes 2.76 m at full 3-sigma
to 2.98 m at none, so navigation error contributes essentially nothing. Take the
wind to zero instead and CEP falls to 0.84 m. Wind drives where it lands; mass
and drag drive how fast it arrives. Different errors, different symptoms.

Each sample is a full solve plus an RK4 replay, so the run count is deliberately
small here — this is the one problem in the viewer that costs seconds per
sample rather than milliseconds.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.monte_carlo import (
    DispersionConfig, sample_dispersions, run_single, summarize,
    MISS_TOL_M, SPEED_TOL_MS,
)
from src.dynamics_6dof import Vehicle6DoF

from ..registry import Problem, register
from ..types import Param, Series, Trajectory, quats_from_pitch


@register
class MonteCarloFlown(Problem):
    slug = "monte-carlo"
    title = "Monte Carlo: the flown trajectory"
    summary = ("What the vehicle did, not what the optimiser planned. "
               "Dispersed, open-loop, through the true dynamics.")
    phase = "Day 9"
    scene_scale = 700.0
    # This is a flown trajectory, not an optimiser's promise. It is allowed to
    # miss the pad and arrive moving -- that is the measurement.
    enforces_terminal_state = False

    def params(self) -> list[Param]:
        return [
            Param("n_runs", "Samples", 6, kind="int", min=2, max=24, step=1,
                  group="Fleet",
                  help="Each one is a full solve plus an RK4 replay, so this "
                       "costs seconds per sample. The published statistics "
                       "come from 250."),
            Param("seed", "Random seed", 42, kind="int", min=0, max=9999,
                  step=1, group="Fleet"),
            Param("pick", "Show which sample", "worst speed", kind="choice",
                  choices=["worst speed", "worst miss", "median miss",
                           "best", "first"],
                  group="Fleet",
                  help="'worst speed' is the interesting one: position and "
                       "propellant are rarely the problem."),

            Param("entry_scale", "Entry dispersion", 1.0, min=0.0, max=2.0,
                  step=0.1, group="Dispersion",
                  help="Scales the navigation error the planner is told "
                       "about. Take it to zero and the miss barely moves - "
                       "CEP went 2.76 m to 2.98 m over the sweeps."),
            Param("wind_3sigma", "Wind (3 sigma)", 15.0, min=0.0, max=30.0,
                  step=1.0, unit="m/s", group="Dispersion",
                  help="The planner is never told about this. It is what "
                       "drives the miss: CEP 0.84 m with no wind, 9.12 m at "
                       "30 m/s, very nearly linear."),
            Param("m_prop_3sigma", "Propellant error (3 sigma)", 1500.0,
                  min=0.0, max=3000.0, step=100.0, unit="kg",
                  group="Dispersion",
                  help="Also never told. This is what drives arrival speed "
                       "rather than position."),
            Param("Cd_3sigma", "Drag error (3 sigma)", 0.15, min=0.0, max=0.4,
                  step=0.01, group="Dispersion"),
            Param("isp_3sigma", "Isp error (3 sigma)", 3.0, min=0.0, max=10.0,
                  step=0.5, unit="s", group="Dispersion",
                  help="Under 1% of nominal, and it still moves the arrival - "
                       "mass flow sets the deceleration, and the trajectory "
                       "has no slack to absorb a change in it."),

            Param("z0_nominal", "Entry altitude", 420.0, min=280.0, max=620.0,
                  step=20.0, unit="m", group="Entry state",
                  help="The band is narrow and one-sided at fixed entry "
                       "speed: 100% solved from 360-420 m, 33% by 600 m."),
            Param("vz0_nominal", "Entry descent rate", -130.0, min=-170.0,
                  max=-100.0, step=5.0, unit="m/s", group="Entry state"),
            Param("theta0_nominal", "Entry pitch", 25.0, min=0.0, max=45.0,
                  step=5.0, unit="deg", group="Entry state"),

            Param("N", "Nodes", 50, kind="int", min=30, max=80, step=10,
                  group="Solver",
                  help="Barely affects the statistics - CEP sat between 3.07 "
                       "and 3.73 m across N = 30 to 80."),
        ]

    def solve(self, values: dict[str, Any]) -> Trajectory:
        p = self.merge(values)
        es = float(p["entry_scale"])

        disp = DispersionConfig(
            z0_nominal=float(p["z0_nominal"]),
            vz0_nominal=float(p["vz0_nominal"]),
            theta0_nominal_deg=float(p["theta0_nominal"]),
            wind_x_3sigma=float(p["wind_3sigma"]),
            m_prop_3sigma=float(p["m_prop_3sigma"]),
            Cd_scale_3sigma=float(p["Cd_3sigma"]),
            isp_3sigma=float(p["isp_3sigma"]),
        )
        for field, base in (("x0_3sigma", disp.x0_3sigma),
                            ("z0_3sigma", disp.z0_3sigma),
                            ("vx0_3sigma", disp.vx0_3sigma),
                            ("vz0_3sigma", disp.vz0_3sigma),
                            ("theta0_3sigma_deg", disp.theta0_3sigma_deg),
                            ("omega0_3sigma", disp.omega0_3sigma)):
            setattr(disp, field, base * es)

        n = int(p["n_runs"])
        rng = np.random.default_rng(int(p["seed"]))
        samples = [sample_dispersions(rng, disp) for _ in range(n)]

        t0 = time.perf_counter()
        try:
            results = [run_single(s, N=int(p["N"]), disp=disp, keep_path=True)
                       for s in samples]
        except Exception as exc:      # noqa: BLE001
            return _failed("error", [f"{type(exc).__name__}: {exc}"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        flown = [r for r in results if r["converged"]]
        if not flown:
            return _failed("infeasible", [
                f"None of the {n} samples produced a plan. The entry state is "
                f"probably outside the band the burn can null - the altitude "
                f"band is one-sided at fixed entry speed, and 600 m solves "
                f"only a third of the time."
            ])

        pick = str(p["pick"])
        if pick == "worst speed":
            chosen = max(flown, key=lambda r: r["speed"])
        elif pick == "worst miss":
            chosen = max(flown, key=lambda r: r["miss"])
        elif pick == "median miss":
            chosen = sorted(flown, key=lambda r: r["miss"])[len(flown) // 2]
        elif pick == "best":
            # Worst violation, normalised by each tolerance. Ranking on miss
            # first would pick the sample that tracks position best, and on
            # this problem that is routinely the one that arrives hardest --
            # the two errors are anti-correlated, because a run carrying more
            # deceleration error stops short rather than hitting fast.
            chosen = min(flown, key=lambda r: max(r["miss"] / MISS_TOL_M,
                                                  r["speed"] / SPEED_TOL_MS))
        else:
            chosen = flown[0]

        stats = summarize(results, n, elapsed / 1000.0)
        return _trajectory(chosen, stats, disp, p, elapsed, len(flown), n)


# ----------------------------------------------------------------------
def _trajectory(r, stats, disp, p, elapsed, n_ok, n) -> Trajectory:
    t = np.asarray(r["t_path"])
    y = np.asarray(r["y_path"])
    x, z, vx, vz, th, om, m = (y[:, i] for i in range(7))
    z = np.maximum(z, 0.0)

    veh = Vehicle6DoF()
    plan = r["plan"]
    sig, dlt = plan["sigma"], plan["delta"]
    dt_ctrl = float(plan["t_f"]) / len(sig)
    idx = np.clip((t[:-1] / dt_ctrl).astype(int), 0, len(sig) - 1)
    sigma_t, delta_t = sig[idx], dlt[idx]

    thrust = np.column_stack([
        sigma_t * np.sin(th[:-1] + delta_t),
        sigma_t * np.cos(th[:-1] + delta_t),
        np.zeros(len(idx)),
    ])
    pos = np.column_stack([x, z, np.zeros(len(x))])
    vel = np.column_stack([vx, vz, np.zeros(len(x))])
    s = r["sample"]

    return Trajectory(
        t_state=t.tolist(),
        t_control=t[:-1].tolist(),
        position=pos.tolist(),
        velocity=vel.tolist(),
        thrust=thrust.tolist(),
        attitude=quats_from_pitch(th).tolist(),
        series=[
            Series("altitude", "Altitude", "m", z.tolist()),
            Series("downrange", "Downrange", "m", x.tolist()),
            Series("speed", "Speed", "m/s", np.hypot(vx, vz).tolist()),
            Series("pitch", "Pitch from vertical", "deg",
                   np.degrees(th).tolist()),
            Series("rate", "Pitch rate", "deg/s", np.degrees(om).tolist()),
            Series("mass", "Vehicle mass", "kg", m.tolist()),
            Series("thrust", "Commanded thrust", "N", sigma_t.tolist(),
                   on="control"),
            Series("gimbal", "Commanded gimbal", "deg",
                   np.degrees(delta_t).tolist(), on="control"),
        ],
        status="flown" if r["good"] else r["fail_reason"],
        feasible=True,
        cost=float(r["fuel"]),
        solve_time_ms=elapsed,
        solver=f"{n_ok}/{n} planned, flown open-loop through the true vehicle",
        thrust_max=veh.T_max,
        notes=_notes(r, stats, disp, p, n_ok, n),
        diagnostics={
            "shown_miss_m": float(r["miss"]),
            "shown_speed_ms": float(r["speed"]),
            "shown_pitch_deg": float(r["pitch_deg"]),
            "shown_touched_down": bool(r["touched_down"]),
            "shown_altitude_left_m": float(r["altitude_left"]),
            "shown_planned_miss_m": float(r["planned_miss"]),
            "shown_fuel_kg": float(r["fuel"]),
            "shown_margin_kg": float(r["margin"]),
            "shown_burn_s": float(r["t_f"]),
            "true_m_prop_kg": float(s["m_prop"]),
            "true_isp_s": float(s["isp"]),
            "true_Cd_scale": float(s["Cd_scale"]),
            "true_wind_x_ms": float(s["wind_x"]),
            "fleet_solved_pct": float(stats.get("solve_rate", 0.0)),
            "fleet_landed_pct": float(stats.get("land_rate", 0.0)),
            "fleet_cep_m": float(stats.get("miss_cep", float("nan"))),
            "fleet_speed_mean_ms": float(stats.get("speed_mean",
                                                   float("nan"))),
            "fleet_margin_min_kg": float(stats.get("margin_min",
                                                   float("nan"))),
        },
    )


def _notes(r, stats, disp, p, n_ok, n) -> list[str]:
    s = r["sample"]
    notes = [
        f"Showing the '{p['pick']}' sample of {n}. It was planned for a "
        f"nominal vehicle in calm air, then flown against one carrying "
        f"{s['m_prop']:,.0f} kg of propellant at Isp {s['isp']:.1f} s with "
        f"drag at {s['Cd_scale']:.2f}x nominal and {s['wind_x']:+.1f} m/s of "
        f"wind. None of that was known to the planner.",
    ]

    verdict = ("landed" if r["good"] else r["fail_reason"])
    where = ("reached the pad" if r["touched_down"]
             else f"ran out of trajectory {r['altitude_left']:.2f} m up")
    notes.append(
        f"Outcome: {verdict}. It {where}, {r['miss']:.2f} m from the pad at "
        f"{r['speed']:.2f} m/s and {r['pitch_deg']:.2f} deg off vertical. "
        f"The plan it was flying reported a terminal error of "
        f"{r['planned_miss']:.2e} m - x[N] and z[N] are hard equality "
        f"constraints, so that number says the constraint was enforced, not "
        f"that the vehicle landed."
    )

    notes.append(
        f"Propellant is not the constraint: this run used {r['fuel']:,.0f} kg "
        f"and finished {r['margin']:,.0f} kg above dry mass. Across 250 "
        f"published samples the worst margin was 21,988 kg and no run ever "
        f"ran out. The authority to fix the arrival is aboard; open-loop is "
        f"why it goes unspent."
    )

    if stats.get("n_solved", 0) >= 2:
        notes.append(
            f"This fleet: {stats.get('solve_rate', 0):.0f}% planned "
            f"({n_ok}/{n}), {stats.get('land_rate', 0):.0f}% landed within "
            f"5 m and 5 m/s, CEP {stats.get('miss_cep', float('nan')):.2f} m, "
            f"mean arrival {stats.get('speed_mean', float('nan')):.2f} m/s. "
            f"With {n} samples these are indicative only - the published "
            f"figures are 98.4% planned, 29.6% landed, CEP 3.74 m over 250."
        )

    if float(p["wind_3sigma"]) == 0.0:
        notes.append(
            "Wind is off, so the miss you see is mass, Isp and drag alone. "
            "Over the sweeps that took CEP from 3.74 m down to 0.84 m - which "
            "is how the position error and the speed error were separated."
        )
    zeroed = [k for k, lab in (("wind_3sigma", "wind"),
                               ("m_prop_3sigma", "propellant"),
                               ("Cd_3sigma", "drag"),
                               ("isp_3sigma", "Isp"))
              if float(p[k]) == 0.0]
    if len(zeroed) == 4:
        msg = ("Every model error is off, so the planner's assumptions about "
               "the vehicle are exact.")
        if float(p["entry_scale"]) == 0.0:
            msg += (" With the entry pinned too, the only thing left between "
                    "plan and flight is discretisation - the plan is "
                    "trapezoidal collocation on a coarse grid, the flight is "
                    "RK4 on a fine one. Measured, that floor is about 1.3 m "
                    "and 0.35 m/s, and it does not improve with node count: "
                    "0.32 m/s at N=30 and 0.37 m/s at N=160.")
        else:
            msg += (" The entry state is still dispersed, though, and each "
                    "sample is a differently-shaped plan. Pin the entry too "
                    "if you want the discretisation floor on its own.")
        notes.append(msg)
    if float(p["entry_scale"]) == 0.0:
        notes.append(
            "Entry dispersion is off: every sample starts from the same "
            "state, and the only differences left are the ones the planner is "
            "not told. Whatever miss survives here is model error, not "
            "navigation. It barely moves - 2.76 m to 2.98 m over the sweeps."
        )
    return notes


def _failed(status: str, notes: list[str]) -> Trajectory:
    return Trajectory(
        t_state=[], t_control=[], position=[], velocity=[],
        thrust=[], attitude=[], status=status, feasible=False,
        solver="Monte Carlo", notes=notes,
    )
