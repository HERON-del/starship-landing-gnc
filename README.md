# Starship Flip-and-Land: 6-DoF Trajectory Optimization with Sequential Convex Programming

Real-time, fuel-optimal guidance for the Starship belly-flop-to-vertical landing
maneuver, solved with Sequential Convex Programming (SCvx) and wrapped in a
closed-loop Model Predictive Controller — with an interactive 3-D viewer for
exploring every solution.

**Status:** in development

---

## Interactive trajectory viewer

![3-DoF powered descent in the viewer](results/viewer_3dof.jpg)

```bash
python run_viewer.py
```

Opens a browser at `http://127.0.0.1:8000`. Every parameter is a live control —
move a slider and the problem re-solves in milliseconds and re-animates.

Trajectory colour encodes thrust magnitude (blue = coasting, orange = full
throttle), so the bang-bang structure of a minimum-fuel solution is visible at a
glance. The translucent cone is the glideslope corridor.

| | |
|---|---|
| **Camera** | orbit / chase / side / top |
| **Overlays** | ground grid, approach corridor, flight path, velocity + thrust vectors |
| **Playback** | scrub, play/pause (`space`), ¼×–2× speed |
| **Export** | full run as JSON — parameters plus trajectory |
| **Re-solve** | `r`, or automatically on any control change |

Three problems are registered: the Day 1 1-D optimiser, the Day 2 open-loop
simulation, and the Week 1 3-DoF optimiser. The Day 2 entry propagates the
verified variable-mass model rather than optimising, so the four Day 2
exploration experiments are sliders rather than notebook edits — pick a guidance
law, an integrator and a step size and watch what the physics actually does.
It is also the only problem that can fail *physically*: it will happily fly the
vehicle into the ground at 220 m/s and report the crash.

---

## Motivation

The Starship landing flip is one of the hardest problems in modern guidance and control:

- **6 degrees of freedom** with strongly coupled translational and rotational dynamics
- **Non-convex constraints**: minimum-throttle bounds, gimbal cone limits, glideslope
- **State-triggered events**: engine relight windows, flap deployment
- **Hard real-time requirement**: a solution must be produced onboard in under 100 ms
- **Minimal margins**: propellant reserve for the landing burn is measured in seconds

This repository implements a from-scratch solution and validates it against
published results in the trajectory-optimization literature.

---

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| Day 2 | Variable-mass dynamics + verified RK4 integrator, live in the viewer | done |
| Day 3 | Constrained landing: glideslope, thrust bounds, pointing limit | done |
| Week 1 | 3-DoF convex powered descent, glideslope + tilt cones | done |
| Week 2 | Sequential Convex Programming (SCvx) solver | next |
| Week 3 | 6-DoF rigid-body dynamics with quaternions | |
| Week 4 | Aerodynamics, flap control, 6-DoF SCvx | |
| Week 5 | Closed-loop MPC + Monte Carlo dispersion analysis | |
| Week 6 | Documentation and technical paper | |

---

## Results so far

### 3-DoF powered descent (Week 1)

Full 3-D translation from 700 m altitude and ~470 m downrange, with a 25°
glideslope corridor and a 25° thrust-tilt (gimbal) limit. Solves in ~11 ms.

The minimum-fuel solution is genuinely **bang-bang**: 34 of 60 control steps sit
at exactly zero thrust and the remainder at the 30 m/s² limit. Burning early
wastes propellant holding the vehicle against gravity for longer, so the optimal
profile is an initial pitch-over burn, a ballistic coast, and one hard terminal
burn. This is the mathematical origin of the "suicide burn" profile flown by
Falcon 9.

Every constraint stays convex — the thrust-magnitude bound, the gimbal cone, and
the glideslope are all second-order cones — so the solve is fast and the optimum
is global.

### Constrained landing optimization (Day 3)

Minimum-fuel powered descent with realistic engineering constraints: glideslope
cone, thrust magnitude bounds via lossless convexification, thrust pointing
limit, and variable mass.

![Landing trajectory](results/day3_landing.png)

The vehicle enters at 2,910 m altitude and 385 m downrange doing 285 m/s, and
lands on the pad at rest using 18,073 kg of propellant (60% of the landing load).
The trajectory stays inside the glideslope cone, thrust respects [T_min, T_max],
and the pointing angle never exceeds 30°. The mass-reference iteration converges
in 8 damped steps.

**The relaxation is verified, not assumed.** Lossless convexification replaces
the non-convex `‖T‖ >= T_min` with `‖T‖ <= sigma` plus a box on sigma, and is
only lossless when the optimal solution drives `‖T‖ = sigma`. That does not hold
automatically: on a gentle entry the optimiser parks sigma on T_min and lets
`‖T‖` fall to 0.87 T_min, producing a trajectory that burns minimum-throttle
propellant while generating less than minimum-throttle force. Checking
`‖T‖ <= sigma` passes trivially and hides it, so the suite asserts the two are
*equal* — max gap 0 N on the nominal problem.

**The problem is solved non-dimensionally.** In SI, thrust (~3×10⁶ N) and the
velocity-update coefficient `dt/m` (~3×10⁻⁶) span twelve orders of magnitude, and
Clarabel raises `SolverError` on some instances — indistinguishable from
infeasibility unless you cross-check solvers. Scaling every quantity by a
characteristic value makes the constraint sweeps smooth and monotone.

### Verified dynamics and integration (Day 2)

3-DoF variable-mass planar dynamics with an in-house RK4 integrator, verified
against closed-form solutions and a formal convergence-order study.

![Convergence order](results/day2_convergence_order.png)

| Integrator | Theoretical order | Measured order |
|-----------|------------------|----------------|
| Euler     | 1                | 1.00           |
| RK4       | 4                | 3.99           |

Verification tests: ballistic free-fall against closed-form kinematics
(machine-precision agreement), ideal hover altitude conservation, and
Tsiolkovsky mass-flow validation.

The convergence study references a **closed-form** solution for constant thrust
with variable mass, not a finely-stepped numerical one. Integrating a reference at
`dt = 1e-4` takes 100,000 steps and accumulates ~3e-9 of round-off — worse than
RK4 at `dt = 0.125` — which puts a floor under the measurement and drags the
apparent order down to ~1.6. Measuring convergence requires a reference that does
not itself converge.

**Physical result:** thrust-to-weight at *minimum* throttle is 2.16 at wet mass and
2.81 at dry mass. It exceeds 1 for the entire landing burn, so the vehicle cannot
hover at any point. The single precisely-timed landing burn is forced by engine
sizing rather than chosen. Ideal-hover endurance on 30 t of propellant is
`Isp·ln(m_wet/m_dry) = 85.8 s`, roughly 4.9× a real 15–20 s landing burn.

### 1-D soft landing (Day 1)

![Day 1 result](results/day1_first_landing.png)

The vertical-only ancestor of the problem — and a useful trap.

**Minimum-fuel at fixed final time is degenerate.** Summing the velocity dynamics
telescopes to `dt·Σa = v[N] − v[0] + g·T`, so with the terminal velocity pinned,
total impulse is fixed by the constraints and *every feasible trajectory ties*.
Verified: CLARABEL, SCS and OSQP all return 118.1 m/s, HIGHS returns 118.1 m/s,
and even `Minimize(0)` returns 118.1 m/s.

The bang-bang profile often shown for this problem is therefore a solver
artifact — simplex methods return a vertex of the optimal face, interior-point
methods return an interior point. Both are "optimal". Fuel only becomes a real
objective once the final time is free or mass depletion is modelled, which is why
the 3-DoF case above produces bang-bang for genuine reasons.

---

## Quick start

```bash
git clone https://github.com/HERON-del/starship-landing-gnc.git
cd starship-landing-gnc
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows;  source venv/bin/activate elsewhere
pip install -r requirements.txt
python tests/test_setup.py      # environment check
python tests/test_problems.py   # solver check
python run_viewer.py            # 3-D viewer
```

**Solver note:** this project uses **Clarabel** rather than ECOS. ECOS ships no
Python 3.13 wheel and has been unmaintained since 2023; Clarabel is its successor
and ships with CVXPY.

---

## Repository structure

```
src/
  dynamics.py          3-DoF variable-mass rocket model + closed-form solutions
  integrators.py       Euler and RK4 steppers, fixed-step propagator
  constraints.py       glideslope, thrust bounds, pointing, mass dynamics
  landing_problem.py   constrained minimum-fuel landing (direct transcription)
  gnc/
    types.py           Param + Trajectory contracts shared by solver and viewer
    registry.py        problem plugin registry
    server.py          FastAPI backend (2 endpoints)
    problems/          one file per solvable problem
web/                   Three.js front end, no build step, vendored deps
tests/                 environment and solver verification
docs/                  derivations and the extension guide
results/               generated figures and exported runs
```

Adding a new problem is one Python file — see
[docs/adding-a-problem.md](docs/adding-a-problem.md). The UI builds its controls
from the declared parameter schema, so the front end never changes.

---

## Method

The landing problem is non-convex. SCvx handles this by iteratively linearizing
the dynamics about a reference trajectory, solving the resulting convex
sub-problem inside a trust region, and updating the reference. Virtual control
slack variables guarantee sub-problem feasibility at every iteration.

---

## References

1. Açıkmeşe, B. and Ploen, S. R., "Convex Programming Approach to Powered Descent Guidance for Mars Landing," *Journal of Guidance, Control, and Dynamics*, 2007.
2. Mao, Y., Szmuk, M., and Açıkmeşe, B., "Successive Convexification of Non-Convex Optimal Control Problems," arXiv:1804.06539.
3. Szmuk, M., Reynolds, T. P., and Açıkmeşe, B., "Successive Convexification for Real-Time 6-DoF Powered Descent Guidance with State-Triggered Constraints," *JGCD*, 2020.
4. "Optimization of Flip-Landing Trajectories for Starship," arXiv:2508.06520, 2025.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Author

**[Your Name]** — Aerospace Engineering, [Your College]
