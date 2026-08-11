# Engineering Log — Starship Landing GNC

Daily record of work done, problems hit, and decisions made.
This becomes the methodology section of the final paper.

---

## Day 1 — 2026-08-09

### Done
- Installed Python, VS Code, Git; created and cloned the GitHub repo
- Set up virtual environment; installed cvxpy, numpy, scipy, matplotlib
- Solved first convex optimization: 1-D minimum-fuel soft landing
- Built an interactive 3-D trajectory viewer (FastAPI + Three.js) with a
  problem-registry architecture, so each future week plugs in as one Python file
- Added a 3-DoF convex powered-descent problem with glideslope and thrust-tilt cones

### Key insight
**Minimum-fuel at fixed final time is degenerate.** Summing the velocity dynamics
telescopes to `dt·Σa = v[N] − v[0] + g·T`, so with the terminal velocity pinned,
total impulse is fixed by the constraints and every feasible trajectory ties at
118.1 m/s. Confirmed across CLARABEL, SCS, OSQP and HIGHS — and `Minimize(0)`
returns the same value.

The bang-bang profile usually shown for this problem is a **solver artifact**:
simplex returns a vertex of the optimal face, interior-point returns an interior
point. Both are "optimal". Fuel only becomes a real objective once the final time
is free or mass depletion is modelled.

The 3-DoF case *does* produce genuine bang-bang (34 of 60 steps at zero thrust,
the rest at the 30 m/s² limit), because `sum(‖a‖) ≥ ‖sum(a)‖` leaves the optimiser
real freedom that the 1-D scalar sum did not.

### Environment decision
`ecos` ships no Python 3.13 Windows wheel and has been unmaintained since 2023.
Using **Clarabel** instead — its direct successor, bundled with CVXPY. Everywhere
the guide says `solver=cp.ECOS`, this project uses `solver=cp.CLARABEL`.

### Problems hit
- Guide assumes Python 3.12; this machine has only 3.13 (Microsoft Store build)
- `pip install ecos` tried to compile from source and failed on missing MSVC

---

## Day 2 — 2026-08-10

### Done
- Built `src/integrators.py`: Euler and RK4 steppers + fixed-step propagator
- Built `src/dynamics.py`: 3-DoF variable-mass planar rocket, Starship-class parameters
- Built `tests/test_dynamics.py`: 4 verification tests, all passing
- Measured empirical convergence order: **RK4 = 3.99** (theory 4.0), **Euler = 1.00** (theory 1.0)
- Ran all four exploration experiments in `notebooks/02_dynamics_exploration.ipynb`

### Paper takeaway
> _To write in my own words after the Part 1 read. Cover: (1) what problem
> Açıkmeşe & Ploen 2007 solves, (2) what lossless convexification means,
> (3) why ‖T‖ ≥ T_min is the hard part, (4) one thing I did not understand._

### Verification bug found in the guide's Day 2 test
Test 4 as written measured **RK4 order 1.65**, not 4. The integrator was correct —
the *reference solution* was wrong. The guide builds it with RK4 at `dt = 1e-4`,
which is 100,000 steps and accumulates ~3.4e-9 of round-off:

| reference step | steps | error vs closed form |
|---|---|---|
| `dt = 1e-3` | 10,000 | 2.2e-10 |
| `dt = 1e-4` | 100,000 | 3.4e-09 ← worse |

More steps means more accumulated round-off, not more accuracy. That floor swamped
RK4's real truncation error at `dt ≤ 0.125`, flattening the fitted slope.

**Fix:** derived the closed-form solution for constant thrust with variable mass
(`exact_constant_thrust` in `src/dynamics.py`) and used it as the reference. With
`u = m/m₀`, the velocity integral is the Tsiolkovsky logarithm `L = −ln u` and its
integral is `S = (m₀/ṁ)(u ln u + 1 − u)`. Step sizes restricted to
`[1.0 … 0.0625]`, which is in the asymptotic regime and above the float64 floor.
Measured order is now 4.00, 4.00, 4.00, 4.01, 3.96 across the sweep.

### Key insights
- **The vehicle physically cannot hover.** TWR at *minimum* throttle is 2.16 at wet
  mass and 2.81 at dry mass — above 1 for the entire landing burn, not just near
  touchdown. A hover attempt with 5 t of propellant left climbs 211 m in 5 s and is
  ascending at +85 m/s. The single precisely-timed burn is forced by engine sizing,
  not chosen.
- **Suicide burn (Experiment A).** At −200 m/s entry with full thrust, the stopping
  *distance* is a constant 448.2 m regardless of where the burn starts — same speed,
  same mass, same manoeuvre translated up and down. So the trigger is a distance,
  and the altitude follows from arrival speed. Realistically: free-fall from 5 km at
  −200 m/s, ignite at **1247.7 m** arriving at **337 m/s**, touchdown uses 15,524 kg
  (52% of the landing load).
- **Propellant budget (Experiment C).** Ideal hover gives `dm/dt = −m/Isp`, so
  endurance is `Isp·ln(m_wet/m_dry) = 85.8 s`. Simulation agrees to 0.1 s. Against a
  15–20 s real landing burn that is ~4.9× margin.
- **RK4 vs Euler (Experiment D).** On free-fall at `dt = 2 s`, Euler is wrong by
  196 m; RK4 by 2e-12 m. But that is not a fair fight — free-fall is a quadratic and
  RK4 integrates polynomials up to degree 4 exactly. The variable-mass case in Test 4
  is where RK4's true fourth-order behaviour shows.
- At `dt = 1 s` on the nonlinear case, RK4 is 2.6e7× more accurate than Euler for 4×
  the work per step.

### Viewer integration
Added `descent-sim` to the 3-D viewer so the Day 2 model is explorable directly.
Building it surfaced a real result: **an open-loop suicide burn essentially cannot
land.** With constant thrust at TWR ~6 the engine keeps burning after the descent
is arrested, so the vehicle stops above the pad and climbs away; ignite a metre
lower and it crashes. The trigger is only evaluated once per step, so at −200 m/s
with `dt = 0.05 s` the ignition altitude quantises to a 10 m grid and touchdown
speed swings from 17 m/s to 73 m/s across one step.

Added a closed-loop law for contrast — thrust tracking `a = v²/2z + g`, the
constant deceleration that nulls velocity exactly at the pad, with ignition
emerging from the minimum-throttle bound rather than being scheduled. It lands at
**0.32 m/s** on the same initial conditions where the open-loop burn crashes at
224 m/s. That gap is the argument for computed guidance, and the motivation for
Weeks 2 and 5.

### Problems hit
- Test 4 failed at order 1.65 — root-caused to the reference solution, not the
  integrator (see above)
- The viewer's verification suite asserted every problem lands at rest, which is
  only true of optimisers. Added an `enforces_terminal_state` flag so simulations
  are checked for ground contact and time monotonicity instead.
- The viewer hardcoded "OPTIMAL" on any successful solve, which labelled a crash
  as optimal. It now reports the real status with a severity colour.
- Guide's ANSI `[PASS]` markers render as escape gibberish on Windows consoles;
  added VT-mode detection with a plain-text fallback
- Guide's test writes figures relative to the current directory; switched to paths
  resolved from the repo root so it works from any working directory

### Tomorrow-after (Day 3)
- Re-derive the Day 1 landing problem using the new dynamics module
- Add glideslope and thrust-cone constraints
- Begin the log-mass change of variables from the paper

### Time spent
_X hours_

---

## Day 4 — 2026-08-11

### Done
- Built `src/discretization.py`: vectorised Euler and trapezoidal collocation
- Built `src/landing_free_time.py`: burn duration chosen by search, not assumed
- Built `tests/test_free_time.py`: 5 groups, all passing
- Added `landing-free-time` to the 3-D viewer (five problems now registered)
- Wrote `docs/free-time-and-scvx.md` covering the SCvx loop and troubleshooting

### Headline result
Holding the entry state fixed and letting the optimiser choose the duration:

| Configuration | Burn time | Fuel |
|---|---|---|
| Fixed 20 s (Day 3) | 20.00 s | 18,077 kg |
| Free time, Euler | 16.02 s | 16,670 kg |
| Free time, trapezoidal | 16.46 s | 16,797 kg |

**7.1% of the landing propellant recovered by choosing the duration rather than
guessing it.** Gravity losses are the whole story: every extra second the engine
spends holding the vehicle up is propellant that does nothing for the
trajectory.

### The guide's free-time formulation does not work
It declares `t_f = cp.Variable()` but holds `dt` at a reference value inside
every dynamics constraint. `t_f` then appears **only** in its own bounds and its
own `0.001·t_f` penalty — completely decoupled from the trajectory. Minimising
that penalty drives `t_f` to whichever bound the penalty prefers, and the
reported "optimal burn time" is the lower bound in disguise. It would not change
if the vehicle were twice as heavy.

The failure is quiet, not loud: it compiles, runs, converges, and prints a
plausible number.

**What I did instead.** Time enters the dynamics multiplicatively, so it cannot
be a convex variable. But for *fixed* `t_f` the problem is convex and solves in
milliseconds, and fuel-versus-duration is smooth and bounded by infeasibility at
both ends. So: coarse scan to bracket the feasible interval, then golden-section
to the minimum. Every point evaluated is a global optimum of its own subproblem.
The convex problem compiles once with the duration-dependent coefficients as
`cp.Parameter`s, so 23 solves take under a second.

### Lossless convexification has a boundary, and the optimum sits on it
Sweeping duration, the relaxation gap is exactly zero for `t_f` in 16.5–20.5 s
and jumps to 13% of T_min outside it. Every slack case has the pointing
constraint at its limit — Açıkmeşe & Ploen's magnitude-only proof does not cover
an active pointing constraint.

Saturation turns out to be **necessary but not sufficient**, which I only caught
by measuring: at N = 40 the search settles on 16.03 s with the tilt pegged at
exactly 30.0° and a gap of 90 N — tight by any measure. So the code checks the
gap on every solve instead of inferring it from the tilt.

This is not academic. The *cheapest* duration lies in the slack region, and its
trajectory burns propellant at the σ rate while commanding less force than that:
cheap on paper, unflyable in fact. The search now rejects those and reports how
many. In the viewer, switching the check off moves the answer from 16.03 s
(gap 90 N) to 15.95 s (gap 328,328 N) at the same fuel to the kilogram.

### Euler vs trapezoidal, settled by measurement not assertion
Comparing reported fuel proves nothing — each is optimal for its own discretized
model, and Euler routinely reports *less* fuel precisely because its model is
wrong. So the test flies the commanded thrust through Day 2's verified RK4
integrator:

```
Euler  position error  43.7 m  (1.50% of the descent)
Trapz  position error   6.1 m  (0.21% of the descent)
```

**7.1× smaller miss at the same node count.** Euler's cheaper number was
discretisation error being exploited.

### Problems hit
- Two of my own test assertions were wrong, not the code. I asserted
  trapezoidal wins on terminal velocity as well as position — it does not, and
  the ordering flips between runs, so asserting it was asserting noise. And I
  required Euler to land within 1% of the descent, which contradicts the very
  finding the test exists to demonstrate. Euler now gets a loose sanity bound
  and the tight bound applies only to trapezoidal.
- The guide predicts a U-shaped fuel-vs-duration curve. It is not a U here: it
  rises monotonically across the whole feasible window, so the optimum sits
  against the short-duration feasibility edge rather than in an interior basin.

### Tomorrow (Day 5)
- Trust regions — the missing third pillar of SCvx
- Replace fixed damping on the mass reference with an adaptive step

### Time spent
_X hours_

---

## Day 3 — 2026-08-10

### Done
- Built `src/constraints.py`: glideslope, thrust magnitude (lossless
  convexification), pointing, mass dynamics, log-mass bounds (for Week 2)
- Built `src/landing_problem.py`: constrained 2-D minimum-fuel landing with
  damped mass-reference iteration, converging in 8 iterations
- Built `tests/test_landing.py`: 4 test groups, all passing
- Generated `results/day3_landing.png` — first real constrained trajectory
- Ran all four exploration experiments in
  `notebooks/03_constraints_exploration.ipynb`

### Paper takeaway
> _To write in my own words after the Part 1 read. Cover: (1) what problem
> Açıkmeşe & Ploen 2007 solves, (2) what lossless convexification means,
> (3) why ‖T‖ ≥ T_min is the hard part, (4) one thing I did not understand._

### Four things the plan got wrong, and what they taught me

**1. The glideslope formula is inverted.** The plan writes
`|x| <= z * tan(gamma)` with gamma measured from horizontal, and says 80° means
"within 10° of vertical". Those disagree: at 80° that formula permits 5.7 m
downrange per metre of altitude — nearly horizontal flight. The correct
constraint is `|x| <= z / tan(gamma)`, which is what the Week 1 3-DoF problem
already used. With the sign fixed, the plan's default entry point (800 m
downrange at 1500 m altitude) sits *outside* its own 80° cone.

**2. The default entry state is unreachable.** Minimum throttle is 40% of three
Raptors, so TWR at minimum throttle is 2.16. Once lit, vertical acceleration is
at least +8.6 m/s² — the vehicle can only decelerate. Nulling `vz` over a fixed
20 s burn therefore demands `|vz0| >= 198 m/s`, and the altitude has to match the
resulting drop. The plan's `z0=1500, vz0=-80` fails both. Wrote
`min_arrestable_speed()` to derive the entry state from the burn rather than
guessing it.

**3. Lossless convexification is not automatically lossless.** This is the one
worth remembering. The relaxation only bounds `‖T‖ <= sigma`. When the entry is
gentle, the optimiser parks `sigma` on `T_min` and lets `‖T‖` drift *below* it —
producing a trajectory that burns minimum-throttle propellant while generating
less than minimum-throttle force. Not flyable, and the plan's own test suite
cannot see it, because checking `‖T‖ <= sigma` passes trivially. Measured gap,
as a fraction of T_min:

| entry margin | max(sigma − ‖T‖) | min ‖T‖ |
|---|---|---|
| 1.05 | 12.55 % | 0.874 T_min |
| 1.15 | 8.60 % | 0.914 T_min |
| 1.30 | 3.14 % | 0.969 T_min |
| 1.42 | **0.00 %** | **1.00 T_min** |

The relaxation is tight only when the minimum-thrust bound is *not* binding
across the whole arc. Added `test_relaxation_is_tight` so a regression is
visible.

**4. The problem needs non-dimensionalising.** In SI, thrust is ~3e6 N while the
velocity-update coefficient `dt/m` is ~3e-6 — twelve orders of magnitude in one
constraint matrix. Clarabel did not merely mis-solve it, it raised `SolverError`
on some instances, which reads exactly like physical infeasibility. Sweeping the
glideslope gave 80° infeasible but 84° and 86° fine; sweeping the pointing limit
gave 30° infeasible but 45° fine. Scattered nonsense. After scaling every
quantity by a characteristic value (L, V, M, F), all coefficients are order 1 and
the sweeps came out smooth and monotone.

### Key insights
- The glideslope costs **nothing** in fuel between 50° and 86° and then goes
  infeasible at 88°. It is a feasibility constraint, not a fuel constraint —
  no warning in the cost function, then a hard wall. Same shape for the pointing
  limit: free above 20°, infeasible at 5°.
- Fuel vs initial downrange is **not linear and not monotone**. It is a shallow U
  with a minimum near x0 = 400 m, about 170 kg cheaper than starting directly
  over the pad, because the fixed −40 m/s drift carries the vehicle to the target
  instead of past it.
- There is **no optimal burn duration**. Fuel rises nearly linearly with burn
  time, and the ceiling is arithmetic: minimum throttle flows 861 kg/s, so
  30,000 kg buys 34.9 s of burn. Ask for 34 s and it runs dry.
- The mass-reference iteration oscillated until damped. Cause: `sum(sigma)` is
  linear, so distinct bang-bang switching structures tie on total fuel while
  giving mass histories that differ by ~700 kg. Converge on the objective, not
  the profile. This is the same degeneracy Day 1 had, and the damping is a crude
  trust region — which is precisely what SCvx formalises.
- Infeasibility is a correct answer. The 5 km / 500 m case fails on **geometry
  alone**: that corridor allows |x| <= 88 m. Run the cheap geometric check before
  invoking the solver.

### Problems hit
- All four items above; each was diagnosed rather than worked around
- Clarabel `SolverError` misread as infeasibility until solvers and node counts
  were cross-checked against each other

### Tomorrow (Day 4)
- Free final time: let the optimizer choose when to start the burn
- Compare fixed-time vs free-time fuel cost
- First look at the full SCvx loop structure

### Time spent
_X hours_
