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

## Day 5 — 2026-08-12

### Done
- Built `src/dynamics_6dof.py`: planar 6-DoF with pitch, pitch rate and gimbal torque
- Built `tests/test_6dof.py`: 4 verification groups, exact to machine precision
- Built `src/landing_flip.py`: flip-and-land optimiser with a real SCvx loop
- Built `tests/test_flip.py`: 6 groups including a non-linear replay and a
  constraint-interaction probe
- Wrote `docs/flip-and-scvx.md`; exploration in `notebooks/05_flip_exploration.ipynb`

### Result
60° entry → upright landing, 15 s burn, **14,775 kg**, converged in 18 SCvx
iterations with a linearisation defect of 0.0003 of maximum thrust. Every limit
respected: pitch rate at its 28.6 °/s bound, gimbal at 15°, throttle inside
[T_min, T_max], glideslope clean. The flip tax against an upright entry is
+1,092 kg, about +8%.

### The modelling trap I nearly walked into
The obvious formulation makes `tau` an independent variable bounded by
`±sigma L sin(delta_max)`. That gives the optimiser **free torque with no effect
on thrust direction** — it will rotate the vehicle while thrusting somewhere
unrelated, solve happily, and report a plausible fuel number for a vehicle that
does not exist. The engine is bolted on; torque and thrust tilt come from the
same deflection. I kept the coupling as an equality and linearised it.

### Three things that made SCvx actually converge
All three were found by the loop failing, not by design.

1. **Every solved subproblem must advance the reference.** The textbook
   "reject the step and shrink the region" deadlocks: the region tightens around
   a point the solution is far from, so the next solve is strictly harder, and it
   walks down to the minimum region size and reports infeasible. The defect
   should size the *next* region, not veto the current step.
2. **The torque needs its own trust region.** The gimbal expansion point is the
   previous torque solution, which is bang-bang; unbounded it flips sign between
   iterations. Constraining `theta` alone left the defect oscillating around
   0.12 forever. Adding a torque region drove it to 0.0003.
3. **Expand about the previous gimbal angle, not zero.** Expanding about
   `delta = 0` drops `0.5 sin(theta) delta²`, worth 0.034 of max thrust at the
   15° limit — a floor no iterating can clear, and it showed up as a defect that
   stopped improving at exactly 0.036.

Seeding matters as much as the loop. `linspace(theta0, 0)` across the whole burn
implies a 4.7 °/s rotation; this vehicle flips at 28.6 °/s. That seed is
infeasible at every trust size where a fast-flip seed solves fine.

### The entry-pitch ceiling — the best physical result so far
The optimiser cannot fly a 90° belly-flop, and the reason is physical. The engine
is lit throughout, so while tilted it pushes the vehicle sideways at up to
21 m/s² whether it wants to or not. The pitch rate is capped, so the flip takes
at least `theta0/omega_max` seconds, and the excursion built in that window must
fit the glideslope corridor *and* be nulled by touchdown.

Measured at N = 80, 15 s burn, entry re-sized per attitude:

| configuration | highest feasible entry pitch |
|---|---|
| nominal, glideslope 75° | **60°** (65° infeasible) |
| glideslope loosened to 45° | 65° (+5°) |
| `omega_max` 28.6 → 51.6 °/s | **75°** (+15°) |

Relaxing **either** constraint alone moves the ceiling, which is what proves the
two bind together. The pitch rate is much the stronger lever.

**A real Starship flips before the landing burn, unpowered, on aerodynamic
surfaces.** That is exactly the freedom this model lacks. Best interview answer
the project has produced.

### Also learned
The rotation here is **rate-limited, not torque-limited** — pitch rate pins to
its bound almost immediately while peak torque stays well under maximum. A
stronger gimbal would not flip this vehicle faster; only a higher `omega_max`
would. And the attitude does not settle monotonically: it overshoots past
vertical and returns, because tilting the other way is the optimiser's only means
of cancelling the sideways velocity the flip itself created.

### Problems hit
- The guide's bang-bang flip test spun the vehicle through **−742°**, two full
  revolutions. Its timings assume far weaker control authority; `alpha_max` here
  is 94.7 °/s², so a 90° rest-to-rest rotation takes ~2 s, not 9. Derived the
  timings from the vehicle instead. A symmetric bang-bang also cannot null both
  attitude and rate — thrust differs between the two phases — which is itself
  the argument for solving rather than scheduling.
- The guide's default entry state (`vz0 = -80`, `t_burn = 25`) is infeasible for
  the same minimum-throttle reason as Day 3. A 25 s burn needs `|vz0| ≈ 285` m/s.
- My own `Set-Content -Encoding utf8` wrote a BOM into a source file and broke
  the parse. PowerShell 5.1 needs `UTF8Encoding($false)`.

### Bug found while wiring Day 5 into the viewer
`solve_flip_landing` returned the status of the **last** SCvx iteration rather
than the best one. A single infeasible subproblem at the end of a run — easy to
hit right after the trust region grows — discarded a perfectly good converged
answer and reported the whole problem infeasible. It now keeps the best iterate,
preferring low linearisation defect and breaking ties on fuel.

This invalidated my first ceiling measurement: several "infeasible" points in
that sweep were the bug rather than physics, and the published boundary
(40°/50°) was wrong. Re-measured after the fix it is 60°/65°, and the effect of
each relaxation is larger than I reported. The mechanism was right; the numbers
were not. Corrected in the README and docs.

### Known limitation
The flip optimiser still discretises with forward Euler. Replaying the commanded
control through the verified simulator lands **66.7 m from the pad, 4.0% of the
descent** — attitude is good to 0.37°, it is the translation that drifts. Day 4
already showed trapezoidal cutting this 7×; porting `src/discretization.py` into
`landing_flip.py` is the outstanding action.

### Tomorrow (Day 6)
- Trapezoidal collocation in the flip optimiser, judged by the replay error
- Free final time for the flip

---

## Day 6 — 2026-08-13

### Done
- `src/atmosphere.py`, `src/aero.py`: exponential atmosphere, attitude-dependent
  area and Cd, drag, lift, dynamic pressure
- `src/dynamics_aero.py`: 6-DoF plus air, and the unpowered entry phase
- `src/landing_aero.py`: two-phase pipeline with a searched ignition point
- `tests/test_aero.py`: 6 groups, all passing (8 suites green overall)
- `entry-aero` added to the viewer; `docs/aerodynamics.md`

### Built differently from the guide, for a measured reason
The guide models aerodynamics and a lit engine over the same 25–30 s window.
Built exactly as written that is **infeasible at every entry attitude (0–60°)
and every burn duration (5–15 s)**. I tried three fixes and ruled each out by
measurement: an aero-aware entry sizer, a homotopy ramping drag in from a known
feasible λ = 0, and both together.

The decisive test was that the aero-sized entry state is infeasible **with drag
switched off**. So it was never the drag forcing — a one-dimensional vertical
velocity budget ignores the altitude, attitude and corridor coupling that
actually binds. Sizing the entry on thrust alone and letting drag be a
perturbation solves cleanly at every attitude.

My aero-aware sizer also diverged to 5,584 m/s before that, because I wrote it
as a one-sided fixed point when drag grows as `v²` and the requirement grows as
`v` — there is a *window*, not a floor. Correct diagnosis, irrelevant fix.

### The finding that reshaped the day
Same problem, drag on and off:

| | no aero | with aero |
|---|---|---|
| 60° entry, 15 s burn | 14,783 kg | 14,785 kg |

**Identical**, despite peak aerodynamic deceleration of 86 m/s² — more than the
engines can produce. The throttle floor explains it: minimum throttle flows
861 kg/s and the engines run the whole descent, so propellant is set by *burn
duration*, not by how much work the engines do. Drag lets the optimiser throttle
down, and throttling down is exactly what it cannot do.

If aerodynamics buy nothing while the engines are lit, the guide's single-phase
model is the wrong shape.

### Where the belly-flop actually pays
Unpowered from 12 km to 300 m, engines off:

| configuration | arrival | coast |
|---|---|---|
| no atmosphere | 494.0 m/s | 38 s |
| nose-first | 357.5 m/s | 42 s |
| **belly-flop** | **64.0 m/s** | 131 s |

`Cd·A` is 28.3× larger broadside (540 m² vs 19 m²). The belly-flop removes
**430 m/s for free** — ~16,300 kg by the rocket equation, more than the entire
landing burn costs. The whole value is in the coast, engines off. That is why the
real vehicle flips immediately *before* ignition, not during it.

### Two-phase result
Coasting is free, burning is charged by the second, and burn duration is coupled
to handoff attitude because the flip is rate-limited:

| handoff | shortest burn | propellant |
|---|---|---|
| 0° | 4 s | **3,874 kg** |
| 30° | 6 s | 5,746 kg |
| 60° | 15 s | 14,820 kg |

The pipeline lands on **4,255 kg — 3.5× less than Day 5's 14,775 kg**, with the
ignition point found rather than assumed (same method as Day 2's suicide-burn
trigger).

### Also learned
Dynamic pressure does **not** decay as the vehicle slows — it converges. At
terminal velocity drag balances weight, so `q = mg/(Cd·A)` is pinned by the
vehicle, not the altitude: rising density exactly offsets falling speed.
Measured 2.42 kPa against 2.36 kPa predicted. My first test asserted monotonic
decay and failed; the model was right and the test was wrong.

### Honest limitation
The model still cannot flip 90° under power at terminal velocity: at 64 m/s the
throttle floor allows only ~5 s of burn, and a 90° flip needs longer. So the
pipeline hands over near-upright and phase 2 does not perform the flip. This is
the Day 5 entry-pitch ceiling again, and the resolution is the same — a real
Starship flips on its **flaps**, unpowered. Modelling the flaps is what closes
this gap; nothing in the current model substitutes for them.

### Tomorrow (Day 7)
- Aerodynamic control surfaces, so the flip can happen unpowered
- Trapezoidal collocation in the flip optimiser (still outstanding from Day 5)

### Time spent
_X hours_

---

## Day 7 — 2026-08-14

### Done
- `src/scvx_params.py`, `src/scvx.py`: SCvx with virtual control on all seven
  dynamics rows, an adaptive trust region, and a step quality measured against
  forward-propagated true dynamics
- `tests/test_scvx.py`: 7 groups, 30 checks, all passing
- `src/scvx_experiments.py`: four sweeps, kept because three of the four
  contradict what the guide predicts
- `scvx-landing` added to the viewer (eight problems now)
- `results/day7_scvx.png`, `day7_scvx_convergence.png`, `day7_scvx_sweeps.png`

### Built differently from the guide, for a measured reason
The guide's subproblem controls a free thrust vector `(Tx, Tz)` with
`||T|| <= sigma`, plus an independent torque. That is the model Day 5 rejected,
and transcribing it verbatim shows why: the thrust vector ends up a **mean of
43° and a maximum of 115° off the body axis, at every one of 80 nodes**, against
a 15° gimbal. It lands exactly on the pad at exactly zero speed by thrusting
sideways relative to where it points.

It also does not converge. With the guide's own trust-region logic, iteration 2
returns `unbounded` from both CLARABEL and SCS and the radius collapses to its
floor. Forced to keep iterating, virtual control grows monotonically from 54 to
2,759,333 while the solver reports `optimal` every time. In SI the L1 penalty is
adding metres to kilograms to radians, so no single weight can be right for all
seven rows; scaled, one `eta` and one `w_vc` serve every row.

### The finding that reshaped the day
The guide says the decay of `||nu||` *is* the convergence proof. It is not.
On the nominal problem `||nu||` is at machine zero from iteration 1 while the
**true** nonlinear defect is 0.34, falling to 0.005 only through iteration — the
linear model believes it is perfect ten orders of magnitude before it is. Slack
going to zero proves the model satisfied the dynamics it wrote down; only a
measured defect proves those were the right dynamics.

What virtual control is actually worth is diagnosis. On an infeasible problem it
settles to a constant that *measures* the shortfall: 3.8737e-2, identical to
five significant figures across four decades of penalty weight, and moving 1.11x
for a 10x tighter trust region. Day 5/6 returned `infeasible` and left you
guessing.

### Key insights
- vs the Day 5/6 loop on identical problems: 2–10% less propellant, and
  linearisation error tighter in 5 of 6 cases (2.0e-3 vs 6.6e-2 on the
  nominal). The ad-hoc loop stopped after 7 iterations on one case and **1** on
  another — false convergence, caught.
- `eta_0` has no sweet spot. Every radius from 0.02 to 4.0 converges in 18–23
  iterations to within 0.2% on fuel, because the adaptive rule re-tunes it.
- `w_vc` has a **floor**, not a sweet spot. Pinned at 1, the solver reports a
  635 kg landing — an 11x understatement bought by paying 5e-2 of slack
  cheaply. At 10 and above, everything is identical across four decades.
- Entry pitch decides feasibility once drag is on. Surviving slack: 1e-12 at
  0° and 10°, ~1e-8 at 30°, 0.28–1.63 at 60°, scaling linearly with how much
  drag is switched on.

### Problems hit
- Gated the aero continuation on the dynamics residual, which the aero case
  never reaches, so the ramp stalled at step zero and silently solved the
  aero-free problem. Re-gated on the thrust defect.
- Grew `w_vc` every iteration, hitting 1e7 in nine steps and wrecking the
  conditioning — both solvers then returned `unbounded` on a problem bounded
  below by −1. Now it grows only when slack stops shrinking.
- Priced the incumbent and the prediction at different `w_vc` when the weight
  moved between iterations, corrupting rho. Now re-priced each iteration.

### Time spent
_X hours_

---

## Day 8 — 2026-08-14

### Done
- `src/scvx_complete.py`: trapezoidal collocation, free final time, log-mass
- `tests/test_scvx_complete.py`: 8 groups, 39 checks, all passing
- `src/scvx_complete_experiments.py`: four sweeps
- `results/day8_complete.png`, `day8_convergence.png`, `day8_comparison.png`,
  `day8_sweeps.png`

### The guide's free final time is a no-op
It declares `t_f` a decision variable, bounds it, gives it a trust region and a
`0.1 * t_f` objective term — but computes `dt` from the *reference* `t_f`, so
the variable never enters a single dynamics constraint. Nothing resists the
penalty, so `t_f` is driven to `t_f_min` every iteration and the reference
follows it down. It is a constant dressed as an optimisation.

Making it real means confronting the term the guide avoided. Writing `kt` for
`t_f / t_nom`, every dynamics row carries `kt * f(x, u)` — a product of two
decision quantities. Linearised about the reference, `kt*f ~ kt_r*f + f_r*(kt -
kt_r)`, it is affine in both and exact at the reference, which is the standard
free-final-time treatment. `t_f` then searches 6.0 → 7.5 → 6.8 → 7.55 → 7.76 s
instead of sliding to a bound.

### Key insights
- **Trapezoidal collocation is the day's real result.** Replaying the commanded
  control through the verified nonlinear simulator, with the burn duration
  pinned so discretisation is the only difference: **0.502 m versus Euler's
  3.575 m** on a 473 m descent. Day 5's suite predicted a ~7x improvement and
  left it as the outstanding action; measured, 7.1x.
- Trapezoidal at **N=20** (0.499 m) beats Euler at **N=120** (2.308 m). The
  higher-order rule buys back roughly 6x the node count. Trapz error is then
  flat in N — it has hit the zero-order-hold control floor, not an integration
  floor.
- The fuel figure decomposes, and the decomposition matters more than the
  headline. At a pinned 8 s, trapz costs 7,246 kg against Euler's 7,209: Euler
  was **understating by 37 kg** because its dynamics were wrong. Free time then
  saves 132 kg against that corrected number, for a net 95 kg (1.3%) under
  Day 7. Reporting only "saved 95 kg" would hide an error and a saving that
  partly cancel.
- Log-mass makes the objective linear — minimising propellant is exactly
  maximising `zm[N]` — and `m_wet exp(zm)` matches the mass it represents to
  0.00 kg. It also un-freezes mass in the velocity rows, which Day 7 had held
  fixed from the reference within each iteration.
- The time penalty is unnecessary. With `w_time = 0` the optimum is 7.761 s and
  nothing is degenerate: the throttle floor already makes a longer burn cost
  propellant. Across `w_time` from 0 to 1 the answer moves 0.7%.
- Guess-independence, tested properly: sweeping `t_burn_guess` is meaningless
  here because the entry state is *sized from the guess*, so each guess is a
  different problem. With the entry state pinned, converged durations span
  **0.574 s across guesses from 5 to 12 s**.

### Problems hit
- Selected the cheapest honest iterate rather than the converged one, which
  bought 20 kg of linearisation error and called it a saving. The converged
  iterate is the answer.
- The comparison plot labelled a 1.3% saving as `+1.3%`, and plotted an 11%
  linearisation difference instead of the 4.5x replay result that matters.

### Honest limitation
Three of seven extreme cases and the longer-guess runs still leave residual
slack. Each is the aerodynamic deficit already measured on Day 7 at that entry
pitch and burn time — inherited, not introduced. Free time does not rescue an
infeasible problem; it reports the same shortfall at a slightly better duration.

### Tomorrow (Day 9)
- Monte Carlo: dispersed initial conditions, mass and aero uncertainty
- Where the solver breaks, and how much margin actually exists

### Time spent
_X hours_

---

## Day 9 — 2026-08-14

### Done
- `src/monte_carlo.py`: dispersion engine, splitting what the planner is told
  from the model error it is not
- `tests/test_monte_carlo.py`: 7 groups, 24 checks
- `src/monte_carlo_experiments.py`: four sweeps
- 250 dispersed runs; `results/day9_monte_carlo.png`, `day9_failures.png`,
  `day9_dispersion.png`, `day9_sweeps.png`, `day9_stats.json`

### Two things in the guide had to be fixed before anything could run
Its `run_single` cannot execute against this codebase at all: it assigns to
`vehicle.m_wet`, `vehicle.T_max` and `vehicle.T_min`, which are read-only
properties here, and raises `AttributeError` on the first one.

The larger problem is what it measures. Landing accuracy is read off `x_f, z_f`
in the solver's own solution — but those are hard equality constraints, so the
number is between 5e-10 and 1e-7 m on every dispersed run and a CEP built from
it is zero by construction. The guide's own expected output quotes a CEP of
2.05 m, which is only reachable if the terminal conditions are soft. Measuring
the optimiser against its own constraint reports that the constraint was
enforced, not that the vehicle landed.

So accuracy here comes from flying the plan open-loop through the independently
verified nonlinear simulator with the dispersions actually applied. That splits
the perturbations in two: the **entry state** is navigation error and the solver
is told it, while **mass, Isp, drag and wind** are model error it is never told.
Wind in particular has to enter the truth model rather than the initial
condition — the guide shifts `vx0` and `vz0` and then hands the shifted state to
the planner, which makes the wind something the planner knows about.

### The centre had to move too
`feasible_entry_state` sizes an entry the burn can *just* null, so what it
returns is a point on the feasibility boundary by construction. Sweeping one
axis at a time about its (473 m, −118.3 m/s, 30°) answer: converges at 473 m and
fails at 500, converges at −118.3 m/s and fails at −112, converges at 30° and
fails at 35. Every axis one-sided. A Monte Carlo centred there returned a 33%
success rate — a fact about where it was centred. Recentred at (420 m,
−130 m/s, 25°), which holds at ±80 m, ±20 m/s and ±8°, the solve rate is 98.4%.

### Key results, 250 runs
- Solver converged on **98.4%**. The optimiser is not the weak link.
- Flown miss: **CEP 3.74 m**, p95 10.33 m, max 13.56 m — against a solver-
  reported terminal error that never exceeded 4.82e-08 m. Four orders of
  magnitude between what it promised and what it delivered.
- Propellant margin is a non-issue: 23,678 kg mean, 21,988 kg worst, and not one
  run finished below dry mass. The vehicle has ~22 t of authority it never uses.
- Only **29.6%** landed within 5 m and 5 m/s. The dominant failure is arrival
  speed, not position or fuel: 34.4% arrived too fast, 21.6% missed the pad,
  12.8% both.

### The finding: minimum-fuel plans are knife-edges
Arrival speed came out bimodal — a spike near zero and a second cluster at
15–22 m/s — and the ground-crossing flag separates them almost exactly (125 of
246 crossed). One plan flown against a swept true propellant load shows why,
with nothing random involved:

| true propellant | outcome | speed |
|---|---|---|
| −1,500 kg | stops 4.39 m up | 2.08 m/s |
| nominal | stops 0.10 m up | 0.03 m/s |
| **+200 kg** | reaches the pad | **6.54 m/s** |
| +1,500 kg | reaches the pad | 19.35 m/s |

A 200 kg error, 0.67% of the load, takes touchdown from 0.03 to 6.54 m/s. Drag
error does the same with the opposite sign: `Cd` at 0.85 arrives at 25.96 m/s,
at 1.15 it stops 7.29 m short. A minimum-fuel trajectory is bang-bang and brings
the vehicle to rest exactly at the pad with no slack anywhere, so any error in
net deceleration puts it on one side or the other and nothing open-loop restores
it. Position stays good throughout — under 7.3 m across the whole sweep. It is
the velocity at contact that is uncontrolled.

That is the argument for Day 10 stated quantitatively: the propellant to fix
this is already aboard, and open-loop is why it goes unspent.

### The sweeps say which error matters
- **Navigation error contributes essentially nothing.** Scaling the entry
  dispersion from full 3σ down to *zero*, with the model errors held at full
  strength, moves CEP from 2.76 m to 2.98 m. The solver absorbs being told it is
  somewhere unexpected; what it cannot absorb is being wrong about the vehicle.
- **Wind is the position-error driver.** CEP runs 0.84 m with no wind, 3.10 m at
  10 m/s, 9.12 m at 30 m/s — very nearly linear. Mass and drag barely move the
  miss; as the knife-edge sweep shows, they go into arrival *speed* instead.
  Position error and speed error have different causes.
- **The altitude band is narrow and one-sided**, entry speed held at −130 m/s:
  solve rate is 100% from 360–420 m, 91.7% at 480, 83.3% at 540 and 33.3% at
  600, with good landings falling 41.7% → 0%. That is the vertical slice through
  the feasibility wedge — a slow, high approach over-brakes before it arrives,
  because the throttle floor will not let the engines ease off.
- **Node count is nearly irrelevant to the statistics.** CEP sits between 3.07
  and 3.73 m across N = 30 to 80, at 2.8–3.5 s per run. Trapezoidal collocation
  is why: Day 8 measured its replay error as flat in N past about 20 nodes.

### Problems hit
- Scored landings on a ground-crossing test, which nominally never fires: the
  plan ends at `z = 0` and the RK4 replay of it bottoms out at `z = +0.097 m`.
  Every run was recorded as failing to land. Day 8's 0.502 m replay figure was
  almost all downrange — 0.486 m of it — so the miss is the distance from the
  pad at the end of the flown plan, with the ground crossing handled when it
  does occur.

### Tomorrow (Day 10)
- Closed-loop guidance: replan on a receding horizon and re-run this same sweep
- The number to beat is 29.6% landed and an 8.49 m/s mean arrival

### Time spent
_X hours_

---

## Day 10 — 2026-08-15

### Done
- `src/warm_start.py`: `shift_reference` turns the previous solution into the
  next solve's reference
- `src/scvx_complete.py`: two hooks a guidance loop needs — `initial_ref` to
  inject a warm reference, and `m0` so a replan can start mid-burn rather than
  from a full tank
- `src/closed_loop.py`: MPC loop, OU gust field, open-loop baseline on
  identical gusts
- `tests/test_closed_loop.py`: 7 groups
- `src/closed_loop_experiments.py`: four sweeps

### Warm starting does not do what the guide says
The guide measures the speedup by capping warm solves at four iterations and
comparing against an uncapped cold solve, so a speedup of at least the cap is
guaranteed whether or not warm starting does anything. Run both to the same
tolerance and the effect is absent: at three replan points warm took 16, 28 and
26 iterations against cold's 20, 21 and 23. Tightening the trust region to
"exploit" the good reference made it worse, because the solver then cannot move
far enough per iteration to absorb the tracking error.

The reason is structural. This solver's iteration count is set by the
trust-region schedule annealing down from `eta_0` and by the convergence test,
not by how far the reference sits from the answer. A better starting point does
not shorten a schedule that does not know about it.

What warm starting *does* buy is the thing a guidance loop needs: a usable
command inside a fixed budget. From a 3 m tracking gap given one iteration, the
warm solve commands a gimbal 5.9 degrees from the converged answer and the cold
solve 24.1 degrees — saturated the wrong way. Given three, warm is 0.30 degrees
off and cold is still 5.9. Commanded thrust is identical either way; it is the
steering that is wrong when cold. So the loop runs a 3-iteration budget per
cycle, which costs about 0.25 s against a 0.5 s cycle.

### The result, and it is not the one the guide expects
Twelve wind seeds, closed and open loop flying identical gusts:

| | landed | miss (median) | arrival (median) |
|---|---|---|---|
| open loop | 33% | 3.45 m | 5.76 m/s |
| closed loop | **8%** | **0.60 m** | **15.31 m/s** |

The closed loop lands nearer in 11 of 12 seeds and arrives slower in 1 of 12.
It fixes position — decisively, 5.7x on the median, and the advantage widens
with wind, 5.24 m against 1.02 m at the strongest gusts — and it makes arrival
speed nearly three times worse. By Day 9's own scoring it is a regression:
33% good landings down to 8%.

That is worth stating plainly rather than dressing up. **Day 9 established that
position was never the failure and arrival speed was. The loop as built
improves the error that did not matter and worsens the one that did.**

### Why, and it is a rate problem rather than a concept problem
The guidance-rate sweep is the diagnosis. Shrinking the cycle improves both
numbers monotonically:

| cycle | miss (median) | arrival (median) | replan cost |
|---|---|---|---|
| 1.000 s | 1.13 m | 21.63 m/s | 0.211 s |
| 0.500 s | 0.62 m | 15.31 m/s | 0.242 s |
| 0.250 s | 0.44 m | 12.21 m/s | 0.208 s |
| 0.125 s | 0.06 m | **7.85 m/s** | 0.220 s — does not fit |

So the loop is converging toward the right answer as it is sampled faster; it
is simply under-sampled where it matters. The descent lasts about 5 s and
essentially all the braking is in the last second, so a 0.5 s cycle leaves the
final command up to half a second stale exactly when precision is needed.
Position is a slow state and gets corrected; velocity is fast, and on a
bang-bang trajectory with no slack it does not. At 0.125 s the arrival is
7.85 m/s and still falling — but the replan costs 0.22 s, so that rate is not
real-time on this solver.

### Problems hit
- Held the plan's first control constant across each 0.5 s cycle. The plan's
  own control interval is 0.13 s, so this mis-steered the attitude badly:
  within two cycles the replan went infeasible while position still matched the
  plan to 0.3 m. The gap metric hid it by measuring position only. Fixed by
  following the current plan's control schedule between replans, which is the
  ordinary guidance/control split.
- Shifted a freshly computed plan forward by a full cycle before its first use,
  producing a 62 m phantom tracking gap on the first replan. The plan's clock
  starts at the state it was solved from; its age is zero until time is flown.
- Guessed that the loop was deferring its braking, since each replan
  re-optimises minimum fuel and the cheapest plan always brakes as late as
  possible. Measured it: `t + t_f` holds at 5.13–5.27 s across every cycle, so
  it is not postponing, and capping the duration changed nothing. The
  hypothesis was wrong and the rate sweep found the real cause.

### Honest limitation
Navigation noise breaks this loop. Feeding raw noisy estimates into a
re-optimisation that is bang-bang by construction amplifies them: 1 m of
position noise is harmless, 3 m produces a 109 m worst-case miss, and 8 m gives
an 84 m median miss and 10,056 kg of propellant against a nominal 6,009 kg.
There is no filter between the estimate and the solver, and there needs to be
one before this is flyable.

### Next
- A terminal-phase controller, or a much higher rate over the last second,
  since that is where the sweep says the remaining error lives
- A state estimator, so navigation noise stops being fed straight into a
  bang-bang re-plan
- Re-run Day 9's 250-sample dispersion sweep closed-loop once those land; the
  number to beat is still 29.6% good landings

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
