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

## Day 11 — 2026-08-17

### Done
- `src/sensors.py`: nav and attitude instruments, each on its own clock, with
  an optional gyro bias
- `src/ekf.py`: EKF over the six translational and rotational states, mean by
  RK4 through the true coupled dynamics, covariance by numerical Jacobian,
  Joseph-form update
- `src/navigation_loop.py`: Day 10's guidance rewired onto the estimate, with
  `truth` and `naive` baselines on identical wind *and* identical sensor noise
- `tests/test_ekf.py`: 7 groups
- `src/navigation_experiments.py`: four sweeps
- `navigation` added to the viewer (twelve problems)
- `results/day11_navigation.png`, `day11_sweeps.png`

This closes the limitation Day 10 recorded: guidance no longer reads the truth.

### The result
Same wind and sensor noise, three ways:

| | miss | arrival | propellant | est. error |
|---|---|---|---|---|
| truth (Day 10's privilege) | 0.28 m | 15.41 m/s | 6,003 kg | — |
| **EKF** | **1.71 m** | 22.52 m/s | 5,751 kg | 2.01 m |
| naive | 92.51 m | 28.78 m/s | 11,120 kg | 4.94 m |

Across four realisations the filter estimates three to four times better every
time.

> **Corrected on Day 12.** This entry originally continued: "and the worst-case
> miss falls from 210 m unfiltered to 6.7 m filtered, with the 20 t propellant
> blowout gone." That was wrong. The 210 m and the 20 t were a defect in the
> guidance loop, not in the estimator — once a plan's horizon was spent the
> loop kept flying its last control, which for a landing plan is a lit engine,
> so the vehicle climbed away from the pad still thrusting. Day 12 found it and
> fixed it, and the unfiltered worst case fell to 4.4 m against the filter's
> 6.7 m. Filtering does not bound the tail here either; the tail was mine. What
> survives is the estimate, and it makes the day's actual conclusion stronger
> rather than weaker.

### The finding, which arrived as a failing test
The first `Q` was wrong by two orders of magnitude, and how it surfaced is the
interesting part. Test 7 failed asserting the filter lands nearer: the EKF
estimated 3-4x better in **4 of 4** seeds while landing worse in **3 of 4**.

The process-noise sweep explains it. At the tuned value the miss is 3.13 m; a
hundred times tighter it is 34.45 m -- while mean estimation error goes 1.60 m
to 2.22 m, which is nothing. An under-confident `Q` makes the filter trust its
own dynamics through gusts it cannot see, and the error that produces is a
**lag** rather than noise. Successive replans average noise out and cannot
average out a bias.

So **mean estimation error is the wrong figure of merit for a filter inside a
control loop.** A lagging estimate and a noisy one are indistinguishable by it
and behave nothing alike. The sweep shows the other side too: at 10x and 100x
the tuned value the estimate degrades again (2.37 m, 3.23 m) as the filter
starts chasing sensor noise, so there is a genuine optimum rather than a
monotone.

### The other sweeps
- **Nav rate is a hardware specification, and a sharp one.** 1 Hz gives a 175 m
  miss and 12,785 kg; 2 Hz gives 1.98 m and 5,913 kg. The loop needs position
  aiding at 2 Hz or better and gains almost nothing above it, even though
  estimation error keeps improving out to 20 Hz (0.78 m). Estimate quality and
  control quality are again different things.
- **Position aiding is not optional.** Switching the nav sensor off leaves the
  filter dead-reckoning on attitude alone: estimation error 1.60 m to 7.42 m,
  and nothing bounds the drift.
- **An unestimated bias is invisible where you look for it.** Miss runs 3.13,
  3.43, 10.22 and 13.67 m at gyro biases of 0, 0.5, 1 and 2 deg/s, while
  position estimation error stays flat at 1.38-1.60 m. The filter reports
  health while the control degrades -- the textbook argument for a bias state,
  and it only became visible after the `Q` fix.

### Problems hit
- `EKF(aero=None)` silently substitutes a default `AeroConfig()`, so the
  "known-linear" Jacobian test still had drag in it and failed by 0.22. With
  `AeroConfig(enabled=False)` the Jacobian matches the exact discrete
  transition matrix to 2.4e-11.
- Fed the sensor model end-of-interval truth while the filter sat at the start
  of the interval, injecting a systematic lead that made the filter look 17x
  worse than it is -- 16.40 m against the 0.96 m it actually achieves.
- Test 7 asserted that filtering lands nearer. It does not, in the median. The
  assertion was corrected to what the data supports -- the estimate is better
  every time, and the tail is bounded -- rather than relaxed until it passed.

### Honest limitation
The filter has no bias state, so the gyro-bias degradation above is real and
unaddressed. Guidance also still arrives too fast in every mode, because that
is Day 10's rate problem and no estimator can fix it.

### Next (Day 12)
- Augment the state with a gyro bias term, which the sweep now justifies
- The terminal-phase controller Day 10 asked for, since arrival speed remains
  the binding failure in all three navigation modes

### Time spent
_X hours_

---

## Day 17 — 2026-08-18

### Done
- `src/scvx_3d_validate.py`: a second, independent 3-D SCvx formulation —
  state-transition-matrix discretisation, finite-difference Jacobians, soft
  trust penalty, geometric trust schedule — sharing Days 13–15's physics and
  nothing else with Day 16
- `tests/test_3d_validation.py`: 6 groups, all live re-solves, all passing
- `scvx-3d-validate` added to the viewer (eighteenth problem)

### The physics is validated. The solvers are not.
That is the whole result, and the two halves are separable because today's
solver shares the physics with Day 16 and shares no algorithm with it.

**Planar reduction passes at bit-for-bit zero.** Given a planar initial
condition the solver produces out-of-plane position, out-of-plane velocity,
roll rate, yaw rate and side thrust of **exactly 0.00e+00** — not 1e-12, not
"negligible". The in-plane and out-of-plane subspaces do not mix, in a
formulation whose Jacobians come from central differences rather than Day 16's
hand-derived analytics. Two independent derivations of the same physics agree
that there is no leakage, which is about as strong a statement as this kind of
check can make. **Days 13–15 are not the problem.**

**And the 3-D case genuinely uses the third dimension** — 192 m of cross-range,
93 m/s of out-of-plane velocity, 19.9 deg/s of roll and yaw rate. That matters
because a solver secretly still planar would pass the reduction test trivially.

**Neither formulation converges.** Virtual control bottoms out at 18.3 against
a 1e-1 target, and the plan misses by 334 m at 19.6 m/s when flown. Day 16's
solver fails the same way with completely different algorithmic machinery,
which locates the failure in what they share that is *not* physics: the problem
statement.

### The bug the guide misses, and it is the one that matters
The guide documents three bugs and picks `tf = 18.0 s`. That number is the
problem.

This vehicle cannot throttle below 40 per cent. A lit engine therefore produces
at least

    T_min / m_wet = 2.76e6 / 130,000 = 21.23 m/s^2

against gravity's 9.81 — a net **upward** floor of **+11.42 m/s²**. The engine
cannot push the vehicle down. So from a descent rate of 80 m/s there is exactly
one burn duration that arrives at rest:

    t = 80 / 11.42 = 7.00 s

and the guide's 18 s is **2.6× that ceiling**. A burn that long does not land
softly; it turns the descent into a climb.

The flown results track the arithmetic:

| t_f | flown arrival speed | flown miss |
|---|---|---|
| 6 s | 20.1 m/s | 339 m |
| **7 s** | **19.6 m/s** | **334 m** |
| 8 s | 28.3 m/s | 345 m |
| 10 s | 54.2 m/s | 410 m |
| 14 s | 122.2 m/s | 729 m |
| 18 s | 201.8 m/s | 1,301 m |

Minimised exactly where the closed form says, then monotonic. Shortening the
horizon to the ceiling takes the arrival speed from 201.8 m/s to 19.6 and the
miss from 1,301 m to 334.

### Where the slack sits, which is how it was found
Decomposing the virtual control by state block on the first iterations:

    iter 1:  pos 11.4%   vel 88.4%   quat 0.1%   omega 0.1%
    iter 2:  pos  3.6%   vel 90.4%   quat 5.5%   omega 0.5%

Nearly all of it is in the **velocity** rows. That is not an attitude problem
or a quaternion problem — it is the translational dynamics being unsatisfiable,
which is exactly what a throttle floor fighting a fixed horizon looks like.
Day 16 guessed at an over-constrained sub-problem; this locates it.

### The guide's reported numbers do not survive checking
Its Part 4 claims 6/6 passing with the 3-D case landing at 2.24e-04 m using
**0.00° of gimbal**, steering "via body attitude". In its own solver torque is
`M_B = r_TB × u`, which is identically zero when `u` lies along the body axis.
Zero gimbal means zero torque means the attitude cannot change at all, so that
trajectory cannot have steered anywhere.

Two things make those numbers reachable without anything working. Its
`solve_scvx` returns its reference array whatever happens, so a run whose first
sub-problem is infeasible hands back the straight-line initial guess — which
lands at the origin, upright, at rest, with thrust along the body axis and
therefore zero gimbal, because that is how the guess was constructed. Every one
of those is something its tests accept. And its `initialize_reference` sets the
quaternion to `[1,0,0,0]` at every node, so "lands upright, q_err = 8.89e-08"
is a property of the guess rather than of a solution.

Today's loop reports `ever_solved` and `is_initial_guess`, and Test 2 asserts a
sub-problem actually solved, precisely so this cannot happen here.

Its identical fuel figures across two different problems — 27,098.2 kg and
27,099.4 kg, agreeing to five significant figures on initial conditions
differing by 180 m of cross-range and 20 m/s of out-of-plane velocity — point
the same way.

### What is still broken
Even at the right horizon the plan misses by 334 m and the virtual control sits
at 18.3. So the throttle-floor conflict was a large contributor and not the
only one. The residual miss is roughly flat across `t_f` (339, 334, 345, 410),
which suggests something that does not scale with horizon length — the
linearisation or the discretisation. That is the next thing to isolate.

Both loops also end the same way: the trust radius shrinks geometrically until
the sub-problem goes infeasible, at iteration 8 here. The guide records this as
its third bug and defers the fix to Day 7's accept/reject step-quality
controller. Day 16 already has that controller and still does not converge, so
porting it is unlikely to be the answer.

### Honest note on the guide's premise
It proposes rebuilding the vehicle, quaternion algebra and dynamics standalone,
on the grounds that earlier days "live only in markdown, not as a live module".
That is not true of this repository — every day since Day 1 is an importable
module with its own suite. Re-typing the physics would validate a fresh copy of
it and nothing else, which is the opposite of what a validation day is for. The
algorithmic differences are kept; the physics is imported.

### Tomorrow (Day 18)
The guide points at reproducing Szmuk & Açıkmeşe. That should wait. Reproducing
a paper's numbers with a solver that does not converge would produce agreement
or disagreement that means nothing either way. The residual 334 m first.

### Time spent
_X hours_

---

## Day 16 — 2026-08-18

### Done
- `src/scvx_3d.py`: the 14-state 3-D SCvx sub-problem — exact-convex thrust,
  gimbal, torque and glideslope; linearised quaternion kinematics, gyroscopic
  coupling and rotated thrust; aero as a reference-iteration perturbation;
  trust regions and virtual control over the full state; plus a replay harness
  that flies the answer through Day 15's true model
- `tests/test_scvx_3d.py`: 8 groups, all passing
- `scvx-3d` added to the viewer (seventeen problems) — it draws the plan and
  the trajectory that plan flies under a switch, defaulting to the flown one,
  because a plan that pays slack is not a trajectory and the site should not
  imply otherwise

**The solver does not converge.** That is the headline and it is recorded here
rather than buried. Everything that can be verified about the convex
sub-problem checks out; the outer loop does not drive its own dynamics defect
to tolerance, so what comes out is a plan, not a trajectory.

### What is verified
Every linearisation, against finite differences:

| piece | check | worst error |
|---|---|---|
| Hamilton L(q), R(p) | vs `quat_multiply`, 400 pairs | **1e-12** |
| dR/dq, all four | central differences of the raw form | **2.82e-10** |
| gyroscopic Jacobian | central differences, relative | **1.65e-09** |
| quaternion kinematics | exact at the reference | **5.55e-17** |
| `force_to_gimbal` | round-trips Day 14's trig, 500 commands | **2.22e-16** |

The quaternion-kinematics expansion also halves-to-a-quarter correctly — the
error divides by 4.00 each time the step is halved, which is what a product
rule applied to a bilinear term must do, and a stronger statement than a
single-point check.

The exact-convex set holds in the returned solution: gimbal peaks at 13.2 deg
of 15, glideslope at 60.3 deg of 80, thrust inside [T_min, T_max] throughout.
Boundary conditions are met to **2.5e-09 m** and **1.9e-09 m/s**, upright to
0.00 deg, with 8,970 kg of the 30,000 kg propellant load.

### What is not
Virtual control stalls at **4.16e-01** against a tolerance of 1e-6. Replaying
the plan through the true Day 15 model misses by **247 m at 14.4 m/s**.

Two obvious causes, both ruled out by measurement rather than argument:

- **Not the Euler step.** The miss does not fall with node count: 141 m at
  N=15, 247 at 25, 418 at 40, 240 at 60, 417 at 90. No trend, and the
  qualitatively different outcomes (some runs land at 14 m/s and 18 deg, others
  at 120 m/s and 170 deg) say the solver is finding different local answers
  rather than refining one.
- **Not an under-sized trust region.** The defect *falls* as the radius grows —
  0.557 at eta = 0.2, 0.416 at 0.5, 0.310 at 1.0. A linearisation-validity
  problem has the opposite signature.

What it looks like is an **over-constrained sub-problem**. Hard terminal
equalities on all four state blocks, a 40 per cent throttle floor that puts
minimum deceleration at 21 m/s^2 against gravity's 9.8, and a fixed horizon.
The solver pays slack because it cannot meet them simultaneously. Lengthening
the horizon eases the defect without improving the replay — 0.42 at 8 s, 0.26
at 11 s, 0.18 at 14 s, miss stuck near 250 m — which fits that reading and
fits nothing else I tried.

Next thing to try, in order: free final time (Day 8's extension, which this day
deliberately dropped), then terminal conditions as penalties rather than hard
equalities.

### Two things the sub-problem needed that the guide does not have
- **Variable scaling.** Without it the problem spans seven orders of magnitude
  — position in thousands of metres, quaternion components of order one, mass
  1e5, force 1e6 — and CLARABEL returns `optimal_inaccurate` answers whose
  quaternion norm has wandered to **2.79**. Scaling every block to order one
  took the drift to 2.8e-03 and the solve from 277 s to 55 s. Day 7 needed the
  same thing for the 2-D solver.
- **A linearised unit-norm constraint.** `||q|| = 1` is non-convex, but about a
  unit reference it is the affine tangent plane `q_ref . q = 1`. Nothing in the
  guide's sub-problem stops the quaternion leaving the sphere, and it does —
  which is worse than it sounds, because every linearisation in the file is
  built on a unit q, so off the sphere the dynamics become free and the solver
  helps itself. One line took the drift from 2.8e-03 to 4.5e-04.

### Three defects in the guide
- **The terminal quaternion is the belly-flop, not upright.** The guide sets
  `q[N] == [1, 0, 0, 0]` and calls it "upright at landing". The identity
  quaternion means the body frame equals the inertial frame, so with the long
  axis on body +x the thrust points along inertial +x — horizontal. A vehicle
  in that attitude cannot hover. Both boundary attitudes here go through Day
  14's `attitude_from_pitch` instead, which is the one place that convention
  lives.
- **A silent rate limit.** The guide writes
  `cp.norm(omega, axis=1) <= vehicle.omega_max if hasattr(...) else True`.
  Python binds that as `<= (omega_max if hasattr else True)`, and `Vehicle3D`
  has no `omega_max`, so the constraint quietly becomes `<= 1` rad/s. Stated
  explicitly as a named parameter.
- **The dR/dq test as described would fail.** The guide's Test 1 checks the
  Jacobians against "finite-difference derivatives of the actual rotation
  matrix". Day 13's `quat_to_rotmatrix` normalises its argument, so it is not
  the function being linearised — the two agree to 1e-15 in value and their
  derivatives do not agree at all, because normalising projects out the radial
  direction. Differencing it misses by **1.26**. The matrices themselves are
  correct; the test is the trap. `rotmatrix_unnormalized` exists to make the
  distinction explicit, and Test 2 asserts both that the right check passes and
  that the wrong one fails.

Also: the guide's solver map has only ECOS and SCS, and `SCvxParams.solver` is
CLARABEL, so every solve would silently fall through to SCS.

### A bug in my own loop
The trust radius shrinks on a rejected step, and I had no `eta_min` exit on the
accepted path — so once it collapsed, the trust region pinned the iterate to
its own reference for every remaining iteration and printed thirty rows of an
unchanging number. That reads like convergence and is the opposite of it. Fixed
with an explicit break and a message that says what happened.

### Honest scope note
The guide's own reduction — fixed final time, Euler discretisation, raw mass —
is stated as deliberate, and it is defensible on a day this large. But the
evidence above points at the fixed horizon as part of what is blocking
convergence, so that reduction is not free, and Day 17 should probably take
free final time back before anything else.

### Tomorrow (Day 17)
Free final time in 3-D, or soft terminal constraints — whichever relieves the
defect. Not the benchmark comparison until the solver converges.

### Time spent
_X hours_

---

## Day 15 — 2026-08-18

### Done
- `src/aero_3d.py`: angle of attack and sideslip, area and drag-coefficient
  blending by the angle to the *relative wind*, a drag/lift/side-force
  decomposition, aerodynamic moments through the same `r x F` pattern as
  Day 14's engine torque, and a combined dynamics layer
- `tests/test_aero_3d.py`: 11 groups, all passing
- `src/demo_aero_3d.py`: three figures and five experiments
- `results/day15_coefficients.png`, `day15_crosswind.png`, `day15_day6_gap.png`
- `src/dynamics_3d.py`: an `extra_body_wrench` hook, so today is a layer on
  Day 14 rather than a second copy of it
- `aero-3d` added to the viewer (sixteen problems)

### Composition, not duplication
The guide's Day 15 rebuilds the whole 14-state derivative — gravity, the
body-to-inertial rotation, Euler's equations, the mass flow — inside a new
`dynamics_3d_with_aero_derivative`. That is a second copy of the physics that
Day 14's test suite guards, free to drift away from it silently.

Instead `dynamics_3d_derivative` gained one optional argument: a callable
returning an extra body-frame force and torque. Day 15 supplies the aerodynamic
wrench and nothing else. The test asserts the combined derivative equals Day
14's plus exactly that wrench — measured at **6.94e-18** — and that switching
aero off reproduces Day 14 bit for bit. Given what the Day 14 reduction check
turned up in `dynamics_6dof` and `landing_flip`, two copies of the same physics
is the last thing this project needs.

### The guide's lift direction is backwards, and it is decidable
The guide specifies the lift direction as `[-w, 0, u]` in body coordinates.
That is perpendicular to the wind, which is the part that is derivable, but the
sign is wrong and it is not a matter of convention.

At positive angle of attack the vehicle is moving in `+z` relative to the air,
so the air pushes it in `-z`. The drag term already carries part of that. The
perpendicular component has to keep pushing the same way, not against it. With
the guide's sign, lift at small angle of attack **overwhelms the drag's own
normal component** and the net force turns the vehicle *away* from the wind —
which makes a centre of pressure aft of the centre of mass destabilising. An
arrow with its fletching at the back would fly backwards.

Corrected to `[w, 0, -u]`. The end-to-end check is Test 11: with `x_cp` aft the
moment is restoring at all six angles of attack tested, and with `x_cp` forward
it diverges at all six. Before the fix, both were diverging.

**Day 6 already had it right.** With the correction, the full 3-D force —
drag and lift together — reproduces Day 6's planar force to **2.13e-15**
relative for vertical descent at every pitch angle from 0 to 90 degrees. Before
the fix the magnitudes matched exactly and the directions were **49.58 degrees**
apart, which is what a flipped lift looks like when drag and lift are
orthogonal: the magnitude cannot see it.

### A second defect in Day 6, and this one is not fixed
Day 6 blends reference area and drag coefficient by `theta`, the pitch from
*vertical*. The quantity that decides how much vehicle the air sees is the
angle between the body axis and the *relative wind*. Those coincide only when
the wind is vertical.

Day 6 is inconsistent about it internally, too: `angle_of_attack()` computes
the wind-relative angle correctly and uses it for lift, and then
`effective_area()` and `effective_Cd()` use the vertical-relative angle for
area and drag.

The blend *formula* is identical — fed the same angle the two agree to
**0.00e+00**. The entire disagreement is which angle goes in.

| pitch | velocity off vertical | Day 6 CdA | true CdA | ratio |
|---|---|---|---|---|
| 70° | 0° | 488.9 | 488.9 | 1.000 |
| 70° | 20° | 488.9 | 540.0 | 1.105 |
| 70° | 45° | 488.9 | 461.7 | 0.944 |
| 30° | 24° | 192.6 | 386.8 | **2.008** |
| 90° | 35° | 540.0 | 394.3 | **0.730** |

**The error is not one-directional**, which is the awkward part. Day 6 gives
the vehicle too much drag in some of the envelope and half what it should in
other parts, so it cannot be waved through as a conservative margin. It is a
bias whose sign depends on where the vehicle is — exactly the kind of thing an
optimiser finds and exploits. Over a full unpowered descent from 6 km the ratio
averages 1.04x and reaches 4.28x at worst.

Not fixed. `aero.py` is load-bearing for Days 6 to 12 — the aero entry, the
two-phase descent, the SCvx aero ramp, Monte Carlo, the closed loop. Same
decision as yesterday's sign error, and the same reason.

### The crosswind result is not the one I expected
Set a crosswind with a lateral component, hold the yaw gimbal at exactly zero,
and the vehicle ends **393 m out of plane**. The obvious reading is that the
aerodynamic side force pushed it there. Splitting the out-of-plane impulse says
otherwise:

- aerodynamic: **+4.56 MN·s**, downwind, as expected
- thrust: **−13.89 MN·s**, upwind, three times larger

The first two seconds do drift downwind under the side force alone. By then the
aerodynamic yaw moment has swung the body about **20 degrees** out of plane,
and 4.8 MN of thrust pointed 20 degrees wrong overwhelms every aerodynamic
force in the model. The vehicle finishes hundreds of metres **upwind**, carried
there by its own engine.

So the thing 3-D aero adds is not a correction to the planar answer. It is that
the wind can *turn* the vehicle, and the engine then does the damage. That is a
control problem rather than an aerodynamic one, and the planar model cannot
express it at all.

### Other results
- Purely axial flow gives no side force, no lift and no moment, to
  **3.27e-16** relative — the same geometry as Day 14's zero-gimbal result, a
  force collinear with the offset arm has no moment about it.
- Broadside flow reproduces Day 6's belly values exactly.
- Doubling the relative speed multiplies the force by **4.000000000000** and
  does not rotate it by a microradian; the coefficients depend on direction
  alone.
- Force scales with density as it should — the sea-level to 8500 m ratio is
  **2.7183**, which is `e`, one scale height.
- Best lift-to-drag is **0.46 at 31.8 degrees**, not at the 45 degrees where
  the lift coefficient peaks, because the drag coefficient is still climbing
  there. A falling cylinder is not a wing.
- Aero cannot roll this vehicle either — `r x F` with `r` along the long axis,
  worst roll moment 0.00e+00 over 500 random forces. Neither can the gimbal.
  The roll axis has no authority from any source in the model.

### Problems hit
- Four of my own test assertions were wrong on the first run. Two were absolute
  tolerances of 1e-12 applied to residues of `sin(2*pi)` on forces of order
  1e5 N; rewritten as relative tolerances, and they now read 3.27e-16. One had
  the cross-product signs transposed. The fourth asserted weathervaning by
  propagating for 25 s and reading the final angle, which conflates the
  restoring moment with an undamped oscillation — replaced by checking the sign
  of the moment at a perturbed angle of attack, which needs no integration and
  is decisive.
- The guide's claim that "Day 6 had no lift term at all" is simply wrong.
  `aero.py` has `Cl_max = 0.4` and the same `Cl_max sin(2 alpha)` curve. What
  Day 6 genuinely lacks is sideslip, a side force, and any moment whatsoever.

### Honest limitation
There is a restoring moment but **no aerodynamic damping** — no moment
proportional to body rate. A disturbed vehicle therefore oscillates about the
wind direction forever instead of settling. Day 16's solver should not read
that oscillation as physical. Adding a rate-damping term is a small change and
probably belongs before any controller is tuned against this model.

The centre of pressure is also a single fixed point, which real aerodynamics
does not have — it moves with angle of attack and Mach number. And `x_cp`
being aft, making the vehicle passively stable, is a *choice*: a real Starship
in the belly-flop is not passively stable, which is why it carries flaps.

### Tomorrow (Day 16)
3-D SCvx: linearising the quaternion kinematics, 3-D glideslope as a
second-order cone, thrust pointing in 3-D, trust regions over the full 14-state
vector — Days 13 to 15 combined into one convex subproblem.

### Time spent
_X hours_

---

## Day 14 — 2026-08-18

### Done
- `src/dynamics_3d.py`: `Vehicle3D` with a real inertia tensor, a two-axis
  gimbal in exact trig, Euler's rotational equations with the gyroscopic
  coupling term, RK4 with renormalisation, and the bridge functions that
  express Day 5's single pitch angle as a quaternion
- `tests/test_dynamics_3d.py`: 8 groups, all passing
- `src/demo_3d_dynamics.py`: two figures and five experiments
- `results/day14_poinsot_tumble.png`, `day14_gyroscopic.png`
- `rigid-body-3d` added to the viewer (fifteen problems) — the gyroscopic
  term as a switch, with the divergence against the other setting reported
  on every solve

Thrust and gravity only. Aero is Day 15, laid on top of a validated rigid body
rather than built into it, which is the same order Day 6 used.

### The headline: Euler's equations change nothing behind them, and everything ahead

The gyroscopic term `omega x (I omega)` is the whole point of today, and the
honest measurement of it is a bifurcation at zero rather than a smooth
correction.

At **exactly zero roll rate the term contributes exactly nothing** — flying the
same burn with it included and with it dropped gives 0.0000 deg of attitude
difference and 0.0000 m of position difference. That is not a small number, it
is an identity: with omega on a single principal axis, omega and `I omega` are
parallel, the cross product vanishes, and a pitch-axis torque never moves omega
off that axis. This project's flip is roll-free, so **Euler's equations reduce
to Day 5's scalar tau = I alpha exactly**, and none of Days 1–12 is affected by
today's physics.

Just above zero it stops being negligible immediately:

| roll rate [rad/s] | \|w x Iw\| / tau_max | d attitude | d position |
|---|---|---|---|
| 0.00 | 0.00e+00 | 0.0000° | 0.000 m |
| 0.02 | 1.63e-03 | 3.61° | 3.39 m |
| 0.10 | 8.14e-03 | 18.07° | 16.89 m |
| 0.50 | 4.07e-02 | 90.67° | 80.31 m |
| 1.00 | 8.14e-02 | 146.00° | 136.53 m |
| 10.00 | 8.14e-01 | 20.63° | 71.74 m |

0.1 rad/s is under 6 deg/s of roll — nothing — and it moves the attitude 18
degrees over a five-second burn. The divergence peaks near 1 rad/s and falls
away beyond it, which is gyroscopic stiffening: a fast enough spin resists
being turned at all.

The uncomfortable part is that this vehicle **has no roll authority**. The
gimbal torque is `r x F` with `r` along body x, so its x component is
identically zero — swept over 10,201 deflection pairs at full thrust, the worst
roll torque is 0.00e+00, and it is zero by construction rather than by
tolerance. So the model can be *disturbed* into a roll it cannot remove, and
the disturbance then costs 18 degrees of attitude per 6 deg/s. Real vehicles
get roll control by throttling several engines differentially; this project has
modelled the engines as one effective thruster since Day 2, and that is the
choice that costs the axis.

### A defect in Day 5, found by the reduction check

The reduction test is the load-bearing one — Days 1–12 all rest on
`dynamics_6dof`, so a 3-D layer that disagreed with it would put them in
question. It disagrees.

`dynamics_6dof` uses `Tx = T sin(theta + delta)` for the thrust tilt and
`tau = +T L sin(delta)` for the torque. **Those two are not compatible.** Given
that thrust convention, working out `r x F` with the engine at `r = -L b` gives
`tau = -T L sin(delta)`. Day 5 has the opposite sign.

Verified three independent ways:

- a planar derivation written from vectors inside the test file, referencing
  neither model: `-0.38977985 rad/s^2` for a +5 deg deflection
- the new 3-D model, which agrees with it to **7.22e-16**
- the physical check: deflecting the nozzle toward +x pushes the tail toward
  +x, so the nose must go toward −x. Day 5 has the nose go toward +x.

This is not a sign convention on `delta`. Flipping `delta`'s sign changes the
thrust tilt as well, so no relabelling reconciles them — what is wrong is the
*relative* sign between the tilt and the torque, and that is
convention-independent.

Consequence: in Day 5's model, gimballing to rotate the vehicle tilts the
thrust the *same* way the vehicle is turning, so tilt and translation reinforce
each other. In reality they fight — a gimballed rocket is non-minimum phase,
and to translate one way you must first accelerate the other way. Day 5's own
docstring states the difficulty correctly ("you cannot buy one without paying
for the other") and then codes its opposite. Same entry state and same open-loop
gimbal profile, the two models rotate the vehicle in opposite directions:
+232° against −91° after six seconds at 2 deg of deflection.

**Why twelve days of tests did not catch it.** `landing_flip.py` carries the
same pairing — its header states `Tx = sigma sin(theta + delta)` and
`tau = sigma L sin(delta)`, and its linearisation is built on them. So the
optimiser and the simulator agree with each other perfectly, and every
verification in this project compares one against the other: the SCvx loop
against the simulator, Monte Carlo against the simulator, the closed loop
against the simulator, the EKF against the simulator. Nothing was ever compared
against an independent derivation of the physics. Two mutually consistent
models are indistinguishable from two correct ones until a third opinion turns
up, and today's reduction check is the first third opinion this project has
had.

That is the transferable lesson, more than the sign itself: a self-consistent
stack tests its own internal agreement, which is not the same thing as testing
whether it is right.

**Left unfixed today, deliberately.** `dynamics_6dof` is load-bearing for
Days 5–12 — the flip optimiser, SCvx, the complete solver, Monte Carlo, the
closed loop, the EKF, the bias filter, and eight viewer entries. Correcting it
means re-deriving the linearisation in `landing_flip` as well as the simulator,
then re-running everything downstream, which is a day of work in its own right
and would invalidate published numbers. It is recorded here, asserted in `test_dynamics_3d.py` as an
exact sign flip so it cannot drift unnoticed, and it is the first thing to
decide about before Day 16 puts a solver on top of the 3-D model.

Direction of the error, unquantified: Day 5's model is *easier* to control than
a real vehicle, so the fuel numbers, the feasibility band and the landing
accuracy reported on Days 5–12 are all likely optimistic. I have not re-run
them to say by how much.

### The checks that are theorems
Torque-free rigid-body motion conserves two things exactly, and both are far
stronger than any tuned tolerance:

- **angular momentum as a vector in the inertial frame**: relative drift
  **5.15e-12** over a 15 s tumble, direction fixed to 1.48e-06 deg — while the
  body-frame omega swings **119 degrees** in body axes over the same run. That
  second number is what gives the test teeth; a check on `|L|` alone would pass
  with the rotation wired wrong.
- **rotational kinetic energy**: relative drift **5.61e-15**.

The guide's tolerances for these were `atol=1e-2` and `< 0.1%`. The first is
worse than it looks — `np.allclose` applies its default `rtol=1e-5` to a
quantity of order 1e7, so it would have accepted a drift of a hundred units
while an `atol` of 1e-2 sat in the call looking strict. Both were rewritten as
explicit relative tolerances, and the measured values beat them by ten orders
of magnitude.

### The tennis racket theorem, which nobody coded
Break axisymmetry (`I_yaw = 1.3 I_pitch`) and spin the body about each
principal axis in turn with a 1 per cent nudge off it. The minimum-inertia axis
is stable, the maximum-inertia axis is stable, and the **intermediate axis
flips completely** — the axis fraction runs from +1.0000 to −1.0000. Nothing in
the code knows about this; it falls straight out of `omega x (I omega)`, which
makes it about the best available check that the term is right.

The axisymmetric body is the clean contrast: the precession cone's half-angle
is **21.46 deg and never moves** (spread 0.00), while the tri-axial body's
breathes between 14.80 and 28.46.

### Problems hit
- Test 6 as written in the guide recomputes `domega` inline with the same
  formula it is testing, so it tests numpy rather than the code. Rewritten to
  go through `dynamics_3d_derivative`, and extended: the roll decoupling has to
  *disappear* when axisymmetry is removed, otherwise the check is passing for
  the wrong reason. It does — the roll-axis gyroscopic term goes from 0.00e+00
  to 2.43e+07.
- The Poinsot figure's middle panel needs its axis limits set by hand. Left to
  autoscale, matplotlib zooms in on a drift of one part in 1e12 and draws the
  conserved vector as an impressive cloud — a plot that says the opposite of
  the truth. Fixed at ±5% of `|L|`, it reads as the single point it is.

### Honest limitation
The inertia tensor is constant while the mass depletes. Burning 30 t of 130 t
does change a real vehicle's inertia and this does not track it. Day 5 made the
same simplification with a scalar `I_pitch`, so the two stay comparable, but it
is a simplification in both.

### Tomorrow (Day 15)
3-D aerodynamics on top of today's validated rigid body: angle of attack,
sideslip, force decomposition, and aerodynamic moments rather than forces alone.

### Time spent
_X hours_

---

## Day 13 — 2026-08-18

### Done
- `src/quaternion.py`: Hamilton algebra, DCM and Euler bridges, kinematics,
  with the convention stated once and used everywhere
- `src/dynamics_3d_kinematics.py`: the 13-state kinematic model, RK4 with
  mandatory renormalisation
- `tests/test_quaternion.py`: 7 groups
- `src/demo_3d_kinematics.py`: two figures and the four experiments
- `results/day13_tumble.png`, `day13_gimbal_lock.png`
- `attitude-3d` added to the viewer (fourteen problems) — the first entry with
  no optimiser behind it and no landing to fly. Building it turned up the
  correction below.

First day that leaves the planar model behind. Deliberately kinematics only —
no inertia tensor, no forces, no torques — because every three-dimensional bug
from here on will be a frame confusion or a sign error, and those are far
easier to find in a model whose answers can be checked against closed-form
rotations than in one where the forces are also in question.

### Gimbal lock, measured rather than described
The interesting number came out of rewriting a test that was asserting the
wrong thing. Flying a perfectly smooth rotation at a constant **0.6 rad/s**
straight through pitch = 90 degrees, the rate the Euler description demands
peaks at **1570.8 rad/s — 2618x the physical rate** — while the quaternion
path shows no discontinuity anywhere along the same trajectory. That ratio is
what would actually break a controller reading angles, and it is why the rest
of this project will store a quaternion.

My first version of that test probed the singularity statically, perturbing
roll and yaw by 1e-6 at a fixed pitch. It measured a ratio of 1.0, because at
that pitch `quat_to_euler` had already switched to its locked branch — the
probe was sitting inside the singularity rather than approaching it. The
failure is dynamic and had to be measured dynamically.

### The other correction
The renormalisation test asserted the norm drifts without it. At `dt = 0.01`
it does not, to nine decimal places, and asserting otherwise would have been
overclaiming to make a point. Measured across step sizes the drift is
**2.5e-03 at dt=0.5, 2.6e-05 at 0.2, 2.6e-08 at 0.05 and 8.2e-12 at 0.01** —
RK4 truncation error, shrinking with the step as it should. The honest
argument for renormalising is not that the solution falls apart quickly but
that the error is **one-sided**: it accumulates monotonically rather than
averaging out, so it only ever grows, and an un-normalised quaternion silently
stops representing a pure rotation.

### Key results
- Sandwich product and rotation matrix agree to **9.5e-16** over 300 random
  cases — two different derivations, so this is the real check that both treat
  the quaternion as body-to-inertial.
- Round trips through the matrix are exact to **0.0 deg**, and 155 of 300
  returned `-q` rather than `q`. That is double cover, not a bug, and the
  comparison helper checks up to sign for exactly this reason.
- **The 3-D model reduces to Day 5's planar case exactly**: a single-axis spin
  matches the closed form to 0.0 deg over 10 s, yaw advances linearly at
  omega, and nothing leaks into the other two quaternion components. Days 1-12
  all rest on the planar model, so a 3-D layer that disagreed would have put
  them in question.
- Day 5's 70 degree flip, expressed as a relative quaternion, comes out at
  **70.0000 deg** about the expected axis and flies to vertical with zero
  residual.

### The experiment worth keeping
Feeding angular velocity in the wrong frame is invisible for a single-axis
spin — 50.60 deg either way — and wrong once the axis tilts: 31.96 deg against
50.61. That is precisely why a frame bug survives a test suite built on planar
motion, which is the suite this project has had for twelve days.

> **Correction, added while building the Day 13 viewer entry.** The tilted half
> of that experiment does not show what I said it did. It compares a body rate
> of (0.7, 0, 0.9) against an *inertial* rate of (0, 0, 0.9) — two different
> rate vectors — so the 31.96-against-50.61 difference mixes the frame with the
> vector and does not isolate the frame at all. The single-axis half stands.
>
> Holding the vector fixed and changing only how it is read gives a sharper
> result than the one I claimed. A body rate composes onto the **right** of the
> initial attitude and an inertial rate onto the **left**, so the two readings
> are not merely similar on easy cases — they are bit-for-bit identical
> whenever those commute. That rules out three whole families: a zero rate, an
> upright start, and a rate parallel to the axis the vehicle is already tilted
> about. Three of the four scenarios in the viewer entry are in that family, as
> is every planar case this project has ever run. Only a tumble started from a
> tilt about a different axis separates them, and it does so by **28.1 deg**.
>
> So the conclusion survives and gets stronger — a frame bug is invisible to
> the obvious test cases — but the mechanism is commutation, not
> single-versus-multi-axis, and the original experiment could not have
> established it.

### Honest limitation
This is bookkeeping, not physics. Nothing here computes a force, and the
angular acceleration is prescribed rather than derived, so none of it can be
wrong in an interesting aerodynamic way yet. Day 14's Euler equations are
where the inertia tensor arrives and where these conventions get their first
real test.

### Time spent
_X hours_

---

## Day 12 — 2026-08-17

### Done
- `src/imu_bias.py`: the true gyro bias as a slow random walk, and the rate
  channel it corrupts
- `src/ekf_bias.py`: `BiasEKF`, Day 11's filter with the bias promoted to a
  seventh state
- `src/ekf.py`: refactored to read its dimension off the state, so the
  augmented filter subclasses it instead of copying it
- `src/navigation_loop.py`: a bias-aware mode, on the same wind and sensor
  streams as the rest
- `tests/test_imu_bias.py`: 8 groups
- `src/bias_experiments.py`: three sweeps; `results/day12_bias.png`

### It works, at the thing it was built for
Rotating under a constant commanded torque with a 1.5 deg/s bias on the gyro,
the bias-blind filter carries a **1.309 deg/s standing rate error** and the
augmented one **0.226 deg/s** — 5.8x better. The bias estimate converges from
nothing to within 0.094 deg/s of a true 1.449, with its uncertainty falling
1.434 → 0.054 deg/s and not collapsing to false certainty. Given nothing to
estimate it does not invent a bias (−0.145 deg/s) and does not degrade.

The magnitude sweep is the clean version. The blind filter's bias error simply
*is* the bias — 0.48, 0.99, 1.98, 3.99 deg/s at 0.5, 1, 2 and 4 — because it
estimates nothing; the augmented filter holds it under 0.39 deg/s throughout.
In landing terms that only pays at large bias: at 4 deg/s the blind loop misses
by 17.52 m against 3.49 m aware, and below about 2 deg/s the two are inside the
seed-to-seed noise. Attitude rate matters up to about 20 Hz (bias error 0.217
at 5 Hz, 0.117 at 20, unchanged at 50), and a filter told the bias drifts ten
times faster than it does loses most of the benefit (bias error 0.913 against
0.117).

### The correction, which matters more than the feature
Wiring the augmented filter in produced a 569 m miss and a 14 s flight. The
cause was not the filter: once a plan's horizon was spent, the guidance loop
kept flying its **last commanded control**, and for a landing plan that is a
lit engine. One run descended to 4.8 m, reversed to +17.8 m/s, and tumbled
through 878 degrees on the way back up to 174 m. The loop now stops when the
plan is spent, as the open-loop baseline always did.

That bug was in Day 11's loop and in Day 10's, and **it invalidated a published
Day 11 claim**. Day 11 reported that filtering bounds the tail: worst miss 210 m
unfiltered against 6.7 m filtered, with a 20 t propellant blowout. Those
catastrophes were the vehicle climbing away on a spent plan, not the unfiltered
estimator. With the bug fixed the unfiltered worst case is **4.4 m** against the
filter's 6.7 m, so filtering does not bound the tail either. Day 11's LOG entry
and README section are corrected in place rather than quietly rewritten.

What survives is the estimate: four times better in four of four seeds. And the
correction strengthens Day 11's actual conclusion — a better estimate is not a
better landing — rather than weakening it, since the tail was the one place the
filter had appeared to pay off downstream.

### Honest limitation
The descent lasts about five seconds and the bias takes one to two of them to
resolve, so a third of the flight is spent learning a constant. That is most of
why augmentation buys so little downstream at realistic bias levels. Arrival
speed remains the binding failure in every mode, which is Day 10's guidance-rate
problem and is not an estimation problem at all.

### The pattern generalises, which is the real claim
Experiment C extends the state once more, to eight, with a bias on the nav
sensor's downrange-velocity channel — an accelerometer error once the
navigation system has integrated it. Nothing structural changed: one more row
of `f` that stays at zero, one more diagonal entry in `Q`, and one more column
in the measurement matrices with a 1 wherever the instrument cannot separate
the error from the signal.

With both errors active at once, the filter resolves **both**: the gyro bias to
0.135 deg/s of a true 1.405, and the velocity bias to 0.100 m/s of a true
2.487, with the uncertainties falling to 0.052 deg/s and 0.319 m/s and the 8x8
covariance staying positive semi-definite.

They are separable because they are seen by *different instruments*. The gyro
bias appears only in the attitude sensor's rate channel and the velocity bias
only in the nav sensor's downrange channel, so there is no path by which one
can be mistaken for the other — which is why adding the second one does not
disturb the first. Two errors on the *same* channel would not be separable at
all, and that limit is worth knowing before reaching for another state.

### Time spent
_X hours_

---

## Day 11 — 2026-08-16

### Done
- `src/sensors.py`: nav and attitude instruments, each on its own clock
- `src/ekf.py`: EKF over the real coupled dynamics, numerical Jacobian,
  Joseph-form update
- `src/navigation_loop.py`: Day 10's guidance rewired onto the estimate, with
  truth and naive baselines on identical wind *and* identical sensor noise
- `tests/test_ekf.py`: 7 groups
- `src/navigation_experiments.py`: four sweeps
- `navigation` added to the viewer (twelve problems)
- `results/day11_navigation.png`, `day11_sweeps.png`

This closes the limitation Day 10 recorded: raw estimates were being fed
straight into a re-optimisation that is bang-bang by construction, and 3 m of
position noise produced a 109 m worst-case miss.

### The result
One seed, three ways, identical wind and identical sensor noise:

| | miss | arrival | fuel | est. error |
|---|---|---|---|---|
| truth (Day 10's privilege) | 0.28 m | 15.41 m/s | 6,003 kg | — |
| **EKF** | **1.71 m** | 22.52 m/s | 5,751 kg | 2.01 m |
| naive | 92.51 m | 28.78 m/s | 11,120 kg | 4.94 m |

Across four seeds the worst miss is **6.7 m filtered against 210 m
unfiltered**, and the 20.2-tonne propellant blowout disappears.

### The finding, which arrived as a failing test
The first `Q` was wrong by two orders of magnitude, and the way that surfaced
is the whole lesson. The test asserting "filtering lands nearer" failed: the
EKF estimated three to four times better in **4 of 4** seeds and landed worse
in **3 of 4**. Rather than weaken the assertion, the sweep was run.

Scaling `Q` over the closed loop:

| Q scale | est. error | miss |
|---|---|---|
| x0.01 | 2.22 m | 34.45 m |
| x0.1 | 1.55 m | 11.74 m |
| **x1** | **1.60 m** | **3.13 m** |
| x10 | 2.37 m | 4.76 m |
| x100 | 3.23 m | 3.28 m |

Note what barely moves. A filter tuned a hundred times too tight estimates
about as accurately *on average* and lands eleven times further away, because
an under-confident `Q` makes it trust its own dynamics through gusts it cannot
see and the resulting error is a **lag** rather than noise. Successive replans
average noise out and cannot average out a bias. **Mean estimation error is the
wrong metric for a control loop** — a lagging estimate and a noisy one are
indistinguishable by it and behave nothing alike. The default was corrected and
the same seed went from 22.40 m to 1.71 m.

### The other sweeps
- **Sensor rate is a hardware spec, not a trend.** 1 Hz gives a 175 m miss and
  12,785 kg; 2 Hz gives 1.98 m. The loop needs at least 2 Hz position aiding
  and gains nothing above it, while estimation error keeps improving to 20 Hz —
  the same divergence between estimate quality and control quality.
- **Attitude alone is not navigation.** With the nav sensor off the filter
  dead-reckons and estimation error goes 1.60 m to 7.42 m, with nothing to
  bound the drift.
- **An unestimated bias is invisible where you look for it.** A gyro bias takes
  the miss from 3.13 m to 13.67 m at 2 deg/s while the *position* estimate
  stays flat near 1.5 m. The filter reports good health while the steering
  degrades, which is the argument for augmenting the state with a bias term.

### Problems hit
- `EKF(aero=None)` silently substitutes a default `AeroConfig()`, so the
  "known-linear" Jacobian test still had drag in it and failed by 0.22. With
  `AeroConfig(enabled=False)` the Jacobian matches the exact transition matrix
  to 2.4e-11.
- Fed end-of-interval truth to measurements taken mid-interval, which injected
  a systematic lead and made the filter look 17x worse than it is — 16.40 m
  against its real 0.96 m. Sampling truth at the filter's own rate fixed it.
- Called the bias experiment inconclusive on the first pass. It was, but only
  because the mistuned `Q` was swamping it; with the corrected filter the trend
  is clean and monotonic.

### Next
- Augment the state with a gyro bias term, which experiment D now argues for
  quantitatively
- Re-run Day 9's dispersion sweep with the full stack — guidance, control and
  navigation — against the standing 29.6% figure

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
