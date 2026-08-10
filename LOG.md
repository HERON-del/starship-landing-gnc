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

### Tomorrow (Day 3)
- Re-derive the Day 1 landing problem using the new dynamics module
- Add glideslope and thrust-cone constraints
- Begin the log-mass change of variables from the paper

### Time spent
_X hours_
