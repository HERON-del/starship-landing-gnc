# Free final time, and where SCvx comes in

Day 4 notes. Part 1 covers why burn duration matters, Part 6 covers the loop
that Days 7–8 will build, Part 7 is troubleshooting for the code as written.

---

## Why the burn duration is worth optimising

Every second the engine burns, some of its thrust is spent holding the vehicle
up rather than changing its trajectory. That is the **gravity loss**, and it is
proportional to burn duration:

```
fuel = fuel to change the trajectory + fuel to fight gravity
```

Fix the duration too long and the vehicle throttles down and hovers, paying
gravity losses for nothing. Fix it too short and there is not enough impulse to
null the velocity, and the problem is infeasible. Only one duration is optimal,
and guessing it costs propellant.

Measured on this vehicle, with the entry state held fixed:

| Configuration | Burn time | Fuel |
|---|---|---|
| Fixed 20 s (Day 3) | 20.00 s | 18,077 kg |
| Free time, Euler | 16.02 s | 16,670 kg |
| Free time, trapezoidal | 16.46 s | 16,797 kg |

**7.1% of the landing propellant, recovered by choosing the duration instead of
assuming it.** That is why real landing burns are short, late and hard.

---

## How free final time is actually implemented

The obvious approach does not work, and the failure is quiet rather than loud.

Time enters the dynamics multiplicatively — every update looks like
`x[k+1] = x[k] + dt·v[k]` with `dt = t_f/N`. Making `t_f` a variable makes
`t_f·v` a product of two unknowns: bilinear, non-convex, rejected by CVXPY.

The tempting workaround is to hold `dt` at a reference value, declare
`t_f = cp.Variable()`, bound it, and add a small penalty like `0.001·t_f` to the
objective. That compiles and runs. But look at where `t_f` appears: only in its
own bounds and its own penalty. The dynamics use the reference `dt`. So `t_f` is
**decoupled from the trajectory entirely**, and minimising its penalty drives it
straight to whichever bound the penalty prefers. The reported "optimal burn
time" is the lower bound wearing a disguise, and it would not change if the
vehicle were twice as heavy.

What is actually true is more useful:

- For a **fixed** `t_f` the problem is convex and solves in milliseconds.
- Fuel versus duration is smooth and unimodal, bounded by infeasibility at both
  ends.

So the honest formulation is a one-dimensional search wrapping the convex solve:
bracket the feasible interval on a coarse grid, then golden-section to the
minimum. Every point evaluated is a global optimum of its own subproblem, which
is the guarantee worth keeping. `src/landing_free_time.py` builds the convex
problem once with the duration-dependent coefficients as `cp.Parameter`s, so the
entire search costs solve time only — about 23 solves, well under a second.

---

## The losslessness boundary is not decorative

Lossless convexification replaces the non-convex floor `‖T‖ ≥ T_min` with
`‖T‖ ≤ σ` and `T_min ≤ σ ≤ T_max`. The substitution is only honest while
`σ = ‖T‖` at the optimum. When a gap opens, the trajectory burns propellant at
the `σ` rate while producing less than that much force — cheaper on paper,
unflyable in fact.

Sweeping burn duration makes the boundary visible:

| `t_f` | fuel | gap / T_min | peak tilt |
|---|---|---|---|
| 15.5 s | 16,798 kg | 0.040 | 30.0° |
| 16.0 s | 16,772 kg | **0.133** | **30.0°** |
| 16.5 s | 16,809 kg | 0.000 | 5.2° |
| 20.0 s | 18,007 kg | 0.000 | 5.5° |
| 21.0 s | 18,564 kg | **0.130** | **28.7°** |

Every slack case has the **pointing constraint at its limit**. Açıkmeşe &
Ploen's magnitude-only proof does not cover an active pointing constraint —
that case needs the separate treatment in Açıkmeşe, Carson & Blackmore (2011),
under its own conditions. Here the theorem's boundary shows up as a number.

Saturation is *necessary but not sufficient*, and it is worth being careful
about the difference. At N = 40 the search settles on 16.03 s with the tilt
pegged at exactly 30.0° and a relaxation gap of 90 N — 0.003% of T_min, tight
by any measure. So an active pointing constraint permits a gap; it does not
force one. Which is precisely why the code measures the gap on every solve
rather than inferring it from whether the tilt is saturated.

The practical consequence: the *cheapest* duration is in the slack region and
its trajectory is not flyable. `solve_landing_free_time` therefore rejects any
duration whose relaxation has gone slack, and reports how many it discarded. The
flyable optimum costs about 30 kg more than the paper optimum — a rounding error
in fuel, and the difference between a trajectory that flies and one that does
not.

---

## Euler versus trapezoidal, measured rather than asserted

Forward Euler is first-order (global error `O(dt)`); the trapezoidal rule
averages the derivative at both ends of the interval and is second-order
(`O(dt²)`). In simulation the implicitness would cost an inner iteration; in
optimisation it is free, because the state at `k+1` is already a variable.

Comparing fuel numbers between the two proves nothing — each is optimal for its
own discretized model, and Euler routinely reports *less* fuel precisely because
its model is wrong. The test that settles it is to take the commanded thrust
profile and fly it through the independently verified RK4 integrator from Day 2:

```
Euler  position error  43.7 m,  velocity error 0.47 m/s
Trapz  position error   6.1 m,  velocity error 1.25 m/s
```

Same node count, **7.1× smaller miss**. Euler's lower fuel number was
discretization error being exploited, not efficiency.

---

## Part 6 — the SCvx loop

Three pillars of Sequential Convex Programming; two are already in the code.

| Pillar | What it does | Where |
|---|---|---|
| Mass-reference iteration | Linearises `T/m` by fixing `m` each iteration | Day 3, Day 4 |
| Time-reference search | Handles `t_f × dynamics` by fixing `t_f` each solve | Day 4 |
| **Trust regions** | Bounds how far a step may stray from the reference | **Not yet** |

Without a trust region the iteration can oscillate: the new solution is computed
from the old reference, so a large step lands somewhere the linearisation was
never valid. The Day 3 solver already needs damping (0.5 by default) on the mass
update for exactly this reason — undamped, the bang-bang switching times flip
between iterations and it never settles. Damping is a crude, fixed trust region.

The real loop, for Days 7–8:

1. Choose an initial reference `x_ref, u_ref, t_f_ref`.
2. Repeat:
   1. Linearise the dynamics about the reference.
   2. Add a trust region `‖x − x_ref‖ ≤ δ`.
   3. Solve the convex subproblem.
   4. If `‖x_new − x_ref‖ < tol`, converged.
   5. If the cost improved, accept and grow `δ`; if it worsened, reject and
      shrink `δ`.
   6. Set `x_ref = x_new`.

This is a trust-region Newton method. The adaptive `δ` is what makes it robust
where fixed damping is not, and it is what the harder problems — aerodynamics,
6-DoF attitude, state-triggered constraints — will require.

---

## Part 7 — troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Every duration reports infeasible | The entry state is not reachable at any duration in the range | Entry altitude and speed are coupled through the minimum-throttle floor. Use `feasible_entry_state`, or widen `t_f_min`/`t_f_max`. |
| Everything is rejected for a slack relaxation | The pointing limit is saturated across the whole feasible window | Relax `theta_max_deg`. If a slack solution is genuinely wanted for comparison, pass `require_lossless=False` — but do not fly it. |
| Free-time fuel exceeds fixed-time fuel | The two runs used different entry states | The entry state must be held fixed across the comparison; `run_comparison` derives one and shares it. |
| Trapezoidal reports *more* fuel than Euler | Expected, not a bug | Euler's discretization error lets the optimiser cheat. Judge by the RK4 replay, not the reported cost. |
| `SolverError` from Clarabel | Bad scaling — SI units span twelve orders of magnitude here | The problem is solved non-dimensionally for this reason. If you add a constraint, scale it too. |
| `DPPError` after adding a parameter | Two parameters multiplying, or a variable divided by a parameter | Every parameter must appear linearly. Fold products into a single parameter — this is why the mass reference lives inside the velocity coefficient. |
| Mass-reference iteration oscillates | Missing trust region | Lower `damping`. This is the gap that SCvx closes. |
| The search is slow | The problem is being rebuilt per evaluation | It should compile once; only `cp.Parameter` values change between solves. |
