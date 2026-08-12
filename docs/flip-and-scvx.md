# The flip: rotational dynamics and a working SCvx loop

Day 5 notes. Part 1 is the physics, then what the optimiser actually does, then
troubleshooting for the code as written.

---

## Why rotation changes the problem

Days 1–4 treated the thrust vector as free: the optimiser could point it
anywhere and pay a pointing-cone penalty. That is false. The engine is bolted to
the vehicle, so the thrust direction **is** the attitude, plus at most a 15°
gimbal:

```
Tx  = sigma sin(theta + delta)
Tz  = sigma cos(theta + delta)
tau = sigma L sin(delta)
```

Torque and thrust tilt come from the same deflection. Gimbaling to rotate the
vehicle simultaneously tilts the thrust that is decelerating it. There is no way
to buy one without paying for the other, and that is the whole difficulty.

The state grows from 5 to 7: `[x, z, vx, vz, theta, omega, m]`.

### The modelling trap

Writing `tau` as an independent variable bounded by `±sigma L sin(delta_max)` is
the obvious move, and it is wrong. It gives the optimiser free torque with no
effect on thrust direction, so it will happily rotate the vehicle while
thrusting somewhere unrelated. The problem solves, reports a plausible fuel
number, and describes a vehicle that does not exist.

Here the coupling is kept as an equality and linearised about a reference
attitude, throttle and gimbal, which keeps the subproblem convex while making
the torque carry its own thrust-tilt cost.

---

## Three things that make the SCvx loop actually converge

All three were found by the loop failing, not by design.

**1. Every solved subproblem advances the reference.** The textbook description
says to reject a step whose linearisation defect is too large and re-solve with a
smaller trust region. Implemented literally that deadlocks: the region tightens
around a point the solution is far from, so the next solve is strictly *harder*,
and it walks straight down to the minimum region size and reports infeasible.
The defect should size the next region, not veto the current step.

```
iter 1: fuel 14,179 kg  step 34.66 deg  defect 0.1960  REJECT
iter 2: infeasible   trust -> 10.00 deg
iter 3: infeasible   trust ->  5.00 deg      ... and so on to failure
```

**2. The torque needs its own trust region.** The gimbal expansion point is the
previous torque solution, and that solution is bang-bang. Left unbounded it
flips sign between iterations, moving the linearisation point further than the
step it was supposed to validate. Constraining `theta` alone left the defect
oscillating around 0.12 indefinitely; adding a torque region drove it to 0.0003.

**3. Expand about the previous gimbal angle, not about zero.** Expanding about
`delta = 0` drops a `0.5 sin(theta) delta²` term, which at the 15° limit is
0.034 of maximum thrust. That is a floor no amount of iterating can clear, and
it is visible as a defect that stops improving at exactly 0.036.

With all three, convergence is clean:

```
iter  1: fuel 14,179 kg  step 34.64 deg  defect 0.1958  trust -> 24.00 deg
iter  8: fuel 14,785 kg  step  1.12 deg  defect 0.0150  trust ->  0.67 deg
iter 18: fuel 14,775 kg  step  0.13 deg  defect 0.0003  trust ->  0.21 deg
Converged after 18 iterations.
```

### Seeding matters as much as the loop

Linearising about a reference the vehicle would never fly gives an infeasible
subproblem even where the true problem is fine. Seeding `theta_ref` with
`linspace(theta0, 0)` across the whole burn implies a 4.7 °/s rotation; this
vehicle flips at 28.6 °/s. At a 40° entry that seed is infeasible at every trust
size, while a fast-flip seed solves at 13,742 kg.

Also worth stating: **lossless convexification is gone and needs no replacement.**
Once the thrust direction is pinned to the attitude there is no free vector to
relax — `sigma` is simply the throttle, bounded directly. The non-convexity moved
from the magnitude to the direction.

---

## The entry-pitch ceiling

The optimiser cannot fly a full 90° belly-flop, and the reason is physical
rather than numerical.

The engine is lit throughout — minimum throttle is 40% and there is no coast —
so while tilted, `sin(theta)` of a very large thrust pushes the vehicle sideways
whether it wants that or not, up to 21 m/s². The pitch rate is capped at 28.6 °/s,
so the flip takes at least `theta0 / omega_max` seconds, and the lateral
excursion accumulated in that window must fit inside the glideslope corridor and
still be nulled by touchdown.

Holding the entry state fixed and sweeping only attitude, 65° solves and 70°
does not. Removing **either** the glideslope **or** the pitch-rate limit makes
70° solve — which is what identifies the two as binding together rather than
one being the culprit:

| configuration | 70° entry |
|---|---|
| nominal | infeasible |
| no glideslope | optimal, 14,736 kg |
| no pitch-rate limit | optimal, 13,692 kg |
| `omega_max` × 1.5 | optimal, 14,194 kg |

Letting the entry state be re-sized per attitude — what the defaults do — the
ceiling drops to 40° feasible, 50° not, because a more tilted entry also
decelerates less and demands a different arrival state. Raising `omega_max` to
51 °/s lifts that ceiling from 40° to 55°.

**A real Starship flips before the landing burn, unpowered, on aerodynamic
surfaces.** That is precisely the freedom this model does not have, and it is the
most useful thing the ceiling teaches.

---

## Known limitation: the flip optimiser still uses Euler

Day 4 measured forward Euler at a 1.5% terminal miss on the 3-DoF problem and
showed trapezoidal collocation cutting it 7×. The flip optimiser has not been
upgraded yet, and rotation compounds the error: replaying the commanded throttle
and gimbal through the verified non-linear simulator lands **66.7 m from the pad,
4.0% of the descent**, with attitude good to 0.37° and residual rate 0.04 °/s.

The attitude tracks well; it is the translation that drifts. Porting
`src/discretization.py` into `landing_flip.py` is the outstanding action, and
that number is how it should be judged.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Infeasible at every trust size, from iteration 1 | The seed reference is a trajectory the vehicle cannot fly | Seed `theta_ref` with a flip no faster than `theta0/omega_max` and no slower than a few times that. `linspace` across the whole burn is usually far too slow. |
| Infeasible at every entry pitch, including 0° | The entry state violates the minimum-throttle energy budget, exactly as on Day 3 | Use `feasible_entry_state`; a fixed `(z0, vz0)` guess almost never satisfies it. |
| Defect stalls at ~0.036 and will not improve | Expanding the gimbal about `delta = 0` | Expand about `delta_ref` from the previous solution. |
| Defect oscillates between 0.08 and 0.20 | Torque reference flipping sign between iterations | Add a trust region on `tau`, not just on `theta`. |
| Trust region collapses and the run reports infeasible | Steps are being rejected rather than accepted | Accept every solved subproblem; use the defect to size the *next* region. |
| Converges but `step` never falls below tolerance | The trust region is still binding at the optimum, so `step` tracks the region size | Test convergence on objective stability as well as step size, and set `trust_min` below `tol`. |
| `SolverError` or wildly wrong magnitudes | SI units span twelve orders of magnitude here | The problem is solved non-dimensionally. Scale any constraint you add. |
| `DPPError` after adding a parameter | Two parameters multiplying, or a variable divided by one | Pre-multiply reference products in NumPy so each parameter enters linearly — that is why `push_params` computes eight combined arrays. |
| `SyntaxError: invalid non-printable character U+FEFF` | PowerShell 5.1 `Set-Content -Encoding utf8` writes a BOM | Write files with `UTF8Encoding($false)`, or use an editor. |
| Bang-bang gimbal test spins through several revolutions | Flip timings assumed from a slower vehicle | `alpha_max` here is 94.7 °/s², so a 90° rest-to-rest rotation takes ~2 s. Derive timings from the vehicle. |
