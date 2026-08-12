# Aerodynamics, and why the belly-flop is worth more than the burn

Day 6 notes. Built differently from the guide, for a measured reason.

---

## The guide's model, and what measuring it said

The guide models aerodynamics and a lit engine over the same 25–30 s window.
Built exactly as specified, that is infeasible at every entry attitude (0–60°)
and every burn duration (5–15 s) tried. Three fixes were attempted and each
ruled out by measurement:

1. **Aero-aware entry sizing.** With drag there is a *window*, not a floor — too
   slow and the 40% throttle floor over-decelerates, too fast and drag does. A
   one-sided fixed point diverges to 5,584 m/s because drag grows as `v²` while
   the requirement grows as `v`.
2. **A homotopy on the drag term**, ramping it in from a known-feasible λ = 0.
3. Both together.

The decisive test: the aero-sized entry state is infeasible **with drag switched
off**. So it was never the drag forcing — a one-dimensional vertical velocity
budget takes no account of the altitude, attitude and corridor coupling that
actually binds. Sizing the entry on thrust alone and letting drag be a
perturbation solves cleanly at every attitude from 0° to 60°.

---

## Drag saves nothing during the burn

Same problem, drag on and off:

| | no aero | with aero |
|---|---|---|
| 60° entry, 15 s burn | 14,783 kg | 14,785 kg |
| 30° entry, 15 s burn | 14,162 kg | 14,162 kg |

Identical, despite peak aerodynamic deceleration of 86 m/s² — more than the
engines can produce. The reason is the throttle floor: minimum throttle flows
861 kg/s and the engines must run for the whole descent, so **propellant is set
by burn duration, not by how much work the engines do.** Drag lets the optimiser
throttle down, and throttling down is exactly what it cannot do.

That result is what makes the guide's single-phase model the wrong shape. If
aerodynamics buy nothing while the engines are lit, the interesting phase is the
one before ignition.

---

## Where the belly-flop actually pays

Unpowered descent from 12 km to 300 m, engines off:

| configuration | arrival speed | coast time |
|---|---|---|
| no atmosphere | 494.0 m/s | 38 s |
| nose-first (0°) | 357.5 m/s | 42 s |
| 45° | 82.0 m/s | 108 s |
| **belly-flop (90°)** | **64.0 m/s** | 131 s |

`Cd·A` is **28.3× larger** broadside than base-first — 540 m² against 19 m² —
and terminal velocity falls from 336 m/s to 63 m/s.

The belly-flop removes **430 m/s for free.** By the rocket equation that is
~16,300 kg of propellant not spent, which is *more than the entire landing burn
costs*. The manoeuvre's whole value is in the coast, with the engines off.

### Dynamic pressure does not decay

Worth stating because the intuition is backwards. As the vehicle slows, `q` does
not fall — it converges. At terminal velocity drag balances weight, so

```
q = mg / (Cd A)
```

pinned by the vehicle rather than the altitude: rising density exactly offsets
falling speed. Measured 2.42 kPa against a predicted 2.36 kPa. An earlier test
asserted monotonic decay and failed; the model was right and the test was wrong.

---

## The two-phase structure

Because coasting is free and burning is charged by the second, the optimum is to
coast as long as possible and burn as briefly as possible. But burn duration is
coupled to handoff attitude — the flip is rate-limited, so a more tilted handoff
forces a longer burn:

| handoff attitude | shortest feasible burn | propellant |
|---|---|---|
| 0° | 4 s | **3,874 kg** |
| 20° | 5 s | 4,868 kg |
| 30° | 6 s | 5,746 kg |
| 45° | 10 s | 9,871 kg |
| 60° | 15 s | 14,820 kg |

Handing over near-upright after a full coast costs **4,255 kg — 3.5× less than
the 14,775 kg single-phase flip of Day 5.**

`solve_two_phase` runs it the way the vehicle flies it: coast unpowered to
terminal velocity, search for the highest altitude from which a powered landing
still closes, then hand to the Day 5 optimiser with drag as a perturbation. The
ignition point is found, not assumed — the same method Day 2 used for the
suicide-burn trigger.

---

## Honest limitation

The model still cannot flip 90° under power at terminal velocity: at 64 m/s the
throttle floor gives only ~5 s of burn before it over-decelerates, and a 90° flip
needs longer. So the pipeline hands over near-upright and the flip itself is not
modelled in phase 2.

This is the Day 5 entry-pitch ceiling reappearing, and the resolution is the
same: a real Starship flips using **aerodynamic control surfaces** — the flaps —
before the engines light. Modelling the flaps is what would close this gap, and
nothing in the current model substitutes for them.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Infeasible at every attitude with aero on | Entry sized against the drag it will experience | Size on thrust alone; drag is a perturbation the optimiser trims. |
| Entry sizer returns absurd speeds (thousands of m/s) | One-sided fixed point on a `v²` term | There is a window, not a floor — and the window is not the binding constraint anyway. |
| Aero-sized entry infeasible even with drag off | A 1-D vertical budget ignores the corridor and attitude coupling | Do not size entries from vertical dynamics alone. |
| Two-phase pipeline finds no ignition point | Handoff attitude too large for the burn the arrival speed permits | Lower the handoff attitude, or accept a faster handoff. |
| `q` rises during the coast | Not a bug | At terminal velocity `q = mg/(Cd A)`, pinned by the vehicle. |
| Viewer entry problem is slow | The coast is minutes long at a small step | Raise `dt`; RK4 is exact enough here that 0.05 s costs nothing. |
