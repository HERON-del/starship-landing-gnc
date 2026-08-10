"""
Day 1 - First convex optimization problem.

THE PROBLEM (a toy version of rocket landing):
We have a point mass at height 100 m, falling at 20 m/s downward.
Gravity pulls it down at 9.81 m/s^2.
We have a thruster that can push UP with force between 0 and 25 m/s^2 of accel.
We have 10 seconds.

QUESTION: What thrust profile lands it softly (height=0, velocity=0 at t=10s)
          while using the LEAST total thrust (= least fuel)?

This is the 1-D ancestor of the full 6-DoF Starship problem.
Every concept here reappears in Week 2 and Week 4 - just with more variables.

SOLVER NOTE: the guide uses ECOS. This environment runs Python 3.13, which has
no ECOS wheel, so we use CLARABEL - the interior-point solver that superseded
ECOS and ships with CVXPY. Same problem class, same answer.
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. PROBLEM SETUP
# ----------------------------------------------------------------------
g = 9.81        # gravity, m/s^2
h0 = 100.0      # initial height, m
v0 = -20.0      # initial velocity, m/s (negative = falling)
T_final = 10.0  # time of flight, s
N = 50          # number of discrete time steps
dt = T_final / N
a_max = 25.0    # max thrust acceleration, m/s^2

print(f"Discretizing {T_final} s into {N} steps of {dt:.3f} s each")

# ----------------------------------------------------------------------
# 2. DECISION VARIABLES
#    These are the unknowns the solver will figure out for us.
# ----------------------------------------------------------------------
h = cp.Variable(N + 1)   # height at each time step
v = cp.Variable(N + 1)   # velocity at each time step
a = cp.Variable(N)       # thrust acceleration during each interval

# ----------------------------------------------------------------------
# 3. CONSTRAINTS
#    Rules the solution MUST obey. Physics goes here.
# ----------------------------------------------------------------------
constraints = []

# --- Initial conditions ---
constraints += [h[0] == h0]
constraints += [v[0] == v0]

# --- Dynamics: how the state evolves (simple Euler integration) ---
#     next_velocity = velocity + (thrust - gravity) * dt
#     next_height   = height   + velocity * dt
for k in range(N):
    constraints += [v[k + 1] == v[k] + (a[k] - g) * dt]
    constraints += [h[k + 1] == h[k] + v[k] * dt]

# --- Terminal conditions: soft landing ---
constraints += [h[N] == 0.0]
constraints += [v[N] == 0.0]

# --- Physical limits ---
constraints += [a >= 0.0]      # thruster can only push, not pull
constraints += [a <= a_max]    # max thrust
constraints += [h >= 0.0]      # never go underground

# ----------------------------------------------------------------------
# 4. OBJECTIVE
#    What we want to minimize. Total thrust ~ total fuel burned.
# ----------------------------------------------------------------------
fuel = cp.sum(a) * dt
objective = cp.Minimize(fuel)

# ----------------------------------------------------------------------
# 5. SOLVE
# ----------------------------------------------------------------------
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CLARABEL, verbose=False)

print("-" * 60)
print(f"Solver status : {problem.status}")
print(f"Optimal cost  : {problem.value:.4f} (delta-v in m/s)")
print(f"Solve time    : {problem.solver_stats.solve_time * 1000:.2f} ms")
print("-" * 60)

if problem.status not in ["optimal", "optimal_inaccurate"]:
    print("Problem was not solved. Check constraints for contradictions.")
    raise SystemExit(1)

# ----------------------------------------------------------------------
# 6. VERIFY
#    Never trust a solver without checking the answer yourself.
# ----------------------------------------------------------------------
print(f"Final height   : {h.value[-1]:.6f} m   (target: 0.000000)")
print(f"Final velocity : {v.value[-1]:.6f} m/s (target: 0.000000)")
print(f"Max thrust used: {np.max(a.value):.3f} m/s^2 (limit: {a_max})")
print(f"Min thrust used: {np.min(a.value):.3f} m/s^2 (limit: 0)")

# ----------------------------------------------------------------------
# 7. PLOT
# ----------------------------------------------------------------------
t_state = np.linspace(0, T_final, N + 1)
t_ctrl = np.linspace(0, T_final - dt, N)

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

axes[0].plot(t_state, h.value, linewidth=2, color="tab:blue")
axes[0].set_ylabel("Altitude [m]")
axes[0].set_title("Day 1 - Minimum-Fuel 1-D Soft Landing (Convex Optimization)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_state, v.value, linewidth=2, color="tab:orange")
axes[1].axhline(0, color="k", linestyle=":", linewidth=1)
axes[1].set_ylabel("Velocity [m/s]")
axes[1].grid(True, alpha=0.3)

axes[2].step(t_ctrl, a.value, where="post", linewidth=2, color="tab:red")
axes[2].axhline(a_max, color="k", linestyle="--", linewidth=1, label=f"max = {a_max}")
axes[2].axhline(g, color="g", linestyle="--", linewidth=1, label=f"hover = {g:.2f}")
axes[2].set_ylabel("Thrust accel [m/s^2]")
axes[2].set_xlabel("Time [s]")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/day1_first_landing.png", dpi=150)
print("\nPlot saved to results/day1_first_landing.png")

plt.show()
