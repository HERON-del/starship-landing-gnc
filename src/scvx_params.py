"""
Tunable parameters for the SCvx solver.

Separated from the algorithm so experiments change one object rather than a
function signature, and so a sweep can be written as a list of dataclasses.

A note on units, because it is the difference between these defaults working
and not working. Every quantity the trust region bounds is non-dimensional:
positions are divided by the entry altitude, velocities by `L / t_burn`, mass
by wet mass, thrust by `T_max`, torque by `tau_max`, pitch rate by `omega_max`.
Pitch is left in radians, which is already O(1). So a single scalar `eta`
bounds every state to the same *fractional* excursion, and a single `w_vc`
penalises every dynamics row on the same footing.

Doing this in SI instead requires a separate hand-picked multiplier per state
just to make one radius mean anything -- and then the L1 penalty is summing
metres, metres per second, radians and kilograms into one number, so no single
weight can be right for all seven rows. That was measured, not assumed: in SI
the virtual-control norm grows monotonically from 54 to 2.8e6 while the
optimiser happily reports `optimal` at every step.

References
----------
[1] Mao, Szmuk, Acikmese, "Successive Convexification: A Superlinearly
    Convergent Algorithm for Non-convex Optimal Control Problems," 2020.
[2] Szmuk, Acikmese, "Successive Convexification for 6-DoF Powered Descent
    Guidance with Free-Final-Time," AIAA, 2018.
"""

from dataclasses import dataclass


@dataclass
class SCvxParams:
    """Parameters controlling the SCvx iteration."""

    # --- Iteration limits ---
    max_iter: int = 30           # max outer iterations
    min_iter: int = 3            # min iterations before declaring convergence

    # --- Trust region (non-dimensional; see module docstring) ---
    eta_0: float = 0.5           # initial radius
    eta_min: float = 1e-4        # floor; reaching it means the step died
    eta_max: float = 4.0         # ceiling
    alpha_shrink: float = 0.5    # multiply eta by this on a rejected step
    alpha_expand: float = 1.6    # multiply eta by this on a good step

    # Torque gets its own region. The gimbal expansion point is the previous
    # torque solution and that solution is bang-bang, so left unbounded it
    # flips sign between iterations and moves the linearisation point further
    # than the step it was meant to validate. Day 5 measured the difference:
    # defect stuck at 0.12 without it, 0.0003 with it.
    eta_u_0: float = 2.0         # torque is normalised to [-1, 1]
    eta_u_min: float = 0.02

    # Guard, not a workhorse. A collapsed trust region with virtual control
    # still unpaid would be a stall rather than an answer -- shrinking cannot
    # help when the linearisation is already accurate and it is the *reference*
    # that sits somewhere the constraints cannot be met -- so the region is
    # re-expanded and slack made more expensive. It does not fire in any of the
    # ten perturbed cases measured; those either converge or settle on a
    # genuinely infeasible problem, where re-expanding correctly changes
    # nothing. Kept because the failure mode is real, not because it is hot.
    max_restarts: int = 2

    # --- Step quality thresholds ---
    rho_good: float = 0.7        # above this: accept and expand
    rho_ok: float = 0.05         # above this: accept, hold eta
    #                              below this: reject the step, shrink eta

    # --- Virtual control ---
    w_vc: float = 1e3            # L1 penalty weight on the dynamics slacks
    # Ceiling deliberately low. The penalty sits next to an objective of order
    # 1, so a weight of 1e7 makes the KKT system as badly conditioned as the SI
    # formulation this scaling exists to avoid -- measured: both CLARABEL and
    # SCS then return `unbounded` for a problem bounded below by -1.
    w_vc_max: float = 1e5
    w_vc_grow: float = 3.0       # growth factor, applied only when slack stalls
    vc_tol: float = 1e-6         # convergence tolerance on ||nu||_1
    # Looser bar for calling an *iterate* usable, as opposed to converged.
    # Applied to ||nu||_1, so it reads directly as "the dynamics are violated
    # by less than this, in scaled units."
    dyn_tol: float = 1e-3

    # --- Step convergence ---
    step_tol: float = 1e-3       # max scaled state change between iterations
    fuel_tol_kg: float = 2.0     # objective considered settled below this

    # --- Honesty check on the converged answer ---
    # The largest tolerated disagreement between the linearised thrust
    # direction and the true one, as a fraction of maximum thrust. Virtual
    # control going to zero proves the *linear* dynamics are satisfied; this
    # proves the linear dynamics were the right ones.
    defect_tol: float = 0.01

    # --- Aero continuation ---
    # Day 5/6 needed a 6-step homotopy to switch drag on without the first
    # subproblem going infeasible. Virtual control is supposed to make that
    # unnecessary -- 0 disables the ramp, which is the default because it was
    # measured to work. See the aero-ramp experiment in scvx.py.
    aero_ramp_iters: int = 0

    # --- Reference seeding ---
    # "flip" is the Day 5 rate-limited sweep; "linear" is the naive straight
    # ramp from theta0 to 0 across the whole burn, which Day 5 recorded as
    # producing an infeasible first subproblem. Virtual control should absorb
    # that. Kept switchable so the claim stays testable.
    seed: str = "flip"

    # --- Solver ---
    # ECOS is unavailable on Python 3.13, so CLARABEL leads. Both are interior
    # point; SCS is a first-order fallback that tolerates worse conditioning
    # at the cost of accuracy.
    solver: str = "CLARABEL"
    solver_fallback: str = "SCS"
    solver_verbose: bool = False

    def summary(self) -> str:
        return "\n".join([
            "SCvx parameters",
            f"  max_iter          : {self.max_iter}",
            f"  eta_0             : {self.eta_0}",
            f"  eta_min/max       : [{self.eta_min}, {self.eta_max}]",
            f"  eta shrink/expand : {self.alpha_shrink} / {self.alpha_expand}",
            f"  rho ok/good       : {self.rho_ok} / {self.rho_good}",
            f"  w_vc              : {self.w_vc:.0e} "
            f"(grows x{self.w_vc_grow:.0f} to {self.w_vc_max:.0e})",
            f"  vc_tol / dyn_tol  : {self.vc_tol:.0e} / {self.dyn_tol:.0e}",
            f"  step_tol          : {self.step_tol:.0e}",
            f"  defect_tol        : {self.defect_tol}",
            f"  seed              : {self.seed}",
            f"  aero ramp         : "
            f"{'off' if not self.aero_ramp_iters else str(self.aero_ramp_iters) + ' steps'}",
            f"  solver            : {self.solver} -> {self.solver_fallback}",
        ])


if __name__ == "__main__":
    print(SCvxParams().summary())
