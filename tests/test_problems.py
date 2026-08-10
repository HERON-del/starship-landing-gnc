"""
Verification for the problem registry and every registered solver.

Run:  python tests/test_problems.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from gnc import registry  # noqa: E402

registry.load_all()

ok = True
print("=" * 68)
print("PROBLEM REGISTRY CHECK")
print("=" * 68)

for prob in registry.all_problems():
    print(f"\n[{prob.phase}] {prob.title}  ({prob.slug})")
    print(f"  params: {len(prob.params())}")

    traj = prob.solve(prob.defaults())
    print(f"  status: {traj.status}  solver: {traj.solver}  "
          f"cost: {traj.cost:.4f}" if traj.cost is not None else
          f"  status: {traj.status}")

    if not traj.feasible:
        print("  [FAIL] default parameters should be feasible")
        for n in traj.notes:
            print(f"         {n}")
        ok = False
        continue

    pos = np.asarray(traj.position)
    vel = np.asarray(traj.velocity)
    thr = np.asarray(traj.thrust)
    att = np.asarray(traj.attitude)
    n_state = len(traj.t_state)

    checks = [
        ("position shape", pos.shape == (n_state, 3)),
        ("velocity shape", vel.shape == (n_state, 3)),
        ("attitude shape", att.shape == (n_state, 4)),
        ("control length", thr.shape == (len(traj.t_control), 3)),
        ("quaternions normalised", np.allclose(np.linalg.norm(att, axis=1), 1.0, atol=1e-6)),
        ("no NaNs", np.isfinite(pos).all() and np.isfinite(thr).all()),
    ]

    # Only optimisers promise to arrive at the pad at rest. An open-loop
    # simulation is allowed to fly the vehicle into the ground.
    if prob.enforces_terminal_state:
        checks += [
            ("lands at pad", float(np.linalg.norm(pos[-1])) < 1e-3),
            ("lands at rest", float(np.linalg.norm(vel[-1])) < 1e-3),
        ]
    else:
        checks += [
            ("reaches the ground or times out", float(pos[-1][1]) >= -1e-6),
            ("monotonic time", bool(np.all(np.diff(traj.t_state) > 0))),
        ]
    for name, passed in checks:
        print(f"    [{'OK' if passed else 'FAIL'}] {name}")
        ok &= passed

    for s in traj.series:
        expect = n_state if s.on == "state" else len(traj.t_control)
        good = len(s.values) == expect
        print(f"    [{'OK' if good else 'FAIL'}] series '{s.key}' length")
        ok &= good

# Infeasibility must be reported, not crash.
print("\n" + "-" * 68)
p1 = registry.get("landing-1d")
bad = p1.solve({**p1.defaults(), "a_max": 6.0})
print(f"Infeasible probe (a_max=6): status={bad.status} feasible={bad.feasible}")
ok &= not bad.feasible and bad.status != "error"
print(f"    [{'OK' if not bad.feasible else 'FAIL'}] infeasibility handled gracefully")

print("\n" + "=" * 68)
print("ALL CHECKS PASSED" if ok else "SOMETHING FAILED - see above")
print("=" * 68)
sys.exit(0 if ok else 1)
