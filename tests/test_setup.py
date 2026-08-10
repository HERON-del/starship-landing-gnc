"""
Day 1 environment verification.
Run this to confirm every library installed correctly.

NOTE: the guide lists `ecos` here. This machine runs Python 3.13, for which
ecos ships no Windows wheel (and ecos is unmaintained since 2023), so we check
for `clarabel` instead -- CVXPY's bundled successor to ECOS.
"""

import sys

print("=" * 60)
print("STARSHIP LANDING GNC - ENVIRONMENT CHECK")
print("=" * 60)
print(f"Python version: {sys.version.split()[0]}")
print("-" * 60)

packages = [
    "numpy",
    "scipy",
    "matplotlib",
    "cvxpy",
    "clarabel",
    "osqp",
    "pandas",
    "ambiance",
]

all_ok = True

for name in packages:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  [OK]   {name:<14} {version}")
    except ImportError as e:
        print(f"  [FAIL] {name:<14} {e}")
        all_ok = False

print("-" * 60)

# Check which solvers CVXPY can see
try:
    import cvxpy as cp

    print(f"Available CVXPY solvers: {cp.installed_solvers()}")
except Exception as e:
    print(f"Could not query solvers: {e}")
    all_ok = False

print("=" * 60)
print("ALL SYSTEMS GO" if all_ok else "SOMETHING FAILED - see above")
print("=" * 60)
