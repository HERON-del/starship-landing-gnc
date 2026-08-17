"""
Verification of the gyro-bias process and the bias-aware filter.

Tests:
    1. The true bias process behaves like a slow random walk
    2. The biased sensor is biased, and only on the rate channel
    3. The augmented filter converges on the bias
    4. It estimates rate better than the bias-blind filter, same readings
    5. Its 7x7 covariance stays well formed
    6. A zero-bias case is not penalised for carrying the extra state
    7. Closed-loop guidance still lands on the augmented estimate

Test 4 is the claim that matters and test 3 is not a substitute for it: a
filter can converge on a number and still steer no better. Day 11 spent a day
establishing that estimate quality and control quality are different things.

Run:  python tests/test_imu_bias.py
"""

import os
import sys
import warnings

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.imu_bias import (                                     # noqa: E402
    GyroBiasProcess, measure_attitude_biased, BiasedSensorSuite,
)
from src.ekf_bias import (                                     # noqa: E402
    BiasEKF, H_ATT_BIAS, H_NAV_BIAS, default_process_noise_bias,
)
from src.ekf import EKF                                        # noqa: E402
from src.sensors import SensorConfig, measure_attitude         # noqa: E402
from src.navigation_loop import run_navigation                 # noqa: E402
from src.dynamics_6dof import Vehicle6DoF                      # noqa: E402
from src.aero import AeroConfig                                # noqa: E402
from tests.test_dynamics import PASS, FAIL                     # noqa: E402

warnings.filterwarnings("ignore")

BIAS_DEG = 1.5


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name:<52}"
    if detail:
        msg += f" {detail}"
    print(msg)
    return bool(ok)


# ======================================================================
def test_bias_process():
    print("\nTEST 1 - The true bias process")
    p = GyroBiasProcess(b0=np.radians(1.0), sigma_walk=np.radians(0.01),
                        seed=0)
    walk = [p.step(0.05) for _ in range(2000)]
    ok = report("starts where it was told",
                abs(np.degrees(walk[0]) - 1.0) < 0.02)
    ok &= report("drifts slowly rather than jumping",
                 np.max(np.abs(np.diff(np.degrees(walk)))) < 0.02,
                 f"largest step {np.max(np.abs(np.diff(np.degrees(walk)))):.4f}"
                 f" deg/s")
    # A random walk spreads as sqrt(t); over 100 s at 0.01 deg/s/sqrt(s) that
    # is about 0.1 deg/s, so it should wander a little and not run away.
    drift = abs(np.degrees(walk[-1]) - 1.0)
    ok &= report("wanders, but stays in the same neighbourhood",
                 0.0 < drift < 0.6, f"{drift:.3f} deg/s after 100 s")
    return ok


# ======================================================================
def test_biased_sensor():
    print("\nTEST 2 - The biased sensor")
    cfg = SensorConfig()
    rng = np.random.default_rng(1)
    truth = np.array([0.0, 300.0, 0.0, -90.0, np.radians(10.0), 0.3, 1.2e5])
    b = np.radians(BIAS_DEG)

    got = np.array([measure_attitude_biased(truth, b, cfg, rng)
                    for _ in range(4000)])
    ok = report("rate channel carries the bias",
                abs(np.degrees(got[:, 1].mean() - truth[5]) - BIAS_DEG) < 0.05,
                f"offset {np.degrees(got[:, 1].mean() - truth[5]):.3f} deg/s")
    ok &= report("attitude channel does not",
                 abs(np.degrees(got[:, 0].mean() - truth[4])) < 0.05)
    ok &= report("noise is unchanged from the unbiased sensor",
                 abs(got[:, 1].std() - cfg.sigma_omega) / cfg.sigma_omega
                 < 0.1)
    unb = np.array([measure_attitude(truth, cfg, rng) for _ in range(2000)])
    ok &= report("and the unbiased sensor really is unbiased",
                 abs(np.degrees(unb[:, 1].mean() - truth[5])) < 0.05)
    return ok


# ======================================================================
def test_measurement_matrices():
    print("\nTEST 3 - The augmented measurement model")
    ok = report("nav sensor cannot see the bias",
                np.allclose(H_NAV_BIAS[:, 6], 0.0))
    ok &= report("rate row reads omega and bias together",
                 H_ATT_BIAS[1, 5] == 1.0 and H_ATT_BIAS[1, 6] == 1.0)
    ok &= report("attitude row reads only theta",
                 H_ATT_BIAS[0, 4] == 1.0 and H_ATT_BIAS[0, 6] == 0.0)
    Q = default_process_noise_bias()
    ok &= report("bias has small but non-zero process noise",
                 0.0 < Q[6, 6] < np.radians(1.0) ** 2,
                 f"sigma {np.degrees(np.sqrt(Q[6, 6])):.4f} deg/s per sqrt(s)")
    return ok


# ======================================================================
def _fly_filters(bias_deg=BIAS_DEG, seconds=12.0, dt=0.05, seed=3):
    """
    Both filters, identical readings, a vehicle simply rotating.

    Deliberately not a landing: a constant commanded torque gives the rate
    something to do, so the bias and the true rotation are separable in
    principle, and the test measures whether the filter separates them rather
    than whether the guidance loop happens to cooperate.
    """
    veh, aero = Vehicle6DoF(), AeroConfig(enabled=False)
    cfg = SensorConfig()
    proc = GyroBiasProcess(b0=np.radians(bias_deg),
                           sigma_walk=np.radians(0.01), seed=seed + 99)
    suite = BiasedSensorSuite(cfg, seed=seed, bias_process=proc)

    truth = np.array([0.0, 3000.0, 0.0, -30.0, 0.0, 0.0, veh.m_wet])
    x0 = truth[:6].copy()
    blind = EKF(x0.copy(), truth[6], veh, aero)
    aware = BiasEKF(x0.copy(), truth[6], veh, aero)

    sigma, delta = 3.0e6, np.radians(3.0)
    n = int(seconds / dt)
    errs = {"blind": [], "aware": []}
    b_hist = {"true": [], "est": [], "sigma": []}
    for k in range(n):
        t = (k + 1) * dt
        # Truth: rotate under the commanded torque.
        tau = sigma * veh.L_engine * np.sin(delta)
        truth[5] += tau / veh.I_pitch * dt
        truth[4] += truth[5] * dt
        suite.advance(t, dt)
        for f in (blind, aware):
            f.predict(sigma, delta, dt)
        got = suite.due(t, truth)
        if "att" in got:
            for f in (blind, aware):
                f.update_attitude(got["att"], cfg.R_att())
        if "nav" in got:
            for f in (blind, aware):
                f.update_nav(got["nav"], cfg.R_nav())
        errs["blind"].append(abs(blind.x[5] - truth[5]))
        errs["aware"].append(abs(aware.x[5] - truth[5]))
        b_hist["true"].append(proc.b)
        b_hist["est"].append(aware.bias)
        b_hist["sigma"].append(aware.bias_sigma)
    return {k: np.array(v) for k, v in errs.items()}, \
        {k: np.array(v) for k, v in b_hist.items()}, aware


def test_bias_converges():
    print("\nTEST 4 - The augmented filter converges on the bias")
    _, b, aware = _fly_filters()
    start_err = abs(np.degrees(b["est"][0] - b["true"][0]))
    end_err = abs(np.degrees(b["est"][-1] - b["true"][-1]))
    ok = report("estimate starts ignorant", start_err > 0.5 * BIAS_DEG,
                f"{start_err:.3f} deg/s out")
    ok &= report("and ends close to the truth", end_err < 0.25,
                 f"{end_err:.3f} deg/s out, true "
                 f"{np.degrees(b['true'][-1]):.3f}")
    ok &= report("uncertainty shrinks as it learns",
                 b["sigma"][-1] < 0.25 * b["sigma"][0],
                 f"{np.degrees(b['sigma'][0]):.3f} -> "
                 f"{np.degrees(b['sigma'][-1]):.3f} deg/s")
    ok &= report("...but does not collapse to certainty",
                 b["sigma"][-1] > 0.0)
    return ok


def test_beats_blind():
    """The claim that matters: better rate tracking, identical readings."""
    print("\nTEST 5 - Rate tracking against the bias-blind filter")
    errs, _, _ = _fly_filters()
    half = len(errs["blind"]) // 2
    blind_late = np.degrees(errs["blind"][half:]).mean()
    aware_late = np.degrees(errs["aware"][half:]).mean()
    ok = report("bias-blind filter carries a standing rate error",
                blind_late > 0.5 * BIAS_DEG,
                f"{blind_late:.3f} deg/s in the second half")
    ok &= report("augmented filter does not", aware_late < 0.5 * blind_late,
                 f"{aware_late:.3f} deg/s, {blind_late / max(aware_late, 1e-9):.1f}x better")
    return ok


# ======================================================================
def test_covariance():
    print("\nTEST 6 - The 7x7 covariance stays well formed")
    _, _, aware = _fly_filters()
    P = aware.P
    ok = report("symmetric", float(np.abs(P - P.T).max()) < 1e-9)
    ok &= report("positive semi-definite",
                 float(np.linalg.eigvalsh(P).min()) > -1e-9,
                 f"smallest eigenvalue {np.linalg.eigvalsh(P).min():.2e}")
    ok &= report("bias row is coupled to the rate row",
                 abs(P[5, 6]) > 0.0,
                 "the filter has learned they trade off")
    return ok


def test_no_bias_not_penalised():
    """Carrying the extra state should not hurt when there is no bias."""
    print("\nTEST 7 - A zero-bias case is not penalised")
    errs, b, _ = _fly_filters(bias_deg=0.0)
    half = len(errs["blind"]) // 2
    blind_late = np.degrees(errs["blind"][half:]).mean()
    aware_late = np.degrees(errs["aware"][half:]).mean()
    ok = report("augmented filter is no worse with nothing to estimate",
                aware_late < max(3.0 * blind_late, 0.2),
                f"{aware_late:.3f} vs {blind_late:.3f} deg/s")
    ok &= report("and does not invent a bias",
                 abs(np.degrees(b["est"][-1])) < 0.6,
                 f"estimated {np.degrees(b['est'][-1]):+.3f} deg/s")
    return ok


# ======================================================================
def test_closed_loop():
    print("\nTEST 8 - Guidance lands on the augmented estimate")
    r = run_navigation(mode="ekf", wind_seed=7, sensor_seed=3,
                       bias0_deg_s=BIAS_DEG, bias_aware=True, verbose=False)
    ok = report("run completed", r.get("status") == "flown")
    if not ok:
        return False
    ok &= report("the vehicle came down rather than climbing away",
                 r["sim_time"] < 9.0, f"{r['sim_time']:.2f}s of flight")
    ok &= report("propellant left at touchdown", r["margin"] > 0,
                 f"{r['margin']:,.0f} kg")
    ok &= report("bias resolved during the descent",
                 np.degrees(r["b_err_final"]) < 0.6,
                 f"{np.degrees(r['b_err_final']):.3f} deg/s out of "
                 f"{np.degrees(r['b_true_final']):.3f}")
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 12 - IMU BIAS ESTIMATION VERIFICATION")
    print("=" * 70)
    oks = [test_bias_process(), test_biased_sensor(),
           test_measurement_matrices(), test_bias_converges(),
           test_beats_blind(), test_covariance(),
           test_no_bias_not_penalised(), test_closed_loop()]
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED" if all(oks) else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
