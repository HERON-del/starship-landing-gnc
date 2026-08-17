"""
Verification of the sensors, the EKF, and guidance flown on its estimate.

Tests:
    1. Sensor models have the noise they advertise
    2. The filter tracks truth through an open-loop descent
    3. The filter beats the raw measurements it is fed
    4. Covariance stays symmetric positive semi-definite
    5. The numerical Jacobian matches a known-linear case
    6. Guidance still lands when flown on the estimate
    7. The filter beats no filter, on identical noise

Test 3 is the one that justifies the whole day. A filter that merely tracks
truth is not interesting -- copying the last nav reading also tracks truth,
badly. What matters is that fusing beats the best thing available without it.

Run:  python tests/test_ekf.py
"""

import os
import sys
import warnings

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.sensors import (                                      # noqa: E402
    SensorConfig, SensorSuite, measure_nav, measure_attitude, H_NAV, H_ATT,
)
from src.ekf import EKF, default_process_noise                 # noqa: E402
from src.navigation_loop import run_navigation, compare        # noqa: E402
from src.closed_loop import _plan, _as_state, _fly, WindGusts  # noqa: E402
from src.dynamics_6dof import Vehicle6DoF                      # noqa: E402
from src.aero import AeroConfig                                # noqa: E402
from tests.test_dynamics import PASS, FAIL                     # noqa: E402

warnings.filterwarnings("ignore")


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name:<52}"
    if detail:
        msg += f" {detail}"
    print(msg)
    return bool(ok)


# ======================================================================
def test_sensors():
    print("\nTEST 1 - Sensor models")
    cfg = SensorConfig()
    rng = np.random.default_rng(0)
    truth = np.array([12.0, 300.0, -4.0, -90.0, np.radians(15.0), 0.2, 1.2e5])

    nav = np.array([measure_nav(truth, cfg, rng) for _ in range(4000)])
    att = np.array([measure_attitude(truth, cfg, rng) for _ in range(4000)])
    ok = report("nav readings unbiased",
                np.allclose(nav.mean(0), truth[:4], atol=0.25),
                f"mean offset {np.abs(nav.mean(0) - truth[:4]).max():.3f}")
    ok &= report("nav noise matches its spec",
                 np.allclose(nav.std(0),
                             [cfg.sigma_x, cfg.sigma_z, cfg.sigma_vx,
                              cfg.sigma_vz], rtol=0.12),
                 f"{np.round(nav.std(0), 3).tolist()}")
    ok &= report("attitude noise matches its spec",
                 np.allclose(att.std(0), [cfg.sigma_theta, cfg.sigma_omega],
                             rtol=0.12))
    ok &= report("rate gyro carries its bias when given one",
                 abs(np.mean([measure_attitude(
                     truth, SensorConfig(omega_bias=0.05), rng)[1]
                     for _ in range(2000)]) - (truth[5] + 0.05)) < 0.02)

    suite = SensorSuite(cfg, seed=1)
    got = [set(suite.due(t, truth)) for t in np.arange(0.0, 0.41, 0.05)]
    n_nav = sum(1 for g in got if "nav" in g)
    n_att = sum(1 for g in got if "att" in g)
    ok &= report("each instrument arrives at its own rate",
                 n_att > n_nav, f"{n_att} attitude vs {n_nav} nav in 0.4 s")
    return ok


# ======================================================================
def _open_loop_truth(seed=5, steps=100, dt=0.05):
    """
    Fly one plan blind and return truth samples and the commands.

    Sampled at the filter's own rate rather than the guidance rate. Handing the
    sensor model end-of-interval truth while the filter sits at the start of it
    injects a systematic lead error that has nothing to do with the estimator.
    """
    veh, aero = Vehicle6DoF(), AeroConfig()
    truth = np.array([0.0, 420.0, 0.0, -130.0, np.radians(25.0), 0.0,
                      veh.m_wet])
    plan = _plan(veh, aero, 40, 2 * 420.0 / 130.0, _as_state(truth), 30)
    gusts = WindGusts(6.0, 2.0, 2.0, seed)
    out, t = [(0.0, truth.copy())], 0.0
    for _ in range(steps):
        if t >= plan["t_f"] - 1e-9:
            break
        step = min(dt, plan["t_f"] - t)
        y = _fly(truth, plan, t, step, veh, aero, gusts.step(step))
        if np.any(y[:, 1] <= 0.0):
            break
        truth = y[-1]
        t += step
        out.append((t, truth.copy()))
    return plan, out, veh, aero


def test_tracking():
    print("\nTEST 2 - The filter tracks truth")
    cfg = SensorConfig()
    plan, samples, veh, aero = _open_loop_truth()
    suite = SensorSuite(cfg, seed=2)
    t0, s0 = samples[0]
    first = suite.due(0.0, s0)
    x0 = s0[:6].copy()
    if "nav" in first:
        x0[:4] = first["nav"]
    ekf = EKF(x0, s0[6], veh, aero)

    dt_ctrl = plan["t_f"] / len(plan["sigma"])
    errs = []
    for (ta, _), (tb, sb) in zip(samples[:-1], samples[1:]):
        k = int(np.clip(ta / dt_ctrl, 0, len(plan["sigma"]) - 1))
        sig, dl = float(plan["sigma"][k]), float(plan["delta"][k])
        ekf.predict(sig, dl, tb - ta)
        got = suite.due(tb, sb)
        if "att" in got:
            ekf.update_attitude(got["att"], cfg.R_att())
        if "nav" in got:
            ekf.update_nav(got["nav"], cfg.R_nav())
        errs.append(np.hypot(*(ekf.x[:2] - sb[:2])))
    errs = np.array(errs)
    ok = report("estimate stays near truth", errs.max() < 15.0,
                f"max {errs.max():.2f} m, mean {errs.mean():.2f} m")
    ok &= report("error does not grow without bound",
                 errs[-1] < 4.0 * max(errs.mean(), 0.5),
                 f"final {errs[-1]:.2f} m")
    return ok, errs


# ======================================================================
def test_beats_raw(ekf_errs):
    """The filter against the best thing available without one."""
    print("\nTEST 3 - The filter beats the readings it is fed")
    cfg = SensorConfig()
    _, samples, _, _ = _open_loop_truth()
    rng = np.random.default_rng(2)
    raw = np.array([np.hypot(*(measure_nav(s, cfg, rng)[:2] - s[:2]))
                    for _, s in samples[1:]])
    ok = report("mean error lower than raw nav readings",
                ekf_errs.mean() < raw.mean(),
                f"EKF {ekf_errs.mean():.2f} m vs raw {raw.mean():.2f} m")
    ok &= report("worst case lower than raw nav readings",
                 ekf_errs.max() < raw.max(),
                 f"EKF {ekf_errs.max():.2f} m vs raw {raw.max():.2f} m")
    return ok


# ======================================================================
def test_covariance():
    print("\nTEST 4 - Covariance stays well formed")
    cfg = SensorConfig()
    veh, aero = Vehicle6DoF(), AeroConfig()
    truth = np.array([0.0, 420.0, 0.0, -130.0, np.radians(25.0), 0.0,
                      veh.m_wet])
    ekf = EKF(truth[:6], truth[6], veh, aero)
    rng = np.random.default_rng(4)
    worst_asym, worst_eig = 0.0, np.inf
    for _ in range(400):
        ekf.predict(3.0e6, np.radians(2.0), 0.05)
        ekf.update_attitude(measure_attitude(truth, cfg, rng), cfg.R_att())
        ekf.update_nav(measure_nav(truth, cfg, rng), cfg.R_nav())
        worst_asym = max(worst_asym,
                         float(np.abs(ekf.P - ekf.P.T).max()))
        worst_eig = min(worst_eig, float(np.linalg.eigvalsh(ekf.P).min()))
    ok = report("covariance stays symmetric", worst_asym < 1e-9,
                f"worst asymmetry {worst_asym:.2e}")
    ok &= report("covariance stays positive semi-definite",
                 worst_eig > -1e-9, f"smallest eigenvalue {worst_eig:.2e}")
    ok &= report("uncertainty settles rather than growing",
                 ekf.position_sigma() < 5.0,
                 f"final position sigma {ekf.position_sigma():.3f} m")
    return ok


# ======================================================================
def test_jacobian():
    """
    Against a case whose answer is known.

    With the engine off and no air, the dynamics are exactly linear: position
    integrates velocity, attitude integrates rate, and nothing else moves. The
    Jacobian of an RK4 step through that must be the exact discrete transition
    matrix, so any error here is the differencing rather than the model.
    """
    print("\nTEST 5 - Numerical Jacobian against a known-linear case")
    veh = Vehicle6DoF()
    # AeroConfig(enabled=False), not aero=None: the constructor substitutes a
    # default config for None, so passing None leaves drag switched on and the
    # case is not linear at all.
    ekf = EKF(np.array([0.0, 500.0, 3.0, -40.0, 0.1, 0.02]), veh.m_dry,
              veh, aero=AeroConfig(enabled=False))
    dt = 0.1
    F = ekf._jacobian(ekf.x, 0.0, 0.0, dt)
    exact = np.eye(6)
    exact[0, 2] = exact[1, 3] = exact[4, 5] = dt
    err = float(np.abs(F - exact).max())
    ok = report("matches the exact transition matrix", err < 1e-6,
                f"largest difference {err:.2e}")
    ok &= report("is finite everywhere", bool(np.all(np.isfinite(F))))
    return ok


# ======================================================================
def test_guidance_on_estimate():
    print("\nTEST 6 - Guidance still lands when flown on the estimate")
    r = run_navigation(mode="ekf", wind_seed=7, sensor_seed=3, verbose=False)
    ok = report("run completed", r.get("status") == "flown")
    if not ok:
        return False
    ok &= report("reached the ground", r["margin"] > 0,
                 f"{r['margin']:,.0f} kg left")
    ok &= report("estimation error stayed bounded",
                 r["max_est_pos_err"] < 25.0,
                 f"max {r['max_est_pos_err']:.2f} m, mean "
                 f"{r['mean_est_pos_err']:.2f} m")
    ok &= report("propellant close to the perfect-sensing case",
                 r["fuel"] < 9000.0, f"{r['fuel']:,.0f} kg")
    return ok


# ======================================================================
def test_filter_beats_no_filter():
    """
    The claim Day 10 left open, over several noise realisations.

    Day 10 fed raw estimates straight to the solver and recorded the damage.
    This is the same comparison with a filter in the path, and it is paired:
    every mode flies identical wind and identical sensor noise.
    """
    print("\nTEST 7 - Filtering beats not filtering, paired")
    wins_miss, wins_est, n = 0, 0, 0
    worst_ekf = worst_naive = 0.0
    worst_fuel_ekf = worst_fuel_naive = 0.0
    for s in range(4):
        runs = compare(seed=s, sensor_seed=s + 20, verbose=False)
        e, v = runs["ekf"], runs["naive"]
        if e.get("status") != "flown" or v.get("status") != "flown":
            continue
        n += 1
        wins_miss += e["miss"] < v["miss"]
        wins_est += e["mean_est_pos_err"] < v["mean_est_pos_err"]
        worst_ekf = max(worst_ekf, e["miss"])
        worst_naive = max(worst_naive, v["miss"])
        worst_fuel_ekf = max(worst_fuel_ekf, e["fuel"])
        worst_fuel_naive = max(worst_fuel_naive, v["fuel"])
    ok = report("comparisons available", n >= 3, f"{n} seeds")
    if not n:
        return False
    ok &= report("filtered estimate is better every time", wins_est == n,
                 f"{wins_est}/{n}")

    # What filtering does *not* buy, measured rather than assumed. A better
    # estimate does not give a better landing here -- the naive loop lands
    # nearer in three of four seeds. Its error is close to zero-mean sensor
    # noise, and successive replans average much of it out; the filter's error
    # is correlated, because a filter lags, and a lag biases every replan the
    # same way. Asserting the opposite would be asserting something false.
    ok &= report("...though it does not follow that it lands nearer",
                 True, f"nearer in only {wins_miss}/{n} seeds -- see the "
                       f"process-noise sweep")

    # This assertion used to read "filtering bounds the worst case", on the
    # strength of a 210 m unfiltered miss against 6.7 m filtered. That 210 m
    # was a bug in the guidance loop, not the estimator: once a plan's horizon
    # was spent the loop kept flying its last control, and for a landing plan
    # that is a lit engine, so the vehicle climbed away from the pad. Day 12
    # found it and fixed it, and the unfiltered worst case fell to 4.4 m -- so
    # the tail the filter appeared to be protecting against was mostly mine.
    #
    # What survives is the estimate, and the conclusion it supports is the
    # stronger one Day 11 already drew: a better estimate is not a better
    # landing, because the binding error is guidance rate rather than
    # knowledge of the state.
    ok &= report("...nor does it bound the worst case",
                 True, f"worst miss {worst_ekf:.1f} m filtered vs "
                       f"{worst_naive:.1f} m unfiltered")
    ok &= report("both keep propellant in hand",
                 max(worst_fuel_ekf, worst_fuel_naive) < 12000.0,
                 f"{worst_fuel_ekf:,.0f} kg vs {worst_fuel_naive:,.0f} kg")
    return ok


# ======================================================================
def main():
    print("=" * 70)
    print("DAY 11 - NAVIGATION AND STATE ESTIMATION VERIFICATION")
    print("=" * 70)

    ok1 = test_sensors()
    ok2, errs = test_tracking()
    ok3 = test_beats_raw(errs)
    ok4 = test_covariance()
    ok5 = test_jacobian()
    ok6 = test_guidance_on_estimate()
    ok7 = test_filter_beats_no_filter()

    all_ok = all([ok1, ok2, ok3, ok4, ok5, ok6, ok7])
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
