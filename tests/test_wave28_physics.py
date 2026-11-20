"""Wave 2.7 / 2.8 physics-guided Flow Matching unit tests.

Locks:
* E26 sampling-path fix: per-sample normalisation in, de-normalisation out;
* Wave 2.8 causal physical forcing (inverse barometer, coast-normal wind);
* the implied-clean-estimate identity used by the physical penalty;
* the paired sign-flip / bootstrap event statistics.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


wave27 = _load("wave27_physics_fm", ROOT / "scripts" / "wave27_physics_fm.py")
wave28 = _load("wave28_physics_fm", ROOT / "scripts" / "wave28_physics_fm.py")

CONTEXT, HORIZON = 96, 72
N = 2000


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "msl": 101000.0 + np.cumsum(rng.normal(0, 5, N)),
        "u10": rng.normal(0, 3, N),
        "v10": rng.normal(0, 3, N),
        "speed": np.abs(rng.normal(0, 3, N)),
    })


# ---------------------------------------------------------------------------
# E26 sampling-path regression test
# ---------------------------------------------------------------------------

class _IdentityFM:
    """Stub generator: returns its (normalised) history input as the
    sample. If ``sample_windows`` normalises correctly the round trip
    reproduces the raw input exactly."""

    def sample(self, hn, net, seed=0, physics=None):
        return hn


def test_sample_windows_normalises_history_and_de_normalises_output():
    rng = np.random.default_rng(3)
    hist = (rng.normal(0, 0.2, (6, CONTEXT)) + 0.35).astype(np.float32)
    model = {"net": None, "fm": _IdentityFM(), "physics_dim": 0}
    out = wave27.sample_windows(model, hist, np.zeros((6, 0)),
                                np.arange(6), draws=1)
    np.testing.assert_allclose(out[0], hist, atol=1e-5)


def test_sample_windows_wave28_same_contract():
    rng = np.random.default_rng(4)
    hist = (rng.normal(0, 0.15, (4, CONTEXT)) - 0.1).astype(np.float32)
    model = {"net": None, "fm": _IdentityFM(), "physics_dim": 0}
    out = wave28.sample_windows(model, hist, np.zeros((4, 0)),
                                np.arange(4), draws=2)
    np.testing.assert_allclose(out[0], hist, atol=1e-5)
    np.testing.assert_allclose(out[1], hist, atol=1e-5)


# ---------------------------------------------------------------------------
# Wave 2.8 causal physical forcing
# ---------------------------------------------------------------------------

def test_physics_raw_ignores_future_horizon():
    base = _frame()
    for s in (500, 900):
        starts = np.array([s])
        perturbed = base.copy()
        perturbed.iloc[s + CONTEXT:] = 1.0e6
        a = wave28.build_physics_raw(starts, base, "ny_bight")
        b = wave28.build_physics_raw(starts, perturbed, "ny_bight")
        for k in ("ib_tend_m", "tau_n24", "tau_n72", "wn24"):
            np.testing.assert_allclose(a[k], b[k], atol=1e-8)


def test_ib_sign_pressure_drop_raises_water():
    """Pressure below the trailing 72 h mean -> positive IB residual."""
    df = _frame()
    starts = np.array([500])
    hi = starts[0] + CONTEXT
    df.iloc[hi - 72:hi, df.columns.get_loc("msl")] = 102000.0
    df.iloc[hi - 1, df.columns.get_loc("msl")] = 100000.0
    out = wave28.build_physics_raw(starts, df, "ny_bight")
    assert out["ib_tend_m"][0] > 0
    # magnitude: -(100000 - mean72)/100 hPa * (-0.01 m/hPa); the last
    # hour is inside the 72 h window so mean72 = (71*102000+100000)/72
    mean72 = (71 * 102000.0 + 100000.0) / 72.0
    np.testing.assert_allclose(out["ib_tend_m"][0],
                               -0.01 * (100000.0 - mean72) / 100.0,
                               atol=1e-6)


def test_coast_normal_onshore_wind_gives_positive_stress():
    """Wind blowing onshore (aligned with the onshore unit vector) must
    produce positive coast-normal stress; the mirrored offshore wind
    must produce the exact negative."""
    region = "ny_bight"
    ne, nn = wave28.COAST_NORMAL_ONSHORE[region]
    nrm = float(np.hypot(ne, nn))
    ne, nn = ne / nrm, nn / nrm
    starts = np.array([500])
    hi = starts[0] + CONTEXT

    def frame(wind_u, wind_v):
        df = _frame()
        df.iloc[hi - 24:hi, df.columns.get_loc("u10")] = wind_u
        df.iloc[hi - 24:hi, df.columns.get_loc("v10")] = wind_v
        df.iloc[hi - 24:hi, df.columns.get_loc("speed")] = np.hypot(
            wind_u, wind_v)
        return df

    onshore = wave28.build_physics_raw(starts, frame(10 * ne, 10 * nn),
                                       region)
    offshore = wave28.build_physics_raw(starts, frame(-10 * ne, -10 * nn),
                                        region)
    assert onshore["tau_n24"][0] > 0
    np.testing.assert_allclose(onshore["tau_n24"][0],
                               -offshore["tau_n24"][0], atol=1e-9)
    # magnitude: rho_a * Cd * |W| * W_n = 1.225*1.5e-3*10*10
    np.testing.assert_allclose(onshore["tau_n24"][0],
                               1.225 * 1.5e-3 * 100.0, atol=1e-9)


def test_regression_train_only_and_predict():
    """OLS fitted on train windows predicts Delta from causal forcing."""
    rng = np.random.default_rng(0)
    m = 500
    raw = {"ib_tend_m": rng.normal(0, 0.05, m),
           "tau_n24": np.abs(rng.normal(0, 0.1, m)),
           "tau_n72": np.abs(rng.normal(0, 0.1, m))}
    y = 0.1 + 0.8 * raw["ib_tend_m"] + 0.5 * raw["tau_n24"] \
        + rng.normal(0, 1e-6, m)
    reg = wave28.fit_physics_regression(["ib_tend_m", "tau_n24"], raw, y)
    assert reg["r2_train"] > 0.999
    pred = wave28.physics_predict(reg, raw)
    np.testing.assert_allclose(pred, y, atol=1e-3)
    np.testing.assert_allclose(reg["coef"], [0.1, 0.8, 0.5], atol=1e-3)


# ---------------------------------------------------------------------------
# Penalty mechanics
# ---------------------------------------------------------------------------

def test_x1_hat_identity():
    """x_t + (1-t)*v recovers x_1 exactly when v equals the true velocity."""
    rng = np.random.default_rng(1)
    B = 8
    x0 = torch.randn(B, 1, HORIZON)
    x1 = torch.randn(B, 1, HORIZON)
    t = torch.rand(B, 1, 1) * 0.9 + 0.05
    xt = (1 - t) * x0 + t * x1
    v = x1 - x0
    x1_hat = xt + (1 - t) * v
    torch.testing.assert_close(x1_hat, x1)


def test_penalty_arms_differ_only_by_loss_terms():
    """Arms era5_cond..combined_tail share conditioning; only penalty
    predictors differ (and combined_tail adds tail weighting)."""
    assert wave28.PENALTY_PREDICTORS["era5_cond"] == []
    assert wave28.PENALTY_PREDICTORS["pressure_prior"] == ["ib_tend_m"]
    assert wave28.PENALTY_PREDICTORS["wind_prior"] == ["tau_n24", "tau_n72"]
    assert wave28.PENALTY_PREDICTORS["combined_prior"] == \
        wave28.PENALTY_PREDICTORS["combined_tail"]
    assert set(wave28.ARMS) == {
        "standard", "era5_cond", "pressure_prior", "wind_prior",
        "combined_prior", "combined_tail"}


# ---------------------------------------------------------------------------
# Paired event statistics
# ---------------------------------------------------------------------------

def test_signflip_all_positive_is_significant():
    d = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
    out = wave28.paired_event_stats(d, seed=0)
    # exact two-sided: only the all+ and all- patterns match |mean|>=|obs|
    assert out["signflip_p"] <= 2 / 2 ** 5 + 1e-12
    assert out["ci95"][0] > 0


def test_signflip_symmetric_noise_is_null():
    rng = np.random.default_rng(0)
    d = rng.normal(0, 0.1, 40)
    out = wave28.paired_event_stats(d, seed=0)
    assert out["signflip_p"] > 0.05


def test_signflip_exact_small_n():
    d = np.array([1.0, -1.0, 1.0, -1.0])
    out = wave28.paired_event_stats(d, seed=0)
    assert out["signflip_p"] == 1.0  # mean zero -> p maximal
    np.testing.assert_allclose(out["mean_delta"], 0.0)
