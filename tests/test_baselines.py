"""Tests for PyTorch forecasting baselines (GRU, TCN, data utilities).

All use synthetic data, no GPU required, run under CPU BLAS caps.
"""
import numpy as np
import torch

from tidetail.baselines import (
    GRUForecaster,
    TCNForecaster,
    make_windows,
    seasonal_baseline,
    train_one_epoch,
    train_only_scaler,
)

torch.set_num_threads(2)


# ==========================================================================
# Data utilities
# ==========================================================================

def test_make_windows_basic():
    """Sine wave hourly series produces correct 96→72 windows."""
    n = 1000
    t = np.arange(n)
    series = np.sin(2 * np.pi * t / 24.0) + 0.1 * np.random.default_rng(0).standard_normal(n)
    hist, targ, missing = make_windows(series, lookback=96, horizon=72, stride=1)

    expected_n = n - 96 - 72 + 1  # = 833
    assert hist.shape == (expected_n, 96)
    assert targ.shape == (expected_n, 72)
    assert missing.shape == (expected_n, 96)
    assert hist.dtype == np.float64
    # No NaN in input → no missing
    assert not missing.any()


def test_make_windows_excludes_nan_targets():
    """Window where ANY target cell is NaN must be excluded."""
    n = 300
    series = np.sin(np.arange(n) / 50.0)
    # Insert a NaN in the middle targets
    series[200] = np.nan  # this creates some invalid windows

    _hist, targ, _missing = make_windows(series, lookback=96, horizon=72, stride=1)
    assert np.isfinite(targ).all()  # targets all finite after filtering


def test_make_windows_stride():
    """Larger stride = fewer windows."""
    n = 500
    series = np.sin(np.arange(n) / 50.0)
    h1, _t1, _ = make_windows(series, lookback=96, horizon=72, stride=1)
    h5, _t5, _ = make_windows(series, lookback=96, horizon=72, stride=5)
    # Approximately 5x fewer; exact ratio depends on total length
    assert h5.shape[0] <= h1.shape[0] // 4 + 2  # allow ±1 window tolerance


def test_train_only_scaler():
    train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    mean, std = train_only_scaler(train)
    assert abs(float(mean) - 5.5) < 0.01
    assert abs(float(std) - np.std(train)) < 0.01

    # Edge: constant series
    const_mean, const_std = train_only_scaler(np.ones(100) * 3.0)
    assert float(const_mean) == 3.0
    assert float(const_std) == 1.0  # std=0 replaced with 1.0


# ==========================================================================
# GRUForecaster
# ==========================================================================

def test_gru_output_shape():
    """GRU produces correct (B, 72, 1) output from (B, 96) input."""
    model = GRUForecaster(input_dim=1, hidden_size=32, horizon=72, num_layers=2)
    model.eval()
    h = torch.randn(8, 96)
    with torch.no_grad():
        out = model.predict(h)
    assert out.shape == (8, 72, 1)
    assert torch.isfinite(out).all()


def test_gru_training_decreases_loss():
    """GRU trained on synthetic data should decrease loss."""
    torch.manual_seed(42)
    n = 200
    history = torch.randn(n, 96) * 0.1
    # Target = mean of last 24 history steps (simple deterministic mapping)
    target = history[:, -24:].mean(dim=1, keepdim=True).expand(-1, 72)

    model = GRUForecaster(input_dim=1, hidden_size=24, horizon=72, num_layers=1, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    initial_loss = float(model.forward_loss(history, target))
    for _ in range(10):
        train_one_epoch(model, optimizer, history, target, batch_size=32)
    final_loss = float(model.forward_loss(history, target))

    assert final_loss < initial_loss * 0.8, f"loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"


def test_gru_with_missing_mask():
    """GRU should handle explicit missing_mask."""
    model = GRUForecaster(input_dim=1, hidden_size=16, horizon=72)
    h = torch.randn(4, 96)
    missing = torch.zeros(4, 96, dtype=torch.bool)
    missing[:, 10:20] = True
    with torch.no_grad():
        out1 = model.predict(h, missing=None)
        out2 = model.predict(h, missing=missing)
    # With mask information, outputs can differ; both must be finite
    assert torch.isfinite(out1).all()
    assert torch.isfinite(out2).all()


def test_gru_param_count():
    """GRU param count scales with hidden_size; stays reasonable."""
    model = GRUForecaster(input_dim=1, hidden_size=32, horizon=72, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    assert 2000 <= n_params <= 30000, f"param count {n_params} out of expected range"


# ==========================================================================
# TCNForecaster
# ==========================================================================

def test_tcn_output_shape():
    model = TCNForecaster(input_dim=1, horizon=72, channels=(32, 32, 32, 32))
    model.eval()
    h = torch.randn(8, 96)
    with torch.no_grad():
        out = model.predict(h)
    assert out.shape == (8, 72, 1)
    assert torch.isfinite(out).all()


def test_tcn_training_decreases_loss():
    torch.manual_seed(42)
    n = 200
    history = torch.randn(n, 96) * 0.1
    target = history[:, -24:].mean(dim=1, keepdim=True).expand(-1, 72)

    model = TCNForecaster(input_dim=1, horizon=72, channels=(16, 16, 16))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_loss = float(model.forward_loss(history, target))
    for _ in range(20):
        train_one_epoch(model, optimizer, history, target, batch_size=32)
    final_loss = float(model.forward_loss(history, target))

    assert final_loss < initial_loss * 0.5, f"loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"


def test_tcn_with_missing_mask():
    model = TCNForecaster(input_dim=1, horizon=72, channels=(16, 16, 16))
    h = torch.randn(4, 96)
    missing = torch.zeros(4, 96, dtype=torch.bool)
    missing[:, 10:20] = True
    with torch.no_grad():
        out = model.predict(h, missing=missing)
    assert torch.isfinite(out).all()


def test_tcn_param_count():
    model = TCNForecaster(input_dim=1, horizon=72, channels=(32, 32, 32, 32))
    n_params = sum(p.numel() for p in model.parameters())
    assert 10000 <= n_params <= 40000, f"param count {n_params} out of expected range"


# ==========================================================================
# Rule-based baselines still work
# ==========================================================================

def test_seasonal_baseline_fallback():
    """seasonal_baseline with history < period falls back to persistence."""
    h = np.array([1.0, 2.0, 3.0])
    result = seasonal_baseline(h, horizon=5, period=144)
    assert len(result) == 5
    assert np.allclose(result, 3.0)  # persistence: last value


def test_persistence_short():
    from tidetail.baselines import persistence
    out = persistence([5.0], horizon=3)
    assert list(out) == [5.0, 5.0, 5.0]
