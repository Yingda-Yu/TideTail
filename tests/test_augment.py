"""Tests for training-set augmentation modules (A1 + A2).

All use synthetic numpy data, run under CPU BLAS caps, no real gauge needed.
"""
import numpy as np
import pytest

from tidetail.augment import (
    circular_tidal_shift,
    combine_augmentations,
    extreme_balance_weights,
    residual_noise_augment,
)


def test_extreme_balance_weights_assigns_higher_weight():
    """Extreme-target windows get higher sampling weights."""
    targets = np.array([
        [0.5, 0.3, 0.1],   # low — not extreme
        [2.0, 2.5, 3.0],   # all extreme (>= q95=2.0)
        [0.1, 0.2, 2.1],   # mixed — has extreme
        [0.0, 0.0, 0.0],   # zero
    ])
    weights = extreme_balance_weights(targets, train_q95=2.0, extreme_weight=3.0)
    assert weights.shape == (4,)
    assert weights[0] == 1.0   # low
    assert weights[1] == 3.0   # all extreme
    assert weights[2] == 3.0   # mixed but has extreme
    assert weights[3] == 1.0   # zero


def test_extreme_balance_weight_boundaries():
    """extreme_weight < 1.0 must raise."""
    targets = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError):
        extreme_balance_weights(targets, train_q95=1.5, extreme_weight=0.5)


def test_residual_noise_augment_does_not_modify_targets():
    """Noise augmentation perturbs history ONLY; targets are preserved."""
    rng = np.random.default_rng(42)
    h = rng.standard_normal((100, 96))
    t = rng.standard_normal((100, 72)) + 0.5

    h_aug, t_aug = residual_noise_augment(h, t, noise_std_scale=0.3, seed=0)

    assert t_aug.shape == t.shape
    assert np.allclose(t_aug, t)  # targets EXACTLY preserved
    assert not np.allclose(h_aug, h)  # history perturbed
    assert np.isfinite(h_aug).all()


def test_residual_noise_augment_rejects_zero_or_negative_scale():
    """noise_std_scale must be positive."""
    h = np.zeros((5, 96))
    t = np.ones((5, 72))
    with pytest.raises(ValueError):
        residual_noise_augment(h, t, noise_std_scale=0)
    with pytest.raises(ValueError):
        residual_noise_augment(h, t, noise_std_scale=-0.5)


def test_circular_tidal_shift_phase_shift():
    """Circular shift by 3 hours returns shifted arrays."""
    h = np.zeros((4, 24))
    t = np.zeros((4, 12))
    # Mark history columns
    h[0] = np.arange(24, dtype=float)
    t[0] = np.arange(12, dtype=float)

    h_s, t_s = circular_tidal_shift(h, t, max_shift_hours=3, period_hours=12, seed=42)

    assert h_s.shape == h.shape
    assert t_s.shape == t.shape
    # All values preserved (circular shift, no loss)
    for i in range(4):
        assert set(h_s[i]) == set(h[i])
        assert set(t_s[i]) == set(t[i])


def test_circular_shift_max_boundary_raises():
    """max_shift > period/2 must raise."""
    h = np.zeros((2, 24))
    t = np.zeros((2, 12))
    with pytest.raises(ValueError):
        circular_tidal_shift(h, t, max_shift_hours=7, period_hours=12)


def test_combine_augmentations_concats():
    """Multiple (h, t) pairs concatenate along axis 0."""
    h1 = np.zeros((3, 96))
    t1 = np.ones((3, 72))
    h2 = np.full((2, 96), 2.0)
    t2 = np.full((2, 72), 3.0)

    hc, tc = combine_augmentations((h1, t1), (h2, t2))
    assert hc.shape == (5, 96)
    assert tc.shape == (5, 72)
    assert np.allclose(hc[:3], 0.0)
    assert np.allclose(hc[3:], 2.0)
    assert np.allclose(tc[:3], 1.0)
    assert np.allclose(tc[3:], 3.0)


def test_combine_augmentations_empty_raises():
    with pytest.raises(ValueError):
        combine_augmentations()
