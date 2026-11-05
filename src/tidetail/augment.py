"""Balanced training (A1) and physics-justified data augmentation (A2).

CRITICAL: These augment the TRAINING SET ONLY, never validation or test.
This is the A1 (oversampling/balancing) and A2 (physics augmentation) groups
from the FORECAST_TASK_CONTRACT v1 comparison protocol.

A1 is NOT "add noise to predictions" — it re-samples training windows so
extreme-event windows appear more frequently in each optimizer step.
A2 is NOT "prediction-noise augmentation" — it perturbs training residual
values with physically-motivated noise levels derived from train-only stats.
"""
from __future__ import annotations

import numpy as np
import torch

# ==========================================================================
# A1: Balanced oversampling (real training-set re-weighting)
# ==========================================================================

def extreme_balance_weights(targets: np.ndarray, train_q95: float,
                              extreme_weight: float = 3.0
                              ) -> np.ndarray:
    """Return per-window sampling weights.

    Windows where ANY target value >= q95_train get `extreme_weight`;
    others get weight 1.0. This makes extreme events appear more often per
    optimizer step, which addresses the rare-event training imbalance without
    fabricating new samples.

    Args:
        targets: (n_windows, horizon) array of target water levels.
        train_q95: q95 threshold computed FROM TRAIN ROWS ONLY.
        extreme_weight: how much more frequently extreme windows are sampled.
                        Must be >= 1.0. Common values 2.0–10.0.

    Returns:
        (n_windows,) weight array for torch.utils.data.WeightedRandomSampler.
    """
    if extreme_weight < 1.0:
        raise ValueError("extreme_weight must be >= 1.0")
    targets = np.asarray(targets, dtype=np.float64)
    is_extreme = np.any(targets >= train_q95, axis=1)
    weights = np.where(is_extreme, extreme_weight, 1.0)
    return weights


def build_balanced_sampler(targets: np.ndarray, train_q95: float,
                            extreme_weight: float = 3.0,
                            seed: int = 0
                            ) -> torch.utils.data.WeightedRandomSampler:
    """Construct WeightedRandomSampler for balanced DataLoader.

    Example usage:
        weights = extreme_balance_weights(train_targets, q95)
        sampler = build_balanced_sampler(train_targets, q95, extreme_weight=5.0)
        loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    """
    weights = extreme_balance_weights(targets, train_q95, extreme_weight)
    total = weights.sum()
    probs = weights / total
    rng = torch.Generator().manual_seed(seed)
    torch.multinomial(
        torch.tensor(probs, dtype=torch.float32),
        num_samples=int(total),
        replacement=True,
        generator=rng,
    )
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=int(total),
        replacement=True,
        generator=rng,
    )


# ==========================================================================
# A2: Physics-justified training-set augmentation
# ==========================================================================

def residual_noise_augment(history: np.ndarray, target: np.ndarray,
                             noise_std_scale: float = 0.3,
                             train_residual_std: float | None = None,
                             seed: int = 0
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Add train-residual-scaled Gaussian noise to training history ONLY.

    Physics justification: residual (de-tided surge) measurements have
    instrument uncertainty and small unresolved sub-hourly variability
    that can be modeled as additive Gaussian. This augmentation applies
    ONLY to the input history, NOT the target.

    Args:
        history: (n, lookback) training windows.
        target: (n, horizon) training targets.
        noise_std_scale: noise std as fraction of train_residual_std.
                        0.3 = 30% of typical residual variability.
        train_residual_std: if None, computed from history finite values.
        seed: RNG seed for reproducibility.

    Returns:
        (augmented_history, original_target) — targets NEVER modified.
    """
    if noise_std_scale <= 0:
        raise ValueError("noise_std_scale must be positive")
    rng = np.random.default_rng(seed)
    h = np.asarray(history, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)

    if train_residual_std is None:
        train_residual_std = float(np.nanstd(h))
    noise_std = noise_std_scale * train_residual_std

    # Only perturb history; targets preserved as real observations
    noise = rng.normal(0, noise_std, size=h.shape)
    h_aug = h + noise
    return h_aug, t


def circular_tidal_shift(history: np.ndarray, target: np.ndarray,
                           max_shift_hours: int = 12,
                           period_hours: int = 12,
                           seed: int = 0
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Circular shift in tidal phase to create phase-shifted training examples.

    Physics justification: semi-diurnal tide has ~12h period. Shifting both
    history and target by a fraction of a full tidal cycle creates valid
    phase-shifted training pairs while preserving the residual forecast
    relationship. Unlike additive noise, this is a symmetry of the tide-only
    component; residual (surge) stays in place.

    Args:
        history: (n, lookback) windows.
        target: (n, horizon) windows.
        max_shift_hours: maximum circular shift in hours (≤ period/2).
        period_hours: tidal period (default 12 = M2 semi-diurnal).
        seed: RNG seed.

    Returns:
        (shifted_history, shifted_target) — BOTH shifted together.
    """
    if max_shift_hours > period_hours // 2:
        raise ValueError("max_shift_hours must be <= period/2")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(-max_shift_hours, max_shift_hours + 1))
    if shift == 0:
        return history.copy(), target.copy()

    h = np.asarray(history, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)

    # Shift each row independently by applying np.roll
    h_shifted = np.roll(h, shift, axis=1)
    t_shifted = np.roll(t, shift, axis=1)
    return h_shifted, t_shifted


def combine_augmentations(*pairs: tuple[np.ndarray, np.ndarray]
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate multiple (history, target) pairs along axis 0.

    Example:
        h_orig, t_orig = real_train
        h_noisy, t_noisy = residual_noise_augment(h_orig, t_orig, ...)
        h_shift, t_shift = circular_tidal_shift(h_orig, t_orig, ...)
        h_combined, t_combined = combine_augmentations(
            (h_orig, t_orig), (h_noisy, t_noisy), (h_shift, t_shift)
        )
    """
    if not pairs:
        raise ValueError("at least one (h, t) pair required")
    h = np.concatenate([p[0] for p in pairs], axis=0)
    t = np.concatenate([p[1] for p in pairs], axis=0)
    return h, t
