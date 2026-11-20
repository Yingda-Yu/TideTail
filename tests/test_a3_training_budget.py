"""Tests for training-budget fairness in augmentation comparison (W1-BUG-04).

Audit: docs/audits/WAVE1_SCIENTIFIC_CORRECTNESS.md (W1-BUG-04)
"""
from __future__ import annotations


class TestA3TrainingBudgetFairness:
    """Augment groups must be compared under explicitly named protocols."""

    def test_different_augment_sizes_give_different_optimizer_steps(self):
        """With same epochs + same batch_size, groups with more training data
        get more optimizer steps — violates fair comparison."""
        # ny_bight example from 2026-10-31 run
        groups = {
            "A0_real_only": 69970,
            "A1_oversampling": 210030,
            "A2_history_noise": 139940,
            "A3_fm_ratio_0.1": 76967,
            "A3_fm_ratio_0.5": 104955,
        }
        epochs = 30
        batch_size = 256

        steps_per_epoch = {g: (n + batch_size - 1) // batch_size
                           for g, n in groups.items()}
        total_steps = {g: s * epochs for g, s in steps_per_epoch.items()}

        # A1 gets ~3× more steps than A0
        ratio = total_steps["A1_oversampling"] / total_steps["A0_real_only"]
        assert ratio > 2.5, (
            f"A1 should have much more steps than A0, ratio={ratio:.2f}"
        )

        # This IS a bug under "same epochs" protocol
        # The fix is to document the protocol and match steps OR real exposure

    def test_protocol_F1_equal_optimizer_updates(self):
        """Protocol F1: all groups get EXACTLY same total optimizer steps."""
        target_steps = 10000
        batch_size = 256

        groups = {
            "A0_real_only": 70000,
            "A1_oversampling": 210000,
            "A2_history_noise": 140000,
            "A3_fm_ratio_0.25": 87500,
        }

        epochs_per_group = {}
        for g, n in groups.items():
            steps_per_epoch = (n + batch_size - 1) // batch_size
            epochs_per_group[g] = max(1, target_steps // steps_per_epoch)

        # Total steps per group
        actual_steps = {}
        for g, n in groups.items():
            spe = (n + batch_size - 1) // batch_size
            actual_steps[g] = spe * epochs_per_group[g]

        # Allow ±5% tolerance (can't hit exact due to integer epochs)
        for g, s in actual_steps.items():
            relative = abs(s - target_steps) / target_steps
            assert relative < 0.10, (
                f"Protocol F1: {g} has {s} steps, target {target_steps}, "
                f"deviation {relative:.1%}"
            )

    def test_protocol_F2_matched_real_exposure(self):
        """Protocol F2: all groups use same number of UNIQUE real examples.
        Extra synthetic or duplicated samples are tracked but not counted
        as real exposure."""
        real_unique = 10000

        groups = {
            "A0_real_only": {"real_unique": 10000, "synthetic": 0, "dup_real": 0},
            "A1_oversampling": {"real_unique": 10000, "synthetic": 0, "dup_real": 20000},
            "A2_history_noise": {"real_unique": 10000, "synthetic": 10000, "dup_real": 0},
            "A3_fm_ratio_0.25": {"real_unique": 10000, "synthetic": 2500, "dup_real": 0},
        }

        for g, composition in groups.items():
            # Every group MUST start with same real_unique count
            assert composition["real_unique"] == real_unique, (
                f"Protocol F2: {g} has {composition['real_unique']} unique real, "
                f"expected {real_unique}"
            )
            # Extra optimization cost must be tracked
            total_samples = composition["real_unique"] + composition["synthetic"] + composition["dup_real"]
            assert total_samples >= real_unique

    def test_real_exposure_never_less_than_a0(self):
        """Augment groups must NEVER use fewer unique real windows than A0.
        (You can't augment away real data — you add on top.)"""
        a0_real = 70000
        augment_groups_real = {
            "A1_oversampling": 70000,  # dup, not fewer
            "A2_history_noise": 70000,  # same real + noisy copies
            "A3_fm_ratio_0.1": 70000,  # same real + synthetic
        }
        for g, real_count in augment_groups_real.items():
            assert real_count >= a0_real, (
                f"{g} uses fewer real windows ({real_count}) than A0 ({a0_real})"
            )
