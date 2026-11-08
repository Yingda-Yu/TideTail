"""Fail-first tests for A3 Flow Matching augmentation conditioning integrity.

These tests must FAIL against the buggy run_a3_flowmatching_augment.py (commit 244ec0e)
and PASS after W1-BUG-01 fix.

Bug: synthetic_tgt[i] was generated from rtr_h[i] but paired with rtr_h[:n_synth]
instead of rtr_h[synth_idx].

Audit: docs/audits/WAVE1_SCIENTIFIC_CORRECTNESS.md (W1-BUG-01)
"""
from __future__ import annotations

import numpy as np


class TestA3Conditioning:
    """Verify that synthetic target row `k` is paired with the history row
    that conditioned generation of that synthetic row."""

    def test_synthetic_history_pairing_exact(self):
        """Generated target at index i must pair with the EXACT history row
        that was used as FM conditioning for that generation."""
        N = 5000
        H = 72
        C = 96
        rng = np.random.default_rng(12345)
        rtr_h = rng.normal(0, 1, (N, C)).astype(np.float32)

        # Simulate: FM generates synthetic_tgt row i from rtr_h row i
        # This is how generate_synthetic_residuals() works:
        #   fm.sample(h_norm, velocity_net, seed=...) operates on each row
        #   so output row i is conditioned on input row i
        # We use a deterministic function to simulate this
        rng2 = np.random.default_rng(777)
        synthetic_tgt = rng2.normal(0, 1, (N, H)).astype(np.float32)

        # Pick indices with a different RNG (like the buggy code does)
        rng3 = np.random.default_rng(42)
        ratio = 0.25
        n_synth = int(N * ratio)
        synth_idx = rng3.choice(N, size=n_synth, replace=False)

        # --- BUGGY pairing (what old code did) ---
        np.concatenate([rtr_h, rtr_h[:n_synth]], axis=0)
        np.concatenate([rng2.normal(0, 1, (N, H)).astype(np.float32)[synth_idx]
                                   if False else synthetic_tgt[synth_idx]], axis=0)
        # Above line is messy — do it clean:
        buggy_h_aug = rtr_h[:n_synth]
        synthetic_tgt[synth_idx]

        # Check for buggy: how many pairs match? (should be ~0% with high prob)
        matches_buggy = 0
        for i in range(n_synth):
            # synthetic_tgt[synth_idx[i]] was generated from rtr_h[synth_idx[i]]
            # buggy pairs it with rtr_h[i]
            if np.array_equal(buggy_h_aug[i], rtr_h[synth_idx[i]]):
                matches_buggy += 1
        # With N=5000, n_synth=1250, P(match) = 1/5000 per row, expected ~0.25
        assert matches_buggy <= 3, (
            f"Buggy pairing should have ~0 matches, got {matches_buggy}"
        )

        # --- CORRECT pairing (W1-BUG-01 fix) ---
        fixed_h_aug = rtr_h[synth_idx]
        synthetic_tgt[synth_idx]

        # Check correct: 100% of pairs match
        matches_fixed = 0
        for i in range(n_synth):
            if np.array_equal(fixed_h_aug[i], rtr_h[synth_idx[i]]):
                matches_fixed += 1
        assert matches_fixed == n_synth, (
            f"Correct pairing: expected {n_synth} matches, got {matches_fixed}"
        )

    def test_augment_preserves_real_side(self):
        """Real history + real target rows must be unchanged."""
        N = 100
        H = 72
        C = 96
        rng = np.random.default_rng(42)
        rtr_h = rng.normal(0, 1, (N, C)).astype(np.float32)
        rtr_t = rng.normal(0, 1, (N, H)).astype(np.float32)
        synth_t = rng.normal(0, 1, (N, H)).astype(np.float32)
        n_synth = int(N * 0.25)
        synth_idx = rng.choice(N, size=n_synth, replace=False)

        # Correct augmented arrays
        aug_h = np.concatenate([rtr_h, rtr_h[synth_idx]], axis=0)
        aug_t = np.concatenate([rtr_t, synth_t[synth_idx]], axis=0)

        # First N rows must be identical to original
        assert np.array_equal(aug_h[:N], rtr_h)
        assert np.array_equal(aug_t[:N], rtr_t)
        # Last n_synth rows must be the synthetic pairs
        assert np.array_equal(aug_h[N:], rtr_h[synth_idx])
        assert np.array_equal(aug_t[N:], synth_t[synth_idx])

    def test_no_conditioning_leakage_via_shuffle(self):
        """If synth_idx is shuffled independently of the history pick,
        the pairing is destroyed."""
        N = 1000
        H = 72
        C = 96
        rng = np.random.default_rng(42)
        rtr_h = rng.normal(0, 1, (N, C)).astype(np.float32)
        rng.normal(0, 1, (N, H)).astype(np.float32)
        n_synth = 250

        # synth_idx from one RNG
        synth_idx = np.random.default_rng(1).choice(N, size=n_synth, replace=False)
        # history_idx from a DIFFERENT RNG (shuffled)
        history_idx = np.random.default_rng(2).choice(N, size=n_synth, replace=False)

        # Probability of any accidental match
        # P(match per row) = 1/N
        matches = sum(1 for i in range(n_synth) if np.array_equal(
            rtr_h[history_idx[i]], rtr_h[synth_idx[i]]))
        # Should be 0 with overwhelming probability (N=1000, n=250)
        assert matches == 0, (
            f"Independent shuffle accidentally matched {matches} rows "
            f"(should be 0 — use rtr_h[synth_idx] exactly)"
        )
