"""Conditional diffusion augmentation — CPU smoke-test only.

This is a toy unconditional/D3PM hybrid designed to run on CPU
without torch.  It uses score-based denoising with a Gaussian noise
schedule and a learned conditional bias.  NOT a full implementation
of any published diffusion model — it exists to verify that the
research pipeline can load synthetic data, train a denoiser, and
produce plausible samples BEFORE committing GPU budget.
"""
from __future__ import annotations

import numpy as np


def cosine_schedule(t: np.ndarray, beta_max: float = 0.02) -> np.ndarray:
    """Linear variance schedule scaled by (1 - cos(pi*t/2))."""
    t = np.clip(t, 0, 1)
    return beta_max * (1 - np.cos(np.pi * t / 2)) ** 2


def noise_level(sigmas: np.ndarray) -> np.ndarray:
    """Cumulative product → diffusion levels."""
    return np.cumprod(np.sqrt(sigmas))  # alpha_t


class ConditionalDenoiser:
    """Per-timestep learned bias for a 1D series denoiser.

    For each (timestep_idx, time_index) pair we learn a scalar bias
    conditioned on a history snapshot.  The architecture is intentionally
    small so CPU smoke tests complete in seconds.
    """

    def __init__(self, series_len: int = 60, n_steps: int = 10,
                 cond_len: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.series_len = series_len
        self.n_steps = n_steps
        self.cond_len = cond_len
        self.W_cond = rng.normal(0, 0.01, (n_steps, series_len, cond_len))
        self.b_cond = np.zeros((n_steps, series_len))

    def _step(self, noisy: np.ndarray, cond: np.ndarray, step_idx: int) -> np.ndarray:
        """Learned bias + residual: out = noisy + W @ cond + b."""
        return noisy + self.W_cond[step_idx] @ cond + self.b_cond[step_idx]

    def train(self, X: np.ndarray, cond: np.ndarray, epochs: int = 20,
              lr: float = 0.01, beta_max: float = 0.1) -> float:
        """MSE between denoiser output and clean sample.

        X: (n_samples, series_len)  clean
        cond: (n_samples, cond_len) conditioning snapshot
        """
        rng = np.random.default_rng(1)
        losses: list = []
        for _ in range(epochs):
            idx = rng.choice(X.shape[0], size=min(64, X.shape[0]), replace=False)
            for i in idx:
                step_idx = int(rng.integers(0, self.n_steps))
                noise = rng.normal(0, beta_max * (step_idx + 1), self.series_len)
                noisy = X[i] + noise
                pred = self._step(noisy, cond[i], step_idx)
                loss = float(np.mean((pred - X[i]) ** 2))
                losses.append(loss)
                # tiny central-difference SGD on one param block
                for p in [self.W_cond[step_idx], self.b_cond[step_idx]]:
                    it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
                    eps = 1e-4
                    while not it.finished:
                        m = it.multi_index
                        orig = p[m]
                        p[m] = orig + eps
                        lp = float(np.mean((self._step(noisy, cond[i], step_idx) - X[i]) ** 2))
                        p[m] = orig - eps
                        lm = float(np.mean((self._step(noisy, cond[i], step_idx) - X[i]) ** 2))
                        p[m] = orig - lr * (lp - lm) / (2 * eps)
                        it.iternext()
        return float(np.mean(losses[-50:]))

    def sample(self, cond: np.ndarray, n: int = 4,
               beta_max: float = 0.1, seed: int = 99) -> np.ndarray:
        """Reverse diffusion from pure noise, n samples.  (n, series_len)."""
        rng = np.random.default_rng(seed)
        series_len = self.series_len
        steps = self.n_steps
        out = np.zeros((n, series_len))
        for i in range(n):
            x_t = rng.normal(0, beta_max * steps, series_len)
            cond_i = cond[i % cond.shape[0]]
            for s in range(steps - 1, -1, -1):
                x_t = self._step(x_t, cond_i, s)
            out[i] = x_t
        return out


def memorization_check(real: np.ndarray, gen: np.ndarray,
                       window: int = 20, threshold: float = 0.95) -> dict:
    """Flag any window in `gen` that exactly memorises `real`.

    real: (L,)  gen: (N, L). Returns dict with memorization counts and
    correlation distribution.  Safe against degenerate inputs.
    """
    results: list = []
    gen = np.atleast_2d(gen)
    L = min(real.shape[0], gen.shape[1])
    if L < window * 2:
        return {"n_windows": 0, "memorized": 0, "memorized_rate": 0.0,
                "mean_corr": float("nan"), "p95_corr": float("nan")}
    max_ri = max(1, window // 2)
    for gi in range(gen.shape[0]):
        for start in range(0, L - window, window):
            g_chunk = gen[gi, start:start + window]
            best_corr = -1.0
            for rstart in range(0, real.shape[0] - window, max_ri):
                r_chunk = real[rstart:rstart + window]
                if np.std(g_chunk) < 1e-8 or np.std(r_chunk) < 1e-8:
                    continue
                corr = float(np.corrcoef(g_chunk, r_chunk)[0, 1])
                best_corr = max(best_corr, corr)
            results.append(best_corr)
    if not results:
        return {"n_windows": 0, "memorized": 0, "memorized_rate": 0.0,
                "mean_corr": float("nan"), "p95_corr": float("nan")}
    arr = np.array(results, dtype=float)
    memorized = int((arr >= threshold).sum())
    return {
        "n_windows": len(arr),
        "memorized": memorized,
        "memorized_rate": float(memorized) / max(1, len(arr)),
        "mean_corr": float(np.mean(arr)),
        "p95_corr": float(np.percentile(arr, 95)),
    }


def quality_summary(real: np.ndarray, gen: np.ndarray) -> dict:
    """Compare marginal mean/std/min/max between real and synthetic."""
    return {
        "real_mean": float(np.nanmean(real)),
        "real_std": float(np.nanstd(real)),
        "real_max": float(np.nanmax(real)),
        "real_min": float(np.nanmin(real)),
        "gen_mean": float(np.nanmean(gen)),
        "gen_std": float(np.nanstd(gen)),
        "gen_max": float(np.nanmax(gen)),
        "gen_min": float(np.nanmin(gen)),
    }
