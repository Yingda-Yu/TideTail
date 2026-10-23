"""CPU smoke tests for ConditionalDenoiser + memorization guardrails."""
import numpy as np

from tidetail.diffusion import (
    ConditionalDenoiser, cosine_schedule, memorization_check,
    quality_summary, noise_level,
)


def test_cosine_schedule_monotone():
    t = np.linspace(0, 1, 20)
    b = cosine_schedule(t)
    assert b[0] < b[-1]
    assert np.all(b >= 0)


def test_noise_level_monotone():
    sigmas = np.array([0.9, 0.8, 0.7, 0.6])
    lvl = noise_level(sigmas)
    assert np.all(lvl[:-1] >= lvl[1:])


def test_denoiser_trains_and_generates():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (50, 60))
    cond = rng.normal(0, 0.5, (50, 10))
    dn = ConditionalDenoiser(series_len=60, n_steps=5, cond_len=10, seed=42)
    loss = dn.train(X, cond, epochs=20, lr=0.01)
    assert loss < 5.0, f"loss too high: {loss}"
    samples = dn.sample(cond[:4], n=4)
    assert samples.shape == (4, 60)
    assert np.isfinite(samples).all()


def test_memorization_check_flags_exact_copy():
    rng = np.random.default_rng(0)
    real = rng.normal(0, 1, (200,))
    gen_exact = np.stack([real.copy() for _ in range(5)])
    m = memorization_check(real, gen_exact, window=20, threshold=0.95)
    # exact copies → many memorized windows
    assert m["memorized"] > 0
    assert m["mean_corr"] > 0.9

    # independent noise → near-zero memorization
    gen_noise = rng.normal(0, 1, (5, 200))
    m2 = memorization_check(real, gen_noise, window=20, threshold=0.95)
    assert m2["memorized_rate"] < 0.01


def test_quality_summary_compares_real_vs_gen():
    rng = np.random.default_rng(1)
    real = rng.normal(0, 1, (100,))
    gen = real + 0.01 * rng.normal(0, 1, (100,))
    q = quality_summary(real, gen)
    assert abs(q["real_mean"] - q["gen_mean"]) < 0.1
    assert abs(q["real_std"] - q["gen_std"]) < 0.2
