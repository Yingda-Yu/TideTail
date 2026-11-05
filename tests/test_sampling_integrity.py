"""Deterministic CPU checks; no optimizer, data files or CUDA needed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tidetail.torch_diffusion import Diffusion


class ConstantEpsilon:
    horizon = 24

    def __init__(self, value):
        self.value = value

    def __call__(self, x, t, history):
        return torch.full_like(x, self.value)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_sampler_refuses_nonfinite_prediction(value):
    with pytest.raises(ValueError, match="nonfinite"):
        Diffusion(8).sample(ConstantEpsilon(value), torch.zeros(2, 72))


def test_sampler_refuses_nonfinite_history():
    with pytest.raises(ValueError, match="nonfinite"):
        Diffusion(8).sample(ConstantEpsilon(0), torch.full((2, 72), float("nan")))


def test_forward_indexing_and_reverse_posterior_algebra():
    diffusion = Diffusion(64)
    clean = torch.tensor([[[0.25]], [[-0.5]]], dtype=torch.float64)
    epsilon = torch.tensor([[[0.1]], [[-0.2]]], dtype=torch.float64)
    t = torch.tensor([0, 63])
    bar = diffusion.bar[t].double()[:, None, None]
    x = diffusion.noisy(clean, t, epsilon)
    # noisy() uses float32 schedule coefficients, so compare at that precision.
    torch.testing.assert_close(x, bar.sqrt() * clean + (1-bar).sqrt() * epsilon,
                               rtol=1e-5, atol=1e-7)
    previous = torch.cat((torch.ones(1), diffusion.bar[:-1])).double()[t, None, None]
    beta = diffusion.beta[t].double()[:, None, None]
    alpha = diffusion.alpha[t].double()[:, None, None]
    mean_epsilon = (x - beta / (1-bar).sqrt() * epsilon) / alpha.sqrt()
    mean_clean = previous.sqrt() * beta / (1-bar) * clean
    mean_clean += alpha.sqrt() * (1-previous) / (1-bar) * x
    torch.testing.assert_close(mean_epsilon, mean_clean, rtol=1e-4, atol=1e-4)
    assert diffusion.posterior_variance[0] == 0
    torch.testing.assert_close(diffusion.posterior_variance[t].double()[:, None, None],
                               beta * (1-previous) / (1-bar), rtol=1e-5, atol=1e-7)


def test_sampler_matches_reference_seed_and_variance():
    diffusion = Diffusion(8)
    history = torch.zeros(2, 72)
    rng = torch.Generator().manual_seed(99)
    x = torch.randn(2, 1, 24, generator=rng)
    for i in reversed(range(8)):
        x = (x - diffusion.beta[i] / (1-diffusion.bar[i]).sqrt() * 0.125)
        x = x / diffusion.alpha[i].sqrt()
        if i:
            x += diffusion.posterior_variance[i].sqrt() * torch.randn(x.shape, generator=rng)
    torch.testing.assert_close(diffusion.sample(ConstantEpsilon(0.125), history), x)


def test_terminal_prediction_error_is_strongly_amplified():
    """Historical cosine schedule (not default!) must amplify terminal error >1000×."""
    diffusion = Diffusion(64, schedule="cosine")
    amplification = ((1-diffusion.bar[-1]) / diffusion.bar[-1]).sqrt().item()
    assert amplification > 1000  # epsilon error 0.01 -> >10 normalized clean units
    assert diffusion.beta[-1].item() == pytest.approx(0.999)


def test_default_schedule_has_bounded_amplification():
    """Default cosine1000_subsample keeps terminal amplification in a
    physically-reasonable range (< 100×, vs 1299× for historical cosine)."""
    diffusion = Diffusion(64)  # default = cosine1000_subsample
    amp = diffusion.terminal_amplification
    assert 10 < amp < 100, f"amplification out of range: {amp}"
    # beta[-1] still clamps near 1 for a proper schedule
    assert diffusion.beta[-1].item() == pytest.approx(0.999)
    # alpha_bar must be well above machine epsilon
    assert diffusion.terminal_alpha_bar > 1e-5


def test_finite_exploding_or_collapsed_samples_fail_scale_gate():
    from tidetail.torch_diffusion import sample_diagnostics

    reference = np.linspace(-2, 2, 240)
    normal = sample_diagnostics(reference.reshape(10, 24), reference)
    assert normal["accepted"]
    for bad in (reference * 20, np.zeros_like(reference), reference + 20):
        result = sample_diagnostics(bad.reshape(10, 24), reference)
        assert result["finite"] and not result["accepted"]
    assert not sample_diagnostics(np.array([np.nan]), reference)["accepted"]


def test_invalid_scale_reference_is_rejected():
    from tidetail.torch_diffusion import sample_diagnostics

    for reference in (np.zeros(10), np.array([np.nan])):
        with pytest.raises(ValueError):
            sample_diagnostics(np.ones(10), reference)
