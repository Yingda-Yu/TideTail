"""Regression tests for Wave 2.7 generative-model repairs."""
from __future__ import annotations

import torch

from tidetail.models import ConditionalVAE, TimeGradDenoiser
from tidetail.models.diffusion_ts import DiffusionTS, ResidualDenoiser
from tidetail.models.physics_diffusion import PhysicsDiffusionTS, physics_loss


def test_timegrad_ar_memory_actually_used():
    torch.manual_seed(0)
    d = TimeGradDenoiser(width=24, use_ar_memory=True).eval()
    noisy = torch.zeros(2, 1)
    t = torch.zeros(2, dtype=torch.long)
    hist = torch.zeros(2, 96)
    a = d(noisy, t, hist, torch.zeros(2, 1))
    b = d(noisy, t, hist, torch.full((2, 1), 5.0))
    assert not torch.allclose(a, b), "ar_memory must influence the prediction"


def test_timegrad_legacy_ignores_ar_memory():
    torch.manual_seed(0)
    d = TimeGradDenoiser(width=24, use_ar_memory=False).eval()
    noisy = torch.zeros(2, 1)
    t = torch.zeros(2, dtype=torch.long)
    hist = torch.zeros(2, 96)
    a = d(noisy, t, hist, torch.zeros(2, 1))
    b = d(noisy, t, hist, torch.full((2, 1), 5.0))
    assert torch.allclose(a, b), "legacy path must ignore ar_memory"


def test_vae_latent_has_rank_greater_than_one():
    """Two latent vectors with equal sum must still produce different output."""
    torch.manual_seed(0)
    vae = ConditionalVAE(latent_dim=8, z_init="linear").eval()
    hist = torch.zeros(1, 96)
    z1 = torch.tensor([[1.0, -1.0, 0, 0, 0, 0, 0, 0]])
    z2 = torch.tensor([[-1.0, 1.0, 0, 0, 0, 0, 0, 0]])
    with torch.no_grad():
        out1 = vae.decoder(hist, z1)
        out2 = vae.decoder(hist, z2)
    assert not torch.allclose(out1, out2), "legacy scalar collapse would make these equal"


def test_vae_wrapper_is_nn_module():
    vae = ConditionalVAE(latent_dim=8)
    assert isinstance(vae, torch.nn.Module)
    assert vae.to("cpu") is vae


def test_physics_loss_uses_true_eps():
    torch.manual_seed(0)
    pred = torch.randn(3, 1, 72)
    eps_a = torch.randn(3, 72)
    eps_b = torch.randn(3, 72)
    tgt = torch.randn(3, 72)
    noisy = torch.randn(3, 1, 72)
    hist = torch.randn(3, 96)
    loss_a = physics_loss(tgt, pred, eps_a, noisy, hist)
    loss_b = physics_loss(tgt, pred, eps_b, noisy, hist)
    assert not torch.allclose(loss_a, loss_b), "loss must depend on the supplied noise"
    assert torch.isfinite(loss_a)


def test_diffusion_ts_loss_finite_and_backprops():
    torch.manual_seed(0)
    model = DiffusionTS(steps=64)
    denoiser = ResidualDenoiser()
    hist = torch.randn(3, 96)
    tgt = torch.randn(3, 72)
    loss = model.training_loss(hist, tgt, denoiser, fourier_weight=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in denoiser.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_physics_diffusion_loss_finite():
    torch.manual_seed(0)
    model = PhysicsDiffusionTS(steps=64)
    denoiser = ResidualDenoiser()
    loss = model.training_loss(torch.randn(3, 96), torch.randn(3, 72), denoiser)
    assert torch.isfinite(loss)
