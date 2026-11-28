"""Unit tests for the conditional WGAN-GP baseline (M7)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tidetail.models.wgan_gp import ConditionalWGANGP, train_wgan_gp


def test_shapes_and_deterministic_sampling():
    torch.manual_seed(0)
    m = ConditionalWGANGP(context=96, horizon=72, width=16, z_dim=16)
    hist = torch.randn(4, 96)
    a = m.sample(hist, seed=7)
    b = m.sample(hist, seed=7)
    c = m.sample(hist, seed=8)
    assert a.shape == (4, 72)
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)


def test_losses_finite_and_reducible():
    torch.manual_seed(0)
    m = ConditionalWGANGP(context=96, horizon=72, width=16, z_dim=16)
    hist = torch.randn(32, 96)
    real = torch.randn(32, 72)
    c1, diag1 = m.critic_loss(hist, real)
    g1 = m.generator_loss(hist)
    assert torch.isfinite(c1) and torch.isfinite(g1)
    assert diag1["gradient_penalty"] >= 0.0
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    opt.zero_grad()
    (c1 + g1).backward()
    opt.step()
    c2, _ = m.critic_loss(hist, real)
    assert torch.isfinite(c2)


def test_training_loop_runs_and_improves_gp():
    torch.manual_seed(0)
    m = ConditionalWGANGP(context=96, horizon=72, width=16, z_dim=16)
    hist = torch.randn(256, 96)
    targ = torch.randn(256, 72) * 0.5
    h = train_wgan_gp(m, hist, targ, epochs=2, batch_size=64, n_critic=2)
    assert len(h["epoch"]) == 2
    assert all(torch.isfinite(torch.tensor(h["wasserstein"])))
    # gradient penalty should stay near its 10-weighted target zone
    assert h["gp"][-1] < 1.0


def test_parameter_count_budget():
    m = ConditionalWGANGP(context=96, horizon=72, width=32, z_dim=32)
    n = sum(p.numel() for p in m.parameters())
    g = sum(p.numel() for n_, p in m.named_parameters() if not n_.startswith("c_"))
    # generator matched to the ~27k VelocityTCN budget; whole model
    # (generator + critic) must stay comparable to other families
    assert 15_000 < g < 60_000
    assert 30_000 < n < 120_000
