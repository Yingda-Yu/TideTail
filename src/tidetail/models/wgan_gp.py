"""M7 — Conditional WGAN-GP baseline (Wave 2.8 P3 fair benchmark).

Reference
---------
Gulrajani et al. (2017). "Improved Training of Wasserstein GANs."
https://arxiv.org/abs/1704.00028

Overview
--------
A conditional Wasserstein GAN with gradient penalty for 96 h history →
72 h residual forecast synthesis. The generator mirrors the
``VelocityTCN`` architecture (it maps pure noise, conditioned on the
per-sample-normalised history, to a forecast window in ONE step — the
same mapping the flow-matching ODE learns to integrate). The critic is
a normalisation-free dilated TCN with mean-pooling (batch/group norm is
incompatible with the gradient penalty).

Training follows WGAN-GP: ``n_critic`` critic updates per generator
update, gradient penalty weight 10, Adam(1e-4, betas=(0.0, 0.9)) as
recommended for Wasserstein critics.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class _ResBlock(nn.Module):
    """Pre-activation residual block with optional conditioning."""

    def __init__(self, width: int, dilation: int, cond: bool = True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.SiLU(),
            nn.Conv1d(width, width, 1),
        )
        self.cond = nn.Linear(width, width) if cond else None

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None) -> torch.Tensor:
        h = x + self.cond(emb)[:, :, None] if self.cond is not None else x
        return (x + self.body(h)) / math.sqrt(2)


class ConditionalWGANGP(nn.Module):
    """Container exposing generator, critic and the WGAN-GP loss."""

    def __init__(self, context: int = 96, horizon: int = 72, width: int = 32,
                 z_dim: int = 32):
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.width = width
        self.z_dim = z_dim

        # --- Generator (mirrors VelocityTCN; no time embedding) ---
        self.g_hist = nn.Sequential(nn.Linear(context, width), nn.SiLU(),
                                    nn.Linear(width, width))
        self.g_z = nn.Sequential(nn.Linear(z_dim, width), nn.SiLU(),
                                 nn.Linear(width, width))
        # noise is tiled over the horizon axis and projected like an image
        self.g_in = nn.Conv1d(z_dim, width, 3, padding=1)
        self.g_blocks = nn.ModuleList(
            _ResBlock(width, d) for d in (1, 2, 4, 8))
        self.g_out = nn.Sequential(nn.GroupNorm(4, width), nn.SiLU(),
                                   nn.Conv1d(width, 1, 3, padding=1))

        # --- Critic (no normalisation — WGAN-GP requirement) ---
        self.c_hist = nn.Sequential(nn.Linear(context, width), nn.LeakyReLU(0.2),
                                    nn.Linear(width, width))
        self.c_in = nn.Conv1d(1, width, 3, padding=1)
        self.c_blocks = nn.ModuleList(
            _ResBlock(width, d, cond=True) for d in (1, 2, 4, 8))
        self.c_out = nn.Linear(width, 1)

    def generate(self, history: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """history: (B, context); z: (B, z_dim) -> (B, horizon)."""
        emb = self.g_hist(history) + self.g_z(z)
        x = z.unsqueeze(2).expand(-1, -1, self.horizon) / math.sqrt(self.z_dim)
        h = self.g_in(x)
        for block in self.g_blocks:
            h = block(h, emb)
        return self.g_out(h).squeeze(1)

    def critic_score(self, history: torch.Tensor,
                     sample: torch.Tensor) -> torch.Tensor:
        """history: (B, context); sample: (B, horizon) -> (B, 1)."""
        emb = self.c_hist(history)
        h = self.c_in(sample.unsqueeze(1))
        for block in self.c_blocks:
            h = block(h, emb)
        return self.c_out(h.mean(dim=2))

    def critic_loss(self, history: torch.Tensor, real: torch.Tensor,
                    gp_weight: float = 10.0) -> tuple[torch.Tensor, dict]:
        """Wasserstein critic loss with gradient penalty.

        Returns (total_loss, diagnostics).
        """
        B = real.shape[0]
        z = torch.randn(B, self.z_dim, device=real.device)
        fake = self.generate(history, z)
        d_real = self.critic_score(history, real).mean()
        d_fake = self.critic_score(history, fake.detach()).mean()
        # gradient penalty on random interpolates
        eps = torch.rand(B, 1, device=real.device)
        interp = (eps * real + (1 - eps) * fake.detach()).requires_grad_(True)
        score = self.critic_score(history, interp)
        grad = torch.autograd.grad(score.sum(), interp, create_graph=True)[0]
        gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        loss = d_fake - d_real + gp_weight * gp
        return loss, {"wasserstein_estimate": float((d_real - d_fake).detach()),
                      "gradient_penalty": float(gp.detach())}

    def generator_loss(self, history: torch.Tensor) -> torch.Tensor:
        z = torch.randn(history.shape[0], self.z_dim, device=history.device)
        fake = self.generate(history, z)
        return -self.critic_score(history, fake).mean()

    @torch.no_grad()
    def sample(self, history: torch.Tensor, seed: int = 42) -> torch.Tensor:
        """Deterministic-seed sampling: (B, context) -> (B, horizon)."""
        rng = torch.Generator(device=history.device).manual_seed(seed)
        z = torch.randn(history.shape[0], self.z_dim,
                        device=history.device, generator=rng)
        return self.generate(history, z)


def train_wgan_gp(model: ConditionalWGANGP, hist: torch.Tensor,
                  targ: torch.Tensor, epochs: int, batch_size: int = 256,
                  n_critic: int = 5, lr: float = 1e-4,
                  log_every: int | None = None) -> dict:
    """Standard WGAN-GP alternation. Returns training history.

    ``hist``/``targ`` must already be normalised (per-sample or train
    scaler); sampling code must invert the same transform.
    """
    g_opt = torch.optim.Adam(
        [p for n, p in model.named_parameters() if not n.startswith("c_")],
        lr=lr, betas=(0.0, 0.9))
    c_opt = torch.optim.Adam(
        [p for n, p in model.named_parameters() if n.startswith("c_")],
        lr=lr, betas=(0.0, 0.9))
    n = len(hist)
    history = {"epoch": [], "g_loss": [], "wasserstein": [], "gp": []}
    for epoch in range(epochs):
        perm = torch.randperm(n)
        g_tot = w_tot = gp_tot = 0.0
        n_g = 0
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            hb, tb = hist[idx], targ[idx]
            c_loss, diag = model.critic_loss(hb, tb)
            c_opt.zero_grad()
            c_loss.backward()
            c_opt.step()
            w_tot += diag["wasserstein_estimate"]
            gp_tot += diag["gradient_penalty"]
            if (b // batch_size) % n_critic == n_critic - 1:
                g_loss = model.generator_loss(hb)
                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()
                g_tot += float(g_loss.detach())
                n_g += 1
        history["epoch"].append(epoch)
        history["g_loss"].append(g_tot / max(n_g, 1))
        history["wasserstein"].append(w_tot / (b // batch_size + 1))
        history["gp"].append(gp_tot / (b // batch_size + 1))
        if log_every and epoch % log_every == 0:
            print(f"    wgan-gp epoch {epoch}: "
                  f"W={history['wasserstein'][-1]:.4f} "
                  f"gp={history['gp'][-1]:.4f}")
    return history
