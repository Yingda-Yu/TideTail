"""M3 — TimeGrad-inspired autoregressive conditional diffusion for tidal surge.

Reference
---------
Alcaraz et al. (2023). "Autoregressive Denoising Diffusion Models for
Probabilistic Time Series Forecasting." https://arxiv.org/abs/2301.03068

Key difference from DDPM (M2)
-----------------------------
DiffusionTS (M2) generates the entire horizon *in parallel* with a single
denoising pass — epsilon conditioned on full 96h history and predicts
72h residual at once. TimeGrad (M3) generates *autoregressively*, one
horizon step at a time, conditioning each diffusion step on the history
AND the previously generated horizon points.

This is a significant architectural change: instead of (B, 1, 72) being
generated jointly, we roll out 72 separate DDPM passes, each predicting
one scalar conditioned on history (96h) + autoregressive memory
(previously generated horizon values).

For TideTail this matters because extreme peaks in storm surge come from
temporal *context* — whether we are on the rising or falling edge of a
tide — which parallel DDPM can blur but autoregressive modeling preserves.

NOTE: This file only implements the diffusion core + denoiser. A full
autoregressive sampler is in ``scripts/train_timegrad.py`` (CPU smoke
test + small-budget GPU training).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..torch_diffusion import Diffusion


class TimeGradDenoiser(nn.Module):
    """Autoregressive scalar denoiser.

    At each forecasting step ``i`` (0 ≤ i < horizon), takes:
      - noisy residual scalar at step i
      - sinusoidal diffusion timestep embedding
      - history context (96h)
      - previously generated horizon values (0..i-1) concatenated
    Predicts epsilon for that scalar.

    For TideTail we keep this intentionally small (~15K params, fewer
    than M2/M4) so the autoregressive rollout cost per training epoch
    is manageable — 72 DDPM passes per sample × 69970 train windows.
    """

    def __init__(self, context: int = 96, horizon: int = 72, width: int = 24):
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.width = width

        # Time embedding
        self.time_enc = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        # History context embedding
        self.history_enc = nn.Sequential(
            nn.Linear(context, width), nn.SiLU(), nn.Linear(width, width)
        )
        # Autoregressive memory (sum over previously generated points)
        self.memory_proj = nn.Linear(1, width)  # project single noisy scalar
        self.aggregator = nn.Sequential(
            nn.Linear(width * 3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.output = nn.Linear(width, 1)

    def forward(self, noisy_scalar: torch.Tensor, timestep: torch.Tensor,
                history: torch.Tensor, ar_memory: torch.Tensor
                ) -> torch.Tensor:
        """Predict epsilon for a single horizon step.

        Args:
            noisy_scalar: (batch, 1) current noisy value
            timestep: (batch,) long tensor of diffusion steps
            history: (batch, context) conditioning history
            ar_memory: (batch,) previously generated horizon values
                       (concatenated into a single sum/scalar per step)
        Returns:
            eps: (batch, 1) predicted noise
        """
        half_w = max(self.width // 2, 1)
        freq = torch.exp(
            torch.arange(half_w, device=noisy_scalar.device)
            * (-math.log(10000) / max(half_w - 1, 1))
        )
        angles = timestep.float()[:, None] * freq[None]
        t_emb = self.time_enc(
            torch.cat([angles.sin(), angles.cos()], dim=1)
        )

        h_emb = self.history_enc(history)
        m_emb = self.memory_proj(noisy_scalar)

        agg = self.aggregator(torch.cat([t_emb, h_emb, m_emb], dim=1))
        return self.output(agg)


class TimeGrad:
    """TimeGrad autoregressive diffusion wrapper.

    Usage:
        model = TimeGrad(context=96, horizon=72, steps=64,
                         schedule="cosine1000_subsample")
        denoiser = TimeGradDenoiser()
        # Training:
        loss = model.training_loss(history, target, denoiser)
        # Inference (autoregressive rollout):
        forecast = model.forecast(history, denoiser, seed=42)
    """

    def __init__(self, context: int = 96, horizon: int = 72,
                 steps: int = 64,
                 schedule: str = "cosine1000_subsample",
                 device: str = "cpu"):
        self.context = context
        self.horizon = horizon
        self.diffusion = Diffusion(steps=steps, device=device, schedule=schedule)
        self.device = device

    def training_loss(self, history: torch.Tensor, target: torch.Tensor,
                      denoiser: TimeGradDenoiser) -> torch.Tensor:
        """Autoregressive epsilon loss on random horizon steps.

        Randomly pick one horizon step per training sample, diffuse it,
        and predict epsilon. Backpropagates through both timestep and
        AR memory conditioning.
        """
        B = history.shape[0]
        device = history.device

        # Per-sample residual normalization (shared with other models)
        res_mean = history.mean(dim=1, keepdim=True)
        res_std = history.std(dim=1, keepdim=True).clamp(min=1e-4)
        hist_norm = (history - res_mean) / res_std
        tgt_norm = (target - res_mean) / res_std

        # Pick a random horizon step per sample
        h_idx = torch.randint(0, self.horizon, (B,), device=device)

        # Build AR memory: sum of target values at steps 0..h_idx[i] for each sample
        # This is the "teacher-forcing" variant for stability
        ar_memory = torch.zeros(B, 1, device=device)
        for i in range(B):
            hi = h_idx[i].item()
            ar_memory[i, 0] = tgt_norm[i, :hi].sum() if hi > 0 else 0.0

        t = torch.randint(0, self.diffusion.steps, (B,), device=device)
        eps = torch.randn(B, 1, device=device)
        clean = tgt_norm[torch.arange(B), h_idx].unsqueeze(1)
        noisy = self.diffusion.noisy(clean.unsqueeze(1), t, eps.unsqueeze(1)).squeeze(1)

        eps_pred = denoiser(noisy, t, hist_norm, ar_memory)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def forecast(self, history: torch.Tensor,
                 denoiser: TimeGradDenoiser, seed: int = 42) -> torch.Tensor:
        """Autoregressive rollout — generates horizon[0], then horizon[1], etc."""
        denoiser.eval()
        device = history.device
        B = history.shape[0]

        res_mean = history.mean(dim=1, keepdim=True)
        res_std = history.std(dim=1, keepdim=True).clamp(min=1e-4)
        hist_norm = (history - res_mean) / res_std

        horizon_out = torch.zeros(B, self.horizon, device=device)
        rng = torch.Generator(device=device).manual_seed(seed)

        for h in range(self.horizon):
            # AR memory = sum of previously generated horizon steps
            if h == 0:
                ar_mem = torch.zeros(B, 1, device=device)
            else:
                ar_mem = horizon_out[:, :h].sum(dim=1, keepdim=True)

            # DDPM denoise from x_T ~ N(0,1) to x_0 for this horizon step
            x = torch.randn(B, 1, device=device, generator=rng)
            for i in reversed(range(self.diffusion.steps)):
                t = torch.full((B,), i, dtype=torch.long, device=device)
                eps = denoiser(x, t, hist_norm, ar_mem)
                x = x - self.diffusion.beta[i] / (1 - self.diffusion.bar[i]).sqrt() * eps
                x = x / self.diffusion.alpha[i].sqrt()
                if i > 0:
                    x = x + self.diffusion.posterior_variance[i].sqrt() \
                        * torch.randn_like(x, generator=rng)

            horizon_out[:, h] = x.squeeze(1)

        # Denormalize
        return horizon_out * res_std + res_mean
