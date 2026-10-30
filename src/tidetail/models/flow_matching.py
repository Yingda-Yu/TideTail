"""M4 — Flow Matching / continuous normalising flow baseline.

Reference
---------
Lipman et al. (2023). "Flow Matching for Generative Modeling."
https://arxiv.org/abs/2210.02747

Overview
--------
A continuous normalising flow that learns a velocity field ``v(x_t, t)``
transporting samples from a simple prior (Gaussian) to the data
distribution. Unlike DDPM, the training objective is a *single*
regression loss on the conditional velocity — no beta schedule, no
variance annealing, no posterior variance term.

For TideTail (96 h history → 72 h forecast, standardized):

    x_t = (1 - t) * x_0 + t * x_1,    t ∈ [0, 1]
    loss = ‖v_θ(x_t, t, context) - (x_1 - x_0)‖²

where ``x_0 ~ N(0, I)`` is noise and ``x_1`` is the target forecast.
Sampling integrates the learned ODE from t=0 to t=1 using Euler steps.

This model is a control for whether the *choice* of generative
framework (DDPM vs. CNF) materially affects extreme-value synthesis,
holding the denoiser architecture constant.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Velocity network (same TCN architecture as Diffusion-TS residual denoiser)
# ---------------------------------------------------------------------------

class VelocityTCN(nn.Module):
    """Dilated TCN velocity field predictor.

    Architecture mirrors ``models.diffusion_ts.ResidualDenoiser`` so
    that DDPM and Flow-Matching baselines are comparable.
    """

    def __init__(self, context: int = 96, horizon: int = 72,
                 width: int = 32, hidden_blocks: Sequence[int] = (1, 2, 4, 8)):
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.width = width

        self.history_enc = nn.Sequential(
            nn.Linear(context, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.time_enc = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.input_proj = nn.Conv1d(1, width, 3, padding=1)
        self.blocks = nn.ModuleList([
            _FMResBlock(width, d) for d in hidden_blocks
        ])
        self.output = nn.Sequential(
            nn.GroupNorm(4, width), nn.SiLU(), nn.Conv1d(width, 1, 3, padding=1)
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                history: torch.Tensor) -> torch.Tensor:
        """Predict velocity at (x_t, t) given history context.

        Args:
            x_t: (batch, 1, horizon) — current state along the interpolation
            t: (batch,) scalar time ∈ [0, 1]
            history: (batch, context) — conditioning history
        Returns:
            v: (batch, 1, horizon) — predicted velocity field
        """
        if x_t.ndim != 3 or x_t.shape[1:] != (1, self.horizon):
            raise ValueError(f"x_t must be (batch, 1, {self.horizon})")

        # Sinusoidal time embedding
        half_w = max(self.width // 2, 1)
        freq = torch.exp(torch.arange(half_w, device=x_t.device)
                         * (-math.log(10000) / max(half_w - 1, 1)))
        angles = t.float()[:, None] * freq[None]
        t_emb = self.time_enc(torch.cat([angles.sin(), angles.cos()], dim=1))

        h_emb = self.history_enc(history)
        emb = t_emb + h_emb

        x = self.input_proj(x_t)
        for block in self.blocks:
            x = block(x, emb)
        return self.output(x)


class _FMResBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Conv1d(width, width, 1),
        )
        self.cond = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return (x + self.net(x + self.cond(emb)[:, :, None])) / math.sqrt(2)


# ---------------------------------------------------------------------------
# Full Flow Matching model
# ---------------------------------------------------------------------------

class FlowMatching:
    """Continuous normalising flow for 96h→72h tide forecasting.

    Usage
    -----
        fm = FlowMatching(context=96, horizon=72, euler_steps=64)
        velocity = VelocityTCN(context=96, horizon=72, width=32)
        # Training:
        loss = fm.training_loss(history, target, velocity)
        # Sampling (Euler ODE):
        forecast = fm.sample(history, velocity, seed=42)
    """

    def __init__(self, context: int = 96, horizon: int = 72,
                 euler_steps: int = 64, device: str = "cpu"):
        self.context = context
        self.horizon = horizon
        self.euler_steps = euler_steps
        self.device = device

    def training_loss(self, history: torch.Tensor, target: torch.Tensor,
                      velocity_net: nn.Module) -> torch.Tensor:
        """Conditional Flow Matching loss — single regression objective.

        Sample t ~ Uniform(0, 1), noise x_0 ~ N(0, I), interpolate
        x_t = (1 - t) x_0 + t x_1. Regress velocity target x_1 - x_0.
        """
        B = target.shape[0]
        device = target.device

        t = torch.rand(B, 1, 1, device=device)  # (B, 1, 1) for broadcast
        x_0 = torch.randn(B, 1, self.horizon, device=device)
        x_1 = target.unsqueeze(1)  # (B, 1, horizon)

        x_t = (1 - t) * x_0 + t * x_1
        target_vel = x_1 - x_0

        v_pred = velocity_net(x_t, t.squeeze(-1).squeeze(-1), history)
        return F.mse_loss(v_pred, target_vel)

    @torch.no_grad()
    def sample(self, history: torch.Tensor, velocity_net: nn.Module,
               seed: int = 42, euler_steps: int | None = None) -> torch.Tensor:
        """Integrate ODE dx/dt = v(x, t, history) from t=0 to t=1 (Euler).

        Args:
            history: (batch, context)
            velocity_net: trained VelocityTCN
            seed: RNG seed for initial noise
            euler_steps: override default integration steps

        Returns:
            forecast: (batch, horizon)
        """
        velocity_net.eval()
        device = history.device
        steps = euler_steps or self.euler_steps
        dt = 1.0 / steps

        rng = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn((history.shape[0], 1, self.horizon),
                        device=device, generator=rng)

        for i in range(steps):
            t = torch.full((history.shape[0],), (i + 0.5) * dt,
                           device=device)
            v = velocity_net(x, t, history)
            x = x + v * dt

        return x.squeeze(1)


# ---------------------------------------------------------------------------
# Optional: richer OT-style path (straight line is default)
# ---------------------------------------------------------------------------

def ot_train_loss(history: torch.Tensor, target: torch.Tensor,
                  velocity_net: nn.Module, sigma_min: float = 1e-4,
                  path_sampler: Callable | None = None) -> torch.Tensor:
    """Flow Matching with Optimal-Transport-inspired noise schedule.

    Uses σ(t) = sigma_min^(1-t) to make the interpolation better-behaved
    near t=0 (avoids the `(1-t)` term vanishing derivative issue).

    This is a drop-in replacement for :meth:`FlowMatching.training_loss`
    when you want a more principled noise interpolation.
    """
    B = target.shape[0]
    device = target.device
    t = torch.rand(B, 1, 1, device=device)
    x_0 = torch.randn(B, 1, target.shape[1], device=device)
    x_1 = target.unsqueeze(1)

    sigma_t = sigma_min ** (1 - t)
    x_t = (1 - t) * x_1 + t * x_0 + sigma_t * torch.randn_like(x_0)
    target_vel = x_0 - x_1

    v_pred = velocity_net(x_t, t.squeeze(-1).squeeze(-1), history)
    return F.mse_loss(v_pred, target_vel)
