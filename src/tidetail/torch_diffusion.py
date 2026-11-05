"""Small conditional epsilon-predicting DDPM; no training at import time.

Schedule choices
----------------
``schedule="cosine"`` (default, historical) — direct cosine over [0,1]
with ``steps`` points. Terminal α_bar ≈ 5.9e-07 at 64 steps → 1299×
clean amplification. DO NOT use for new experiments; retained only for
back-compat with historical checkpoint traces.

``schedule="cosine1000_subsample"`` — compute cosine over 1000 points,
then subsample ``steps`` evenly-spaced betas. At 64 steps this gives
terminal α_bar ≈ 5.9e-4 → 41× amplification, physically plausible.
**REQUIRES RETRAINING** (epsilon predictions are schedule-specific).

``schedule="linear"`` — Ho et al. 2020 linear β ∈ [1e-4, 0.02].
"""

import math
from typing import Literal

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Schedule constructors
# ---------------------------------------------------------------------------

def _cosine_beta(steps: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Imagen cosine schedule over [0,1] with `steps` betas."""
    t = torch.linspace(0, 1, steps + 1, device=device)
    curve = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
    beta = (1 - curve[1:] / curve[:-1]).clamp(0.0001, 0.999)
    return beta


def _cosine1000_subsample_beta(steps: int,
                               device: torch.device | str = "cpu") -> torch.Tensor:
    """cosine over 1000 points, subsample `steps` evenly-spaced betas.

    At 64 steps → terminal α_bar ≈ 5.9e-4 (vs 5.9e-7 for direct 64).
    """
    beta_dense = _cosine_beta(1000, device=device)
    idx = np.linspace(0, 999, steps, dtype=int)
    return beta_dense[torch.from_numpy(idx).to(device)]


def _linear_beta(steps: int, beta_min: float = 1e-4, beta_max: float = 0.02,
                 device: torch.device | str = "cpu") -> torch.Tensor:
    beta = torch.linspace(beta_min, beta_max, steps, device=device)
    return beta.clamp(0.0001, 0.999)


SCHEDULE_REGISTRY = {
    "cosine": _cosine_beta,
    "cosine1000_subsample": _cosine1000_subsample_beta,
    "linear": _linear_beta,
}


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Conv1d(width, width, 1),
        )
        self.condition = nn.Linear(width, width)

    def forward(self, x, embedding):
        return (x + self.net(x + self.condition(embedding)[:, :, None])) / math.sqrt(2)


class TemporalDenoiser(nn.Module):
    """Condition on all 72 past residuals, never future weather or levels."""

    def __init__(self, context=72, horizon=24, width=32):
        super().__init__()
        self.context, self.horizon, self.width = context, horizon, width
        self.history = nn.Sequential(nn.Linear(context, width), nn.SiLU(), nn.Linear(width, width))
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.input = nn.Conv1d(1, width, 3, padding=1)
        self.blocks = nn.ModuleList(ResidualBlock(width, d) for d in (1, 2, 4, 8))
        self.output = nn.Sequential(
            nn.GroupNorm(4, width), nn.SiLU(), nn.Conv1d(width, 1, 3, padding=1)
        )

    def forward(self, noisy, timestep, history):
        if noisy.ndim != 3 or noisy.shape[1:] != (1, self.horizon):
            raise ValueError("noisy must have shape (batch, 1, horizon)")
        if history.shape != (len(noisy), self.context) or timestep.shape != (len(noisy),):
            raise ValueError("history/timestep batch shape mismatch")
        frequencies = torch.exp(
            torch.arange(self.width // 2, device=noisy.device)
            * (-math.log(10000) / (self.width // 2 - 1))
        )
        angles = timestep.float()[:, None] * frequencies[None]
        embedding = self.time(torch.cat((angles.sin(), angles.cos()), dim=1))
        embedding = embedding + self.history(history)
        x = self.input(noisy)
        for block in self.blocks:
            x = block(x, embedding)
        return self.output(x)


class Diffusion:
    def __init__(self, steps=64, device="cpu",
                 schedule: Literal["cosine", "cosine1000_subsample", "linear"]
                 = "cosine1000_subsample"):
        if steps < 2:
            raise ValueError("at least two diffusion steps required")
        if schedule not in SCHEDULE_REGISTRY:
            raise ValueError(
                f"unknown schedule {schedule!r}; "
                f"choose from {list(SCHEDULE_REGISTRY)}"
            )
        self.steps = steps
        self.schedule_name = schedule
        self.beta = SCHEDULE_REGISTRY[schedule](steps, device=device)
        self.alpha = 1 - self.beta
        self.bar = self.alpha.cumprod(0)
        previous = torch.cat((torch.ones(1, device=self.bar.device),
                              self.bar[:-1]))
        self.posterior_variance = self.beta * (1 - previous) / (1 - self.bar)

    # --- Convenience diagnostics --------------------------------------------------
    @property
    def terminal_alpha_bar(self) -> float:
        return float(self.bar[-1])

    @property
    def terminal_amplification(self) -> float:
        """sqrt((1 - ᾱ_T) / ᾱ_T)) — clean-signal vs noise std ratio at T."""
        ab = self.terminal_alpha_bar
        return float(math.sqrt(max(1 - ab, 0) / max(ab, 1e-30)))

    def noisy(self, clean, timestep, noise):
        bar = self.bar[timestep][:, None, None]
        return bar.sqrt() * clean + (1 - bar).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, history, seed=99):
        if not torch.isfinite(history).all():
            raise ValueError("nonfinite conditioning history")
        rng = torch.Generator(device=history.device).manual_seed(seed)
        x = torch.randn((len(history), 1, model.horizon), device=history.device, generator=rng)
        for index in reversed(range(self.steps)):
            t = torch.full((len(history),), index, dtype=torch.long, device=x.device)
            epsilon = model(x, t, history)
            if not torch.isfinite(epsilon).all():
                raise ValueError(f"nonfinite epsilon prediction at step {index}")
            x = x - self.beta[index] / (1 - self.bar[index]).sqrt() * epsilon
            x = x / self.alpha[index].sqrt()
            if index:
                noise = torch.randn(x.shape, device=x.device, generator=rng)
                x = x + self.posterior_variance[index].sqrt() * noise
            if not torch.isfinite(x).all():
                raise ValueError(f"nonfinite reverse sample at step {index}")
        return x


def sample_diagnostics(samples, train_reference, max_std_ratio=5.0, min_std_ratio=0.1):
    """Engineering rejection screen in common units, never a realism certificate.

    Reference must be the frozen train-only target, not validation/test data.
    Limits are explicit policy defaults, not estimates of physical plausibility.
    Preserve rejected samples; do not clip them into an apparent success.
    """
    reference = np.asarray(train_reference, dtype=np.float64)
    values = np.asarray(samples, dtype=np.float64)
    if (reference.size < 2 or not np.isfinite(reference).all()
            or reference.std() <= 0):
        raise ValueError("finite, nonconstant training reference required")
    if not 0 < min_std_ratio < max_std_ratio or not np.isfinite(max_std_ratio):
        raise ValueError("invalid scale limits")
    finite = bool(values.size and np.isfinite(values).all())
    ratio = float(values.std() / reference.std()) if finite else None
    offset = float(abs(values.mean() - reference.mean()) / reference.std()) if finite else None
    accepted = bool(finite and min_std_ratio <= ratio <= max_std_ratio
                    and offset <= max_std_ratio)
    return {"finite": finite, "accepted": accepted, "std_ratio": ratio,
            "mean_offset_train_std": offset,
            "min_std_ratio": min_std_ratio, "max_std_ratio": max_std_ratio,
            "claim": "Engineering scale screen only; realism and utility unvalidated"}
