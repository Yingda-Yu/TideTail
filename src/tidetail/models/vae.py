"""Conditional Variational Autoencoder for 96h→72h tide forecasting.

Architecture follows the same TCN-style pattern as DiffusionTS and
FlowMatching so that the VAE is a controlled generative baseline
against DDPM and CNF frameworks.

Core idea
---------
Encoder maps (history, target) → latent (mu, logvar).
Decoder maps (history, z) → target reconstruction.
Loss = MSE reconstruction + beta * KL(z || N(0, I)).

The conditioning history gives the VAE context about which tidal
regime we're in; the latent z captures stochastic variability in
the forecast horizon.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# TCN building blocks (shared pattern with other TideTail models)
# ---------------------------------------------------------------------------

class _VAResBlock(nn.Module):
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
# Conditional VAE Encoder
# ---------------------------------------------------------------------------

class ConditionalVAEEncoder(nn.Module):
    """Encodes (history, target) → latent parameters (mu, logvar)."""

    def __init__(self, context: int = 96, horizon: int = 72,
                 width: int = 32, latent_dim: int = 32):
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.width = width
        self.latent_dim = latent_dim

        # History context embedding
        self.history_enc = nn.Sequential(
            nn.Linear(context, width), nn.SiLU(), nn.Linear(width, width)
        )

        # Target encoder: Conv1d over horizon
        self.target_proj = nn.Conv1d(1, width, 3, padding=1)
        self.blocks = nn.ModuleList([
            _VAResBlock(width, d) for d in (1, 2, 4, 8)
        ])

        # Combine encoded target + history → latent params
        # After TCN we global-average pool horizon dim → width, then concat with history
        self.fc_mu = nn.Linear(width + width, latent_dim)
        self.fc_logvar = nn.Linear(width + width, latent_dim)

    def forward(self, history: torch.Tensor, target: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            history: (batch, context)
            target: (batch, horizon)
        Returns:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        h_emb = self.history_enc(history)  # (B, width)

        x = self.target_proj(target.unsqueeze(1))  # (B, width, horizon)
        for block in self.blocks:
            x = block(x, h_emb)

        # Global average pool
        x_pool = x.mean(dim=2)  # (B, width)

        combined = torch.cat([x_pool, h_emb], dim=1)  # (B, 2*width)
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        return mu, logvar


# ---------------------------------------------------------------------------
# Conditional VAE Decoder
# ---------------------------------------------------------------------------

class ConditionalVAEDecoder(nn.Module):
    """Decodes (history, z) → target reconstruction."""

    def __init__(self, context: int = 96, horizon: int = 72,
                 width: int = 32, latent_dim: int = 32):
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.width = width
        self.latent_dim = latent_dim

        # History context embedding
        self.history_enc = nn.Sequential(
            nn.Linear(context, width), nn.SiLU(), nn.Linear(width, width)
        )

        # Latent conditioning
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, width), nn.SiLU(), nn.Linear(width, width)
        )

        # Combine history + latent → conditioning embedding
        self.cond_proj = nn.Sequential(
            nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, width)
        )

        # Decoder TCN: starts from latent broadcast to horizon length
        self.input_proj = nn.Conv1d(1, width, 3, padding=1)
        self.blocks = nn.ModuleList([
            _VAResBlock(width, d) for d in (1, 2, 4, 8)
        ])
        self.output = nn.Sequential(
            nn.GroupNorm(4, width), nn.SiLU(), nn.Conv1d(width, 1, 3, padding=1)
        )

    def forward(self, history: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (batch, context)
            z: (batch, latent_dim)
        Returns:
            recon: (batch, horizon)
        """
        h_emb = self.history_enc(history)
        z_emb = self.latent_proj(z)
        cond = self.cond_proj(torch.cat([h_emb, z_emb], dim=1))

        # Start from broadcast latent signal: use z as initial input
        # We project z to horizon length via a tiny head
        init_signal = z @ torch.ones(z.shape[1], self.horizon, device=z.device) / math.sqrt(self.horizon)
        x = self.input_proj(init_signal.unsqueeze(1))  # (B, width, horizon)
        for block in self.blocks:
            x = block(x, cond)
        recon = self.output(x).squeeze(1)  # (B, horizon)
        return recon


# ---------------------------------------------------------------------------
# Full Conditional VAE model
# ---------------------------------------------------------------------------

class ConditionalVAE:
    """Conditional VAE for 96h→72h tide forecasting.

    Usage:
        vae = ConditionalVAE(context=96, horizon=72, latent_dim=32, beta=0.001)
        enc = vae.encoder
        dec = vae.decoder
        # Training:
        loss = vae.training_loss(history, target, enc, dec)
        # Sampling:
        forecast = vae.sample(history, dec, seed=42)
    """

    def __init__(self, context: int = 96, horizon: int = 72,
                 width: int = 32, latent_dim: int = 32,
                 beta: float = 0.001, device: str = "cpu"):
        self.context = context
        self.horizon = horizon
        self.width = width
        self.latent_dim = latent_dim
        self.beta = beta
        self.device = device

        self.encoder = ConditionalVAEEncoder(context, horizon, width, latent_dim).to(device)
        self.decoder = ConditionalVAEDecoder(context, horizon, width, latent_dim).to(device)

    def training_loss(self, history: torch.Tensor, target: torch.Tensor,
                      encoder: ConditionalVAEEncoder | None = None,
                      decoder: ConditionalVAEDecoder | None = None,
                      kl_weight: float | None = None) -> torch.Tensor:
        """VAE loss = MSE_recon + beta * KL(z || N(0, I)).

        Args:
            history: (batch, context) — conditioning history (normalized)
            target: (batch, horizon) — target forecast (normalized)
            encoder/decoder: optional overrides; defaults to self.encoder/decoder
            kl_weight: override self.beta
        """
        enc = encoder or self.encoder
        dec = decoder or self.decoder
        beta = kl_weight if kl_weight is not None else self.beta

        mu, logvar = enc(history, target)
        # Reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        recon = dec(history, z)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, target)

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + beta * kl

    @torch.no_grad()
    def sample(self, history: torch.Tensor,
               decoder: ConditionalVAEDecoder | None = None,
               seed: int = 42, n_samples: int | None = None) -> torch.Tensor:
        """Generate forecasts by sampling z ~ N(0, I) and decoding.

        Args:
            history: (batch, context)
            decoder: optional override
            seed: RNG seed
            n_samples: if set, draw multiple z per history and return
                      (batch, n_samples, horizon)
        Returns:
            forecast: (batch, horizon) or (batch, n_samples, horizon)
        """
        dec = decoder or self.decoder
        dec.eval()
        device = history.device

        rng = torch.Generator(device=device).manual_seed(seed)
        B = history.shape[0]

        if n_samples is not None and n_samples > 1:
            # Sample multiple z per history
            z = torch.randn((B, n_samples, self.latent_dim),
                            device=device, generator=rng)
            # Flatten batch + samples
            h_rep = history.unsqueeze(1).expand(B, n_samples, -1).reshape(-1, self.context)
            z_flat = z.reshape(-1, self.latent_dim)
            out = dec(h_rep, z_flat)  # (B*n_samples, horizon)
            return out.reshape(B, n_samples, self.horizon)
        else:
            z = torch.randn((B, self.latent_dim), device=device, generator=rng)
            return dec(history, z)


def vae_count_params(vae: ConditionalVAE) -> int:
    """Count total trainable parameters in encoder + decoder."""
    return (sum(p.numel() for p in vae.encoder.parameters())
            + sum(p.numel() for p in vae.decoder.parameters()))
