"""TideTail generative models — unified baseline implementations.

All models target the 96 h history → 72 h horizon contract and operate
on standardized water-level residuals.

Modules
-------
diffusion_ts   — M2: trend/seasonality decomposition + residual DDPM
flow_matching  — M4: continuous normalizing flow baseline (non-DDPM)
timegrad       — M3: autoregressive conditional diffusion
physics_diffusion — M5: rate-of-change + peak-magnitude constraints
                   — M6: tail-aware extension (primary H1 candidate)
"""
from .diffusion_ts import (
    DiffusionTS,
    ResidualDenoiser,
    decompose,
    extrapolate_seasonal,
    extrapolate_trend,
    fft_seasonal,
    moving_average_trend,
)

__all__ = [
    # M2
    "DiffusionTS",
    "ResidualDenoiser",
    "decompose",
    "extrapolate_seasonal",
    "extrapolate_trend",
    "fft_seasonal",
    "moving_average_trend",
]
