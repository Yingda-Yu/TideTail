"""Oracle test: same trained checkpoint, DIFFERENT inference schedules.

This does NOT retrain. It loads the historical ε-predicting checkpoint and
runs inference with three alternative schedules: cosine(64, historical),
linear(64), sigmoid(64). The trained ε predictions come from the cosine-64
schedule, so this is a controlled probe (not a scientific fix) — but it
directly answers whether terminal schedule drives sample magnitude.

If linear/sigmoid produce physically plausible samples, terminal schedule is
a primary root cause. If not, training budget or architecture dominates.
"""
import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "2"

from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(2)

from tidetail.pilot_data import digest, prepare, safe_file


def make_schedule(name, steps, device="cpu"):
    """Build beta, alpha_bar, posterior_variance for a given schedule name."""
    if name == "cosine":
        t = torch.linspace(0, 1, steps + 1, device=device)
        curve = torch.cos((t + 0.008) / 1.008 * np.pi / 2).square()
        beta = (1 - curve[1:] / curve[:-1]).clamp(0.0001, 0.999)
    elif name == "linear":
        beta = torch.linspace(1e-4, 0.02, steps, device=device).clamp(0.0001, 0.999)
    elif name == "sigmoid":
        steps_arr = torch.linspace(0, 5.0, steps, device=device)
        beta = (torch.sigmoid(steps_arr - 2.5) * 0.02).clamp(0.0001, 0.999)
    elif name == "cosine1000":
        # Full T=1000 cosine, subsample every 1000//64 ≈ 15 steps
        t = torch.linspace(0, 1, 1001, device=device)
        curve = torch.cos((t + 0.008) / 1.008 * np.pi / 2).square()
        beta_full = (1 - curve[1:] / curve[:-1]).clamp(0.0001, 0.999)
        idx = torch.linspace(0, 999, steps, dtype=torch.long)
        beta = beta_full[idx]
    else:
        raise ValueError(f"unknown schedule {name}")

    alpha = 1 - beta
    alpha_bar = alpha.cumprod(0)
    prev = torch.cat((torch.ones(1, device=device), alpha_bar[:-1]))
    posterior_variance = beta * (1 - prev) / (1 - alpha_bar)
    return beta, alpha, alpha_bar, posterior_variance


def sample_with_schedule(model, history, steps, beta, alpha, alpha_bar,
                          posterior_variance, seed=99):
    """Run reverse sampling with manually-provided schedule."""
    rng = torch.Generator(device=history.device).manual_seed(seed)
    x = torch.randn((len(history), 1, model.horizon), device=history.device,
                    generator=rng)
    for index in reversed(range(steps)):
        t = torch.full((len(history),), index, dtype=torch.long, device=x.device)
        eps = model(x, t, history)
        x = x - beta[index] / (1 - alpha_bar[index]).sqrt() * eps
        x = x / alpha[index].sqrt()
        if index:
            x = x + posterior_variance[index].sqrt() * torch.randn(
                x.shape, device=x.device, generator=rng)
    return x


def main():
    CHECKPOINT_SHA = "dee516cf74b50a0ca8aff8ab05815c7ed36a32ea6c49b9dc81d5b532d39ebb68"
    RUN = Path("/data/yingda/spartina-agent/worktrees/spartina/f1cff330/outputs/"
               "issue6-gpu4-pilot-7164")

    checkpoint = safe_file(RUN, "checkpoint.pt")
    assert digest(checkpoint) == CHECKPOINT_SHA, "checkpoint SHA mismatch"

    from tidetail.torch_diffusion import TemporalDenoiser

    history, target, card = prepare()

    model = TemporalDenoiser().eval()
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)

    condition = torch.from_numpy(history[:8])
    results = {}

    for sched_name in ["cosine", "linear", "sigmoid", "cosine1000"]:
        beta, alpha, alpha_bar, posterior_variance = make_schedule(sched_name, 64)
        sample = sample_with_schedule(model, condition, 64, beta, alpha,
                                      alpha_bar, posterior_variance, seed=99)
        physical = sample[:, 0].detach().numpy() * card["std"] + card["mean"]
        results[sched_name] = {
            "terminal_alpha_bar": float(alpha_bar[-1]),
            "amplification": float(np.sqrt((1 - alpha_bar[-1].item()) / alpha_bar[-1].item())),
            "terminal_beta": float(beta[-1]),
            "samples_m": {
                "min": float(physical.min()),
                "max": float(physical.max()),
                "std": float(physical.std()),
                "range": float(physical.max() - physical.min()),
            },
            "train_reference": {
                "target_std_m": float(card["std"]),
                "target_mean_m": float(card["mean"]),
                "range_m": [float(target.min() * card["std"] + card["mean"]),
                            float(target.max() * card["std"] + card["mean"])],
            },
        }
        r = results[sched_name]
        print(f"  {sched_name:12s}: ᾱ_T={r['terminal_alpha_bar']:.3e}, amp={r['amplification']:.1f}×, "
              f"range=[{r['samples_m']['min']:.2f}, {r['samples_m']['max']:.2f}] m, σ={r['samples_m']['std']:.3f} m")

    train_range = results["cosine"]["train_reference"]
    print(f"\n  Train residual range: [{train_range['range_m'][0]:.3f}, {train_range['range_m'][1]:.3f}] m, σ={train_range['target_std_m']:.4f} m")

    # Scientific note — writing schedule without retraining is oracle-only
    results["disclaimer"] = (
        "These are NOT scientific fix demonstrations. The checkpoint was trained "
        "on the historical cosine schedule and ε predictions are schedule-specific. "
        "Oracle results indicate whether terminal schedule drives magnitude; "
        "a proper fix requires retraining on the new schedule and independent "
        "held-out evaluation."
    )

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default="/tmp/ddpm_oracle_schedule.json")
    args = p.parse_args()
    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
