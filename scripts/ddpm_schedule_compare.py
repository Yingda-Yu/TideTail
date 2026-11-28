"""Controlled CPU diagnostic: schedule/objective/reference-comparison for DDPM magnitude failure.

Does NOT train any new model. All comparisons are pure-math forward/reverse
probes that isolate which architectural/parametrization choices drive the
terminal amplification effect. The historical checkpoint is loaded only to
provide the trained epsilon predictions at fixed noise levels.

Runs under CPU BLAS caps, CUDA_VISIBLE_DEVICES="".
"""
import argparse
import json
import os
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "2"

import numpy as np
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(2)


def cosine_schedule(steps=64, device="cpu"):
    """Historical schedule from torch_diffusion.py — beta_t at step t is DIFFUSION STEP, not array index."""
    t = torch.linspace(0, 1, steps + 1, device=device)
    curve = torch.cos((t + 0.008) / 1.008 * np.pi / 2).square()
    beta = (1 - curve[1:] / curve[:-1]).clamp(0.0001, 0.999)
    return beta


def linear_schedule(steps=64, beta_min=1e-4, beta_max=0.02, device="cpu"):
    """Ho et al. 2020 linear schedule."""
    beta = torch.linspace(beta_min, beta_max, steps, device=device)
    return beta.clamp(0.0001, 0.999)


def sigmoid_schedule(steps=64, start=0.0, end=5.0, device="cpu"):
    """Bertschausen & Brox — sigmoid betas."""
    steps_arr = torch.linspace(start, end, steps, device=device)
    beta = torch.sigmoid(steps_arr - end / 2) * 2 * 0.02  # scale to ~0.02 max
    return beta.clamp(0.0001, 0.999)


def compute_alpha_bar(beta):
    alpha = 1 - beta
    return alpha.cumprod(0)


def compare_schedules(steps=64):
    """Compare terminal alpha_bar and amplification across schedules."""
    results = {}
    for name, beta_fn in [("cosine (historical)", cosine_schedule),
                          ("linear", linear_schedule),
                          ("sigmoid", sigmoid_schedule),
                          ("cosine T=1000 (full)", lambda s=64, d="cpu": cosine_schedule(1000, d)[::1000//64])]:
        try:
            beta = beta_fn(steps)
        except TypeError:
            continue
        alpha_bar = compute_alpha_bar(beta)
        terminal_bar = float(alpha_bar[-1])
        amplification = float(np.sqrt((1 - terminal_bar) / max(terminal_bar, 1e-30)))
        results[name] = {
            "terminal_alpha_bar": terminal_bar,
            "clean_amplification_factor": amplification,
            "final_beta": float(beta[-1]),
        }
    return results


def compare_reverse_step_math():
    """Probe: for given epsilon_predicted, do epsilon/x0/v objective reverses
    produce equivalent clean samples? (They should, mathematically, if
    epsilon is perfect. The question is whether training epsilon target
    is the right one for the sampled objective.)

    With perfect epsilon prediction, all three should give same x_{t-1}.
    With imperfect epsilon prediction, they diverge differently at terminal steps.
    """
    # Test at terminal step t=63 (cosine schedule, alpha_bar ≈ 5.93e-7)
    beta = cosine_schedule(64)
    alpha = 1 - beta
    alpha_bar = compute_alpha_bar(beta)

    t_idx = 63
    abar_t = float(alpha_bar[t_idx])
    beta_t = float(beta[t_idx])
    alpha_t = float(alpha[t_idx])

    # True clean sample x0 = 1.0, true noise epsilon = 0.5
    x0 = 1.0
    eps_true = 0.5
    # Forward
    x_t = np.sqrt(abar_t) * x0 + np.sqrt(1 - abar_t) * eps_true  # ≈ 0.002426
    # epsilon objective: model predicts eps ≈ eps_true (assuming perfect)
    eps_pred = eps_true

    # Epsilon-prediction reverse (historical code)
    eps_rev = x_t - beta_t / np.sqrt(1 - abar_t) * eps_pred
    eps_rev /= np.sqrt(alpha_t)

    # x0-prediction reverse: model predicts x0 directly
    # x0_pred = (x_t - sqrt(1-abar_t) * eps_pred) / sqrt(abar_t)
    # But x0-model output → treat as direct prediction
    x0_pred_direct = x0  # perfect
    x0_rev = np.sqrt(alpha_t) * x0_pred_direct + np.sqrt(1 - alpha_t) * (eps_pred)  # approx

    print(f"  t={t_idx}: alpha_bar={abar_t:.4e}, beta={beta_t:.4f}, amp={np.sqrt((1-abar_t)/abar_t):.1f}×")
    print(f"  True clean x0={x0}, true eps={eps_true:.2f}")
    print(f"  x_t = sqrt(abar)*{x0} + sqrt(1-abar)*{eps_true:.2f} = {x_t:.4e}")
    print(f"  eps-reverse → x_{{t-1}} ≈ {eps_rev:.4f} (target ≈ x_t, which is noisy)")
    print(f"  x0-reverse  → x_{{t-1}} ≈ {x0_rev:.4f}")

    return {
        "t": t_idx, "alpha_bar": abar_t, "beta": beta_t,
        "amplification": float(np.sqrt((1 - abar_t) / abar_t)),
        "x_t": x_t, "eps_rev": eps_rev, "x0_rev": x0_rev,
    }


def fixed_noise_trajectory():
    """Run historical checkpoint reverse sampler with fixed noise,
    per-step clean reconstruction, to isolate terminal amplification."""
    from pathlib import Path

    from tidetail.pilot_data import digest, prepare, safe_file
    from tidetail.torch_diffusion import Diffusion, TemporalDenoiser

    CHECKPOINT_SHA = "dee516cf74b50a0ca8aff8ab05815c7ed36a32ea6c49b9dc81d5b532d39ebb68"
    RUN = Path("/data/yingda/spartina-agent/worktrees/spartina/f1cff330/outputs/issue6-gpu4-pilot-7164")

    # Verify checkpoint
    checkpoint = safe_file(RUN, "checkpoint.pt")
    assert digest(checkpoint) == CHECKPOINT_SHA, "checkpoint SHA mismatch"

    history, target, card = prepare()
    model = TemporalDenoiser().eval()
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    diffusion = Diffusion(64)

    # Fixed noise trajectory: seed 99, first 8 train windows
    rng = torch.Generator().manual_seed(99)
    x = torch.randn(8, 1, 24, generator=rng)
    condition = torch.from_numpy(history[:8])

    # Per-step trace with fixed seed
    trace = []
    with torch.no_grad():
        for index in reversed(range(64)):
            t = torch.full((8,), index, dtype=torch.long)
            eps = model(x, t, condition)

            # Implied clean from epsilon prediction
            implied_clean = (x - (1 - diffusion.bar[index]).sqrt() * eps) / diffusion.bar[index].sqrt()

            physical = implied_clean.numpy() * card["std"] + card["mean"]
            trace.append({
                "step": int(index),
                "alpha_bar": float(diffusion.bar[index]),
                "amplification": float(np.sqrt((1 - diffusion.bar[index].item()) / diffusion.bar[index].item())),
                "epsilon_rms_standardized": float(eps.square().mean().sqrt().item()),
                "implied_clean_std_m": float(physical.std()),
                "implied_clean_min_m": float(physical.min()),
                "implied_clean_max_m": float(physical.max()),
            })

            x = (x - diffusion.beta[index] / (1 - diffusion.bar[index]).sqrt() * eps)
            x = x / diffusion.alpha[index].sqrt()
            if index:
                x += diffusion.posterior_variance[index].sqrt() * torch.randn(
                    x.shape, generator=rng)

    return {
        "train_std_m": card["std"], "train_mean_m": card["mean"],
        "train_target_min_m": float(target.min() * card["std"] + card["mean"]),
        "train_target_max_m": float(target.max() * card["std"] + card["mean"]),
        "train_target_std_standardized": float(target.std()),
        "first_8_train_windows": {
            "shape": list(target[:8].shape),
            "residual_min_m": float(target.min() * card["std"] + card["mean"]),
            "residual_max_m": float(target.max() * card["std"] + card["mean"]),
        },
        "reverse_trace": trace,
        "sample_std_final_m": float(x.numpy().std() * card["std"] + card["mean"]),
        "sample_min_final_m": float(x.numpy().min() * card["std"] + card["mean"]),
        "sample_max_final_m": float(x.numpy().max() * card["std"] + card["mean"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default="/tmp/ddpm_schedule_compare.json")
    args = p.parse_args()

    print("=== SCHEDULE COMPARISON (64 steps) ===")
    sched = compare_schedules(64)
    for name, v in sched.items():
        print(f"  {name}: terminal_alpha_bar={v['terminal_alpha_bar']:.3e}, "
              f"amp={v['clean_amplification_factor']:.1f}×, final_beta={v['final_beta']:.4f}")

    print("\n=== MATH VERIFICATION: epsilon vs x0 reverse ===")
    math_probe = compare_reverse_step_math()

    print("\n=== HISTORICAL CHECKPOINT TRACE (terminal steps) ===")
    try:
        trace = fixed_noise_trajectory()
        print(f"  Train target range: [{trace['train_target_min_m']:.3f}, {trace['train_target_max_m']:.3f}] m")
        print(f"  Sample final range: [{trace['sample_min_final_m']:.3f}, {trace['sample_max_final_m']:.3f}] m")
        steps_to_show = [0, 16, 32, 48, 55, 60, 63]
        for entry in trace["reverse_trace"]:
            if entry["step"] in steps_to_show:
                print(f"    step {entry['step']:2d}: ᾱ={entry['alpha_bar']:.3e}, "
                      f"amp={entry['amplification']:.1f}×, "
                      f"implied_clean_σ={entry['implied_clean_std_m']:.3f} m, "
                      f"ε_rms={entry['epsilon_rms_standardized']:.4f}")
    except RuntimeError as exc:
        print(f"  (checkpoint load failed: {exc})")
        trace = {"error": str(exc), "traceback": traceback.format_exc()}

    result = {
        "schedules_64": sched,
        "math_probe": math_probe,
        "checkpoint_trace": trace,
    }
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
