"""Evaluate existing trained checkpoints without retraining.

Usage:
    # Evaluate all Diffusion-TS checkpoints on GPU3
    CUDA_VISIBLE_DEVICES=3 python3 scripts/eval_existing_checkpoints.py --model diffusion_ts

    # Evaluate VAE checkpoints
    CUDA_VISIBLE_DEVICES=3 python3 scripts/eval_existing_checkpoints.py --model vae

    # Evaluate TimeGrad checkpoints  
    CUDA_VISIBLE_DEVICES=3 python3 scripts/eval_existing_checkpoints.py --model timegrad

    # Evaluate all models
    CUDA_VISIBLE_DEVICES=3 python3 scripts/eval_existing_checkpoints.py --model all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tidetail.models import (
    ConditionalVAE,
    DiffusionTS,
    FlowMatching,
    ResidualDenoiser,
    TimeGrad,
    TimeGradDenoiser,
    VelocityTCN,
)

DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
RESULTS_DIR = ROOT / "data" / "results"
CKPT_DIR = ROOT / "data" / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_region(region: str, context: int = 96, horizon: int = 72):
    from scripts.train_gen_bench import load_region as _lr, compute_residual_stats, generate_samples
    return _lr(region, context, horizon)


def evaluate_model(model_name: str, data, region: str, seed: int, device: str):
    """Load checkpoint and evaluate."""
    ckpt_path = CKPT_DIR / f"{model_name}_{region}_seed{seed}.pt"
    if not ckpt_path.exists():
        print(f"  SKIP: {ckpt_path} not found")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = ckpt.get("model_config", {})
    ctx = mc.get("context", 96)
    hrz = mc.get("horizon", 72)

    # Reconstruct model
    if model_name == "vae":
        model = ConditionalVAE(context=ctx, horizon=hrz, width=32, latent_dim=32,
                               beta=0.001, device=device).to(device)
        model.encoder.load_state_dict(ckpt["encoder_state_dict"])
        model.decoder.load_state_dict(ckpt["decoder_state_dict"])
        model_obj = model
        nets = model
    elif model_name == "flow_matching":
        model_obj = FlowMatching(context=ctx, horizon=hrz, euler_steps=64, device=device)
        nets = VelocityTCN(context=ctx, horizon=hrz, width=32).to(device)
        nets.load_state_dict(ckpt["state_dict"])
    elif model_name == "diffusion_ts":
        model_obj = DiffusionTS(context=ctx, horizon=hrz, steps=64,
                                schedule="cosine1000_subsample", device=device)
        nets = ResidualDenoiser(context=ctx, horizon=hrz, width=32,
                                use_full_context=True).to(device)
        nets.load_state_dict(ckpt["state_dict"])
    elif model_name == "timegrad":
        model_obj = TimeGrad(context=ctx, horizon=hrz, steps=64,
                             schedule="cosine1000_subsample", device=device)
        nets = TimeGradDenoiser(context=ctx, horizon=hrz, width=24).to(device)
        nets.load_state_dict(ckpt["state_dict"])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Use generate_samples from train_gen_bench
    from scripts.train_gen_bench import generate_samples
    result, gen, real = generate_samples(
        model_name, model_obj, nets, data, device, seed, n_samples=5000
    )

    # Save samples
    sample_path = RESULTS_DIR / f"{model_name}_{region}_seed{seed}_samples.npz"
    np.savez(sample_path, generated=gen, real=real)

    result["region"] = region
    result["checkpoint_path"] = str(ckpt_path)
    try:
        hist = ckpt.get("history", {})
        result["epochs_trained"] = len(hist.get("epoch", []))
        result["final_train_loss"] = float(hist.get("train_loss", [None])[-1]) if hist.get("train_loss") else None
    except Exception:
        pass

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["vae", "flow_matching", "diffusion_ts", "timegrad", "all"])
    ap.add_argument("--regions", nargs="+",
                    default=["ny_bight", "us_gulf_coast", "south_atlantic_charleston",
                             "southeast_australia", "thames_north_sea"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 99])
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-suffix", default="",
                    help="Suffix for output JSON filename (e.g., '_full_eval')")
    args = ap.parse_args()

    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda" or torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    git_sha = __import__("os").popen("git rev-parse HEAD").read().strip()
    print(f"Git SHA: {git_sha}")

    models_to_eval = {
        "all": ["diffusion_ts", "vae", "timegrad", "flow_matching"],
    }.get(args.model, [args.model])

    all_results = []
    for model_name in models_to_eval:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        for region in args.regions:
            print(f"\n  Region: {region}")
            data = load_region(region)

            for seed in args.seeds:
                print(f"    Seed {seed}...", end=" ", flush=True)
                t0 = time.time()
                result = evaluate_model(model_name, data, region, seed, device)
                if result is None:
                    continue
                result["runtime_s"] = time.time() - t0
                all_results.append(result)
                print(f"scale={result['scale_ratio']:.3f} gate={result['gate']} "
                      f"time={result['runtime_s']:.1f}s")

    # Save
    suffix = args.output_suffix
    out_path = RESULTS_DIR / f"{models_to_eval[0]}{suffix}_96_72.json"
    if len(models_to_eval) == 1 and suffix == "":
        # Overwrite the standard results file
        out_path = RESULTS_DIR / f"{models_to_eval[0]}_96_72.json"
    else:
        out_path = RESULTS_DIR / f"eval_all_models{suffix}_96_72.json"

    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {len(all_results)} results → {out_path}")

    # Summary per model
    by_model = {}
    for r in all_results:
        m = r.get("model", "?")
        by_model.setdefault(m, []).append(r)
    for m, results in by_model.items():
        scales = [r["scale_ratio"] for r in results]
        gates = [r["gate"] for r in results]
        print(f"\n  {m}: {len(results)} ckpts, "
              f"scale=[{min(scales):.3f}, {max(scales):.3f}], "
              f"gates: {dict((g, gates.count(g)) for g in set(gates))}")


if __name__ == "__main__":
    main()
