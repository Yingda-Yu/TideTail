"""Bounded, pinned-checkpoint CPU diagnosis; no optimization or CUDA calls."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

# Set before importing torch; never inherit a visible CUDA allocation.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"

import numpy as np
import torch

from tidetail.pilot_data import digest, prepare, safe_file
from tidetail.torch_diffusion import Diffusion, TemporalDenoiser, sample_diagnostics

RUN = Path("/data/yingda/spartina-agent/worktrees/spartina/f1cff330/outputs/"
           "issue6-gpu4-pilot-7164")
CHECKPOINT_SHA = "dee516cf74b50a0ca8aff8ab05815c7ed36a32ea6c49b9dc81d5b532d39ebb68"
SOURCE_SHA = "5bc646676a08191a77e5d424f2cf52bb1af00c04"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    checkpoint = safe_file(RUN, "checkpoint.pt")
    if checkpoint.stat().st_size > 2 * 1024**2 or digest(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("checkpoint identity/size mismatch; refuse load")
    config = json.loads(safe_file(RUN, "config.json").read_text())
    card = json.loads(safe_file(RUN, "data_card.json").read_text())
    if config["code_sha"] != SOURCE_SHA or config["dirty"]:
        raise ValueError("training source identity mismatch")
    source_hashes = {}
    for recorded_path, expected in config["source_code_hashes"].items():
        relative = ("scripts/train_gpu_pilot.py" if recorded_path.startswith("/")
                    else recorded_path)
        if relative not in ("scripts/train_gpu_pilot.py", "src/tidetail/torch_diffusion.py",
                            "src/tidetail/pilot_data.py"):
            raise ValueError("unexpected source path")
        source = subprocess.check_output(["git", "show", f"{SOURCE_SHA}:{relative}"])
        source_hashes[relative] = hashlib.sha256(source).hexdigest()
        if source_hashes[relative] != expected:
            raise ValueError("recorded source hash mismatch")
    history, target, actual_card = prepare()  # rehash owner data BEFORE checkpoint load
    if actual_card != card:
        raise ValueError("data/scaler/event/split card mismatch")
    # Hash-pinned owner-produced 435KiB checkpoint; restricted weights-only CPU load.
    # TorchVersion is the sole extra class in the saved primitive config.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["config"] != config or state["data_card"] != card or state["steps"] != 1641:
        raise ValueError("checkpoint/config/data identity mismatch")
    model = TemporalDenoiser().eval()
    model.load_state_dict(state["model"], strict=True)
    diffusion = Diffusion(64)
    condition = torch.from_numpy(history[:8])
    sample = diffusion.sample(model, condition, seed=99)
    assert torch.equal(sample, diffusion.sample(model, condition, seed=99))
    physical = sample[:, 0].numpy() * card["std"] + card["mean"]
    with np.load(safe_file(RUN, "samples.npz"), allow_pickle=False) as saved:
        original = saved["residual_m"].copy()
    rng = torch.Generator().manual_seed(99)
    x = torch.randn(8, 1, 24, generator=rng)
    trace = []
    fixed_noise = torch.randn(8, 1, 24, generator=torch.Generator().manual_seed(20261028))
    clean = torch.from_numpy(target[:8, None])
    calibration = []
    with torch.no_grad():
        for index in reversed(range(64)):
            t = torch.full((8,), index, dtype=torch.long)
            epsilon = model(x, t, condition)
            clean_prediction = (x - (1-diffusion.bar[index]).sqrt() * epsilon)
            clean_prediction /= diffusion.bar[index].sqrt()
            trace.append({"index": index, "x_std": x.std(unbiased=False).item(),
                          "epsilon_rms": epsilon.square().mean().sqrt().item(),
                          "predicted_clean_min_m": clean_prediction.min().item()
                          * card["std"] + card["mean"],
                          "predicted_clean_max_m": clean_prediction.max().item()
                          * card["std"] + card["mean"]})
            x = (x - diffusion.beta[index] / (1-diffusion.bar[index]).sqrt() * epsilon)
            x /= diffusion.alpha[index].sqrt()
            if index:
                x += diffusion.posterior_variance[index].sqrt() * torch.randn(
                    x.shape, generator=rng)
        for index in (0, 16, 32, 48, 63):
            t = torch.full((8,), index, dtype=torch.long)
            noisy = diffusion.noisy(clean, t, fixed_noise)
            error = (model(noisy, t, condition)-fixed_noise).square().mean().item()
            amplification = ((1-diffusion.bar[index])/diffusion.bar[index]).sqrt().item()
            calibration.append({"index": index, "epsilon_mse": error,
                                "clean_error_amplification": amplification,
                                "implied_clean_rmse_m": error**0.5 * amplification * card["std"]})
    torch.testing.assert_close(x, sample, rtol=0, atol=0)
    changed = diffusion.sample(model, condition + 1, seed=99)
    result = {
        "training_source_sha": SOURCE_SHA,
        "diagnostic_source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                          text=True).strip(),
        "diagnostic_working_file_hashes": {
            path: digest(Path(path)) for path in (
                "scripts/diagnose_ddpm_cpu.py", "src/tidetail/torch_diffusion.py",
                "src/tidetail/pilot_data.py")},
        "checkpoint_sha256": CHECKPOINT_SHA, "source_hashes_verified": source_hashes,
        "source_data_hashes_verified": actual_card["source_hashes"],
        "data_card_equal": True, "checkpoint_steps": state["steps"],
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
        "device": "cpu", "seed": 99, "calibration_seed": 20261028,
        "train_mean_m": card["mean"], "train_std_m": card["std"],
        "train_rows": card["train_rows"], "train_windows": len(target),
        "station_ids": card["station_ids"], "independent_storm_count": None,
        "train_event_ids_sha256": hashlib.sha256(
            json.dumps(card["threshold_run_ids"]).encode()).hexdigest(),
        "train_target_range_m": [float(target.min()*card["std"]+card["mean"]),
                                 float(target.max()*card["std"]+card["mean"])],
        "cpu_samples_m": {"min": float(physical.min()), "max": float(physical.max()),
                          "std": float(physical.std())},
        "original_cuda_samples_m": {"min": float(original.min()), "max": float(original.max()),
                                    "std": float(original.std())},
        "cpu_cuda_max_abs_difference_m": float(abs(physical-original).max()),
        "seed_repeatable": True,
        "history_shift_mean_abs_standardized": float((sample-changed).abs().mean()),
        "sample_scale": sample_diagnostics(sample.numpy(), target),
        "calibration_first_eight_train_windows_only": calibration, "reverse_trace": trace,
        "claim": "Negative engineering diagnosis; no realism or augmentation benefit",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("cpu_samples_m", "original_cuda_samples_m",
                     "cpu_cuda_max_abs_difference_m", "sample_scale")}))


if __name__ == "__main__":
    main()
