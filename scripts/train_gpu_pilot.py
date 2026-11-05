"""Issue #6: bounded GPU-only training; preparation and tests never run CPU gradients."""

import argparse
import csv
import fcntl
import json
import os
import resource
import subprocess
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "2"

import numpy as np

from tidetail.pilot_data import digest, prepare


def query(arguments):
    return subprocess.check_output(["nvidia-smi", *arguments], text=True, timeout=10)


def gpu_status(index, own_pid=None):
    """Query physical GPU independently of CUDA_VISIBLE_DEVICES; fail on conflicts."""
    fields = "index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu"
    rows = list(
        csv.reader(
            query([f"--query-gpu={fields}", "--format=csv,noheader,nounits"]).strip().splitlines()
        )
    )
    row = next(row for row in rows if int(row[0]) == index)
    row = [field.strip() for field in row]
    _, uuid, name, total, used, utilization, temperature = row
    if "RTX 3090" not in name or int(total) != 24576:
        raise RuntimeError("requires one 24GB RTX3090")
    if int(temperature) >= 80 or int(used) > 8192:
        raise RuntimeError("GPU temperature/8GB pilot VRAM budget exceeded")
    processes = list(
        csv.reader(
            query(
                [
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
            .strip()
            .splitlines()
        )
    )
    foreign = [p for p in processes if p and p[0].strip() == uuid and int(p[1]) != own_pid]
    if foreign or (own_pid is None and (int(used) > 128 or int(utilization) != 0)):
        raise RuntimeError(f"GPU resource conflict: {foreign}")
    return {
        "physical_index": index,
        "uuid": uuid,
        "name": name,
        "total_mib": int(total),
        "used_mib": int(used),
        "utilization_percent": int(utilization),
        "temperature_c": int(temperature),
    }


def canonical_uuid(value):
    return str(UUID(str(value).removeprefix("GPU-")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument(
        "--reservation-record",
        type=Path,
        help="Owner/scheduler evidence that selected physical GPU is unreserved",
    )
    parser.add_argument("--epochs", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--max-steps", type=int, help="Optional total optimizer-step limit")
    parser.add_argument("--resume", type=Path, help="Trusted local pilot checkpoint")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 3:
        parser.error("max-steps must be at least 3")
    if not 1 <= args.batch <= 128 or not 1 <= args.max_seconds <= 1800:
        parser.error("batch <=128 and time budget <=1800 seconds required")
    args.output.mkdir(parents=True, exist_ok=False)
    history, target, card = prepare()
    (args.output / "data_card.json").write_text(json.dumps(card, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps({k: v for k, v in card.items() if k != "threshold_run_ids"}, indent=2))
        return
    if args.gpu is None or args.gpu <= 0:
        parser.error("explicit nonzero physical --gpu required; CPU training forbidden")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        parser.error("CUDA_VISIBLE_DEVICES must equal the single physical --gpu index")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        parser.error("CUDA_DEVICE_ORDER=PCI_BUS_ID required for physical index mapping")
    if not args.reservation_record or not args.reservation_record.is_file():
        parser.error(
            "reservation/ownership evidence required; idle utilization alone is insufficient"
        )
    reservation = json.loads(args.reservation_record.read_text())
    if reservation.get("physical_index") != args.gpu or reservation.get("unreserved") is not True:
        parser.error("reservation record must confirm this physical index is unreserved")
    with ExitStack() as stack:
        lock = stack.enter_context(open(f"/tmp/tidetail-gpu-{args.gpu}.lock", "a+"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = gpu_status(args.gpu)
        time.sleep(3)
        gpu_status(args.gpu)
        import torch

        from tidetail.torch_diffusion import Diffusion, TemporalDenoiser, sample_diagnostics

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one visible CUDA device required; no CPU fallback")
        properties = torch.cuda.get_device_properties(0)
        mapped_uuid = str(properties.uuid)
        if canonical_uuid(mapped_uuid) != canonical_uuid(initial["uuid"]):
            raise RuntimeError(f"physical CUDA mapping mismatch: {mapped_uuid}, {initial}")
        torch.cuda.set_per_process_memory_fraction(0.30, 0)
        torch.manual_seed(20261025)
        np.random.seed(20261025)
        device = torch.device("cuda:0")
        model = TemporalDenoiser().to(device)
        diffusion = Diffusion(device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        # Entire ~27MB training window arrays stay on the chosen GPU: no worker pools.
        x = torch.from_numpy(history).to(device)
        y = torch.from_numpy(target[:, None]).to(device)
        amp = torch.cuda.is_bf16_supported()
        config = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "seed": 20261025,
            "epochs": args.epochs,
            "batch": args.batch,
            "max_seconds": args.max_seconds,
            "max_steps": args.max_steps,
            "resume": str(args.resume) if args.resume else None,
            "physical_gpu": initial,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "mapped_uuid": mapped_uuid,
            "mapped_device": "cuda:0",
            "threads": torch.get_num_threads(),
            "data_loader_workers": 0,
            "amp": "bfloat16" if amp else "float32",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "diffusion_steps": 64,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
            "reservation_record": reservation,
            "source_code_hashes": {
                str(p): digest(p)
                for p in [
                    Path(__file__),
                    Path("src/tidetail/torch_diffusion.py"),
                    Path("src/tidetail/pilot_data.py"),
                ]
            },
        }
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        started, cpu_start = time.monotonic(), time.process_time()
        step, examples, stopped = 0, 0, False
        next_epoch, next_offset = 0, 0
        if args.resume:
            state = torch.load(args.resume, map_location=device, weights_only=False)
            if state["data_card"] != card:
                raise RuntimeError("resume data card mismatch")
            for key in ("batch", "seed", "diffusion_steps", "amp"):
                if state["config"][key] != config[key]:
                    raise RuntimeError(f"resume config mismatch: {key}")
            if state["config"]["source_code_hashes"] != config["source_code_hashes"]:
                raise RuntimeError("resume source mismatch")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            step, examples = state["steps"], state["examples"]
            next_epoch, next_offset = state["next_epoch"], state["next_offset"]
            torch.set_rng_state(state["torch_rng_state"].cpu())
            torch.cuda.set_rng_state(state["cuda_rng_state"].cpu())
        initial_step = step

        def checkpoint():
            temporary = args.output / "checkpoint.tmp"
            torch.save(
                {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": config, "data_card": card, "steps": step, "examples": examples,
                    "next_epoch": next_epoch, "next_offset": next_offset,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(),
                }, temporary,
            )
            temporary.replace(args.output / "checkpoint.pt")
        monitor = stack.enter_context(open(args.output / "resources.jsonl", "w", buffering=1))
        loss_file = stack.enter_context(open(args.output / "loss.csv", "w", buffering=1))
        writer = csv.writer(loss_file)
        writer.writerow(["step", "epoch", "examples", "epsilon_mse", "wall_seconds"])

        def measure():
            stats = gpu_status(args.gpu, os.getpid())
            stats.update(
                step=step,
                wall_seconds=time.monotonic() - started,
                process_cpu_seconds=time.process_time() - cpu_start,
                load_average=os.getloadavg(),
                rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            )
            monitor.write(json.dumps(stats) + "\n")
            print(json.dumps(stats), flush=True)

        measure()
        try:
            for epoch in range(next_epoch, args.epochs):
                rng = torch.Generator(device=device).manual_seed(20261025 + epoch)
                order = torch.randperm(len(y), device=device, generator=rng)
                offset = next_offset if epoch == next_epoch else 0
                for indices in order[offset:].split(args.batch):
                    if time.monotonic() - started >= args.max_seconds or (
                        args.max_steps is not None and step >= args.max_steps
                    ):
                        stopped = True
                        break
                    if step % 20 == 0:
                        measure()
                    clean, condition = y[indices], x[indices]
                    t = torch.randint(diffusion.steps, (len(indices),), device=device)
                    noise = torch.randn_like(clean)
                    noisy = diffusion.noisy(clean, t, noise)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                        predicted = model(noisy, t, condition)
                        loss = (predicted.float() - noise).square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite training loss")
                    loss.backward()  # Every parameter and gradient is CUDA, never CPU.
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(gradient_norm):
                        raise RuntimeError("nonfinite gradient")
                    optimizer.step()
                    step += 1
                    examples += len(indices)
                    offset += len(indices)
                    next_epoch = epoch + int(offset == len(y))
                    next_offset = 0 if offset == len(y) else offset
                    writer.writerow(
                        [step, epoch + 1, examples, loss.item(), time.monotonic() - started]
                    )
                    if step % 100 == 0:
                        checkpoint()
                if stopped:
                    break
            torch.cuda.synchronize()
            if step < 3:
                raise RuntimeError("insufficient optimizer steps")
            model.eval()
            sample = diffusion.sample(model, x[:8], seed=99)
            changed = diffusion.sample(model, x[:8] + 1, seed=99)
            sensitivity = (sample - changed).abs().mean().item()
            if (
                sample.shape != (8, 1, 24)
                or not torch.isfinite(sample).all()
                or sensitivity <= 1e-7
            ):
                raise RuntimeError("sampler finite/shape/conditional-sensitivity check failed")
            # Full train-window nearest neighbours on the GPU, in standardized residual units.
            generated = sample[:, 0]
            distances = torch.cdist(
                generated, y[:, 0], p=2, compute_mode="donot_use_mm_for_euclid_dist"
            )
            nearest = distances.min(dim=1).values / (24**0.5)
            checkpoint()
            np.savez(
                args.output / "samples.npz",
                residual_m=generated.cpu().numpy() * card["std"] + card["mean"],
            )
            measure()
            wall, cpu = time.monotonic() - started, time.process_time() - cpu_start
            summary = {
                "sample_scale": sample_diagnostics(
                    generated.cpu().numpy(), target
                ),
                "steps": step,
                "session_steps": step - initial_step,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "examples_seen": examples,
                "complete_epochs": next_epoch,
                "next_offset": next_offset,
                "timebox_reached": stopped,
                "wall_seconds": wall,
                "process_cpu_seconds": cpu,
                "mean_process_cpu_percent": cpu / wall * 100,
                "sensitivity_mean_abs": sensitivity,
                "nearest_train_rmse_standardized": nearest.cpu().tolist(),
                "exact_or_near_copies_1e_minus_4": int((nearest < 1e-4).sum()),
                "generated_samples": 8,
                "independent_storm_count": None,
                "claim": "Engineering smoke only; no generalization or predictive-skill evidence",
                "checkpoint_sha256": digest(args.output / "checkpoint.pt"),
            }
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
        except Exception as error:
            if step > initial_step:
                checkpoint()
            (args.output / "failure.json").write_text(
                json.dumps({"step": step, "error": str(error)}) + "\n"
            )
            raise
        finally:
            monitor.close()
            loss_file.close()
            lock.close()


if __name__ == "__main__":
    main()
