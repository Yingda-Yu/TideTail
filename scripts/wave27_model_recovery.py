"""Wave 2.7 controlled generative-model recovery pilots (CPU only).

Each pilot isolates ONE defect and compares:

  legacy config  ->  repaired config  ->  existing production checkpoint

No GPU is used. The cohort is a deterministic subsample of Charleston
(one region) so the comparison is controlled, not a claim about all regions.

Usage:
    python3 scripts/wave27_model_recovery.py --region south_atlantic_charleston

Output:
    data/results/wave27_model_recovery.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tidetail.models import (
    ConditionalVAE,
    DiffusionTS,
    ResidualDenoiser,
    TimeGrad,
    TimeGradDenoiser,
)
from tidetail.quality import evaluate
from train_gen_bench import load_region

CKPT = ROOT / "data" / "checkpoints"
RESULTS = ROOT / "data" / "results"


def subsample(data: dict, n_train: int, n_val: int, n_test: int) -> dict:
    rng = np.random.default_rng(7)
    out = dict(data)
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        for base in ("hist_res", "tgt_res", "hist_wl", "tgt_wl"):
            key = f"{split}_{base}"
            arr = data[key]
            idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
            out[key] = arr[np.sort(idx)]
    return out


def normalize(hist: torch.Tensor, tgt: torch.Tensor):
    mean = hist.mean(dim=1, keepdim=True)
    std = hist.std(dim=1, keepdim=True).clamp(min=1e-4)
    return (hist - mean) / std, (tgt - mean) / std, mean, std


def fidelity(real: np.ndarray, gen: np.ndarray, train_ref: np.ndarray) -> dict:
    report = evaluate(real, gen, train_ref=train_ref)
    out = {
        "status": report.status,
        "scale_ratio": float(report.scale_ratio),
        "wasserstein1": float(report.wasserstein1),
        "autocorr_rmse": float(report.autocorr_rmse),
        "q95_exceed_ratio": float(report.q95_exceedance_ratio),
        "peak_ks_pvalue": float(report.peak_ks_pvalue),
        "gen_std_ratio": float(np.std(gen) / np.std(real)),
        "failures": list(report.failures),
        "warnings": list(report.warnings)[:4],
    }
    return out


# ---------------------------------------------------------------------------
# VAE pilot
# ---------------------------------------------------------------------------

def train_vae_pilot(data, z_init: str, kl_warmup_epochs: int, beta: float,
                    epochs: int, batch_size: int, device: str) -> dict:
    torch.manual_seed(42)
    vae = ConditionalVAE(context=96, horizon=72, width=32, latent_dim=32,
                         beta=beta, device=device, z_init=z_init)
    opt = torch.optim.Adam(
        list(vae.encoder.parameters()) + list(vae.decoder.parameters()), lr=1e-3
    )
    Xh = torch.from_numpy(data["train_hist_res"]).to(device)
    Xt = torch.from_numpy(data["train_tgt_res"]).to(device)
    hn, tn, _, _ = normalize(Xh, Xt)
    Vh = torch.from_numpy(data["val_hist_res"]).to(device)
    Vt = torch.from_numpy(data["val_tgt_res"]).to(device)
    vhn, vtn, _, _ = normalize(Vh, Vt)

    n = len(hn)
    hist = {"epoch": [], "loss": [], "recon": [], "kl": [], "kl_weight": []}
    for epoch in range(epochs):
        if kl_warmup_epochs > 0 and epoch < kl_warmup_epochs:
            kl_w = beta * (epoch + 1) / kl_warmup_epochs
        else:
            kl_w = beta
        perm = torch.randperm(n)
        tot = rec = kl = 0.0
        vae.train()
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            loss = vae.training_loss(hn[idx], tn[idx], kl_weight=kl_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                mu, logvar = vae.encoder(hn[idx], tn[idx])
                z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
                recon = vae.decoder(hn[idx], z)
                rec += float(F.mse_loss(recon, tn[idx]) * len(idx))
                kl += float((-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())) * len(idx))
                tot += float(loss) * len(idx)
        hist["epoch"].append(epoch)
        hist["loss"].append(tot / n)
        hist["recon"].append(rec / n)
        hist["kl"].append(kl / n)
        hist["kl_weight"].append(kl_w)

    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encoder(vhn, vtn)
        recon = vae.decoder(vhn, mu)
        val_recon = float(F.mse_loss(recon, vtn))
        dim_var = logvar.exp().mean(dim=0)
        active = int((dim_var > 0.1).sum())
        gen = vae.sample(
            torch.from_numpy(data["test_hist_res"]).to(device), n_samples=10, seed=42
        ).cpu().numpy()
    real = data["test_tgt_res"]
    g2 = gen.reshape(-1, real.shape[1])
    ref = data["train_tgt_res"]
    return {
        "config": {"z_init": z_init, "kl_warmup_epochs": kl_warmup_epochs, "beta": beta},
        "params": int(sum(p.numel() for p in vae.encoder.parameters())
                      + sum(p.numel() for p in vae.decoder.parameters())),
        "val_recon_mse": val_recon,
        "val_kl": float((-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())).item()),
        "latent_active_dims": active,
        "latent_total_dims": 32,
        "sample_within_history_std": float(np.std(gen, axis=1).mean()),
        "fidelity": fidelity(real, g2, ref),
        "history": hist,
        "model": vae,
    }


# ---------------------------------------------------------------------------
# TimeGrad pilot
# ---------------------------------------------------------------------------

def train_timegrad_pilot(data, use_ar_memory: bool, width: int, ar_mode: str,
                         epochs: int, batch_size: int, device: str) -> dict:
    torch.manual_seed(42)
    model = TimeGrad(context=96, horizon=72, steps=64,
                     schedule="cosine1000_subsample", device=device)
    den = TimeGradDenoiser(context=96, horizon=72, width=width,
                           use_ar_memory=use_ar_memory).to(device)
    opt = torch.optim.Adam(den.parameters(), lr=1e-3)
    Xh = torch.from_numpy(data["train_hist_res"]).to(device)
    Xt = torch.from_numpy(data["train_tgt_res"]).to(device)
    n = len(Xh)
    hist = {"epoch": [], "loss": []}
    for epoch in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        den.train()
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            loss = model.training_loss(Xh[idx], Xt[idx], den, ar_mode=ar_mode)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        hist["epoch"].append(epoch)
        hist["loss"].append(tot / n)

    # Generate 3 draws for 200 test histories
    th = torch.from_numpy(data["test_hist_res"][:200]).to(device)
    draws = []
    for seed in (42, 99, 7):
        draws.append(model.forecast(th, den, seed=seed, ar_mode=ar_mode).cpu().numpy())
    gen = np.stack(draws, axis=0).reshape(-1, 72)
    real = data["test_tgt_res"][:200]
    return {
        "config": {"use_ar_memory": use_ar_memory, "width": width, "ar_mode": ar_mode},
        "params": int(sum(p.numel() for p in den.parameters())),
        "fidelity": fidelity(real, gen, data["train_tgt_res"]),
        "history": hist,
        "denoiser": den,
    }


# ---------------------------------------------------------------------------
# Diffusion-TS pilot
# ---------------------------------------------------------------------------

def train_diffusion_pilot(data, fourier_target: str, epochs: int,
                          batch_size: int, device: str) -> dict:
    torch.manual_seed(42)
    model = DiffusionTS(context=96, horizon=72, steps=64,
                        schedule="cosine1000_subsample", device=device)
    den = ResidualDenoiser(context=96, horizon=72, width=32,
                           use_full_context=True).to(device)
    opt = torch.optim.Adam(den.parameters(), lr=1e-3)
    Xh = torch.from_numpy(data["train_hist_wl"]).to(device)
    Xt = torch.from_numpy(data["train_tgt_wl"]).to(device)
    n = len(Xh)
    hist = {"epoch": [], "loss": []}
    for epoch in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        den.train()
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            loss = model.training_loss(Xh[idx], Xt[idx], den,
                                       fourier_weight=0.1,
                                       fourier_target=fourier_target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        hist["epoch"].append(epoch)
        hist["loss"].append(tot / n)

    th = torch.from_numpy(data["test_hist_wl"][:200]).to(device)
    draws = [model.forecast(th, den, seed=s).cpu().numpy() for s in (42, 99, 7)]
    gen = np.stack(draws, axis=0).reshape(-1, 72)
    real = data["test_tgt_wl"][:200]
    # Compare against residual space too (target semantics: residual = water_level_m)
    return {
        "config": {"fourier_target": fourier_target},
        "params": int(sum(p.numel() for p in den.parameters())),
        "fidelity": fidelity(real, gen, data["train_tgt_wl"]),
        "history": hist,
        "denoiser": den,
    }


def load_existing_metrics() -> dict:
    out = {}
    for model in ("vae", "timegrad", "diffusion_ts"):
        path = RESULTS / f"{model}_96_72.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        out[model] = [
            {
                "region": r.get("region"),
                "seed": r.get("seed"),
                "status": r.get("status"),
                "scale_ratio": r.get("scale_ratio"),
                "wasserstein1": r.get("wasserstein1"),
            }
            for r in rows
        ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="south_atlantic_charleston")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--epochs-vae", type=int, default=15)
    ap.add_argument("--epochs-timegrad", type=int, default=6)
    ap.add_argument("--epochs-diffusion", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    t0 = time.time()
    raw = load_region(args.region)
    data = subsample(raw, args.n_train, args.n_val, args.n_test)
    print(f"region={args.region} train={len(data['train_hist_res'])} "
          f"val={len(data['val_hist_res'])} test={len(data['test_hist_res'])}")

    results = {"region": args.region, "cohort": "deterministic subsample",
               "device": args.device, "existing_checkpoint_metrics": load_existing_metrics()}

    print("\n[VAE]")
    for label, z_init, warm, beta in (
        ("legacy_scalar", "scalar", 0, 0.001),
        ("repaired_linear_warmup", "linear", 5, 0.005),
    ):
        print(f"  training {label} ...")
        res = train_vae_pilot(data, z_init, warm, beta, args.epochs_vae,
                              args.batch_size, args.device)
        del res["model"]
        res["label"] = label
        results.setdefault("vae", {})[label] = res
        print(f"    recon={res['val_recon_mse']:.4f} kl={res['val_kl']:.2f} "
              f"active_dims={res['latent_active_dims']}/32 "
              f"scale={res['fidelity']['scale_ratio']:.2f} status={res['fidelity']['status']}")

    print("\n[TimeGrad]")
    for label, use_mem, width, mode in (
        ("legacy_ignored_memory_w24", False, 24, "sum"),
        ("repaired_ar_memory_sum_w64", True, 64, "sum"),
        ("repaired_ar_memory_mean_w64", True, 64, "mean"),
    ):
        print(f"  training {label} ...")
        res = train_timegrad_pilot(data, use_mem, width, mode,
                                   args.epochs_timegrad,
                                   args.batch_size, args.device)
        del res["denoiser"]
        res["label"] = label
        results.setdefault("timegrad", {})[label] = res
        print(f"    params={res['params']} scale={res['fidelity']['scale_ratio']:.2f} "
              f"status={res['fidelity']['status']}")

    print("\n[Diffusion-TS]")
    for label, ft in (("legacy_fourier_noise", "noise_legacy"),
                      ("repaired_fourier_x0", "x0")):
        print(f"  training {label} ...")
        res = train_diffusion_pilot(data, ft, args.epochs_diffusion,
                                    args.batch_size, args.device)
        del res["denoiser"]
        res["label"] = label
        results.setdefault("diffusion_ts", {})[label] = res
        print(f"    scale={res['fidelity']['scale_ratio']:.2f} "
              f"status={res['fidelity']['status']}")

    results["elapsed_s"] = round(time.time() - t0, 1)
    results["note"] = (
        "Controlled single-region CPU pilot. Diagnoses implementation defects; "
        "it is not evidence that any model family is intrinsically ineffective."
    )
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "wave27_model_recovery.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}  ({results['elapsed_s']}s)")


if __name__ == "__main__":
    main()
