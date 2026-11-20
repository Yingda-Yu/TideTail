"""Wave 2.8 P3 — fair generative-model benchmark (CPU pilot).

Seven families on identical residual-target windows, splits, seeds and
epoch budgets (two-region pilot):

  flow_matching   standard FM (VelocityTCN)
  tail_aware_fm   FM + tail-weighted loss
  cvae            conditional VAE (default beta; E22 repairs refuted)
  timegrad        TimeGrad (ar_mode="sum", the partial E22 repair)
  ddpm_cond       conditional DDPM (torch_diffusion, cosine1000_subsample)
  wgan_gp         conditional WGAN-GP (new, src/tidetail/models/wgan_gp.py)
  diffusion_ts    Diffusion-TS on TOTAL water by design; outputs are
                  reconstructed into the residual space by subtracting
                  the known astronomical tide, so every family is scored
                  in the same evaluation space (E25 comparability limit
                  resolved by an explicit justified transformation)

Fairness contract
-----------------
* Same windows, same per-sample normalisation convention, same epochs,
  same batch size, same downstream budget (2000 synthetic windows,
  GRU+TCN proxies, 3 downstream seeds).
* No hyperparameter search for any family: fixed defaults, documented
  (search budget = 0 for all families, so none is disadvantaged).
* Training/optimizer-step counts, parameter counts and wall-clock are
  reported per family so compute-matched and converged comparisons can
  be separated.
* Failed families are reported, not excluded.

Usage:
    python3 scripts/wave28_gen_benchmark.py \
        --regions south_atlantic_charleston ny_bight

Output:
    data/results/wave28_gen_benchmark.json (incremental, per family)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tidetail.models.diffusion_ts import DiffusionTS, ResidualDenoiser
from tidetail.models.flow_matching import FlowMatching, VelocityTCN
from tidetail.models.timegrad import TimeGrad, TimeGradDenoiser
from tidetail.models.vae import ConditionalVAE
from tidetail.models.wgan_gp import ConditionalWGANGP, train_wgan_gp
from tidetail.physics import tail_weight
from tidetail.quality import evaluate
from tidetail.torch_diffusion import Diffusion, TemporalDenoiser

# reuse the Wave 2.8 P1 downstream proxies + paired statistics
_spec = importlib.util.spec_from_file_location(
    "wave28_physics_fm", ROOT / "scripts" / "wave28_physics_fm.py")
w28 = importlib.util.module_from_spec(_spec)
sys.modules["wave28_physics_fm"] = w28
_spec.loader.exec_module(w28)

DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
RESULTS = ROOT / "data" / "results"
CKPT_DIR = RESULTS / "checkpoints" / "wave28_bench"
CONTEXT, HORIZON = 96, 72
FAMILIES = ["flow_matching", "tail_aware_fm", "cvae", "timegrad",
            "ddpm_cond", "wgan_gp", "diffusion_ts"]


def load_region(region: str) -> dict:
    """Residual windows + tide windows + total-water windows."""
    base = DATA_ROOT / region
    target = pd.read_csv(base / "target_timeline_v1.csv")
    tide = pd.read_csv(base / "tide_timeline_v1.csv")
    merged = target[["timestamp_utc", "water_level_m"]].merge(
        tide[["timestamp_utc", "tide_level_m"]], on="timestamp_utc", how="inner"
    )
    res = merged["water_level_m"].to_numpy(dtype=np.float32)
    tot = (merged["water_level_m"] + merged["tide_level_m"]).to_numpy(
        dtype=np.float32)
    tid = merged["tide_level_m"].to_numpy(dtype=np.float32)
    wi = np.load(base / "window_index_v1.npz")

    def windows(starts, series):
        starts = np.asarray(starts, dtype=np.int64)
        h = np.empty((len(starts), CONTEXT), dtype=np.float32)
        t = np.empty((len(starts), HORIZON), dtype=np.float32)
        for i, s in enumerate(starts):
            h[i] = series[s:s + CONTEXT]
            t[i] = series[s + CONTEXT:s + CONTEXT + HORIZON]
        return h, t

    out = {}
    for split, key in (("train", "train_start_indices"),
                       ("test", "test_start_indices")):
        for name, series in (("res", res), ("tot", tot), ("tide", tid)):
            h, t = windows(wi[key], series)
            out[f"{split}_{name}_h"], out[f"{split}_{name}_t"] = h, t
    return out


def _norm(hist: torch.Tensor, target: torch.Tensor):
    mean = hist.mean(dim=1, keepdim=True)
    std = hist.std(dim=1, keepdim=True).clamp(min=1e-4)
    return (hist - mean) / std, (target - mean) / std, mean, std


# ---------------------------------------------------------------------------
# Family trainers. Each returns a dict with a `gen(hist_raw, idx, draws)`
# callable producing (draws, len(idx), horizon) samples in residual metres.
# ---------------------------------------------------------------------------

def train_fm(train_h: np.ndarray, train_t: np.ndarray, tail: bool,
             epochs: int, batch_size: int, seed: int) -> dict:
    torch.manual_seed(seed)
    fm = FlowMatching(context=CONTEXT, horizon=HORIZON, euler_steps=64)
    net = VelocityTCN(context=CONTEXT, horizon=HORIZON, width=32)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xh = torch.from_numpy(train_h)
    Xt = torch.from_numpy(train_t)
    hn, tn, _, _ = _norm(Xh, Xt)
    tail_w = None
    if tail:
        flat = np.abs(train_t.reshape(-1))
        q95, q99 = np.percentile(flat, 95), np.percentile(flat, 99)
        tail_w = torch.from_numpy(
            tail_weight(np.abs(train_t), q95, q99).mean(axis=1)
            .astype(np.float32))
    n = len(hn)
    steps = 0
    t0 = time.time()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            hb, tb = hn[idx], tn[idx]
            B = hb.shape[0]
            tt = torch.rand(B, 1, 1)
            x0 = torch.randn(B, 1, HORIZON)
            x1 = tb.unsqueeze(1)
            xt = (1 - tt) * x0 + tt * x1
            v = net(xt, tt.squeeze(-1).squeeze(-1), hb)
            if tail_w is None:
                loss = F.mse_loss(v, x1 - x0)
            else:
                per = ((v - (x1 - x0)) ** 2).mean(dim=(1, 2))
                loss = (per * tail_w[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1

    def gen(hist_raw, idx, draws, seed_base=1000):
        h = torch.from_numpy(np.ascontiguousarray(hist_raw[idx]))
        mean = h.mean(dim=1, keepdim=True)
        std = h.std(dim=1, keepdim=True).clamp(min=1e-4)
        hn_ = (h - mean) / std
        out = []
        for k in range(draws):
            g = fm.sample(hn_, net, seed=seed_base + k)
            out.append((g * std + mean).numpy())
        return np.stack(out)

    return {"gen": gen, "params": int(sum(p.numel() for p in net.parameters())),
            "opt_steps": steps, "train_s": time.time() - t0,
            "conditioning": "96h residual history (per-sample normalised)"}


def train_cvae(train_h, train_t, epochs, batch_size, seed) -> dict:
    torch.manual_seed(seed)
    vae = ConditionalVAE(context=CONTEXT, horizon=HORIZON, width=32,
                         latent_dim=32, beta=0.001)
    opt = torch.optim.Adam(
        list(vae.encoder.parameters()) + list(vae.decoder.parameters()),
        lr=1e-3)
    Xh, Xt = torch.from_numpy(train_h), torch.from_numpy(train_t)
    hn, tn, _, _ = _norm(Xh, Xt)
    n = len(hn)
    steps = 0
    t0 = time.time()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            loss = vae.training_loss(hn[idx], tn[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1

    def gen(hist_raw, idx, draws, seed_base=2000):
        h = torch.from_numpy(np.ascontiguousarray(hist_raw[idx]))
        mean = h.mean(dim=1, keepdim=True)
        std = h.std(dim=1, keepdim=True).clamp(min=1e-4)
        hn_ = (h - mean) / std
        out = []
        for k in range(draws):
            g = vae.sample(hn_, seed=seed_base + k)
            out.append((g * std + mean).numpy())
        return np.stack(out)

    p = sum(p.numel() for p in vae.encoder.parameters()) + \
        sum(p.numel() for p in vae.decoder.parameters())
    return {"gen": gen, "params": int(p), "opt_steps": steps,
            "train_s": time.time() - t0,
            "conditioning": "96h residual history (per-sample normalised)"}


def train_timegrad(train_h, train_t, epochs, batch_size, seed) -> dict:
    torch.manual_seed(seed)
    model = TimeGrad(context=CONTEXT, horizon=HORIZON)
    den = TimeGradDenoiser(context=CONTEXT, horizon=HORIZON)
    opt = torch.optim.Adam(den.parameters(), lr=1e-3)
    Xh, Xt = torch.from_numpy(train_h), torch.from_numpy(train_t)
    n = len(Xh)
    steps = 0
    t0 = time.time()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            loss = model.training_loss(Xh[idx], Xt[idx], den, ar_mode="sum")
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1

    def gen(hist_raw, idx, draws, seed_base=3000):
        h = torch.from_numpy(np.ascontiguousarray(hist_raw[idx]))
        out = []
        for k in range(draws):
            g = model.forecast(h, den, seed=seed_base + k, ar_mode="sum")
            out.append(g.numpy())
        return np.stack(out)

    return {"gen": gen,
            "params": int(sum(p.numel() for p in den.parameters())),
            "opt_steps": steps, "train_s": time.time() - t0,
            "conditioning": "96h residual history + AR memory (internal norm)"}


def train_ddpm(train_h, train_t, epochs, batch_size, seed) -> dict:
    torch.manual_seed(seed)
    diff = Diffusion(steps=64, schedule="cosine1000_subsample")
    den = TemporalDenoiser(context=CONTEXT, horizon=HORIZON, width=32)
    opt = torch.optim.Adam(den.parameters(), lr=1e-3)
    Xh, Xt = torch.from_numpy(train_h), torch.from_numpy(train_t)
    hn, tn, _, _ = _norm(Xh, Xt)
    n = len(hn)
    steps = 0
    t0 = time.time()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            hb, tb = hn[idx], tn[idx]
            B = hb.shape[0]
            t_ix = torch.randint(0, diff.steps, (B,))
            eps = torch.randn(B, 1, HORIZON)
            noisy = diff.noisy(tb.unsqueeze(1), t_ix, eps)
            pred = den(noisy, t_ix, hb)
            loss = F.mse_loss(pred, eps)
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1

    def gen(hist_raw, idx, draws, seed_base=4000):
        h = torch.from_numpy(np.ascontiguousarray(hist_raw[idx]))
        mean = h.mean(dim=1, keepdim=True)
        std = h.std(dim=1, keepdim=True).clamp(min=1e-4)
        hn_ = (h - mean) / std
        out = []
        for k in range(draws):
            g = diff.sample(den, hn_, seed=seed_base + k)
            out.append((g.squeeze(1) * std + mean).numpy())
        return np.stack(out)

    return {"gen": gen,
            "params": int(sum(p.numel() for p in den.parameters())),
            "opt_steps": steps, "train_s": time.time() - t0,
            "conditioning": "96h residual history (per-sample normalised)"}


def train_wgan(train_h, train_t, epochs, batch_size, seed) -> dict:
    torch.manual_seed(seed)
    model = ConditionalWGANGP(context=CONTEXT, horizon=HORIZON, width=32,
                              z_dim=32)
    Xh, Xt = torch.from_numpy(train_h), torch.from_numpy(train_t)
    hn, tn, _, _ = _norm(Xh, Xt)
    t0 = time.time()
    hist = train_wgan_gp(model, hn, tn, epochs=epochs, batch_size=batch_size,
                         n_critic=5)

    def gen(hist_raw, idx, draws, seed_base=5000):
        h = torch.from_numpy(np.ascontiguousarray(hist_raw[idx]))
        mean = h.mean(dim=1, keepdim=True)
        std = h.std(dim=1, keepdim=True).clamp(min=1e-4)
        hn_ = (h - mean) / std
        out = []
        for k in range(draws):
            g = model.sample(hn_, seed=seed_base + k)
            out.append((g * std + mean).numpy())
        return np.stack(out)

    g_params = sum(p.numel() for n_, p in model.named_parameters()
                   if not n_.startswith("c_"))
    c_params = sum(p.numel() for n_, p in model.named_parameters()
                   if n_.startswith("c_"))
    n_batches = (len(hn) + batch_size - 1) // batch_size
    return {"gen": gen, "params": int(g_params + c_params),
            "gen_params": int(g_params), "critic_params": int(c_params),
            "opt_steps": int(n_batches * epochs),   # critic steps; gen = /5
            "train_s": time.time() - t0,
            "final_wasserstein": float(hist["wasserstein"][-1]),
            "final_gp": float(hist["gp"][-1]),
            "conditioning": "96h residual history + z noise (WGAN-GP)"}


def train_diffusion_ts(train_tot_h, train_tot_t, epochs, batch_size, seed) -> dict:
    torch.manual_seed(seed)
    model = DiffusionTS(context=CONTEXT, horizon=HORIZON, steps=64,
                        schedule="cosine1000_subsample")
    den = ResidualDenoiser(context=CONTEXT, horizon=HORIZON, width=32,
                           use_full_context=True)
    opt = torch.optim.Adam(den.parameters(), lr=1e-3)
    Xh = torch.from_numpy(train_tot_h)
    Xt = torch.from_numpy(train_tot_t)
    n = len(Xh)
    steps = 0
    t0 = time.time()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            loss = model.training_loss(Xh[idx], Xt[idx], den,
                                       fourier_weight=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1

    def gen(hist_tot_raw, idx, draws, tide_t, seed_base=6000):
        """Returns RESIDUAL samples: total-water generation minus tide."""
        h = torch.from_numpy(np.ascontiguousarray(hist_tot_raw[idx]))
        out = []
        for k in range(draws):
            g = model.forecast(h, den, seed=seed_base + k).numpy()
            out.append(g - tide_t[idx])          # -> residual space
        return np.stack(out)

    return {"gen": gen,
            "params": int(sum(p.numel() for p in den.parameters())),
            "opt_steps": steps, "train_s": time.time() - t0,
            "conditioning": "96h TOTAL-water history (decomposition design); "
                            "output reconstructed to residual via tide"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+",
                    default=["south_atlantic_charleston", "ny_bight"])
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-aug", type=int, default=2000)
    ap.add_argument("--downstream-n-train", type=int, default=8000)
    ap.add_argument("--downstream-steps", type=int, default=120)
    ap.add_argument("--downstream-seeds", type=int, nargs="+",
                    default=[42, 99, 7])
    ap.add_argument("--n-train", type=int, default=0,
                    help="cap on FM-family training windows (0 = all)")
    args = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "wave28_gen_benchmark.json"
    payload = {
        "audit_version": "wave28_gen_benchmark_v1",
        "fairness_contract": {
            "target_space": "non-tidal residual for all families; "
                            "Diffusion-TS trained on total water (by design) "
                            "and reconstructed to residual via tide "
                            "subtraction before scoring",
            "hyperparameter_search_budget": "zero for every family — fixed "
                                            "defaults, no search, so no "
                                            "family is disadvantaged",
            "selection": "fixed epoch budget; no validation-based early "
                         "stopping in this pilot (documented limitation)",
            "matched": "same windows, batch size, epochs, seeds, downstream "
                       "budget (n_aug synthetic windows, GRU+TCN proxies)",
        },
        "regions": {},
    }

    for region in args.regions:
        print(f"\n{'='*70}\nREGION {region}")
        data = load_region(region)
        if args.n_train and len(data["train_res_h"]) > args.n_train:
            ridx = np.sort(np.random.default_rng(7).choice(
                len(data["train_res_h"]), size=args.n_train, replace=False))
            for k in list(data):
                if k.startswith("train_"):
                    data[k] = data[k][ridx]
        print(f"  train windows: {len(data['train_res_h'])}")

        n_down = min(args.downstream_n_train, len(data["train_res_h"]))
        down_idx = np.sort(np.random.default_rng(17).choice(
            len(data["train_res_h"]), size=n_down, replace=False))
        down_data = {"train_h": data["train_res_h"][down_idx],
                     "train_t": data["train_res_t"][down_idx],
                     "test_h": data["test_res_h"],
                     "test_t": data["test_res_t"]}

        truth = down_data["test_t"]
        ids = w28.load_event_ids(region, len(truth))

        base_pred = {k: w28.train_downstream(
            k, down_data, None, tuple(args.downstream_seeds),
            args.downstream_steps, args.batch_size)
            for k in ("gru", "tcn")}
        base = {k: (w28.event_level_mae(base_pred[k], truth, ids)
                    if ids is not None else
                    {"all_window_mae": float(np.abs(base_pred[k].mean(0)
                                                    - truth).mean())})
                for k in ("gru", "tcn")}
        for k in ("gru", "tcn"):
            m = base[k].get("extreme_unit_mae", base[k]["all_window_mae"])
            print(f"  A0 {k.upper()} MAE={m:.4f}")

        n_test = min(args.n_test, len(data["test_res_h"]))
        real = data["test_res_t"][:n_test]
        fam_out = {}

        for fam in args.families:
            print(f"\n  FAMILY {fam}")
            try:
                if fam in ("flow_matching", "tail_aware_fm"):
                    model = train_fm(data["train_res_h"], data["train_res_t"],
                                     tail=(fam == "tail_aware_fm"),
                                     epochs=args.epochs,
                                     batch_size=args.batch_size, seed=args.seed)
                    gen = model["gen"](data["test_res_h"], np.arange(n_test),
                                       3)
                elif fam == "cvae":
                    model = train_cvae(data["train_res_h"], data["train_res_t"],
                                       args.epochs, args.batch_size, args.seed)
                    gen = model["gen"](data["test_res_h"], np.arange(n_test), 3)
                elif fam == "timegrad":
                    model = train_timegrad(data["train_res_h"],
                                           data["train_res_t"], args.epochs,
                                           args.batch_size, args.seed)
                    gen = model["gen"](data["test_res_h"], np.arange(n_test), 3)
                elif fam == "ddpm_cond":
                    model = train_ddpm(data["train_res_h"], data["train_res_t"],
                                       args.epochs, args.batch_size, args.seed)
                    gen = model["gen"](data["test_res_h"], np.arange(n_test), 3)
                elif fam == "wgan_gp":
                    model = train_wgan(data["train_res_h"],
                                       data["train_res_t"], args.epochs,
                                       args.batch_size, args.seed)
                    gen = model["gen"](data["test_res_h"], np.arange(n_test), 3)
                elif fam == "diffusion_ts":
                    model = train_diffusion_ts(data["train_tot_h"],
                                               data["train_tot_t"],
                                               args.epochs, args.batch_size,
                                               args.seed)
                    gen = model["gen"](data["test_tot_h"], np.arange(n_test),
                                       3, data["test_tide_t"])
                else:
                    raise ValueError(fam)
            except Exception as exc:                       # noqa: BLE001
                fam_out[fam] = {"status": "ENGINEERING FAILURE",
                                "error": str(exc)[:500]}
                print(f"    ENGINEERING FAILURE: {exc}")
                continue

            g_flat = gen.reshape(-1, HORIZON)
            report = evaluate(real, g_flat, train_ref=data["train_res_t"])
            entry = {"params": model["params"],
                     "opt_steps": model["opt_steps"],
                     "train_s": round(model["train_s"], 1),
                     "conditioning": model["conditioning"],
                     "fidelity_status": report.status,
                     "scale_ratio": float(report.scale_ratio),
                     "wasserstein1": float(report.wasserstein1),
                     "q95_exceedance_ratio": float(report.q95_exceedance_ratio),
                     "q99_exceedance_ratio": float(report.q99_exceedance_ratio),
                     "peak_ks_pvalue": float(report.peak_ks_pvalue),
                     "peak_timing_mae": float(report.peak_timing_mae),
                     "warnings": list(report.warnings)[:3]}
            if "final_wasserstein" in model:
                entry["final_wasserstein"] = model["final_wasserstein"]
                entry["final_gp"] = model["final_gp"]
                entry["gen_params"] = model["gen_params"]
                entry["critic_params"] = model["critic_params"]

            # downstream augmentation (residual space for every family)
            n_aug = min(args.n_aug, n_down)
            gidx = np.random.default_rng(7).choice(n_down, size=n_aug,
                                                   replace=False)
            if fam == "diffusion_ts":
                aug = model["gen"](data["train_tot_h"][down_idx], gidx, 1,
                                   data["train_tide_t"][down_idx])[0]
            else:
                aug = model["gen"](down_data["train_h"], gidx, 1)[0]
            agg = {}
            for k in ("gru", "tcn"):
                pred = w28.train_downstream(
                    k, down_data,
                    {"hist": down_data["train_h"][gidx], "target": aug},
                    tuple(args.downstream_seeds), args.downstream_steps,
                    args.batch_size)
                ev = w28.event_level_mae(pred, truth, ids) if ids is not None \
                    else {"all_window_mae": float(np.abs(pred.mean(0) - truth)
                                                  .mean())}
                bm = base[k].get("extreme_unit_mae", base[k]["all_window_mae"])
                vm = ev.get("extreme_unit_mae", ev["all_window_mae"])
                delta_pct = (vm - bm) / bm * 100
                stats_ev = None
                if "per_unit_mae" in ev and "per_unit_mae" in base[k]:
                    stats_ev = w28.paired_event_stats(
                        np.array(ev["per_unit_mae"]) -
                        np.array(base[k]["per_unit_mae"]))
                agg[k] = {"delta_pct_vs_A0": float(delta_pct),
                          "extreme_unit_mae": vm, "paired_event_stats": stats_ev}
                print(f"    {k} delta = {delta_pct:+.2f}%  "
                      f"fidelity={report.status} "
                      f"scale={report.scale_ratio:.3f}")
            entry["downstream"] = agg
            entry["status"] = "OK"
            fam_out[fam] = entry

            payload["regions"][region] = {
                "n_train": int(len(data["train_res_h"])),
                "n_test": int(len(data["test_res_h"])),
                "n_eval_windows": int(n_test),
                "epochs": args.epochs, "batch_size": args.batch_size,
                "seed": args.seed,
                "downstream_seeds": args.downstream_seeds,
                "baseline_A0": base,
                "families": fam_out,
            }
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"  [incremental save] {out_path}")

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
