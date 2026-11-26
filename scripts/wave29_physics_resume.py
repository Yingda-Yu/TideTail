#!/usr/bin/env python3
"""Wave 2.9 Physics Resume — OOM-safe incremental completion of wave28_physics_fm.

Wave 2.8 wave28physics systemd unit was OOM-killed on 2026-11-08 23:00 after
3h25min CPU. Charleston 4/6 arms (standard, era5_cond, pressure_prior,
wind_prior) are COMPLETE with FM training, fidelity, and downstream eval.
Missing: combined_prior, combined_tail (charleston) + entire ny_bight region.

THIS SCRIPT:
1. Loads existing wave28_physics_fm.json
2. Identifies (region, arm) tuples that need completion
3. For each missing tuple, imports wave28_physics_fm internals and runs
   only that region's remaining arms
4. Increments JSON on completion of each arm (OOM-safe)
5. Uses GPU3 exclusively (25GB free, no other users)

Run:
    CUDA_VISIBLE_DEVICES=3 python3 scripts/wave29_physics_resume.py
"""
from __future__ import annotations
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "data" / "results"
CKPT_DIR = ROOT / "data" / "results" / "checkpoints" / "wave28_physics"
OUT = RESULTS / "wave28_physics_fm.json"

torch.set_num_threads(min(8, torch.get_num_threads()))

# Import wave28_physics_fm internals (skip main)
_spec = importlib.util.spec_from_file_location(
    "w28", ROOT / "scripts" / "wave28_physics_fm.py")
w28 = importlib.util.module_from_spec(_spec)
sys.modules["wave28_physics_fm"] = w28
_spec.loader.exec_module(w28)

ARMS = ["standard", "era5_cond", "pressure_prior", "wind_prior",
        "combined_prior", "combined_tail"]

# Resume state
if OUT.exists():
    payload = json.loads(OUT.read_text())
    print(f"Resuming from {OUT}")
else:
    payload = None

# Determine what's complete
def arm_complete(region_data, arm_name):
    """An arm is complete when ALL fm_seeds have downstream results."""
    arms = region_data.get("arms", {})
    arm = arms.get(arm_name, {})
    per_seed = arm.get("per_fm_seed", {})
    dagg = arm.get("downstream_agg", {})
    # Need at least fm_seeds present, all with downstream, and agg non-empty
    return (len(per_seed) >= 2 and
            all("downstream" in sd for sd in per_seed.values()) and
            bool(dagg.get("gru", {})) and bool(dagg.get("tcn", {})))

# Load or init payload
if payload is None:
    payload = {
        "audit_version": "wave28_physics_v1",
        "causal_contract": "ERA5 covariates from 96h history only; "
                           "no future reanalysis in conditioning or penalty",
        "design": {
            "arms": ARMS,
            "conditioning": "arms era5_cond..combined_tail share the same 9 "
                            "causal ERA5 features; only the penalty differs",
            "penalty": "MSE on implied clean estimate x1_hat = x_t + (1-t)v "
                       "between Delta(x1_hat) and a train-only OLS physical "
                       "prediction of the residual level change across the "
                       "forecast origin",
            "lambda_rule": "auto-calibrated on first batch as "
                           "0.1*L_FM/R clipped to [0.01, 1.0], then frozen",
            "coast_normal": "approximate onshore unit vectors, see script",
        },
        "regions": {},
    }

# Wave 2.9 adds only the two regions we know ERA5 exists for
TARGET_REGIONS = ["south_atlantic_charleston", "ny_bight"]

# Figure out missing work
missing = []
for region in TARGET_REGIONS:
    rd = payload["regions"].get(region, {})
    if not rd:
        # entire region missing
        missing.extend([(region, arm) for arm in ARMS])
    else:
        for arm in ARMS:
            if not arm_complete(rd, arm):
                missing.append((region, arm))

print(f"\nResume plan: {len(missing)} arm-runs needed")
for r, a in missing:
    print(f"  {r} / {a}")

if not missing:
    print("\n✅ Wave 2.8 physics FM COMPLETE — nothing to do")
    sys.exit(0)

# Reusable state per region (load once)
region_cache = {}

def get_region_state(region):
    """Load region once — shared across arms."""
    if region in region_cache:
        return region_cache[region]
    data = w28.load_region(region)
    n_down = min(2740, len(data["train_h"]))  # default downstream budget
    down_idx = np.sort(np.random.default_rng(17).choice(
        len(data["train_h"]), size=n_down, replace=False))
    down_data = {"train_h": data["train_h"][down_idx],
                 "train_t": data["train_t"][down_idx],
                 "test_h": data["test_h"],
                 "test_t": data["test_t"]}

    physics = w28.align_physics(region, data["ts"])
    tr_feats, stats = w28.build_cond_features(data["train_starts"], physics,
                                             use_era5=True)
    feats = {
        "train": tr_feats,
        "val": w28.build_cond_features(data["val_starts"], physics, True, stats)[0],
        "test": w28.build_cond_features(data["test_starts"], physics, True, stats)[0],
    }
    feats_none = {k: np.zeros((len(v), 0), dtype=np.float32) for k, v in feats.items()}
    raw = {s: w28.build_physics_raw(data[f"{s}_starts"], physics, region)
           for s in ("train", "val", "test")}
    d_real = {s: w28.delta_real(data[f"{s}_h"], data[f"{s}_t"])
              for s in ("train", "val", "test")}
    truth = down_data["test_t"]
    ids = w28.load_event_ids(region, len(truth))
    region_cache[region] = {
        "data": data, "down_data": down_data, "physics": physics,
        "feats": feats, "feats_none": feats_none, "raw": raw,
        "d_real": d_real, "truth": truth, "ids": ids, "n_down": n_down,
    }
    return region_cache[region]

# Run one arm
def run_arm(region, arm_name):
    t_start = time.time()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"RUNNING: {region} / {arm_name}")
    print(f"{'='*60}")

    st = get_region_state(region)
    data = st["data"]
    n_down = st["n_down"]
    reg_by_arm = {}
    arms_out = {}

    # If region doesn't exist in payload yet, init it
    region_data = payload["regions"].get(region, {})
    if not region_data:
        # A0 baselines
        base = {}
        truths = st["truth"]
        for k in ("gru", "tcn"):
            model = w28.train_downstream(k, st["down_data"], None, [42, 99, 7], 8000, 256)
            ev = w28.event_level_mae(model, truths, st["ids"]) if st["ids"] is not None else \
                {"all_window_mae": float(np.abs(model.mean(0) - truths).mean())}
            base[k] = ev
        region_data = {
            "n_train_fm": len(data["train_h"]),
            "n_train_downstream": n_down,
            "n_test": len(data["test_h"]),
            "n_eval_windows": len(data["test_h"]),
            "era5_join_missing_before_interp": st["physics"].attrs.get("era5_missing_after_join", 0),
            "coast_normal_onshore": list(w28.COAST_NORMAL_ONSHORE[region]),
            "baseline_A0": base,
            "fm_seeds": [42, 99],
            "downstream_seeds": [42, 99, 7],
            "arms": {},
        }

    # Build physics features (if ERA5 arm)
    if arm_name in ("standard",):
        arm_feats = st["feats_none"]
        full_pred_test = None
    else:
        arm_feats = st["feats"]
        _, stats = w28.build_cond_features(data["train_starts"], st["physics"], True)
        test_phys = st["raw"]["test"]
        full_pred_test = w28.ols_predict(test_phys, data[f"test_h"][:len(test_phys)], stats, region)

    # Per FM seed
    per_seed = {}
    fm_seeds = region_data["fm_seeds"]
    downstream_seeds = region_data["downstream_seeds"]
    arm_seed = region_data["arms"].get(arm_name, {})
    existing_per_seed = arm_seed.get("per_fm_seed", {})

    for seed in fm_seeds:
        skey = str(seed)
        # Skip if complete
        if skey in existing_per_seed and "downstream" in existing_per_seed[skey]:
            print(f"  seed={seed}: ALREADY DONE, skipping")
            per_seed[skey] = existing_per_seed[skey]
            continue

        ckpt_path = CKPT_DIR / f"{region}_{arm_name}_seed{seed}.pt"
        model = w28.train_fm(arm_name, arm_feats, data, seed, ckpt_path, arm_name)
        reg_by_arm[arm_name] = model.get("reg_coefs") if isinstance(model, dict) else None

        gen = w28.sample_windows(model, data["test_h"], arm_feats["test"],
                                  np.arange(len(data["test_h"])),
                                  draws=10, seed_base=1000 * seed)
        real = st["truth"]
        g_flat = gen.reshape(-1, 72)
        report = w28.evaluate(real, g_flat, train_ref=data["train_t"])

        # Physics consistency
        d_gen = gen.mean(axis=0)[:, :24].mean(axis=1) - data["test_h"][:, 72:96].mean(axis=1)
        pc = w28.physics_consistency(st["d_real"]["test"][:len(d_gen)], d_gen, full_pred_test[:len(d_gen)]) \
            if full_pred_test is not None else None

        per_seed[skey] = {
            "params": model["params"],
            "lambda_phys": model.get("lambda_phys", 0),
            "final_train_loss": model["hist"]["loss"][-1],
            "fidelity_status": report.status,
            "scale_ratio": float(report.scale_ratio),
            "wasserstein1": float(report.wasserstein1),
            "q95_exceedance_ratio": float(report.q95_exceedance_ratio),
            "q99_exceedance_ratio": float(report.q99_exceedance_ratio),
            "peak_ks_pvalue": float(report.peak_ks_pvalue),
            "peak_timing_mae": float(report.peak_timing_mae),
            "physics_consistency": pc,
            "train_s": round(model["train_s"], 1),
            "warnings": list(report.warnings)[:3],
        }
        print(f"  seed={seed}: fidelity={report.status} scale={report.scale_ratio:.3f} W1={report.wasserstein1:.4f}")

        # Downstream
        gidx = np.random.default_rng(7).choice(len(st["down_data"]["train_h"]), size=min(2740, n_down), replace=False)
        if arm_name != "standard":
            tr_phys_idx = w28.feats["train"][gidx]
        else:
            tr_phys_idx = np.zeros((len(gidx), 0), dtype=np.float32)
        gen_train = w28.sample_windows(model, st["down_data"]["train_h"],
                                        tr_phys_idx, gidx, draws=1,
                                        seed_base=2000 * seed)[0]
        pred = {k: w28.train_downstream(k, st["down_data"],
                    {"hist": st["down_data"]["train_h"][gidx], "target": gen_train},
                    tuple(downstream_seeds), 8000, 256) for k in ("gru", "tcn")}
        ev = {k: (w28.event_level_mae(pred[k], st["truth"], st["ids"])
                  if st["ids"] is not None else
                  {"all_window_mae": float(np.abs(pred[k].mean(0) - st["truth"]).mean())})
              for k in ("gru", "tcn")}
        per_seed[skey]["downstream"] = ev
        for k in ("gru", "tcn"):
            bm = region_data["baseline_A0"][k].get("extreme_unit_mae",
                    region_data["baseline_A0"][k].get("all_window_mae"))
            vm = ev[k].get("extreme_unit_mae", ev[k].get("all_window_mae"))
            print(f"    {k} delta = {(vm-bm)/bm*100:+.2f}%")

    # Aggregate
    agg = {}
    base = region_data["baseline_A0"]
    for k in ("gru", "tcn"):
        puv = np.mean([np.array(per_seed[s]["downstream"][k]["per_unit_mae"])
                       for s in per_seed if "per_unit_mae" in per_seed[s]["downstream"][k]], axis=0) \
            if all("per_unit_mae" in per_seed[s]["downstream"][k] for s in per_seed) else None
        pua0 = np.array(base[k]["per_unit_mae"]) if "per_unit_mae" in base[k] else None
        stats = w28.paired_event_stats(puv - pua0) if puv is not None and pua0 is not None and len(puv) else None
        deltas = [(per_seed[s]["downstream"][k].get("extreme_unit_mae",
                    per_seed[s]["downstream"][k].get("all_window_mae"))
                   - base[k].get("extreme_unit_mae",
                    base[k].get("all_window_mae")))
                  / base[k].get("extreme_unit_mae", base[k].get("all_window_mae")) * 100
                  for s in per_seed]
        agg[k] = {
            "mean_delta_pct": float(np.mean(deltas)),
            "per_seed_delta_pct": deltas,
            "paired_event_stats": stats,
        }

    arms_out[arm_name] = {
        "per_fm_seed": per_seed,
        "augmentation_ratio": min(2740, n_down) / n_down,
        "physics_regression": reg_by_arm.get(arm_name),
        "downstream_agg": agg,
    }

    # Update payload and save
    region_data["arms"][arm_name] = arms_out[arm_name]
    payload["regions"][region] = region_data
    OUT.write_text(json.dumps(payload, indent=2))
    elapsed = time.time() - t_start
    print(f"\n  ✅ {region}/{arm_name} saved ({elapsed:.0f}s)")
    return arms_out[arm_name]


# Run missing arms
print(f"\n{'='*60}")
print(f"WAVE 2.9 PHYSICS RESUME — GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
print(f"Target regions: {TARGET_REGIONS}")
print(f"Missing: {len(missing)} arm-runs")
print(f"{'='*60}")

results = []
for region, arm in missing:
    try:
        res = run_arm(region, arm)
        results.append((region, arm, "OK", res))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ FAILED {region}/{arm}: {e}")
        results.append((region, arm, f"FAIL: {e}", None))
        # Continue to next — partial progress better than nothing

print(f"\n{'='*60}")
print("RESUME SUMMARY")
print(f"{'='*60}")
for r, a, s, _ in results:
    print(f"  {r:30s} {a:20s} {s}")

print(f"\n✅ Output -> {OUT}")
print("Now run: python3 scripts/wave28_physics_analysis.py")
