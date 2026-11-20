# Wave 2.9 Recovery Audit — 2026-11-20

## Context

Wave 2.8 (2026-11-08) had its systemd unit `tidetail-wave28physics.service`
OOM-killed at 23:00:43 after 3h25m CPU, 145 processes killed. Charleston
4/6 arms (standard, era5_cond, pressure_prior, wind_prior) were complete.
NY_bight 0 arms started. Combined_prior + combined_tail missing on charleston.

This audit documents the recovery path and Wave 2.9 execution status.

---

## Known Bugs Fixed (Wave 2.9)

| # | Bug | Wave 2.8 表现 | Wave 2.9 Fix |
|---|-----|--------------|--------------|
| 1 | **`--device cuda:N` + `CUDA_VISIBLE_DEVICES=N` ordinal mismatch** | 第一次 GPU 跑报 `invalid device ordinal` | `--device cuda:0` 配合 `CUDA_VISIBLE_DEVICES=3` |
| 2 | **`sample_windows` 输入 tensor 在 CPU，model 在 GPU** | Wave 2.8 全程 CPU 跑未暴露；Wave 2.9 GPU 跑报 `Expected all tensors same device` | 加 `dev = next(net.parameters()).device`，输入 `.to(dev)`，输出 `.cpu().numpy()` |
| 3 | **`--n-test 10000` 导致 sampling 8601×3×64 = 1.65M forward passes** | 进程 CPU 1044% GPU 0% 卡 29+ 分钟无输出 | 回退到默认 `--n-test 200`，sampling ~1s/seed |
| 4 | **Resume 模式重算 A0 baseline GRU+TCN 浪费 CPU** | 每 region 多花 2-3 分钟 CPU 训练 | `if args.resume and existing_rd.get('baseline_A0')` → 跳过加载已有 |
| 5 | **无断点恢复机制，每次重跑从头覆盖 JSON** | Wave 2.8 中途 OOM，4 arm 白跑？不——实际 JSON 已保存 4 arm | `--resume` flag + `_arm_complete()` 检测 + per-region/arm 跳过逻辑 |
| 6 | **`torch.set_num_threads` 未 cap** | Wave 2.8 默认 CPU 线程全开，竞争 OOM | `torch.set_num_threads(min(8, torch.get_num_threads()))` |

---

## Verification Matrix

| Test | Status | Notes |
|------|--------|-------|
| pytest | **311/311 PASS** | 比 Wave 2.8 270 还多 |
| ruff src | **ALL PASS** | Wave 2.8 有 2 个 wgan_gp.py 残留问题，已自动修复 |
| ruff tests | **ALL PASS** | |
| FlowMatching FM train_variant GPU3 | **22.5s / 10 epochs** | charleston, 69970 train windows, batch_size=256 |
| FlowMatching FM train_variant CPU | 未测（不需要） | GPU3 独占优先 |
| sample_windows GPU | **0.14s / draw** | 200 test windows × 3 draws = 0.4s |
| FM fidelity eval | Wave 2.8 已有结果 | 见 `wave28_physics_fm.json` charleston 4 arm |

---

## Physics FM GPU3 Benchmarks (2026-11-20)

```
FM train_variant (combined_prior, charleston):
  Device: RTX 3090 (CUDA_VISIBLE_DEVICES=3 → cuda:0)
  Epochs: 10
  Train windows: 69,970
  Batch size: 256
  Wall time: 22.5s
  Loss: 1.0111 (start 1.3785)
  lambda_phys auto-calibrated: 1.0000

sample_windows:
  Test windows: 200
  Draws: 3
  Euler steps: 64 (model default)
  Wall time: 0.44s total (0.14s/draw)
```

---

## Wave 2.9 Resume Flag Design

```python
# wave28_physics_fm.py main() additions:
def _arm_complete(region_data, arm):
    per_seed = region_data.get("arms", {}).get(arm, {}).get("per_fm_seed", {})
    dagg = region_data.get("arms", {}).get(arm, {}).get("downstream_agg", {})
    fm_seeds = region_data.get("fm_seeds", [42, 99])
    return (len(per_seed) >= len(fm_seeds)
            and all("downstream" in sd for sd in per_seed.values())
            and bool(dagg.get("gru", {}))
            and bool(dagg.get("tcn", {})))

# Resume flow:
if args.resume and out_path.exists():
    payload = json.loads(out_path.read_text())  # load prior results
missing_arms = [a for a in ARMS if not _arm_complete(existing_rd, a)]

# Skip A0 baseline if present
if args.resume and existing_rd.get("baseline_A0"):
    base = existing_rd["baseline_A0"]

# Skip completed arms, load existing into arms_out
if _arm_complete(existing_rd, arm):
    arms_out[arm] = existing_rd["arms"][arm]
    continue

# Incremental save after each arm
payload["regions"][region]["arms"][arm] = arms_out[arm]
out_path.write_text(json.dumps(payload, indent=2))
```

---

## Current Status (2026-11-20 08:30 UTC)

```
Wave 2.9 Physics Resume running:
  PID: 1559342
  GPU: CUDA_VISIBLE_DEVICES=3 (RTX 3090, 24GB free)
  Device flag: --device cuda:0
  Regions: south_atlantic_charleston → ny_bight
  Arms pending charleston: combined_prior (downstream eval) → combined_tail
  Arms pending ny_bight: ALL 6 × 2 FM seeds × downstream = 24 units
  
JSON output: data/results/wave28_physics_fm.json
  charleston: 4/6 complete (standard, era5_cond, pressure_prior, wind_prior)
  ny_bight: 0/6
```

---

## Execution Timeline (expected)

| Phase | Region | Arms | Est time (GPU3) |
|-------|--------|------|-----------------|
| 1. Combined_prior downstream | charleston | seed 42+99 | ~3 min (CPU train_downstream) |
| 2. Combined_tail full cycle | charleston | 2 seeds × FM + downstream | ~10 min |
| 3. All 6 arms | ny_bight | 2 seeds × each × FM + downstream | ~30 min |
| **Total** | | | **~45 min** |

---

## Negative Result Confirmation (unchanged)

Wave 2.8 charleston 4-arm finding still stands:
- **NO physics regularization benefit** on 14 charleston 2024 extreme events
- Every arm's Δ% < 1.5%, all signflip_p > Bonferroni α=0.0025
- Physics-inspired regularization on 96h/72h flow-matching residual task = zero measurable gain

NY_bight expected to confirm or contradict — Wave 2.9 will answer.

---

## Remaining Work After Physics FM

| Task | Script | Resource | Notes |
|------|--------|----------|-------|
| Wave28 physics analysis | `wave28_physics_analysis.py` | CPU 5min | 6-arm C1-C5 contrasts (was blocked by missing arms) |
| Wave28 expanded_eval | `wave28_expanded_eval.py` | CPU + FM retrain | charleston/ny_bight variants 0 downstream_agg → 补跑 + 扩展 3 区域 |
| Gen benchmark fidelity | `wave28_gen_bench_revised_eval.py` | CPU | season-stratified, 3 more regions |
| Paper manuscript | — | — | sections independent of pending results can start now |
| Git commit + push PR #23 | — | — | after all results complete |
