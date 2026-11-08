# BLOCKERS — TideTail

**Last updated**: 2026-11-01
**Branch**: `trae/global-data-generative-research-20261031`
**Head**: `c6a703af4462e05bfc4a746b29d623e92e84e107`

---

## Blocker (a) — ERA5 CDS Authentication Missing

| Field | Value |
|---|---|
| **Severity** | `BLOCKING` |
| **Stage blocked** | S4 (Data Acquisition), S6 (Physics-guided models — conditional FM needs U10/V10/MSLP) |
| **Owner needed** | **Human user action required** |
| **Symptom** | `~/.cdsapirc` does not exist; `cdsapi.Client()` will fail with authentication error |
| **Root cause** | No Copernicus CDS API token has been configured |
| **Required action** | 1. Register at https://cds.climate.copernicus.eu/user/ 2. Get personal API key from user page 3. `echo "url: https://cds.climate.copernicus.eu/api\nkey: YOUR-TOKEN" > ~/.cdsapirc` 4. `chmod 600 ~/.cdsapirc` |
| **Workarounds** | None for ERA5. GEBCO regional subsets can proceed via `download.gebco.net` or CEDA OPeNDAP without CDS. Wind/pressure forcing can be obtained post-auth only. |
| **Affected files** | `scripts/download_real_data.py` (ERA5 section), `src/tidetail/data_sources/download_manager.py` |
| **Estimated resolution time** | ~15 min once user provides token |

---

## Blocker (b) — 10 Invalidated FM Checkpoints Need Retrain

| Field | Value |
|---|---|
| **Severity** | `BLOCKING` (for P5 rerun) |
| **Stage blocked** | S5 (Controlled Recovery — P5), S6 (all downstream generative variants) |
| **Owner needed** | Agent with GPU access |
| **Symptom** | All 10 `data/checkpoints/flow_matching_*_seed*.pt` trained on `residual = water_level_m - tide_level_m` — double-subtraction of astronomical tide |
| **Root cause** | P0 target-semantics bug (fc45703 fixed loader; checkpoints are pre-fix) |
| **Required action** | Retrain 10 FM checkpoints: 5 regions × 2 seeds, correct residual target (`residual = water_level_m`, not `water - tide`) |
| **Workarounds** | None — downstream A3 results depend entirely on generator fidelity |
| **Affected files** | `data/checkpoints/flow_matching_*_seed*.pt` (10 files — DO NOT delete, just supersede) |
| **Estimated compute** | ~30 GPU hours (15 epochs × 10 runs, ~3h each on RTX 3090) |
| **Status** | **RESOLVED 2026-11-03 21:27–21:47**. All 10 FM checkpoints retrained on correct residual target (`residual = water_level_m`). wave17 pipeline now running with corrected FM on 5 regions × 3 ratios × 2 FM seeds × 2 downstream models × 2 downstream seeds. |

---

## Blocker (c) — GPUs Saturated (GPU4 with Carla Simulator)

| Field | Value |
|---|---|
| **Severity** | `BLOCKING` (at peak contention) / `DELEGATED` (Carla coexistence) |
| **Stage blocked** | S5, S6, S7 — all GPU-requiring tasks |
| **Owner needed** | Agent must verify coexistence |
| **Symptom** | GPUs 5–9 saturated (< 1 GB free). GPU1/2 partially used. GPU3 has ~15.8 GB free. **GPU4 has 15.7 GB free but Carla simulator is already running** on same card. |
| **Root cause** | Multi-user GPU server — prior jobs + Carla occupy most cards |
| **Required action** | 1. `nvidia-smi` immediately before starting 2. If GPU4 Carla uses < 2 GB VRAM → safe to co-run FM training 3. If Carla uses > 8 GB → use GPU3 instead 4. Never use GPU0 |
| **Workarounds** | CPU-only analysis tasks (W17-T2, W19-T7) can run immediately without GPU. Carla can be paused if user approves. |
| **Policy** | `CUDA_VISIBLE_DEVICES=4` (or 3), verify with `nvidia-smi` first, no GPU0 |
| **Checkpoints** | ~16 GB each RTX 3090 available memory window for FM training |

---

## Blocker (d) — Disk at 97% Full

| Field | Value |
|---|---|
| **Severity** | `DELEGATED` (monitor) |
| **Stage blocked** | S4 (large downloads), S5 (multiple FM checkpoints + prediction saves) |
| **Owner needed** | Agent (can defer to user if threshold crossed) |
| **Symptom** | 281 GB free; new-download budget is 100 GB. ERA5 + GEBCO regional subsets could exceed 50 GB. |
| **Root cause** | Long-running experiment checkpoints + logs accumulating |
| **Required action** | 1. `df -h /data` before any download 2. Keep raw data outside repo per AGENTS.md 3. Clean old logs/checkpoints in `data/logs/` if needed |
| **Workarounds** | Download minimal regional subsets (5 regions only, ~5 GB total). Compress old checkpoints. |
| **Threshold** | Abort downloads if free space < 50 GB |

---

## Blocker (e) — HURDAT2 STATUS Field for Atlantic Needed for Typing Improvement

| Field | Value |
|---|---|
| **Severity** | `LOW` (enhancement, not blocking) |
| **Stage blocked** | None — current code has fallback to peak wind |
| **Owner needed** | Agent |
| **Symptom** | Some Atlantic storms parsed without STATUS field fall back to peak-wind heuristic; slight typing ambiguity (~5% of storms) |
| **Root cause** | HURDAT2 Atlantic file has STATUS field but older rows may be missing; code uses STATUS when present, falls back otherwise |
| **Required action** | Optionally: cross-reference HURDAT2 missing STATUS with IBTrACS basin info. Document which storms lack STATUS. |
| **Workarounds** | Current fallback works — peak wind > 64 kt → HU, > 34 kt → TS. Sufficient for S5 rerun. |
| **Impact** | Minor — affects ~5% of Atlantic storm typing; downstream metrics use per-event groups which are already robust |

---

## Blocker Summary Table

| ID | Blocker | Severity | Blocks | Owner | GPU Needed? |
|---|---|---|---|---|---|
| (a) | ERA5 CDS auth | **BLOCKING** | S4, S6 | User | No |
| (b) | FM checkpoint retrain | ~~**BLOCKING**~~ **RESOLVED** | S5, S6 | Agent | **YES** (10h, completed 21:27-21:47) |
| (c) | GPU contention (Carla) | BLOCKING/DELEGATED | S5, S6, S7 | Agent | N/A |
| (d) | Disk 97% full | **DELEGATED** | S4 downloads | Agent | No |
| (e) | HURDAT2 STATUS missing | **LOW** | None | Agent | No |

---

## Immediate Unblock Priority

1. **Launch G8 pilot on GPU4** — verify nvidia-smi first, coexist with Carla if VRAM allows
2. **User action** — obtain CDS token for ERA5
3. **Clean disk** if free < 200 GB before large downloads
4. FM retrain + A3 rerun unlocks S6 and S7
