# LATEST PROGRESS — TideTail

**Last updated**: 2026-11-03 Wave 1.8
**Head SHA**: `d124bcc`
**Branch**: `trae/global-data-generative-research-20261031`
**PR**: #23 (Draft)

---

## Current Status

### ✅ COMPLETE
- **Wave 1.7 P0 audit**: 190 prediction files verified, 10 FM checkpoints verified, event-level bootstrap CI computed
- **Physics equations**: Full math derivation for inverse barometer, wind stress, coastal setup, tail weights, lag kernel
- **Publication figures**: 6/14 generated (F1, F3, F4, F6, F8, F9) from verified artifacts only
- **Dataset coverage matrix**: All 10 datasets catalogued with provenance
- **Task queue**: W17-T1 marked COMPLETE, scientific_gate RESOLVED

### 🟡 IN PROGRESS / PENDING
- **Generative benchmark**: GPU3 and GPU4 available (24 GB free each) — next immediate priority
- **Manuscript updates**: Need event-level CI + data quality notes + figure references
- **GEBCO download**: Available without auth — low bandwidth, no blocker
- **ERA5**: AUTH_REQUIRED (user action: register CDS → get token → write ~/.cdsapirc)

### ❌ BLOCKED
- **Physics-guided generative models**: Need ERA5 wind/pressure data
- **Full shallow-water PDE claims**: Not within scope of TideTail

---

## Key Verified Numbers

| Metric | Value |
|---|---|
| Total prediction files | 190/190 ✅ |
| FM checkpoints (correct target) | 10/10 ✅ |
| Pytest | 270 passed, 0 fail |
| Unique pseudo-events | 360×4 + 301×1 = 1741 calendar-day groups |
| Valid A3 configs | 12 per region (6 FM seeds × 3 ratios × 2 downstream seeds) |
| **Best A3 overall** | ny_bight, -15.3% GRU (s99_ratio_0.5) |
| **Worst region** | thames_north_sea, all A3 configs hurt, +2.8% best-case |
| A1 oversampling | 0/20 comparisons help — UNIVERSAL NEGATIVE |
| Event-level CI | Slightly more conservative than window-level but directionally identical |

---

## GPU Status (2026-11-03)

| GPU | VRAM | Usage | Policy |
|---|---:|---|---|
| **0** | 24 GB | 22 MiB | ❌ PROHIBITED (other user) |
| 1 | 24 GB | 18.5 GB | LOCKED |
| 2 | 24 GB | 22.3 GB | SATURATED |
| **3** | 24 GB | **24 MiB** | ✅ **FREE** |
| **4** | 24 GB | **18 MiB** | ✅ **FREE** |
| 5-9 | 24 GB | 17-18 GB | 100% saturated |

**Immediate**: `CUDA_VISIBLE_DEVICES=3,4` for tail-aware FM training.

---

## Immediate Next Commands

```bash
# 1. Push everything
git add -A && git commit -m "feat: Wave 1.8 — physics docs, figures, acceptance report"
git push origin trae/global-data-generative-research-20261031

# 2. Verify PR #23 CI
gh pr view 23 --repo Yingda-Yu/TideTail --json checks

# 3. Start tail-aware FM training (GPU3)
CUDA_VISIBLE_DEVICES=3 python3 scripts/train_tail_aware_fm.py --region ny_bight --seed 42

# 4. Download GEBCO regional subsets (no auth)
python3 scripts/download_gebco.py --regions ny_bight,us_gulf_coast,south_atlantic_charleston,southeast_australia,thames_north_sea
```

---

## Blockers

| Blocker | Resolution | Status |
|---|---|---|
| ERA5 CDS auth | User register + get token → write ~/.cdsapirc | ❌ AUTH_REQUIRED |
| Canonical storm IDs | Wire storm_events.py into pipeline | ❌ DEFERRED |
| Thames NaN source gaps | NOAA QC flags likely maintenance | ❌ DOCUMENTED (test set clean) |

---

## Evidence Traceability

Every claim in this session traces to:
- Git SHA `d124bcc` (HEAD at time of writing)
- `data/results/wave17_analysis_v2.json` (190 records)
- `data/predictions/v2/*.npz` (190 files, all with event_ids + timestamps)
- `data/checkpoints/flow_matching_{region}_seed{42,99}.pt` (10 checkpoints)
- `docs/status/WAVE17_AUDIT.md` (independent P0 audit)
- `docs/status/WAVE1_8_ACCEPTANCE.md` (this wave's acceptance report)
