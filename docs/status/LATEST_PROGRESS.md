# LATEST PROGRESS — TideTail Wave 2.0–2.4

**Last updated**: 2026-11-03 Wave 2.4 (unified orchestration checkpoint)
**Head SHA**: `108eba3`
**Branch**: `trae/global-data-generative-research-20261031`
**PR**: #23 (Draft — DO NOT MERGE main)

---

## Unified Dependency-Aware Task Queue

| Task | PID | GPU | Status | Artifacts |
|---|---|---|---|---|
| **Standard FM A3 downstream** | 4019786 | GPU4 | 🔄 RUNNING | → append to `a3_flowmatching_augment_96_72.json` |
| **VAE full training (5×2×100)** | 4022380 | GPU1 | 🔄 RUNNING | → `data/checkpoints/vae_*_seed{42,99}.pt` |
| **TimeGrad full training (5×2×50)** | 4022383 | GPU3 | 🔄 RUNNING | → `data/checkpoints/timegrad_*.pt` |
| **ERA5 download (2 workers)** | 1868415/16 | CPU | 🔄 RUNNING | 253/600 verified, 786 MB |
| **CDS auth tiny test** | 4070374 | — | 🔄 RUNNING | → `/tmp/era5_auth_result.json` |
| **Storm event census** | — | CPU | ✅ COMPLETE | `data/results/storm_event_census_2015_2024.json` |
| **POT+TC bootstrap** | — | CPU | ✅ COMPLETE | `data/results/wave20_pot_storm_bootstrap.json` |
| **Tail-aware FM A3** | done | GPU4 | ✅ COMPLETE | 242 records in `a3_flowmatching_augment_96_72.json` |

---

## Resource Status

| Resource | Value |
|---|---|
| **GPU0** | 🚫 PROHIBITED (llama-server + starvla, 22MB free) |
| **GPU1** | VAE full training (19.4GB/24GB, 30% util) |
| **GPU2** | comfy-h3 (22.3GB), FREE otherwise |
| **GPU3** | TimeGrad (419MB used — initializing) |
| **GPU4** | Standard FM (5.4GB used — initializing) |
| **GPUs 5-9** | OCCUPIED by others |
| **Disk free** | 933 GB (after ERA5 download + 136 GB increase) |
| **pytest** | 270 PASS |

---

## ERA5 Download Pipeline

| Metric | Value |
|---|---|
| CDS config | `~/.cdsapirc`, perms 600, ✅ auth tested |
| cdsapi | 0.7.7 ✅ |
| Workers | 2 PID=1868415 (ny_bight, se_aus) + 1868416 (gulf, charleston, thames) |
| Verified | 253 / 600 (42%), 786 MB |
| Manifest | `/data/yingda/tidetail-data/real-sources-v1/era5/MANIFEST.json` |
| Integrity | xarray verify: vars present, hour count, lat/lon bounds, physical plausibility |
| Rate limit | 3s cooldown per chunk, CDS protection notice acknowledged |
| Test data | `/data/yingda/tidetail-data/real-sources-v1/era5-test/` 1.4 MB verified |

---

## Generative Model Status

| Model | Checkpoints | Training | Results |
|---|---|---|---|
| Diffusion-TS | 10/10 ✅ | ✅ 1 region seed42 done | `diffusion_ts_96_72.json` 1 record |
| Flow Matching (standard) | 10/10 ✅ | 🔄 GPU4 A3 eval running | Pending |
| Flow Matching (tail-aware) | 10/10 ✅ | ✅ Complete | 242 records |
| Conditional VAE | 1/10 | 🔄 GPU1 5×2×100 running | Pending |
| TimeGrad | 0/10 | 🔄 GPU3 5×2×50 running | Pending |
| WGAN-GP | Not started | Needs code | — |

---

## Storm Event Independence — CRITICAL SCIENTIFIC FINDING

POT q95 + 168h support embargo declustering across all 5 regions, 2015-2024:

| Region | per-2024 | per-2023-24 | per-2022-24 | 10yr total | TC coverage |
|---|---:|---:|---:|---:|---:|
| ny_bight | 15 ✅ | 33 ✅ | 51 ✅ | 171 | 10 TCs |
| us_gulf_coast | 11 | 26 | 37 | 110 | 37 TCs (rich!) |
| charleston | 13 | 27 | 40 | 133 | 25 TCs |
| se_aus | 6 ❌ | 15 | 20 | 101 | 0 TCs |
| thames | 10 | 18 | 34 | 119 | 1 TC |
| **TOTAL** | **55** | **119** | **182** | **634** | |

### Key conclusions:
1. Wave 1.9's 2024-only finding was exploratory due to low power (se_aus only 6 units)
2. **expanded_3yr_test (2022-2024, 182 units)** is now the formal evaluation protocol
3. Charleston's 2024 finding (FM aug statistically significant) needs expanded-cohort reconfirmation
4. Thames' significant harm finding also needs expanded-cohort validation
5. us_gulf_coast has rich canonical TC coverage (37 TCs 10yr) → best for TC-conditioned analysis

---

## Wave 2.0 POT+TC Bootstrap — 2024-only results

| Region | Independent Units | Best A3 Δ% | Verdict |
|---|---:|---:|---|
| **charleston** | 15 | -16.0% | ✅ 4 significant improvements (CI95 entirely below A0 MAE=0.1023) |
| ny_bight | 16 | -14.5% | 📉 numeric, CI overlaps |
| us_gulf | 16 | -10.3% | 📉 numeric, CI overlaps |
| se_aus | 7 | -6.7% | Descriptive only |
| **thames** | 10 | +35% worst | ❌ Significant harm |

---

## Next Three Executable Tasks

1. **When GPU4 FM finishes → diffusing on expanded_3yr_test protocol** (2022-2024 test years, 182 independent units)
2. **Physics-guided FM ablation** (once ≥3 regions ERA5 complete): pressure-only prior, wind-only prior, combined
3. **Expanded-cohort bootstrap** for charleston/thames formal inference on 2022-2024 units

---

## Git & CI

| Item | Value |
|---|---|
| Current branch | `trae/global-data-generative-research-20261031` |
| Base | `d7fdb5a` |
| Head | `108eba3` |
| New commits this session | 7 |
| pytest | 270 PASS |
| Merges | NO — DRAFT PR #23 |

---

## CLAIMS_LEDGER E16a Status

- ❌ Wave 1.9: exploratory (0-1 observ segments/year)
- ✅ Wave 2.0: mixed scientific finding (charleston verified, thames harm confirmed)
- ⏳ Wave 2.4: awaiting expanded-cohort reconfirmation on 182 3yr units

---

**Blockers**: None currently. All pipelines running or queued.
**Independent units**: 634 POT clusters confirmed 2015-2024 across 5 regions.
**ERA5 progress**: 42% downloaded, 2 workers active 14+ hours.
**Next checkpoint**: After standard FM / VAE / TimeGrad training completes (~1-2h).
