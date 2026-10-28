# Evidence-linked claims ledger

| ID | Claim | Status | Artifact / commands | Commit | Qualification |
|---|---|---|---|---|---|
| H1 | Extreme-aware diffusion improves held-out real extreme-event accuracy vs strong augmentation baselines | UNTESTED HYPOTHESIS | none | none | Needs event-level independent CI |
| H2 | Physical and multi-gauge coherence improves synthesis usefulness | UNTESTED HYPOTHESIS | none | none | Optional extension |

Status changes require reproducible artifacts, unambiguous split IDs and review. Negative and inconclusive results remain visible.

| E6 | Temporal conditional DDPM executes 1,641 CUDA optimizer steps over three train-only epochs; checkpoint reload and finite conditional 24h sampling verified | ENGINEERING ONLY / SAMPLE QUALITY NEGATIVE | [GPU execution report](../docs/experiments/issue6/GPU_RUN_7164.md), `gpu_7164_aggregate.json`, `gpu_7164_reload.json` | 5bc646676a08191a77e5d424f2cf52bb1af00c04 | Owner-authorized GPU4; samples have implausible amplitudes despite falling training loss; independent storms unknown; license/datum/G1/G2 incomplete; no H1/H2 evidence. Earlier blocked record preserved in REPORT.md |
| E6-CPU | Pinned checkpoint independently reproduces implausible residual amplitudes on CPU; terminal epsilon error is strongly amplified; finite-only acceptance was insufficient | NEGATIVE ENGINEERING DIAGNOSIS | [CPU recovery report](../docs/experiments/issue6/cpu_recovery/REPORT.md), `diagnostics.json`, `tests_before.txt`, `tests_after.txt` | Parent e2d93fba65f670f9750aeae117ee9345c3ae1b47; training source 5bc646676a08191a77e5d424f2cf52bb1af00c04; repair commit in PR | CPU RNG differs from CUDA; generator remains unapproved; scale screen is not physical realism; no independent-storm utility evidence or H1/H2 change |
