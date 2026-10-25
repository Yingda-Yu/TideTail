# Evidence-linked claims ledger

| ID | Claim | Status | Artifact / commands | Commit | Qualification |
|---|---|---|---|---|---|
| H1 | Extreme-aware diffusion improves held-out real extreme-event accuracy vs strong augmentation baselines | UNTESTED HYPOTHESIS | none | none | Needs event-level independent CI |
| H2 | Physical and multi-gauge coherence improves synthesis usefulness | UNTESTED HYPOTHESIS | none | none | Optional extension |

Status changes require reproducible artifacts, unambiguous split IDs and review. Negative and inconclusive results remain visible.

| E6 | A genuine temporal conditional DDPM implementation passes CPU forward/sampler gates; real GPU training is not established | ENGINEERING ONLY / GPU BLOCKED | [Issue #6 report](../docs/experiments/issue6/REPORT.md), `pytest.txt`, `preparation.json` | See PR head; initial base 554dfc2 | Zero optimizer steps; unreserved GPU ownership unverified; G1/generalization sign-off incomplete; no H1/H2 evidence |
