# TideTail Claims Ledger — v0

> Every positive claim in any draft paper must have an evidence row here.
> Evidence = run logs, commit SHA, test artifact, figure number.
> Negative findings are also recorded (required).

| ID | Claim | Status | Evidence | License / Constraints |
|----|-------|--------|----------|----------------------|
| C001 | Persistence baseline MAE ≤ 0.5 m on real events | PENDING | — | Train-only thresholds |
| C002 | Seasonal baseline beats persistence | PENDING | — | — |
| C003 | A3 standard diffusion equals A0 on real events | PENDING | — | No synthetic val/test |
| C004 | A4 tail-aware diffusion beats A0 on real storms | PENDING | — | Matched budget |
| C005 | A4 tail-aware diffusion beats A3 on real storms | PENDING | — | Matched budget |
| C006 | No leakage present in split protocol | PENDING | `tidetail.splits.partition_chronological` + 5 passing leakage tests | |
| N001 | (Negative finding: to be recorded after runs) | — | — | — |
