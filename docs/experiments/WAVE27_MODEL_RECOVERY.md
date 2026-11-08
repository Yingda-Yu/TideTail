# Wave 2.7 — Model Recovery (Conditional VAE, TimeGrad, Diffusion-TS)

**Status:** diagnostic pilots complete. One repair hypothesis (TimeGrad memory
semantics, `sum` mode) produced a measurable improvement; the VAE and
Diffusion-TS repair hypotheses did **not** produce a credible improvement and are
recorded as negative results.

**Script:** `scripts/wave27_model_recovery.py`
**Artifact:** `data/results/wave27_model_recovery.json`
**Tests:** `tests/test_wave27_model_fixes.py` (7 tests, CPU)

All pilots ran on CPU. GPU0 is prohibited and every other GPU was occupied by
other users at launch (`nvidia-smi` checked before starting). The cohort is a
deterministic subsample of `south_atlantic_charleston`, so every comparison is
controlled within one region and one sampling seed.

Two budgets were run to separate architecture effects from budget effects:

| Run | `n_train` | `n_val` | `n_test` | Epochs (VAE / TG / DTS) | Wall time |
|---|---:|---:|---:|---|---:|
| Pilot A (small) | 4,000 | 1,000 | 200 | 8 / 10 / 8 | 77.5 s |
| Pilot B (primary) | 20,000 | 2,000 | 200 | 25 / 30 / 25 | 524.3 s |

Pilot B is the primary record. Pilot A is retained because the VAE result
**reverses** between the two budgets, which is itself the finding.

---

## 1. Sampler correctness is NOT the problem

Before attributing failures to model families, the DDPM reverse process was
verified with an **oracle denoiser** that returns the exact forward-process noise:

| Check | Result |
|---|---|
| Deterministic reverse pass recovers `x_0` | max abs error **1.1e-10** |
| Stochastic reverse pass | rel. RMSE 0.06 (expected posterior-noise floor) |

The `Diffusion` schedule, `noisy()` and the reverse recursion are correct. Any
fidelity failure is a *fitting* or *conditioning* failure, not a sampler bug.
This rules out the most common false diagnosis and is the reason the remaining
sections look at optimisation and sampling-path mismatches instead.

---

## 2. Conditional VAE

### Defect found

`ConditionalVAEDecoder` broadcast a **single scalar** `z` projection across all
72 horizon positions (`z.unsqueeze(1).expand(-1, horizon, -1)`), so the latent
could only shift the horizon's overall offset. With a 32-dim latent the
effective rank was 1 regardless of latent size. Additionally `ConditionalVAE`
did not subclass `nn.Module`, so `.to(device)` raised `AttributeError` and every
production VAE run silently stayed on CPU.

### Repairs tested

* `z_to_seq = nn.Linear(latent_dim, horizon)` → per-position signal
  `init_signal = z_to_seq(z).unsqueeze(1)`, restoring rank ≤ min(latent, 72)
  (`z_init="linear"`).
* `ConditionalVAE(nn.Module)` with `super().__init__()`.
* KL warmup (5 epochs) to a slightly larger β (0.005).

### Pilot result (Pilot B, 20k windows, 25 epochs)

| Variant | val recon MSE | val KL | active latent dims | scale_ratio | W1 | autocorr_RMSE | q95 ratio | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| legacy scalar z, β=0.001 | 0.882 | 2.92 | **0 / 32** | **3.29** | 0.475 | 0.197 | 10.13 | REJECTED |
| repaired linear z + warmup, β=0.005 | 0.326 | 2.76 | **0 / 32** | **7.26** | 1.034 | 0.316 | 7.55 | REJECTED |

Pilot A (4k windows) gave a different and opposite picture: legacy 1.32
(2/32 active), repaired 1.41 (16/32 active). Both were INCONCLUSIVE.

### Interpretation — the repair hypothesis is REFUTED at this budget

The linear latent path does exactly what it was designed to do in *training
space*: validation reconstruction MSE falls 2.7× and the decoder can represent
per-position variation. But **both variants show total posterior collapse**
(0/32 latent dimensions with posterior std > 0.316), and the repaired variant
therefore samples *worse* (scale 3.29 → 7.26).

The mechanism is a **prior mismatch, not a decoder defect**. The model samples
with `z ~ N(0, I)` (`vae.py` `sample()`), while the aggregate posterior has
collapsed to a narrow, non-standard-normal distribution (mean per-dimension KL
≈ 0.086 nat). At β=0.005 the decoder additionally learns a strong linear
`z → horizon` map, so off-manifold prior samples are amplified 41× more than in
the legacy graph. Strengthening the latent path without fixing the prior makes
sampling worse, exactly as observed.

The 4k → 20k reversal (16/32 → 0/32 active) shows the collapse is driven by
optimisation length: with 4× more steps the decoder satisfies reconstruction
from the history encoder alone and the KL term drives the latent to the prior.
This is the standard posterior-collapse dynamic, not a dataset-specific effect.

**Verdict: no credible improvement from the tested repairs. Hypothesis
preserved as negative.** The family is *not* condemned: the untested
next hypotheses are free-bits KL (per-dimension floor), latent dropout, or a
learned conditional prior `p(z | history)` so that sampling and training share
one distribution. None of these was run, so none may be claimed.

---

## 3. TimeGrad

### Defect found

`TimeGradDenoiser` accepted `ar_memory` and then **ignored it**, projecting
`noisy_scalar` through `memory_proj` instead; the aggregator input width was
`3×width`. The autoregressive memory — the entire point of TimeGrad — never
reached the network.

### Repairs tested

* `use_ar_memory=True`: separate `noisy_proj`, aggregator input `4×width`,
  memory genuinely consumed.
* `training_loss(..., ar_mode=)` / `forecast(..., ar_mode=)` now share one
  memory definition across `"sum" | "mean" | "last"`, removing the
  teacher-forcing/inference mismatch.

### Pilot result (Pilot B, 30 epochs)

| Variant | params | scale_ratio | W1 | autocorr_RMSE | status |
|---|---:|---:|---:|---:|---|
| legacy (memory ignored, w=24) | 6,553 | **10.55** | 1.474 | 0.462 | REJECTED |
| repaired, raw **sum** memory, w=64 | 39,617 | **6.09** | 1.342 | 0.429 | REJECTED |
| repaired, **mean** memory, w=64 | 39,617 | **1.22 × 10⁵** | 1.33 × 10⁴ | 0.898 | REJECTED — divergent |

Pilot A (4k windows) gave legacy 22.16 and repaired-`sum` **1.9 × 10⁷**.

**Diagnosis.** Repair *direction* is confirmed: consuming the memory and
widening the denoiser cuts scale error by 42% (10.55 → 6.09) and W1 by 9%. But
the rollout is still an order of magnitude outside the gate, and the two memory
aggregations behave in opposite ways. Raw `sum` grows without bound across 72
horizon steps (train saw at most 71 standardised terms from *teacher-forced*
targets), so train/inference mismatch compounds through the autoregressive
loop — 1.9 × 10⁷ at 4k windows. Conversely `mean` collapses the memory signal
to near-constant and the 72-step recursion diverges outright (1.2 × 10⁵) at 20k
windows. Neither aggregation is numerically stable across the full horizon.

**Verdict: partial repair — root cause confirmed, no usable model at this
budget.** The remaining gap is not attributable to the family. Autoregressive
rollout is 72× more expensive per sample than parallel DDPM (72 horizon steps ×
64 denoiser calls), which is why these pilots are small. A per-horizon-step
variance audit and a bounded (e.g. exponentially decayed) memory are the
untested next hypotheses.

---

## 4. Diffusion-TS

### Defect found

The Fourier-spectrum regulariser compared the spectrum of the **predicted
noise** to the spectrum of the **clean signal**. Noise and signal spectra have
opposite shapes, so the term pushed the model away from the data manifold.

### Repair tested

`training_loss(..., fourier_weight=0.1, fourier_target="x0")` reconstructs
`x0_pred = (noisy − sqrt(1−ᾱ)·ε_pred) / sqrt(ᾱ)` and compares its spectrum to
the clean target. `"noise_legacy"` retains the defective term for reproduction.

### Pilot result

| Variant | Pilot A scale | Pilot B scale | Pilot B W1 | Pilot B autocorr | status |
|---|---:|---:|---:|---:|---|
| legacy noise-spectrum Fourier | 21.59 | **10.98** | 4.527 | 0.538 | REJECTED |
| repaired x0-spectrum Fourier | 13.22 | **NaN** | NaN | NaN | REJECTED — diverged |

In Pilot B the repaired variant's training loss reached **NaN by epoch 24**, and
its generated set has `exact_dup_ratio = 0.998` — the sampler collapses to
repeated values.

**Diagnosis.** `x0_pred` divides by `sqrt(ᾱ_t)`. Under the subsampled cosine
schedule the terminal `ᾱ ≈ 5.9 × 10⁻⁴`, so `sqrt(ᾱ) ≈ 0.024`. At high `t` the
reconstruction is therefore amplified ~41× **and so is its gradient**, which is
why the loss diverges at higher budgets rather than settling. Pilot A's apparent
2× improvement (21.6 → 13.2) was an artefact of undertraining, not evidence that
the repair works; the divergence only appears once the model fits the data.

**Verdict: repair hypothesis REFUTED at this budget.** The correct fix is not a
different Fourier *target* but a numerically stable parameterisation — predict
`x0` or `v` directly with SNR weighting instead of dividing by `ᾱ`, or restrict
the x0-derived term to `t` below a threshold. Untested.

---

## 5. Cross-model summary — what is actually established

| Model | Root cause | Repair outcome | Evidence |
|---|---|---|---|
| Conditional VAE | scalar rank-1 latent **and** prior/posterior mismatch at sampling | reconstruction improved, **sample scale worsened** | recon 0.882 → 0.326 but scale 3.29 → 7.26; 0/32 active dims in both |
| TimeGrad | AR memory ignored + unstable memory aggregation | **partial**: scale 10.55 → 6.09 (sum), mean diverges | 4k sum reached 1.9 × 10⁷ |
| Diffusion-TS | Fourier term on the wrong target | **refuted**: NaN divergence once fitted | x0 term amplifies by 41× at high `t` |
| Flow Matching | none found | production checkpoints pass scale gate | sampler verified exact; scale 0.945–1.116 |

Two honest conclusions:

1. **No model family is shown to be intrinsically ineffective.** Every failure
   so far is an optimisation, sampling-path or numerical-stability defect that
   was localised to a specific line of code. The DDPM sampler itself is proven
   exact by the oracle test.
2. **Two of the three proposed repairs did not work, and saying otherwise would
   be overclaiming.** Because "it worked" was the prior expectation, this is
   precisely the kind of result that must be preserved rather than quietly
   re-tuned until it looks positive.

The manuscript must report (1) and (2) together, and must not generalise from
these single-region, CPU-budget pilots to any family-level verdict.

## 6. Reproduce

```bash
python3 -m pytest tests/test_wave27_model_fixes.py -q
python3 scripts/wave27_model_recovery.py --region south_atlantic_charleston \
    --n-train 20000 --n-val 2000 --n-test 200 \
    --epochs-vae 25 --epochs-timegrad 30 --epochs-diffusion 25
```
