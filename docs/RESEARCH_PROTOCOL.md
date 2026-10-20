# TideTail preregistration-style research protocol v0.1

## Core question
Can tail-aware conditional diffusion augmentation improve prediction of independent real extreme coastal water-level events relative to original-data training, oversampling and standard diffusion, under matched forecaster training budget? Treat all expected gains as hypotheses.

## Target and observations
Specify gauge datum, units, tide versus surge residual, frequency, sampling period, horizon, forecast issue time and station set explicitly. Candidate sources include GESLA, NOAA and ERA5; acquisition, coverage, attribution and redistribution rights are NOT yet verified.

## Leakage-safe splitting
Partition independent event groups and chronological blocks before window creation, imputation fit, scaling, extreme-threshold estimation or generative training. Apply an embargo sufficient to prevent input/target overlap across partitions. Keep storm groups together; hold out sites separately for cross-station generalization. Select q95/q99 extreme thresholds from TRAIN only, then apply those fixed thresholds to real validation/test targets. If event definitions need hindsight, ensure they are only used for evaluation labels, not operational forecast inputs.

## Conditions and physics
Prefer generating de-tided residuals and explicitly restoring tidal components when scientifically warranted. Conditioning may use gauge history, calendar/site metadata and meteorological forecasts known at issuance; observed future weather is forbidden in operational experiments (oracle experiments must be segregated). Validate spectra, amplitude, duration, tail mass, gauge-to-gauge propagation lags, forcing-response consistency, physical units and nearest-neighbor memorization. Report both sample realism and forecast utility.

## Precommitted comparisons
A0 real-only; A1 balanced sampler / duplication; A2 perturbation or carefully justified interpolation; A3 standard conditional diffusion; A4 tail-aware conditional diffusion; A5 A4 plus physical and/or spatial constraints if feasible. Match model capacity, training steps, total input examples and tuning budget, with sensitivity for synthetic fraction and seeds. Include naive/persistence and at least one strong neural forecasting baseline.

## Measures
Overall MAE/RMSE; real-event q95/q99 tail MAE/RMSE; event peak magnitude and timing error; threshold event precision/recall/false alarms; lead-time and station-level breakdown; event-unit paired bootstrapped CIs; computation/latency. Document denominator, sample count and handling of zero-event test folds.

## Integrity
No train-validation-test contamination or selection based on test; no synthetic validation/test; no invented outcomes. Preserve run logs including failures. Positive claims must survive held-out event analysis with uncertainty, strong baselines and fair budgets. Freeze a data/split card before expensive GPU experiments.
