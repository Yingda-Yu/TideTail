# Agent working agreement
Read README.md, docs/RESEARCH_PROTOCOL.md, docs/ROADMAP.md and open Issues before coding. Stay inside Yingda-Yu/TideTail.

Each task must be scoped to one Issue and an isolated branch, with a draft PR explaining exact tests, commands, and evidence. Do not merge, force-push, delete experiments, overwrite another agent's branch or push directly to main. Do not make publication claims without evidence.

Integrity first: split by event and time BEFORE making windows; no leakage through normalization, station alignment, threshold selection, augmentation or hyperparameter search. Never condition a real-time forecast on unknown future covariates. Preserve genuine negative results and independent sample counts.

GPU usage is opt-in. Specify CUDA_VISIBLE_DEVICES; check nvidia-smi and available memory before training. Do not assume GPU 0 is free, terminate other users' jobs, download huge weights without approval, or use paid APIs.

Log configuration, git SHA, dependencies, source data license/hash, split manifest, event IDs, station IDs, seeds, scripts and exact result artifacts. Never commit raw licensed datasets, credentials or checkpoints. Use pytest on CPU for PR gate. Cite literature precisely. Any paper claim requires traceability through paper/CLAIMS_LEDGER.md.
