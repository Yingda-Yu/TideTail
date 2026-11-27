#!/usr/bin/env python3
"""Validate real generator checkpoints against cell metadata.

Checks per cell with a checkpoint_path:
  - file exists, sha256 matches cell.checkpoint_sha256
  - torch.load succeeds (CPU), state_dict non-empty
  - tensor count and total element count recorded
Exit code 1 on any failure. Read-only; never touches running processes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
JSON_IN = ROOT / "data/results/wave30_generator_fidelity.json"
FAMILIES = ["flow_matching", "tail_aware_fm", "cvae", "timegrad",
            "ddpm_cond", "wgan_gp", "diffusion_ts"]


def main() -> int:
    d = json.loads(JSON_IN.read_text())
    failures = []
    rows = []
    n_expected = 0
    for region, reg in sorted(d["regions"].items()):
        for fam in FAMILIES:
            for seed in ("42", "99"):
                n_expected += 1
                cell = reg["families"][fam]["per_gen_seed"][seed]
                cp = cell.get("checkpoint_path")
                if not cp:
                    failures.append(f"{region}/{fam}/s{seed}: no checkpoint_path")
                    rows.append((region, fam, seed, "NO-PATH", None, None, None))
                    continue
                p = ROOT / cp
                if not p.exists():
                    failures.append(f"{region}/{fam}/s{seed}: file missing {cp}")
                    rows.append((region, fam, seed, "MISSING", None, None, None))
                    continue
                sha = hashlib.sha256(p.read_bytes()).hexdigest()
                sha_ok = sha == cell.get("checkpoint_sha256")
                try:
                    sd = torch.load(p, map_location="cpu", weights_only=True)
                except TypeError:
                    sd = torch.load(p, map_location="cpu")
                # checkpoint layout: {ModuleName: OrderedDict(state_dict)}
                n_tensors = 0
                n_elems = 0
                for v in sd.values():
                    if torch.is_tensor(v):
                        n_tensors += 1
                        n_elems += int(v.numel())
                    elif isinstance(v, dict):
                        for t in v.values():
                            if torch.is_tensor(t):
                                n_tensors += 1
                                n_elems += int(t.numel())
                empty = n_tensors == 0 or n_elems == 0
                status = "OK" if (sha_ok and not empty) else \
                    ("EMPTY" if empty else "SHA-MISMATCH")
                if status != "OK":
                    failures.append(f"{region}/{fam}/s{seed}: {status}")
                rows.append((region, fam, seed, status, n_tensors, n_elems,
                             "sha_ok" if sha_ok else "sha_MISMATCH"))
    print(f"{'region':26s} {'family':14s} {'seed':4s} {'status':12s} "
          f"{'tensors':>7s} {'params':>9s} sha")
    for r in rows:
        print(f"{r[0]:26s} {r[1]:14s} {r[2]:4s} {str(r[3]):12s} "
              f"{str(r[4]):>7s} {str(r[5]):>9s} {r[6] or ''}")
    n_with_ckpt = sum(1 for r in rows if r[4] is not None)
    print(f"\ncells expected: {n_expected}; checkpoints validated: "
          f"{n_with_ckpt}; failures: {len(failures)}")
    for f in failures:
        print(" FAIL:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
