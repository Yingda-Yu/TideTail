#!/usr/bin/env python3
"""Strict validation of Wave 3.0 generator checkpoints against cell metadata.

Acceptance requirements (owner-mandated 2026-11-27):
  - exactly the 5 predefined regions, all 7 predefined families,
    both seeds 42 and 99 -> exactly 70 expected cells
  - fail on any missing cell or missing checkpoint
  - verify checkpoint SHA256 against cell metadata
  - verify torch.load succeeds and state_dict is non-empty
  - record tensor count and total parameter elements
  - compare against cell param_count where definitions match; distinguish
    generator-only parameters from total trainable parameters when the
    checkpoint contains auxiliary modules
  - per-checkpoint loading exceptions are caught and reported explicitly

Read-only; never touches running processes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_REGIONS = ["south_atlantic_charleston", "ny_bight", "us_gulf_coast",
                    "southeast_australia", "thames_north_sea"]
REQUIRED_FAMILIES = ["flow_matching", "tail_aware_fm", "cvae", "timegrad",
                     "ddpm_cond", "wgan_gp", "diffusion_ts"]
REQUIRED_SEEDS = ["42", "99"]
EXPECTED_CELLS = len(REQUIRED_REGIONS) * len(REQUIRED_FAMILIES) \
    * len(REQUIRED_SEEDS)  # 70

# family -> module name captured in the generating closure that IS the
# generator (everything else in the checkpoint is auxiliary)
GENERATOR_MODULE = {
    "flow_matching": "VelocityTCN",
    "tail_aware_fm": "VelocityTCN",
    "cvae": "ConditionalVAE",
    "timegrad": "TimeGradDenoiser",
    "ddpm_cond": "TemporalDenoiser",
    "wgan_gp": "ConditionalWGANGP",  # includes critic (G+C by design)
    "diffusion_ts": "ResidualDenoiser",
}


def _count_state_dict(sd: dict) -> tuple[int, int]:
    """(tensor_count, total_elements) for {ModuleName: state_dict} or flat."""
    n_t = n_e = 0
    for v in sd.values():
        if torch.is_tensor(v):
            n_t += 1
            n_e += int(v.numel())
        elif isinstance(v, dict):
            for t in v.values():
                if torch.is_tensor(t):
                    n_t += 1
                    n_e += int(t.numel())
    return n_t, n_e


def check_cell(cell: dict, ckpt_path: Path, fam: str) -> dict:
    """Validate one cell. Returns a result dict; 'status' in {OK, FAIL}."""
    res: dict = {"status": "FAIL", "failures": []}
    cp_meta = cell.get("checkpoint_path")
    if not cp_meta:
        res["failures"].append("cell has no checkpoint_path metadata")
        return res
    if not ckpt_path.exists():
        res["failures"].append(f"checkpoint file missing: {ckpt_path}")
        return res
    sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    if sha != cell.get("checkpoint_sha256"):
        res["failures"].append(
            f"sha256 mismatch: file {sha[:16]}... vs cell "
            f"{str(cell.get('checkpoint_sha256'))[:16]}...")
        return res
    try:
        try:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:  # older torch without weights_only
            sd = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:  # explicit per-checkpoint error reporting
        res["failures"].append(f"torch.load failed: {type(exc).__name__}: {exc}")
        return res
    if not isinstance(sd, dict) or not sd:
        res["failures"].append("checkpoint state_dict empty or not a dict")
        return res
    n_t, n_e = _count_state_dict(sd)
    if n_t == 0 or n_e == 0:
        res["failures"].append("checkpoint contains zero tensors/elements")
        return res
    res["tensors"] = n_t
    res["total_elements"] = n_e

    gen_mod = GENERATOR_MODULE.get(fam)
    gen_elements = None
    aux_elements = 0
    if gen_mod is not None and gen_mod in sd and isinstance(sd[gen_mod], dict):
        gen_elements = sum(int(t.numel()) for t in sd[gen_mod].values()
                           if torch.is_tensor(t))
        for name, sub in sd.items():
            if name != gen_mod and isinstance(sub, dict):
                aux_elements += sum(int(t.numel()) for t in sub.values()
                                    if torch.is_tensor(t))
    res["generator_module"] = gen_mod
    res["generator_elements"] = gen_elements
    res["auxiliary_elements"] = aux_elements

    pc = cell.get("param_count")
    if pc is not None and int(pc) > 0:
        if gen_elements == int(pc):
            res["param_comparison"] = "PARAMS_MATCH (generator == param_count)"
        elif n_e == int(pc):
            res["param_comparison"] = (
                "MATCH_VIA_AUXILIARY (param_count includes auxiliary modules; "
                f"generator {gen_elements} + auxiliary {aux_elements} = {n_e})")
        else:
            res["failures"].append(
                f"param_count mismatch: cell param_count={pc}, "
                f"generator_elements={gen_elements}, total={n_e}")
            return res
    res["status"] = "OK"
    return res


def run_validation(payload: dict, root: Path) -> tuple[list[dict], list[str]]:
    """Validate the full payload. Returns (rows, failures)."""
    rows: list[dict] = []
    failures: list[str] = []
    regions = payload.get("regions", {})
    # structural completeness: exactly 5 x 7 x 2
    for reg in REQUIRED_REGIONS:
        if reg not in regions:
            failures.append(f"MISSING REGION: {reg}")
            continue
        fams = regions[reg].get("families", {})
        for fam in REQUIRED_FAMILIES:
            if fam not in fams:
                failures.append(f"MISSING FAMILY: {reg}/{fam}")
                continue
            seeds = fams[fam].get("per_gen_seed", {})
            for seed in REQUIRED_SEEDS:
                if seed not in seeds:
                    failures.append(f"MISSING CELL: {reg}/{fam}/s{seed}")
                    continue
                cell = seeds[seed]
                ckpt = root / cell["checkpoint_path"] \
                    if cell.get("checkpoint_path") else None
                res = check_cell(cell, ckpt, fam) if ckpt else {
                    "status": "FAIL",
                    "failures": ["cell has no checkpoint_path metadata"]}
                if res["status"] != "OK":
                    for f in res["failures"]:
                        failures.append(f"{reg}/{fam}/s{seed}: {f}")
                rows.append({"region": reg, "family": fam, "seed": seed,
                             **res})
    if len(rows) != EXPECTED_CELLS:
        failures.append(f"expected {EXPECTED_CELLS} cells, saw {len(rows)}")
    return rows, failures


def main(argv: list[str]) -> int:
    json_path = Path(argv[1]) if len(argv) > 1 else \
        ROOT / "data/results/wave30_generator_fidelity.json"
    payload = json.loads(json_path.read_text())
    rows, failures = run_validation(payload, ROOT)
    print(f"{'region':26s} {'family':14s} {'seed':4s} {'status':5s} "
          f"{'tensors':>7s} {'total':>9s} {'gen':>9s} {'aux':>7s} params")
    for r in rows:
        print(f"{r['region']:26s} {r['family']:14s} {r['seed']:4s} "
              f"{r['status']:5s} {str(r.get('tensors', '-')):>7s} "
              f"{str(r.get('total_elements', '-')):>9s} "
              f"{str(r.get('generator_elements', '-')):>9s} "
              f"{str(r.get('auxiliary_elements', '-')):>7s} "
              f"{r.get('param_comparison', '')}")
    n_ok = sum(1 for r in rows if r["status"] == "OK")
    print(f"\ncells: {len(rows)}/{EXPECTED_CELLS} present, {n_ok} OK, "
          f"{len(failures)} failures")
    for f in failures:
        print(" FAIL:", f)
    return 0 if not failures and len(rows) == EXPECTED_CELLS else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
