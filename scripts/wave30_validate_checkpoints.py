#!/usr/bin/env python3
"""Strict validation of Wave 3.0 generator checkpoints (v3, hardened).

Owner-mandated hardening (2026-11-27):
  - set-equality: exactly the 5 predefined regions, 7 families, seeds 42/99;
    EXTRA region/family/seed entries are failures; exactly 70 unique cells
  - duplicate JSON keys are detected via strict loading (object_pairs_hook)
  - state_dict tensor ELEMENTS are reported as such — never relabelled as
    "trainable parameters". Trainable parameters are computed by
    instantiating the real architecture and using named_parameters();
    buffers are counted separately via named_buffers()
  - WGAN-GP: generator (g_ prefix) and critic (c_ prefix) are reported
    separately; param_count is compared against G+C (its recorded
    definition), and the generator-only count is always shown
  - per-checkpoint exceptions are caught and reported explicitly

Read-only; never touches running processes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED_REGIONS = ["south_atlantic_charleston", "ny_bight", "us_gulf_coast",
                    "southeast_australia", "thames_north_sea"]
REQUIRED_FAMILIES = ["flow_matching", "tail_aware_fm", "cvae", "timegrad",
                     "ddpm_cond", "wgan_gp", "diffusion_ts"]
REQUIRED_SEEDS = ["42", "99"]
EXPECTED_CELLS = len(REQUIRED_REGIONS) * len(REQUIRED_FAMILIES) \
    * len(REQUIRED_SEEDS)  # 70

CONTEXT, HORIZON = 96, 72


def _arch_flow_matching():
    from tidetail.models.flow_matching import VelocityTCN
    return VelocityTCN(context=CONTEXT, horizon=HORIZON, width=32)


def _arch_cvae():
    from tidetail.models.vae import ConditionalVAE
    return ConditionalVAE(context=CONTEXT, horizon=HORIZON, width=32,
                          latent_dim=32, beta=0.001)


def _arch_timegrad():
    from tidetail.models.timegrad import TimeGradDenoiser
    return TimeGradDenoiser(context=CONTEXT, horizon=HORIZON)


def _arch_ddpm():
    from tidetail.torch_diffusion import TemporalDenoiser
    return TemporalDenoiser(context=CONTEXT, horizon=HORIZON, width=32)


def _arch_wgan():
    from tidetail.models.wgan_gp import ConditionalWGANGP
    return ConditionalWGANGP(context=CONTEXT, horizon=HORIZON, width=32,
                             z_dim=32)


def _arch_dts():
    from tidetail.models.diffusion_ts import ResidualDenoiser
    return ResidualDenoiser(context=CONTEXT, horizon=HORIZON, width=32,
                            use_full_context=True)


# family -> (module name expected in the checkpoint, arch factory,
#            param groups as prefix rules, param_count definition)
ARCH_SPECS = {
    "flow_matching": ("VelocityTCN", _arch_flow_matching,
                      {"generator": ("",)},
                      "all_module_params"),
    "tail_aware_fm": ("VelocityTCN", _arch_flow_matching,
                      {"generator": ("",)},
                      "all_module_params"),
    "cvae": ("ConditionalVAE", _arch_cvae,
             {"encoder": ("encoder.",), "decoder": ("decoder.",),
              "auxiliary": ()},
             "encoder_plus_decoder"),
    "timegrad": ("TimeGradDenoiser", _arch_timegrad,
                 {"generator": ("",)},
                 "all_module_params"),
    "ddpm_cond": ("TemporalDenoiser", _arch_ddpm,
                  {"generator": ("",)},
                  "all_module_params"),
    "wgan_gp": ("ConditionalWGANGP", _arch_wgan,
                {"generator": ("g_",), "critic": ("c_",)},
                "all_module_params"),
    "diffusion_ts": ("ResidualDenoiser", _arch_dts,
                     {"generator": ("",)},
                     "all_module_params"),
}


def load_json_strict(path: Path) -> tuple[dict, list[str]]:
    """Load JSON flagging duplicate keys (json.loads keeps the last one
    silently — that would hide duplicated cells)."""
    dups: list[str] = []

    def hook(pairs):
        seen = Counter(k for k, _ in pairs)
        for k, n in seen.items():
            if n > 1:
                dups.append(k)
        return dict(pairs)

    payload = json.loads(path.read_text(), object_pairs_hook=hook)
    return payload, dups


def _group_of(key: str, groups: dict) -> str:
    for label, prefixes in groups.items():
        for p in prefixes:
            if p and key.startswith(p):
                return label
    # empty-prefix group ("") is the catch-all for whole-module families
    for label, prefixes in groups.items():
        if "" in prefixes:
            return label
    return "auxiliary"


def analyse_module(sub_sd: dict, arch: torch.nn.Module,
                   groups: dict) -> dict:
    """Classify a checkpoint module's entries via the live architecture.

    Returns tensor/param/buffer element counts and per-group trainable
    parameter counts computed with named_parameters()/named_buffers().
    """
    param_keys = {n for n, _ in arch.named_parameters()}
    buffer_keys = {n for n, _ in arch.named_buffers()}
    sd_keys = set(sub_sd.keys())
    missing_from_arch = sorted(sd_keys - param_keys - buffer_keys)
    if missing_from_arch:
        raise KeyError(
            f"state_dict keys not present in live architecture: "
            f"{missing_from_arch[:5]}{'...' if len(missing_from_arch) > 5 else ''}")

    sd_tensor_elements = sum(int(v.numel()) for v in sub_sd.values()
                             if torch.is_tensor(v))
    trainable_by_group: dict[str, int] = {g: 0 for g in groups}
    trainable_by_group["auxiliary"] = trainable_by_group.get("auxiliary", 0)
    buffers_elements = 0
    for name, t in sub_sd.items():
        if not torch.is_tensor(t):
            continue
        if name in buffer_keys:
            buffers_elements += int(t.numel())
        else:  # a parameter entry
            trainable_by_group[_group_of(name, groups)] += int(t.numel())
    total_trainable = sum(v for k, v in trainable_by_group.items()
                          if k != "auxiliary") + \
        trainable_by_group.get("auxiliary", 0)
    return {
        "sd_tensor_elements": sd_tensor_elements,
        "trainable_by_group": trainable_by_group,
        "total_trainable_params": total_trainable,
        "buffer_elements": buffers_elements,
        "n_params": sum(1 for k in sd_keys if k in param_keys),
        "n_buffers": sum(1 for k in sd_keys if k in buffer_keys),
    }


def check_cell(cell: dict, ckpt_path: Path | None, fam: str) -> dict:
    res: dict = {"status": "FAIL", "failures": []}
    if ckpt_path is None or not cell.get("checkpoint_path"):
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
    except Exception as exc:
        res["failures"].append(
            f"torch.load failed: {type(exc).__name__}: {exc}")
        return res
    if not isinstance(sd, dict) or not sd:
        res["failures"].append("checkpoint state_dict empty or not a dict")
        return res

    mod_name, factory, groups, pc_def = ARCH_SPECS[fam]
    unexpected = sorted(set(sd.keys()) - {mod_name})
    if unexpected:
        res["failures"].append(
            f"unexpected checkpoint modules: {unexpected}")
        return res
    sub = sd[mod_name]
    if not isinstance(sub, dict) or not sub:
        res["failures"].append(f"module {mod_name} state_dict empty")
        return res
    try:
        arch = factory()
        info = analyse_module(sub, arch, groups)
    except Exception as exc:
        res["failures"].append(
            f"architecture analysis failed: {type(exc).__name__}: {exc}")
        return res

    res.update(info)
    gen_only = info["trainable_by_group"].get("generator", 0)
    res["generator_trainable_params"] = gen_only
    if pc_def == "encoder_plus_decoder":
        enc = info["trainable_by_group"].get("encoder", 0)
        dec = info["trainable_by_group"].get("decoder", 0)
        expected_pc = enc + dec
        res["param_comparison"] = (
            f"expected encoder+decoder={expected_pc} "
            f"(encoder {enc}, decoder {dec})")
    else:
        expected_pc = info["total_trainable_params"]
        res["param_comparison"] = f"expected all_module_params={expected_pc}"
        if "critic" in info["trainable_by_group"]:
            res["param_comparison"] += (
                f"; generator-only={gen_only}, "
                f"critic={info['trainable_by_group']['critic']}")

    pc = cell.get("param_count")
    if pc is not None and int(pc) > 0:
        if int(pc) == expected_pc:
            res["param_comparison"] += " -> PARAMS_MATCH"
        else:
            res["failures"].append(
                f"param_count mismatch: cell param_count={pc} vs "
                f"{res['param_comparison']}")
            return res
    res["status"] = "OK"
    return res


def run_validation(payload: dict, root: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    failures: list[str] = []
    regions = payload.get("regions", {})

    # set-equality: no extra entries
    extra_regions = sorted(set(regions) - set(REQUIRED_REGIONS))
    if extra_regions:
        failures.append(f"EXTRA REGION(s): {extra_regions}")
    for reg in REQUIRED_REGIONS:
        if reg not in regions:
            failures.append(f"MISSING REGION: {reg}")
            continue
        fams = regions[reg].get("families", {})
        extra_fams = sorted(set(fams) - set(REQUIRED_FAMILIES))
        if extra_fams:
            failures.append(f"EXTRA FAMILY in {reg}: {extra_fams}")
        for fam in REQUIRED_FAMILIES:
            if fam not in fams:
                failures.append(f"MISSING FAMILY: {reg}/{fam}")
                continue
            seeds = fams[fam].get("per_gen_seed", {})
            extra_seeds = sorted(set(seeds) - set(REQUIRED_SEEDS))
            if extra_seeds:
                failures.append(f"EXTRA SEED in {reg}/{fam}: {extra_seeds}")
            for seed in REQUIRED_SEEDS:
                if seed not in seeds:
                    failures.append(f"MISSING CELL: {reg}/{fam}/s{seed}")
                    continue
                cell = seeds[seed]
                ckpt = root / cell["checkpoint_path"] \
                    if cell.get("checkpoint_path") else None
                res = check_cell(cell, ckpt, fam)
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
    payload, dups = load_json_strict(json_path)
    if dups:
        print(f"FAIL: duplicate JSON keys detected: {sorted(set(dups))}")
        return 1
    rows, failures = run_validation(payload, ROOT)
    print(f"{'region':26s} {'family':14s} {'seed':4s} {'st':4s} "
          f"{'sd_elem':>9s} {'gen_par':>9s} {'aux_par':>9s} {'buf':>7s} "
          f"param_check")
    for r in rows:
        groups = r.get("trainable_by_group", {})
        aux = sum(v for k, v in groups.items()
                  if k not in ("generator", "encoder", "decoder"))
        print(f"{r['region']:26s} {r['family']:14s} {r['seed']:4s} "
              f"{r['status'][:4]:4s} {str(r.get('sd_tensor_elements', '-')):>9s} "
              f"{str(r.get('generator_trainable_params', '-')):>9s} "
              f"{str(aux):>9s} {str(r.get('buffer_elements', '-')):>7s} "
              f"{r.get('param_comparison', '')}")
    n_ok = sum(1 for r in rows if r["status"] == "OK")
    print(f"\ncells: {len(rows)}/{EXPECTED_CELLS} present, {n_ok} OK, "
          f"{len(failures)} failures")
    for f in failures:
        print(" FAIL:", f)
    ok = not failures and len(rows) == EXPECTED_CELLS and not dups
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
