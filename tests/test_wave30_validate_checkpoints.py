"""Lightweight regression tests for wave30_validate_checkpoints.py (v3).

Covers: missing region/seed, SHA mismatch, empty checkpoint, param-count
mismatch, EXTRA region/family/seed, duplicate JSON keys, buffer-vs-parameter
distinction, WGAN-GP generator/critic split, happy path. Tiny synthetic
modules only — safe to run alongside the CPU-heavy refresh.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import wave30_validate_checkpoints as V  # noqa: E402


def _tiny_arch(group_prefixes: dict[str, tuple], with_buffer: bool = False):
    """Module whose named_parameters() start with the given prefixes."""
    mod = torch.nn.Module()
    for label, prefixes in group_prefixes.items():
        for i, p in enumerate(prefixes or ("",)):
            name = (p.rstrip(".") if p else f"{label}{i}")
            blk = torch.nn.Module()
            blk.w = torch.nn.Parameter(torch.zeros(12))
            blk.b = torch.nn.Parameter(torch.zeros(4))
            mod.add_module(name, blk)
    if with_buffer:
        mod.register_buffer("running", torch.zeros(8))
    return mod


TINY_SPECS = {
    "flow_matching": ("TinyFM", lambda: None, {"generator": ("",)},
                      "all_module_params"),
    "tail_aware_fm": ("TinyTAFM", lambda: None, {"generator": ("",)},
                      "all_module_params"),
    "cvae": ("TinyCVAE", lambda: None,
             {"encoder": ("encoder.",), "decoder": ("decoder.",)},
             "encoder_plus_decoder"),
    "timegrad": ("TinyTG", lambda: None, {"generator": ("",)},
                 "all_module_params"),
    "ddpm_cond": ("TinyDDPM", lambda: None, {"generator": ("",)},
                  "all_module_params"),
    "wgan_gp": ("TinyWGAN", lambda: None,
                {"generator": ("g_",), "critic": ("c_",)},
                "all_module_params"),
    "diffusion_ts": ("TinyDTS", lambda: None, {"generator": ("",)},
                     "all_module_params"),
}


@pytest.fixture(autouse=True)
def tiny_specs(monkeypatch):
    """Point ALL families at tiny architectures mirroring real groups."""
    for fam, (mod, _f, groups, pc_def) in TINY_SPECS.items():
        monkeypatch.setitem(V.ARCH_SPECS, fam,
                            (mod, lambda g=groups: _tiny_arch(g, with_buffer=True), groups,
                             pc_def))


def _family_param_count(fam: str, with_buffer: bool = False) -> int:
    groups = TINY_SPECS[fam][2]
    arch = _tiny_arch(groups, with_buffer=with_buffer)
    if TINY_SPECS[fam][2] == "encoder_plus_decoder":
        return sum(p.numel() for n, p in arch.named_parameters()
                   if n.startswith(("encoder.", "decoder.")))
    return sum(p.numel() for _, p in arch.named_parameters())


def _write_ckpt(path: Path, fam: str, with_buffer: bool = False,
                empty: bool = False, wrong_module: str | None = None) -> str:
    arch = _tiny_arch(TINY_SPECS[fam][2], with_buffer=with_buffer)
    mod = wrong_module or TINY_SPECS[fam][0]
    sd = arch.state_dict() if not empty else {}
    torch.save({mod: sd}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(root: Path, buffer_fams: set | None = None,
             empty: bool = False, bad_sha: bool = False,
             param_override: int | None = None,
             wrong_module: str | None = None) -> dict:
    buffer_fams = buffer_fams or set()  # accepted for API compat; unused
    fams = {}
    for fam in V.REQUIRED_FAMILIES:
        cp = root / f"ckpt_{fam}.pt"
        sha = _write_ckpt(cp, fam, with_buffer=True, empty=empty,
                          wrong_module=wrong_module)
        if bad_sha:
            sha = "0" * 64
        pc = param_override if param_override is not None \
            else _family_param_count(fam, with_buffer=True)
        cell = {"checkpoint_path": str(cp), "checkpoint_sha256": sha,
                "param_count": pc}
        fams[fam] = {"per_gen_seed": {s: dict(cell)
                                      for s in V.REQUIRED_SEEDS}}
    return {"regions": {r: {"families": json.loads(json.dumps(fams))}
                        for r in V.REQUIRED_REGIONS}}


def test_happy_path_all_70_cells_ok(tmp_path):
    rows, failures = V.run_validation(_payload(tmp_path), tmp_path)
    assert len(rows) == V.EXPECTED_CELLS == 70
    assert failures == []
    assert all(r["status"] == "OK" for r in rows)
    fm = rows[0]
    assert fm["total_trainable_params"] == 16
    assert fm["sd_tensor_elements"] == 24  # params + buffer in raw state_dict
    assert fm["buffer_elements"] == 8      # buffer excluded from trainable


def test_missing_region_fails(tmp_path):
    payload = _payload(tmp_path)
    del payload["regions"]["thames_north_sea"]
    rows, failures = V.run_validation(payload, tmp_path)
    assert len(rows) == 56
    assert any("MISSING REGION: thames_north_sea" in f for f in failures)
    assert any("expected 70 cells, saw 56" in f for f in failures)


def test_missing_seed_fails(tmp_path):
    payload = _payload(tmp_path)
    del payload["regions"]["ny_bight"]["families"]["cvae"]["per_gen_seed"]["99"]
    rows, failures = V.run_validation(payload, tmp_path)
    assert len(rows) == 69
    assert any("MISSING CELL: ny_bight/cvae/s99" in f for f in failures)


def test_extra_region_fails(tmp_path):
    payload = _payload(tmp_path)
    payload["regions"]["bogus_region"] = \
        json.loads(json.dumps(payload["regions"]["ny_bight"]))
    rows, failures = V.run_validation(payload, tmp_path)
    assert any("EXTRA REGION" in f and "bogus_region" in f for f in failures)
    assert len(rows) == 70  # extra region's cells are NOT validated/counted


def test_extra_family_fails(tmp_path):
    payload = _payload(tmp_path)
    extra_cell = dict(payload["regions"]["ny_bight"]["families"]
                      ["flow_matching"]["per_gen_seed"]["42"])
    payload["regions"]["ny_bight"]["families"]["bogus_family"] = \
        {"per_gen_seed": {"42": extra_cell}}
    _, failures = V.run_validation(payload, tmp_path)
    assert any("EXTRA FAMILY in ny_bight" in f
               and "bogus_family" in f for f in failures)


def test_extra_seed_fails(tmp_path):
    payload = _payload(tmp_path)
    extra_cell = dict(payload["regions"]["ny_bight"]["families"]
                      ["cvae"]["per_gen_seed"]["42"])
    payload["regions"]["ny_bight"]["families"]["cvae"]["per_gen_seed"]["7"] = \
        extra_cell
    _, failures = V.run_validation(payload, tmp_path)
    assert any("EXTRA SEED in ny_bight/cvae" in f and "'7'" in f
               for f in failures)


def test_sha_mismatch_fails(tmp_path):
    rows, failures = V.run_validation(_payload(tmp_path, bad_sha=True),
                                      tmp_path)
    assert all(r["status"] == "FAIL" for r in rows)
    assert any("sha256 mismatch" in f for f in failures)


def test_empty_checkpoint_fails(tmp_path):
    rows, failures = V.run_validation(_payload(tmp_path, empty=True,
                                               param_override=None),
                                      tmp_path)
    assert all(r["status"] == "FAIL" for r in rows)
    assert any(("empty" in f) or ("zero" in f) for f in failures)


def test_param_count_mismatch_fails(tmp_path):
    rows, failures = V.run_validation(_payload(tmp_path, param_override=999),
                                      tmp_path)
    assert all(r["status"] == "FAIL" for r in rows)
    assert any("param_count mismatch" in f for f in failures)


def test_missing_checkpoint_file_fails(tmp_path):
    payload = _payload(tmp_path)  # create files once
    (tmp_path / "ckpt_flow_matching.pt").unlink()
    rows, failures = V.run_validation(payload, tmp_path)
    fm_rows = [r for r in rows if r["family"] == "flow_matching"]
    assert all(r["status"] == "FAIL" for r in fm_rows)
    assert sum(1 for r in rows if r["status"] == "OK") == 60  # 70-10 fm cells
    assert any("checkpoint file missing" in f for f in failures)


def test_duplicate_json_keys_detected(tmp_path):
    block = json.dumps(_payload(tmp_path))
    text = block[:-1] + ',"regions":{}}'  # duplicate 'regions' key
    p = tmp_path / "dup.json"
    p.write_text(text)
    _, dups = V.load_json_strict(p)
    assert "regions" in dups


def test_buffer_counted_separately_from_trainable(tmp_path):
    rows, failures = V.run_validation(
        _payload(tmp_path), tmp_path)
    assert failures == []
    fm = rows[0]
    assert fm["total_trainable_params"] == 16      # params only
    assert fm["buffer_elements"] == 8              # buffer NOT a parameter
    assert fm["sd_tensor_elements"] == 24          # raw state_dict elements
    assert fm["n_params"] == 2 and fm["n_buffers"] == 1


def test_unexpected_module_fails(tmp_path):
    rows, failures = V.run_validation(
        _payload(tmp_path, wrong_module="NotTheRightModule"), tmp_path)
    assert all(r["status"] == "FAIL" for r in rows)
    assert any("unexpected checkpoint modules" in f for f in failures)


def test_wgan_generator_critic_split(tmp_path):
    """Generator (g_) and critic (c_) are reported separately; param_count
    is compared against G+C; generator-only count is always shown."""
    fam = "wgan_gp"
    cp = tmp_path / "ckpt_wgan_gp.pt"
    sha = _write_ckpt(cp, fam)
    pc = _family_param_count(fam)
    cell = {"checkpoint_path": str(cp), "checkpoint_sha256": sha,
            "param_count": pc}
    res = V.check_cell(cell, cp, fam)
    assert res["status"] == "OK"
    g = sum(p.numel() for n, p in
            _tiny_arch(TINY_SPECS[fam][2]).named_parameters()
            if n.startswith("g_"))
    c = sum(p.numel() for n, p in
            _tiny_arch(TINY_SPECS[fam][2]).named_parameters()
            if n.startswith("c_"))
    assert res["trainable_by_group"]["generator"] == g
    assert res["trainable_by_group"]["critic"] == c
    assert "critic" in res["param_comparison"]
    assert res["generator_trainable_params"] != res["total_trainable_params"]
