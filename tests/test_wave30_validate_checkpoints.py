"""Lightweight regression tests for wave30_validate_checkpoints.py.

Covers: missing region, missing seed, SHA mismatch, empty checkpoint,
parameter-count mismatch, plus the happy path. Tiny tensors only — safe to
run alongside the CPU-heavy refresh.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import wave30_validate_checkpoints as V  # noqa: E402


def _make_ckpt(path: Path, n_elem: int = 12, empty: bool = False) -> str:
    """Write a nested {module: state_dict} checkpoint; return sha256 hex."""
    if empty:
        sd: dict = {"VelocityTCN": {}}
    else:
        sd = {"VelocityTCN": {"w": torch.zeros(n_elem),
                              "b": torch.zeros(4)}}
    torch.save(sd, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(root: Path, fam: str = "flow_matching", n_elem: int = 12,
          param_count: int | None = 16, empty: bool = False,
          bad_sha: bool = False) -> dict:
    cp = root / f"ckpt_{fam}.pt"
    sha = _make_ckpt(cp, n_elem=n_elem, empty=empty)
    if bad_sha:
        sha = "0" * 64
    cell = {"checkpoint_path": str(cp), "checkpoint_sha256": sha,
            "checkpoint_path_rel": str(cp)}
    if param_count is not None:
        cell["param_count"] = param_count
    return cell


def _payload(root: Path, cell: dict) -> dict:
    fams = {}
    for fam in V.REQUIRED_FAMILIES:
        fams[fam] = {"per_gen_seed": {s: dict(cell) for s in V.REQUIRED_SEEDS}}
    return {"regions": {r: {"families": json.loads(json.dumps(fams))}
                        for r in V.REQUIRED_REGIONS}}


def test_happy_path_all_70_cells_ok(tmp_path):
    cell = _cell(tmp_path, n_elem=12, param_count=16)
    rows, failures = V.run_validation(_payload(tmp_path, cell), tmp_path)
    assert len(rows) == V.EXPECTED_CELLS == 70
    assert failures == []
    assert all(r["status"] == "OK" for r in rows)
    ok = rows[0]
    assert ok["tensors"] == 2 and ok["total_elements"] == 16
    assert ok["generator_elements"] == 16


def test_missing_region_fails(tmp_path):
    cell = _cell(tmp_path)
    payload = _payload(tmp_path, cell)
    del payload["regions"]["thames_north_sea"]
    rows, failures = V.run_validation(payload, tmp_path)
    assert len(rows) == 56
    assert any("MISSING REGION: thames_north_sea" in f for f in failures)
    # cell count discrepancy must also be flagged
    assert any("expected 70 cells, saw 56" in f for f in failures)


def test_missing_seed_fails(tmp_path):
    cell = _cell(tmp_path)
    payload = _payload(tmp_path, cell)
    del payload["regions"]["ny_bight"]["families"]["cvae"]["per_gen_seed"]["99"]
    rows, failures = V.run_validation(payload, tmp_path)
    assert len(rows) == 69
    assert any("MISSING CELL: ny_bight/cvae/s99" in f for f in failures)


def test_sha_mismatch_fails(tmp_path):
    cell = _cell(tmp_path, bad_sha=True)
    rows, failures = V.run_validation(_payload(tmp_path, cell), tmp_path)
    assert all(r["status"] == "FAIL" for r in rows)
    assert any("sha256 mismatch" in f for f in failures)


def test_empty_checkpoint_fails(tmp_path):
    cell = _cell(tmp_path, empty=True, param_count=None)
    rows, failures = V.run_validation(_payload(tmp_path, cell), tmp_path)
    assert all(r["status"] == "FAIL" for r in rows)
    assert any(("zero tensors" in f) or ("empty" in f) for f in failures)


def test_param_count_mismatch_fails(tmp_path):
    # generator has 16 elements, cell claims 999 -> mismatch
    cell = _cell(tmp_path, n_elem=12, param_count=999)
    rows, failures = V.run_validation(_payload(tmp_path, cell), tmp_path)
    assert all(r["status"] == "FAIL" for r in rows)
    assert any("param_count mismatch" in f for f in failures)


def test_auxiliary_module_match_is_reported_not_failed(tmp_path):
    # cell param_count equals generator+auxiliary total -> MATCH_VIA_AUXILIARY
    root = tmp_path
    cp = root / "ckpt_aux.pt"
    sd = {"VelocityTCN": {"w": torch.zeros(16)},
          "SomeCritic": {"c": torch.zeros(8)}}
    torch.save(sd, cp)
    cell = {"checkpoint_path": str(cp),
            "checkpoint_sha256": hashlib.sha256(cp.read_bytes()).hexdigest(),
            "param_count": 24}
    rows, failures = V.run_validation(_payload(root, cell), root)
    assert failures == []
    assert rows[0]["generator_elements"] == 16
    assert rows[0]["auxiliary_elements"] == 8
    assert "MATCH_VIA_AUXILIARY" in rows[0]["param_comparison"]


def test_missing_checkpoint_file_fails(tmp_path):
    cell = _cell(tmp_path)
    (tmp_path / "ckpt_flow_matching.pt").unlink()
    rows, failures = V.run_validation(_payload(tmp_path, cell), tmp_path)
    assert all(r["status"] == "FAIL" for r in rows)
    assert any("checkpoint file missing" in f for f in failures)
