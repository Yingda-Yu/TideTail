"""Regression tests for POT/TC set-union arithmetic (Wave 2.7).

Locks in the identity ``|POT ∪ TC| = |POT| + |TC| - |POT ∩ TC|`` and prevents a
recurrence of the Wave 2.5 claim ``634 + 73 - 73 = 707``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tidetail.event_union import classify_event_id, match_one_to_one, union_cardinality

ROOT = Path(__file__).resolve().parent.parent
DECLUSTERING = ROOT / "data" / "manifests" / "multiyear_declustering.json"


def _cluster(cid, start, end):
    return {"cluster_id": cid, "start": start, "end": end}


def test_union_identity_simple_disjoint():
    pot = [_cluster(1, "2024-01-01T00:00:00+00:00", "2024-01-03T00:00:00+00:00")]
    tc = ["2024-06-01"]
    result = union_cardinality(match_one_to_one(tc, pot))
    assert result == {
        "pot_total": 1,
        "tc_total": 1,
        "matched_overlap": 0,
        "pot_only": 1,
        "tc_only": 1,
        "union": 2,
    }


def test_union_identity_full_overlap():
    pot = [_cluster(1, "2024-01-01T00:00:00+00:00", "2024-01-03T00:00:00+00:00")]
    tc = ["2024-01-02"]
    result = union_cardinality(match_one_to_one(tc, pot))
    assert result["matched_overlap"] == 1
    assert result["union"] == 1  # 1 + 1 - 1
    assert result["union"] == result["pot_total"] + result["tc_total"] - result["matched_overlap"]


def test_one_to_one_prevents_double_absorption():
    """Two distinct TCs inside one long cluster must not both collapse into it."""
    pot = [_cluster(1, "2024-01-01T00:00:00+00:00", "2024-01-10T00:00:00+00:00")]
    tc = ["2024-01-03", "2024-01-07"]
    result = union_cardinality(match_one_to_one(tc, pot))
    assert result["matched_overlap"] == 1
    assert result["tc_only"] == 1
    assert result["union"] == 2  # cluster + one unmatched TC


def test_tolerance_only_adds_matches():
    pot = [_cluster(1, "2024-01-01T00:00:00+00:00", "2024-01-02T00:00:00+00:00")]
    tc = ["2024-01-03"]
    strict = union_cardinality(match_one_to_one(tc, pot, tolerance_hours=0.0))
    widened = union_cardinality(match_one_to_one(tc, pot, tolerance_hours=48.0))
    assert strict["matched_overlap"] == 0
    assert widened["matched_overlap"] == 1
    assert widened["union"] < strict["union"]


@pytest.mark.parametrize(
    "event_id,kind",
    [(-999, "background"), (0, "tc"), (5, "tc"), (-1, "pot"), (-16, "pot")],
)
def test_classify_event_id(event_id, kind):
    assert classify_event_id(event_id) == kind


@pytest.mark.skipif(not DECLUSTERING.exists(), reason="declustering manifest absent")
def test_real_manifest_union_is_not_707():
    manifest = json.loads(DECLUSTERING.read_text())
    pot: list[dict] = []
    tc: list[str] = []
    for region in manifest:
        for year in range(2015, 2025):
            pot.extend(manifest[region]["pot_clusters"].get(str(year), []))
            for entry in manifest[region]["tcs"].get(str(year), []):
                tc.append(entry[2])

    result = union_cardinality(match_one_to_one(tc, pot))
    assert result["pot_total"] == 634
    assert result["tc_total"] == 73
    assert result["union"] == 634 + 73 - result["matched_overlap"]
    assert result["union"] != 707  # the Wave 2.5 arithmetic error
    # Explicit set-partition identities
    assert result["matched_overlap"] + result["pot_only"] == result["pot_total"]
    assert result["matched_overlap"] + result["tc_only"] == result["tc_total"]


@pytest.mark.skipif(not DECLUSTERING.exists(), reason="declustering manifest absent")
def test_2024_manifest_matches_bootstrap_extreme_counts():
    """2024 union must equal the bootstrap extreme units, except Thames."""
    import numpy as np

    manifest = json.loads(DECLUSTERING.read_text())
    preds = ROOT / "data" / "predictions" / "v2"
    expected = {
        "ny_bight": 15,
        "us_gulf_coast": 15,
        "south_atlantic_charleston": 14,
        "southeast_australia": 6,
        "thames_north_sea": 9,  # one cluster has no 2024 test windows
    }
    for region, exp in expected.items():
        pot = manifest[region]["pot_clusters"].get("2024", [])
        tc = [e[2] for e in manifest[region]["tcs"].get("2024", [])]
        union = union_cardinality(match_one_to_one(tc, pot))["union"]

        files = sorted(preds.glob(f"{region}_A0_real_only_gru_seed*.npz"))
        if not files:
            pytest.skip(f"no bootstrap prediction file for {region}")
        with np.load(files[0], allow_pickle=True) as d:
            ids = np.unique(d["event_ids"].astype(np.int64))
        bootstrap_extremes = int(np.sum(ids != -999))

        if region == "thames_north_sea":
            assert union >= bootstrap_extremes
        else:
            assert union == exp
        assert bootstrap_extremes == exp
