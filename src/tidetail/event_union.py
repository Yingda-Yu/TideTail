"""Pure set-arithmetic helpers for POT cluster / tropical-cyclone event unions.

Wave 2.5 reported POT=634, TC=73, overlap=73, total=707. That identity is
impossible: |POT ∪ TC| = |POT| + |TC| - |POT ∩ TC| = 634 + 73 - 73 = 634.
The error came from adding a TC count that had never been de-duplicated.

This module makes the arithmetic explicit and one-to-one:

* ``POT``     - declustered q95 peak-over-threshold clusters.
* ``TC``      - canonical tropical-cyclone tracks matched to a region.
* ``matched`` - one-to-one TC <-> POT pairs that describe the SAME physical
  event. A TC and a POT cluster may overlap in time, but a single POT cluster
  cannot absorb two distinct TC tracks.
* ``union``   - ``|POT| + |TC| - |matched|``.

Only temporal information is used. A TC has a single date in the manifest
(no track start/end), so matching uses ``start - tol <= date <= end + tol``.
This is deliberately conservative: a TC is never assumed to be a local surge
event merely because it exists in the basin.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from dateutil import parser as date_parser


def parse_ts(value: str) -> datetime:
    """Parse a manifest timestamp and force UTC tz-awareness."""
    dt = date_parser.parse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def classify_event_id(event_id: int) -> str:
    """Classify a wave20 ``event_ids`` entry.

    Encoding convention (``wave20_pot_storm_bootstrap.assign_pot_units``):
      * ``>= 0``      canonical TC track (TRACK_VALIDATED_STORM)
      * ``< 0, != -999`` POT q95 cluster, encoded ``-(cluster_id + 1)``
      * ``== -999``   background / non-extreme fallback
    """
    if event_id == -999:
        return "background"
    if event_id >= 0:
        return "tc"
    return "pot"


@dataclass(frozen=True)
class Cluster:
    cluster_id: int
    start: datetime
    end: datetime
    label: str  # unique across years: cluster_id restarts each year


def _clusters_from_manifest(raw: list[dict]) -> list[Cluster]:
    return [
        Cluster(
            cluster_id=int(c["cluster_id"]),
            start=parse_ts(c["start"]),
            end=parse_ts(c["end"]),
            label=f"{i}:{int(c['cluster_id'])}",
        )
        for i, c in enumerate(raw)
    ]


def match_one_to_one(
    tc_dates: list[str],
    pot_raw: list[dict],
    tolerance_hours: float = 0.0,
) -> dict:
    """Greedy one-to-one temporal matching of TC dates to POT clusters.

    Returns explicit sets, not just counts:
      * ``matched``: list of ``(tc_index, cluster_label)`` pairs
      * ``pot_only``: cluster labels with no TC assigned
      * ``tc_only``: tc indices with no cluster assigned

    Greedy order is deterministic: TCs sorted by date, clusters sorted by
    start. A cluster may absorb at most one TC; an unmatched TC stays in
    ``tc_only`` even if it overlaps a cluster that already absorbed another
    TC. This prevents one long cluster from silently deleting many storms.
    """
    tol = timedelta(hours=tolerance_hours)
    clusters = _clusters_from_manifest(pot_raw)
    tc_parsed = [(i, parse_ts(d)) for i, d in enumerate(tc_dates)]
    tc_parsed.sort(key=lambda x: (x[1], x[0]))

    used_clusters: set[str] = set()
    matched: list[tuple[int, str]] = []
    tc_only: list[int] = []

    for tc_index, tc_dt in tc_parsed:
        best = None
        for c in clusters:
            if c.label in used_clusters:
                continue
            if c.start - tol <= tc_dt <= c.end + tol:
                # nearest cluster start wins on ties; deterministic
                key = abs((tc_dt - c.start).total_seconds())
                if best is None or key < best[0]:
                    best = (key, c.label)
        if best is None:
            tc_only.append(tc_index)
        else:
            used_clusters.add(best[1])
            matched.append((tc_index, best[1]))

    pot_only = [c.label for c in clusters if c.label not in used_clusters]
    return {
        "matched": matched,
        "pot_only": pot_only,
        "tc_only": tc_only,
        "pot_total": len(clusters),
        "tc_total": len(tc_dates),
    }


def union_cardinality(result: dict) -> dict:
    """Derive and verify the set identity for a matching result."""
    pot_total = result["pot_total"]
    tc_total = result["tc_total"]
    matched = len(result["matched"])
    pot_only = len(result["pot_only"])
    tc_only = len(result["tc_only"])
    union = pot_total + tc_total - matched

    # Hard identities. If any fails the audit is invalid, so raise.
    assert matched + tc_only == tc_total, "matched + tc_only must equal TC total"
    assert matched + pot_only == pot_total, "matched + pot_only must equal POT total"
    assert union == pot_total + tc_only, "union must equal POT + TC-only"
    assert union == pot_only + tc_total, "union must equal POT-only + TC"

    return {
        "pot_total": pot_total,
        "tc_total": tc_total,
        "matched_overlap": matched,
        "pot_only": pot_only,
        "tc_only": tc_only,
        "union": union,
    }
