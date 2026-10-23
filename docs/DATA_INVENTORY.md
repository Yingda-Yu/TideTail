# TideTail Data Inventory — v0

> Dataset rows added here once each source is inspected, licensed, and
> checksummed.  GESLA/ERA5/NOAA are expected; license TBD.

| # | Source | Access | License | Coverage | Status | Action Taken |
|---|--------|--------|---------|----------|--------|-------------|
| D001 | GESLA-3 | pending download | TBD | global tide gauges | 🔴 PENDING | — |
| D002 | NOAA tidesandcurrents | pending | TBD | US gauges | 🔴 PENDING | — |
| D003 | ERA5 | pending | TBD | atmospheric forcing | 🔴 PENDING | — |
| D004 | Rongjia coastal research | 🔒 ACL group::--- denied | permission pending | TBD | 🔒 BLOCKED | Owner: run `chmod g+rx /data/rongjia` or `setfacl -m g:rongjia:rx` as Rongjia |

## Provenance fields required for every ingested dataset

- `license` text + redistribution rights
- `download_manifest.json` — URL, ISO-8601 download time, SHA-256
- `gauges.json` — station_id, lat/lon, datum, units, sampling_period, time coverage
- `split_manifest.json` — event_group ids + start/end + train/val/test + embargo

## Integrity gates (from docs/RESEARCH_PROTOCOL.md)

1. Dataset downloaded, licensed, checksummed BEFORE QC pipeline runs.
2. Station metadata audited — correct gauge datum and units.
3. Timestamp aligned to UTC, monotonic, no duplicates.
4. QC pipeline flags gaps/outliers/impossible values — flagged rows never contribute to threshold estimation.
5. Train-only q95/q99 thresholds computed AFTER event-grouped splits.
6. Synthetic data never enters val or test partitions.
