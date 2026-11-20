"""Read-only CPU audit of the migrated Prime copy; writes aggregate evidence only."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def numbers(records, key):
    return np.array([float(r[key]) if r.get(key) else np.nan for r in records])


def audit(root):
    # Never follow manifest paths into the original collaborator workspace.
    root = root.resolve(strict=True)
    authorized = Path('/data/yingda/tidetail-data').resolve(strict=True)
    if not root.is_relative_to(authorized):
        raise ValueError('Only the migrated owner data workspace is authorized')
    paths = sorted(root.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('Symlinks are not permitted in this audit')
    inventory = {str(p.relative_to(root)): {'bytes': p.stat().st_size,
                                          'sha256': digest(p)}
                 for p in paths if p.is_file()}
    declared = json.loads((root / 'file_inventory_sha256.json').read_text())['files']
    checks = []
    for item in declared:
        actual = inventory.get(item['path'])
        checks.append({'path': item['path'], 'present': actual is not None,
                       'matches': actual is not None and all(
                           actual[k] == item[k] for k in ('bytes', 'sha256'))})
    out = {'environment': {'python': platform.python_version(),
                           **{p: importlib.metadata.version(p) for p in ('numpy', 'pytest')}},
           'audit_base_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'data_root': str(root), 'file_count': len(inventory),
           'bytes': sum(p['bytes'] for p in inventory.values()),
           'migration_manifest_present': (root / 'MIGRATION_MANIFEST.json').is_file(),
           'inventory': inventory, 'declared_checksum_checks': checks, 'regions': []}
    for base in sorted((root / 'derived/prime_coast').iterdir()):
        if not base.is_dir():
            continue
        target = rows(base / 'target_timeline_v1.csv')
        forcing = rows(base / 'forcing_timeline_v1.csv')
        tide = rows(base / 'tide_timeline_v1.csv')
        events = rows(base / 'event_table_v1.csv')
        meta = json.loads((base / 'train_only_metadata_v1.json').read_text())
        manifest = json.loads((base / 'region_manifest_v1.json').read_text())
        ts = np.array([r['timestamp_utc'].removesuffix('Z') for r in target],
                      dtype='datetime64[s]')
        delta = np.diff(ts).astype('timedelta64[s]').astype(int)
        obs = numbers(target, 'observed_total_water_level_m')
        missing = np.array([int(r['is_missing']) for r in target], dtype=bool)
        a, b = int(meta['train_end_index']), int(meta['validation_end_index'])
        clean = np.where(missing, np.nan, obs)
        thresholds = {str(q): {'computed': float(np.nanquantile(obs[:a], q)),
                               'declared': meta['thresholds'][f'total_water_q{int(q*100)}_m']}
                      for q in (.95, .99)}
        scalers = {}
        for key, stored in meta['scalers'].items():
            source = target if key in target[0] else tide if key in tide[0] else forcing
            vals = numbers(source, key)[:a]
            finite = vals[np.isfinite(vals)]
            scalers[key] = {'computed_mean': float(np.mean(finite)),
                            'computed_std_ddof0': float(np.std(finite)),
                            'computed_count': len(finite), 'declared': stored}
        event_sets = {s: {e['event_id'] for e in events if e['split'] == s}
                      for s in ('train', 'validation', 'test')}
        crossing = [e['event_id'] for e in events
                    if any(int(e['start_index']) < cut <= int(e['end_index']) for cut in (a, b))]
        # Reproduce PR17's 72/24 rolling test windows. These are dependent windows,
        # not independent storms, and do not certify the migrated 96/72 contract.
        segment = clean[b:]
        starts = np.array([s for s in range(len(segment) - 95)
                           if np.isfinite(segment[s:s + 96]).all()], dtype=int)
        y = segment[starts[:, None] + np.arange(72, 96)]
        pred_p = np.repeat(segment[starts + 71, None], 24, axis=1)
        pred_s = segment[starts[:, None] + np.arange(48, 72)]
        q95 = thresholds['0.95']['declared']
        metrics = {}
        for name, prediction in [('persistence', pred_p), ('seasonal24', pred_s)]:
            error = prediction - y
            mask = y >= q95
            metrics[name] = {'mae_m': float(np.mean(np.abs(error))),
                             'rmse_m': float(np.sqrt(np.mean(error ** 2))),
                             'q95_tail_mae_m': float(np.mean(np.abs(error[mask]))),
                             'tail_cells': int(mask.sum())}
        with np.load(base / 'window_index_v1.npz', allow_pickle=False) as idx:
            window_summary = {k: {'shape': list(idx[k].shape), 'dtype': str(idx[k].dtype),
                                  'first': idx[k].reshape(-1)[:3].tolist(),
                                  'last': idx[k].reshape(-1)[-3:].tolist()}
                              for k in idx.files}
        out['regions'].append({
            'region': base.name, 'rows': len(target), 'first': str(ts[0]), 'last': str(ts[-1]),
            'anchor_ids': sorted({r['anchor_id'] for r in target}),
            'source_names': sorted({r['source_name'] for r in target}),
            'qc_flags': dict(Counter(r['qc_flag'] for r in target)),
            'missing_flags': int(missing.sum()), 'nonfinite_observed': int((~np.isfinite(obs)).sum()),
            'duplicate_timestamps': int((delta == 0).sum()),
            'nonhourly_steps': int((delta != 3600).sum()),
            'tide_aligned': [r['timestamp_utc'] for r in tide] == [r['timestamp_utc'] for r in target],
            'forcing_aligned': [r['timestamp_utc'] for r in forcing] == [r['timestamp_utc'] for r in target],
            'forcing_nonfinite': {k: int((~np.isfinite(numbers(forcing, k))).sum())
                                  for k in forcing[0] if k != 'timestamp_utc'},
            'thresholds': thresholds, 'scalers': scalers,
            'split_indices_half_open': [0, a, b, len(target)],
            'events_per_split': {s: len(ids) for s, ids in event_sets.items()},
            'crossing_cut_events': crossing,
            'shared_event_ids': sorted((event_sets['train'] & event_sets['validation']) |
                                       (event_sets['train'] & event_sets['test']) |
                                       (event_sets['validation'] & event_sets['test'])),
            'event_durations_hours': dict(Counter(
                int(e['end_index']) - int(e['start_index']) + 1 for e in events)),
            'manifest': manifest, 'window_index_summary': window_summary,
            'pr17_reproduction': {'input_hours': 72, 'horizon_hours': 24,
                                  'stride_hours': 1, 'windows': len(starts),
                                  'forecast_cells': int(y.size),
                                  'unique_target_hours': int(np.unique(
                                      starts[:, None] + np.arange(72, 96)).size),
                                  'independent_storm_count': None, 'metrics': metrics}})
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"Audited {result['file_count']} files and {len(result['regions'])} regions")
