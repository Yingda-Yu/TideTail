"""Small deterministic probes of remaining PR17 integrity limitations."""
import json

import numpy as np

from tidetail.evaluate import threshold_metrics
from tidetail.splits import EventGroup, embargo_filter, group_by_boolean_flags

metrics = threshold_metrics(np.array([np.nan]), np.array([2.]), 1.)
result = {
    'missing_observation_threshold_metrics': {
        k: v if np.isfinite(v) else None for k, v in metrics.items()
    },
    'embargo_accepts_interval_containing_protected_row': bool(
        embargo_filter(np.array([0]), 1)(EventGroup('x', 0, 100))
    ),
    'transitive_run_merge': group_by_boolean_flags(
        np.array([1, 0, 1, 0, 1], dtype=bool), gap_tolerance=1
    ),
}
print(json.dumps(result, indent=2, allow_nan=False))
