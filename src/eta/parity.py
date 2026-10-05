"""Offline vs online feature parity: replay a window through the online store and compare."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import polars as pl

from . import config as C
from .features import FEATURES, OnlineFeatureStore, offline_features, replay_events, stream_queries, to_sec


def run_parity(trips: pl.DataFrame, meta, start, end, n_sample=10_000, seed=C.SEED, tol=1e-4) -> dict:
    q = trips.filter(pl.col("label_ok") & (pl.col("request_datetime") >= start) & (pl.col("request_datetime") < end))
    rng = np.random.default_rng(seed)
    q = q[np.sort(rng.choice(q.height, size=min(n_sample, q.height), replace=False))]
    off = offline_features(trips, q, meta)

    queries = defaultdict(list)
    for rid, (z, t, w) in enumerate(zip(q["zone"].to_numpy(), to_sec(q["request_datetime"]), q["wav"].to_numpy())):
        queries[int(t)].append((rid, int(z), int(w)))
    store = OnlineFeatureStore(meta)
    on = {rid: f for rid, f in stream_queries(store, replay_events(trips, start, end), queries)}

    mismatch_rows, max_diff = 0, {}
    for i in range(q.height):
        row_bad = False
        for k in FEATURES:
            a, b = float(off[k][i]), float(on[i][k])
            if np.isnan(a) and np.isnan(b):
                continue
            d = abs(a - b) if not (np.isnan(a) or np.isnan(b)) else np.inf
            max_diff[k] = max(max_diff.get(k, 0.0), d)
            row_bad |= d > tol
        mismatch_rows += row_bad
    return {"n": q.height, "rows_matching": q.height - mismatch_rows,
            "parity_rate": (q.height - mismatch_rows) / q.height,
            "max_abs_diff": {k: float(v) for k, v in max_diff.items()}}


if __name__ == "__main__":
    from .data import load_trips, load_zones
    from .features import ZoneMeta
    trips = load_trips()
    r = run_parity(trips, ZoneMeta.from_zones(load_zones()), C.TEST_START, C.TEST_END)
    (C.REPORTS / "parity.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
