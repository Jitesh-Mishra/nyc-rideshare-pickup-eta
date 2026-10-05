from datetime import datetime

import numpy as np
import polars as pl
from fastapi.testclient import TestClient

from eta.features import FEATURES, OnlineFeatureStore, offline_features, replay_events, stream_queries
from eta.models import HistoricalMedian, LGBQuantile
from eta.serve import Predictor, build_app


def _predictor(trips, meta):
    q = trips.filter(pl.col("label_ok"))
    f = offline_features(trips, q, meta)
    y = q["wait_min"].to_numpy()
    X = f.select(FEATURES).to_numpy().astype(np.float32)
    b0 = HistoricalMedian().fit(f["zone"].to_numpy(), f["how"].to_numpy(), y)
    m = LGBQuantile().fit(X[:15000], y[:15000], X[15000:], y[15000:], rounds=30)
    store = OnlineFeatureStore(meta)
    end = datetime(2025, 3, 10, 8)
    for _ in stream_queries(store, replay_events(trips, datetime(2025, 3, 10, 6), end), {}):
        pass
    return Predictor(store, b0, m, "lgb")


def test_predict_and_fallback(trips, meta):
    p = _predictor(trips, meta)
    with TestClient(build_app(p)) as c:
        r = c.post("/predict", json={"zone": 4}).json()
        assert r["fallback"] is False and 0 <= r["p50_min"] <= r["p90_min"]
        # request 20 minutes past the last event the store saw -> stale -> baseline
        r = c.post("/predict", json={"zone": 4, "request_ts": "2025-03-10T08:20:00"}).json()
        assert r["fallback"] is True and r["fallback_reason"] == "stale_features"
        assert c.post("/predict", json={"zone": 999}).status_code == 422


def test_model_error_falls_back(trips, meta):
    p = _predictor(trips, meta)
    p.model = None  # simulate a broken model
    r = p.predict(4, p.store.clock)
    assert r["fallback"] is True and r["fallback_reason"].startswith("error")


def test_event_ingest_moves_clock(trips, meta):
    p = _predictor(trips, meta)
    with TestClient(build_app(p)) as c:
        before = c.get("/health").json()["store_clock"]
        c.post("/events", json=[{"kind": "request", "zone": 4, "ts": "2025-03-10T08:30:00"}])
        assert c.get("/health").json()["store_clock"] > before
