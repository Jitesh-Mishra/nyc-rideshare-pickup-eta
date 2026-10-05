"""FastAPI pickup-ETA service over the in-memory online feature store.

uvicorn eta.serve:app --port 8000
Env: ETA_MODEL=lgb|torch (default lgb), ETA_REPLAY_UNTIL=2025-03-27T18:00 (warm the store
from the staged data up to that instant, for demos and the load test).
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import config as C
from .features import FEATURES, OnlineFeatureStore, ZoneMeta
from .models import HistoricalMedian, LGBQuantile, TorchQuantile, torch_inputs


class Predictor:
    """Online features -> model -> P50/P90, with fallback to the B0 baseline."""

    def __init__(self, store: OnlineFeatureStore, b0: HistoricalMedian, model, kind="lgb"):
        self.store, self.b0, self.model, self.kind = store, b0, model, kind

    def predict(self, zone: int, t: int, wav: int = 0) -> dict:
        stale = self.store.staleness_min(t)
        how = int((((t // 86400) + 3) % 7) * 24 + (t // 3600) % 24)
        b50, b90 = (float(v[0]) for v in self.b0.predict([zone], [how]))
        base = {"p50_min": b50, "p90_min": b90, "model": "B0_zone_how_median", "fallback": True,
                "feature_staleness_min": stale}
        if stale > C.STALE_AFTER_MIN:
            return {**base, "fallback_reason": "stale_features"}
        try:
            f = self.store.features(zone, t, wav)
            if self.kind == "lgb":
                x = np.array([[f[k] for k in FEATURES]], dtype=np.float32)
                p50, p90 = (float(v[0]) for v in self.model.predict(x))
            else:
                num, cat = torch_inputs({k: [v] for k, v in f.items()}, np.array([b50], np.float32))
                p50, p90 = (float(v[0]) for v in self.model.predict(num, cat))
            return {"p50_min": max(p50, 0.0), "p90_min": max(p90, p50, 0.0), "model": self.kind,
                    "fallback": False, "feature_staleness_min": stale}
        except Exception as e:  # any model/feature error -> baseline, never a 500
            return {**base, "fallback_reason": f"error:{type(e).__name__}"}


class PredictIn(BaseModel):
    zone: int = Field(ge=1, le=265)
    request_ts: datetime | None = None   # defaults to the store's event-time clock
    wav: int = 0


class Event(BaseModel):
    kind: str            # request | dropoff | arrival
    zone: int
    ts: datetime
    wait_min: float = 0.0


def _sec(dt: datetime) -> int:
    return int((dt - datetime(1970, 1, 1)).total_seconds())


def build_app(predictor: Predictor | None = None) -> FastAPI:
    app = FastAPI(title="NYC rideshare pickup-ETA")
    state = {}

    @app.on_event("startup")
    def _load():
        if predictor is not None:
            state["p"] = predictor
            return
        from .data import load_trips, load_zones
        meta = ZoneMeta.from_zones(load_zones())
        kind = os.getenv("ETA_MODEL", "lgb")
        model = (LGBQuantile.load(str(C.MODEL_DIR / "lgb")) if kind == "lgb"
                 else TorchQuantile.load(C.MODEL_DIR / "torch_eta.pt"))
        store = OnlineFeatureStore(meta)
        until = os.getenv("ETA_REPLAY_UNTIL")
        if until:
            from .features import replay_events, stream_queries
            end = datetime.fromisoformat(until)
            from datetime import timedelta
            ev = replay_events(load_trips(), end - timedelta(hours=2), end)
            for _ in stream_queries(store, ev, {}):
                pass
        state["p"] = Predictor(store, HistoricalMedian.load(C.MODEL_DIR / "b0.npz"), model, kind)

    @app.get("/health")
    def health():
        s = state["p"].store
        return {"ok": True, "store_clock": s.clock, "model": state["p"].kind}

    @app.post("/events")
    def events(evs: list[Event]):
        s = state["p"].store
        for e in evs:
            t = _sec(e.ts)
            {"request": lambda: s.add_request(e.zone, t), "dropoff": lambda: s.add_dropoff(e.zone, t),
             "arrival": lambda: s.add_arrival(e.zone, t, e.wait_min)}[e.kind]()
        return {"ingested": len(evs), "store_clock": s.clock}

    @app.post("/predict")
    def predict(req: PredictIn):
        t0 = time.perf_counter()
        p = state["p"]
        t = _sec(req.request_ts) if req.request_ts else p.store.clock
        out = p.predict(req.zone, t, req.wav)
        out["server_ms"] = (time.perf_counter() - t0) * 1000
        return out

    return app


app = build_app()
