"""Point-in-time features, one module for training (offline) and serving (online).

Every window aggregate is computed over events with timestamp in [t - w, t), where t is
the request time. Strictly-before means a request never sees itself, its own label, or
anything that happened at or after t.

Event streams per pickup zone:
  request  - a ride request in the zone, known at request_datetime        (demand)
  dropoff  - a trip ending in the zone, known at dropoff_datetime           (supply PROXY:
             a driver just freed up there; TLC has no driver GPS)
  arrival  - a driver reaching a rider in the zone, known at on_scene_datetime,
             carries that rider's observed wait                             (recent ETA)
  open     - requests made before t whose driver has not arrived by t       (queue)
             = requests so far minus arrivals so far, per zone. Known at t: a
             dispatcher always knows which requests are still waiting.

Offline: vectorised with numpy searchsorted per zone.
Online:  OnlineFeatureStore keeps per-zone sorted buffers and answers the same query.
Both feed raw window aggregates into derive(), so derived features cannot drift apart.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from . import config as C

FEATURES = [
    "req_5", "req_15", "req_60", "drop_15", "drop_60", "ds_ratio_15",
    "wait_mean_30", "wait_n_30", "wait_mean_10", "open_reqs", "open_age_mean", "hour", "dow", "how", "minute_of_day",
    "zone", "borough_id", "is_airport", "wav",
]
CAT_FEATURES = ["zone", "borough_id", "how"]
N_ZONES = 266
BOROUGHS = ["Bronx", "Brooklyn", "EWR", "Manhattan", "Queens", "Staten Island", "Unknown", "N/A"]


def to_sec(col: pl.Series) -> np.ndarray:
    """Naive local timestamps -> int64 seconds (treated as UTC; only differences matter)."""
    return col.dt.epoch("s").to_numpy().astype(np.int64)


@dataclass
class ZoneMeta:
    borough_id: np.ndarray   # index = zone id
    is_airport: np.ndarray

    @classmethod
    def from_zones(cls, zones: pl.DataFrame) -> "ZoneMeta":
        b = np.full(N_ZONES, BOROUGHS.index("Unknown"), dtype=np.int16)
        a = np.zeros(N_ZONES, dtype=np.int8)
        for z, bor in zones.select("zone", "borough").iter_rows():
            b[z] = BOROUGHS.index(bor) if bor in BOROUGHS else BOROUGHS.index("Unknown")
            a[z] = int(z in C.AIRPORT_ZONES)
        return cls(b, a)


def derive(raw: dict, t_sec: np.ndarray, zone: np.ndarray, wav: np.ndarray, meta: ZoneMeta) -> dict:
    """Raw window aggregates -> model features. Shared by offline and online paths."""
    t_sec = np.asarray(t_sec, dtype=np.int64)
    zone = np.asarray(zone, dtype=np.int64)
    hour = (t_sec // 3600) % 24
    dow = ((t_sec // 86400) + 3) % 7          # 1970-01-01 was a Thursday; Monday = 0
    n = np.asarray(raw["wait_n_30"], dtype=np.float64)
    s = np.asarray(raw["wait_sum_30"], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        wait_mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    n10 = np.asarray(raw["wait_n_10"], dtype=np.float64)
    s10 = np.asarray(raw["wait_sum_10"], dtype=np.float64)
    on = np.asarray(raw["open_n"], dtype=np.float64)
    osum = np.asarray(raw["open_req_sum"], dtype=np.float64)   # sum of open requests' timestamps
    with np.errstate(invalid="ignore", divide="ignore"):
        wait10 = np.where(n10 > 0, s10 / np.maximum(n10, 1), np.nan)
        open_age = np.where(on > 0, (t_sec * on - osum) / np.maximum(on, 1) / 60.0, 0.0)
    req15 = np.asarray(raw["req_15"], dtype=np.float64)
    drop15 = np.asarray(raw["drop_15"], dtype=np.float64)
    return {
        "req_5": np.asarray(raw["req_5"], dtype=np.float32),
        "req_15": req15.astype(np.float32),
        "req_60": np.asarray(raw["req_60"], dtype=np.float32),
        "drop_15": drop15.astype(np.float32),
        "drop_60": np.asarray(raw["drop_60"], dtype=np.float32),
        "ds_ratio_15": ((req15 + 1.0) / (drop15 + 1.0)).astype(np.float32),
        "wait_mean_30": wait_mean.astype(np.float32),
        "wait_n_30": n.astype(np.float32),
        "wait_mean_10": wait10.astype(np.float32),
        "open_reqs": on.astype(np.float32),
        "open_age_mean": open_age.astype(np.float32),
        "hour": hour.astype(np.int16),
        "dow": dow.astype(np.int16),
        "how": (dow * 24 + hour).astype(np.int16),
        "minute_of_day": ((t_sec // 60) % 1440).astype(np.int16),
        "zone": zone.astype(np.int16),
        "borough_id": meta.borough_id[zone].astype(np.int16),
        "is_airport": meta.is_airport[zone].astype(np.int8),
        "wav": np.asarray(wav, dtype=np.int8),
    }


# ----------------------------------------------------------------------------- events
@dataclass
class Events:
    """Per-zone sorted event arrays."""
    req: dict = field(default_factory=dict)        # zone -> times
    drop: dict = field(default_factory=dict)       # zone -> times
    arr_t: dict = field(default_factory=dict)      # zone -> times
    arr_cs: dict = field(default_factory=dict)     # zone -> cumsum of waits (len n+1)
    open_t: dict = field(default_factory=dict)     # zone -> request times of labelled trips
    open_cs: dict = field(default_factory=dict)    # zone -> cumsum of those request times
    arr_rcs: dict = field(default_factory=dict)    # zone -> cumsum of arrivals' request times


def build_events(trips: pl.DataFrame) -> Events:
    ev = Events()

    def group(z, t):
        order = np.lexsort((t, z))
        z, t = z[order], t[order]
        cuts = np.flatnonzero(np.diff(z)) + 1
        return {int(zz[0]): tt for zz, tt in zip(np.split(z, cuts), np.split(t, cuts)) if len(zz)}, order, cuts

    ev.req, _, _ = group(trips["zone"].to_numpy(), to_sec(trips["request_datetime"]))
    d = trips.filter(pl.col("dropoff_datetime").is_not_null())
    ev.drop, _, _ = group(d["do_zone"].to_numpy(), to_sec(d["dropoff_datetime"]))
    a = trips.filter(pl.col("label_ok"))
    az, at, aw = a["zone"].to_numpy(), to_sec(a["on_scene_datetime"]), a["wait_min"].to_numpy()
    ar = to_sec(a["request_datetime"])
    order = np.lexsort((at, az))
    az, at, aw, arq = az[order], at[order], aw[order], ar[order]
    cuts = np.flatnonzero(np.diff(az)) + 1
    for zz, tt, ww, rr in zip(np.split(az, cuts), np.split(at, cuts), np.split(aw, cuts), np.split(arq, cuts)):
        if len(zz):
            ev.arr_t[int(zz[0])] = tt
            ev.arr_cs[int(zz[0])] = np.concatenate([[0.0], np.cumsum(ww, dtype=np.float64)])
            ev.arr_rcs[int(zz[0])] = np.concatenate([[0], np.cumsum(rr, dtype=np.int64)])
    oz, ot = a["zone"].to_numpy(), ar
    order = np.lexsort((ot, oz))
    oz, ot = oz[order], ot[order]
    cuts = np.flatnonzero(np.diff(oz)) + 1
    for zz, tt in zip(np.split(oz, cuts), np.split(ot, cuts)):
        if len(zz):
            ev.open_t[int(zz[0])] = tt
            ev.open_cs[int(zz[0])] = np.concatenate([[0], np.cumsum(tt, dtype=np.int64)])
    return ev


def _count(times: np.ndarray | None, t: np.ndarray, w_sec: int) -> np.ndarray:
    if times is None:
        return np.zeros(len(t), dtype=np.int64)
    return np.searchsorted(times, t, "left") - np.searchsorted(times, t - w_sec, "left")


def offline_raw(ev: Events, zone: np.ndarray, t_sec: np.ndarray) -> dict:
    n = len(zone)
    raw = {f"req_{w}": np.zeros(n, np.int64) for w in C.REQ_WINDOWS_MIN}
    raw |= {f"drop_{w}": np.zeros(n, np.int64) for w in C.DROP_WINDOWS_MIN}
    raw |= {k: np.zeros(n, np.int64) for k in ("wait_n_30", "wait_n_10", "open_n", "open_req_sum")}
    raw |= {k: np.zeros(n, np.float64) for k in ("wait_sum_30", "wait_sum_10")}
    order = np.argsort(zone, kind="stable")
    zs = zone[order]
    cuts = np.flatnonzero(np.diff(zs)) + 1
    for idx in np.split(order, cuts):
        if not len(idx):
            continue
        z, t = int(zone[idx[0]]), t_sec[idx]
        for w in C.REQ_WINDOWS_MIN:
            raw[f"req_{w}"][idx] = _count(ev.req.get(z), t, w * 60)
        for w in C.DROP_WINDOWS_MIN:
            raw[f"drop_{w}"][idx] = _count(ev.drop.get(z), t, w * 60)
        at = ev.arr_t.get(z)
        if at is not None:
            hi = np.searchsorted(at, t, "left")
            lo = np.searchsorted(at, t - C.WAIT_WINDOW_MIN * 60, "left")
            raw["wait_n_30"][idx] = hi - lo
            cs = ev.arr_cs[z]
            raw["wait_sum_30"][idx] = cs[hi] - cs[lo]
            lo10 = np.searchsorted(at, t - 10 * 60, "left")
            raw["wait_n_10"][idx] = hi - lo10
            raw["wait_sum_10"][idx] = cs[hi] - cs[lo10]
            a_n, a_rs = hi, ev.arr_rcs[z][hi]
        else:
            a_n, a_rs = 0, 0
        ot = ev.open_t.get(z)
        if ot is not None:
            oh = np.searchsorted(ot, t, "left")
            raw["open_n"][idx] = oh - a_n
            raw["open_req_sum"][idx] = ev.open_cs[z][oh] - a_rs
    return raw


def offline_features(trips: pl.DataFrame, queries: pl.DataFrame, meta: ZoneMeta,
                     ev: Events | None = None) -> pl.DataFrame:
    """Features for each query row (needs zone, request_datetime, wav)."""
    ev = ev or build_events(trips)
    zone = queries["zone"].to_numpy().astype(np.int64)
    t = to_sec(queries["request_datetime"])
    raw = offline_raw(ev, zone, t)
    return pl.DataFrame(derive(raw, t, zone, queries["wav"].to_numpy(), meta))


# ----------------------------------------------------------------------------- online
class _Buf:
    """Sorted timestamps (+ optional values) for one zone and one stream."""
    __slots__ = ("t", "v")

    def __init__(self):
        self.t: list[int] = []
        self.v: list[float] = []

    def add(self, ts: int, val: float = 0.0):
        if not self.t or ts >= self.t[-1]:
            self.t.append(ts); self.v.append(val)
        else:  # out-of-order event
            i = bisect.bisect_right(self.t, ts)
            self.t.insert(i, ts); self.v.insert(i, val)

    def evict(self, before: int):
        i = bisect.bisect_left(self.t, before)
        if i:
            del self.t[:i]; del self.v[:i]

    def window(self, t: int, w_sec: int) -> tuple[int, int]:
        return bisect.bisect_left(self.t, t - w_sec), bisect.bisect_left(self.t, t)


class OnlineFeatureStore:
    """In-memory, event-time feature store. Fed by the replay streamer (or a live stream)."""

    RETAIN_SEC = (C.MAX_WINDOW_MIN + 30) * 60

    def __init__(self, meta: ZoneMeta):
        self.meta = meta
        self.req = [_Buf() for _ in range(N_ZONES)]
        self.drop = [_Buf() for _ in range(N_ZONES)]
        self.arr = [_Buf() for _ in range(N_ZONES)]
        self.open_n = np.zeros(N_ZONES, np.int64)        # waiting requests per zone
        self.open_sum = np.zeros(N_ZONES, np.int64)      # sum of their request timestamps
        self.clock = -1          # latest event time seen
        self._n = 0

    def _tick(self, ts: int):
        if ts > self.clock:
            self.clock = ts
        self._n += 1
        if self._n % 50_000 == 0:
            cut = self.clock - self.RETAIN_SEC
            for bufs in (self.req, self.drop, self.arr):
                for b in bufs:
                    b.evict(cut)

    def add_request(self, zone: int, ts: int):
        self.req[zone].add(ts); self._tick(ts)

    def add_dropoff(self, zone: int, ts: int):
        self.drop[zone].add(ts); self._tick(ts)

    def add_arrival(self, zone: int, ts: int, wait_min: float, req_ts: int | None = None):
        self.arr[zone].add(ts, wait_min)
        if req_ts is not None:          # closes an open request
            self.open_n[zone] -= 1; self.open_sum[zone] -= req_ts
        self._tick(ts)

    def add_open(self, zone: int, ts: int):
        """A request that is now waiting for a driver (labelled trips only in replay)."""
        self.open_n[zone] += 1; self.open_sum[zone] += ts; self._tick(ts)

    def raw(self, zone: int, t: int) -> dict:
        out = {}
        for w in C.REQ_WINDOWS_MIN:
            lo, hi = self.req[zone].window(t, w * 60); out[f"req_{w}"] = [hi - lo]
        for w in C.DROP_WINDOWS_MIN:
            lo, hi = self.drop[zone].window(t, w * 60); out[f"drop_{w}"] = [hi - lo]
        b = self.arr[zone]
        lo, hi = b.window(t, C.WAIT_WINDOW_MIN * 60)
        out["wait_n_30"] = [hi - lo]
        out["wait_sum_30"] = [float(np.sum(b.v[lo:hi], dtype=np.float64)) if hi > lo else 0.0]
        lo10, _ = b.window(t, 10 * 60)
        out["wait_n_10"] = [hi - lo10]
        out["wait_sum_10"] = [float(np.sum(b.v[lo10:hi], dtype=np.float64)) if hi > lo10 else 0.0]
        out["open_n"] = [int(self.open_n[zone])]
        out["open_req_sum"] = [int(self.open_sum[zone])]
        return out

    def features(self, zone: int, t: int, wav: int = 0) -> dict:
        f = derive(self.raw(zone, t), np.array([t]), np.array([zone]), np.array([wav]), self.meta)
        return {k: v[0].item() for k, v in f.items()}

    def staleness_min(self, t: int) -> float:
        return float("inf") if self.clock < 0 else (t - self.clock) / 60.0


# ----------------------------------------------------------------------------- replay
KIND_QUERY, KIND_REQ, KIND_DROP, KIND_ARR, KIND_OPEN = 0, 1, 2, 3, 4


def replay_events(trips: pl.DataFrame, start, end) -> np.ndarray:
    """Event-time ordered stream covering [start - 60 min warm-up, end).

    Structured array (ts, kind, zone, val, aux). At equal timestamps stream_queries answers
    queries before applying events, so a query never sees its own instant (offline [t-w, t)).
    Open-queue state is seeded with every labelled request still waiting at warm-up start.
    """
    warm = pl.lit(start) - pl.duration(minutes=C.MAX_WINDOW_MIN)
    lab = pl.col("label_ok")
    r = trips.filter((pl.col("request_datetime") >= warm) & (pl.col("request_datetime") < end))
    d = trips.filter((pl.col("dropoff_datetime") >= warm) & (pl.col("dropoff_datetime") < end))
    a = trips.filter(lab & (pl.col("on_scene_datetime") >= warm) & (pl.col("on_scene_datetime") < end))
    o = trips.filter(lab & (pl.col("on_scene_datetime") >= warm) & (pl.col("request_datetime") < end))
    parts = [
        (to_sec(r["request_datetime"]), KIND_REQ, r["zone"], np.zeros(r.height), np.zeros(r.height)),
        (to_sec(d["dropoff_datetime"]), KIND_DROP, d["do_zone"], np.zeros(d.height), np.zeros(d.height)),
        (to_sec(a["on_scene_datetime"]), KIND_ARR, a["zone"], a["wait_min"].to_numpy(), to_sec(a["request_datetime"])),
        (to_sec(o["request_datetime"]), KIND_OPEN, o["zone"], np.zeros(o.height), np.zeros(o.height)),
    ]
    dt = np.dtype([("ts", np.int64), ("kind", np.int8), ("zone", np.int16), ("val", np.float64), ("aux", np.int64)])
    arrs = []
    for ts, kind, zone, val, aux in parts:
        x = np.empty(len(ts), dt)
        x["ts"], x["kind"], x["zone"], x["val"], x["aux"] = ts, kind, zone.to_numpy(), val, aux
        arrs.append(x)
    ev = np.concatenate(arrs)
    return ev[np.lexsort((ev["kind"], ev["ts"]))]


def stream_queries(store: OnlineFeatureStore, events: np.ndarray, queries: dict[int, list]):
    """Push events through the store; at each query timestamp yield online features.

    queries: ts -> list of (row_id, zone, wav). Yields (row_id, features dict).
    """
    qts = np.array(sorted(queries), dtype=np.int64)
    qi = 0
    for ts, kind, zone, val, aux in events:
        while qi < len(qts) and qts[qi] <= ts:   # answer queries before same-instant events
            for rid, z, wav in queries[int(qts[qi])]:
                yield rid, store.features(int(z), int(qts[qi]), int(wav))
            qi += 1
        if kind == KIND_REQ:
            store.add_request(int(zone), int(ts))
        elif kind == KIND_DROP:
            store.add_dropoff(int(zone), int(ts))
        elif kind == KIND_ARR:
            store.add_arrival(int(zone), int(ts), float(val), int(aux))
        elif kind == KIND_OPEN:
            store.add_open(int(zone), int(ts))
    for q in qts[qi:]:
        for rid, z, wav in queries[int(q)]:
            yield rid, store.features(int(z), int(q), int(wav))
