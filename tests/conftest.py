"""Synthetic trips with the load_trips() schema. Used only by unit tests, never for results."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eta.features import ZoneMeta  # noqa: E402


def make_trips(n=20_000, seed=0, start=datetime(2025, 3, 10), hours=12, zones=(1, 4, 132, 161, 236)):
    rng = np.random.default_rng(seed)
    req = np.datetime64(start, "us") + pd_seconds(rng.integers(0, hours * 3600, n))
    wait = rng.gamma(2.0, 2.0, n)
    on_scene = req + pd_seconds((wait * 60).astype(int))
    pickup = on_scene + pd_seconds(rng.integers(0, 90, n))
    drop = pickup + pd_seconds(rng.integers(300, 3000, n))
    df = pl.DataFrame({
        "request_datetime": req, "on_scene_datetime": on_scene, "pickup_datetime": pickup,
        "dropoff_datetime": drop,
        "zone": rng.choice(zones, n).astype(np.int32), "do_zone": rng.choice(zones, n).astype(np.int32),
        "wav": (rng.random(n) < 0.01).astype(np.int8),
    })
    raw = (pl.col("on_scene_datetime") - pl.col("request_datetime")).dt.total_seconds() / 60
    return df.with_columns(raw.alias("wait_raw")).with_columns(
        (pl.col("wait_raw") >= 0).alias("label_ok"), pl.col("wait_raw").clip(0, 30).alias("wait_min"),
    ).sort("request_datetime")


def pd_seconds(s):
    return np.asarray(s, dtype=np.int64).astype("timedelta64[s]").astype("timedelta64[us]")


@pytest.fixture(scope="session")
def trips():
    return make_trips()


@pytest.fixture(scope="session")
def meta():
    z = pl.DataFrame({"zone": [1, 4, 132, 161, 236], "borough": ["EWR", "Manhattan", "Queens", "Manhattan", "Manhattan"]})
    return ZoneMeta.from_zones(z.with_columns(pl.col("zone").cast(pl.Int32)))
