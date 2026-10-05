from datetime import datetime, timedelta

import numpy as np
import polars as pl

from eta.features import offline_features
from eta.parity import run_parity


def _q(zone, t, wav=0):
    return pl.DataFrame({"zone": [zone], "request_datetime": [t], "wav": [wav]}).with_columns(
        pl.col("zone").cast(pl.Int32), pl.col("wav").cast(pl.Int8))


def test_window_is_strictly_before_request(meta):
    t = datetime(2025, 3, 10, 12, 0, 0)
    rows = {  # request times in zone 4; dropoffs into zone 4; one arrival
        "request_datetime": [t - timedelta(minutes=16), t - timedelta(minutes=4), t, t + timedelta(minutes=1)],
    }
    n = 4
    df = pl.DataFrame({
        **rows,
        "on_scene_datetime": [t - timedelta(minutes=10), t, t + timedelta(minutes=3), t + timedelta(minutes=5)],
        "pickup_datetime": [t - timedelta(minutes=9)] * n,
        "dropoff_datetime": [t - timedelta(minutes=14), t - timedelta(seconds=1), t, t + timedelta(minutes=2)],
        "zone": [4] * n, "do_zone": [4] * n, "wav": [0] * n,
        "wait_min": [6.0, 4.0, 3.0, 4.0], "label_ok": [True] * n,
    }).with_columns(pl.col("zone").cast(pl.Int32), pl.col("do_zone").cast(pl.Int32), pl.col("wav").cast(pl.Int8))
    f = offline_features(df, _q(4, t), meta).row(0, named=True)
    assert f["req_5"] == 1      # t-4min only; the request at t itself and the future one are excluded
    assert f["req_15"] == 1     # t-16 is outside 15 min
    assert f["req_60"] == 2
    assert f["drop_15"] == 2    # t-14 and t-1s; drop at exactly t excluded
    assert f["wait_n_30"] == 1  # only the arrival at t-10; arrival at exactly t excluded
    assert abs(f["wait_mean_30"] - 6.0) < 1e-6
    assert f["wait_mean_10"] == 6.0   # t-10 arrival is inside [t-10, t)
    # open queue at t: t-16 request arrived at t-10 (closed); t-4 request's driver arrives at t (still open)
    assert f["open_reqs"] == 1 and abs(f["open_age_mean"] - 4.0) < 1e-6


def test_future_events_do_not_change_features(trips, meta):
    """Leakage test: rewrite everything at/after a cut time; features before the cut must not move."""
    cut = datetime(2025, 3, 10, 6, 0)
    q = trips.filter((pl.col("request_datetime") >= cut - timedelta(hours=1)) & (pl.col("request_datetime") < cut))
    q = q.head(2000)
    base = offline_features(trips, q, meta)
    fut = pl.col("request_datetime") >= cut
    scrambled = trips.with_columns(
        pl.when(fut).then(pl.lit(1)).otherwise(pl.col("zone")).cast(pl.Int32).alias("zone"),
        pl.when(pl.col("dropoff_datetime") >= cut).then(pl.lit(1)).otherwise(pl.col("do_zone")).cast(pl.Int32).alias("do_zone"),
        pl.when(pl.col("on_scene_datetime") >= cut).then(pl.lit(29.0)).otherwise(pl.col("wait_min")).alias("wait_min"),
    )
    after = offline_features(scrambled, q, meta)
    for c in base.columns:
        a, b = base[c].to_numpy(), after[c].to_numpy()
        assert np.allclose(a, b, equal_nan=True), c


def test_offline_online_parity(trips, meta):
    r = run_parity(trips, meta, datetime(2025, 3, 10, 4), datetime(2025, 3, 10, 10), n_sample=3000)
    assert r["parity_rate"] == 1.0, r
