"""Load the staged Lyft (HV0005) TLC trips, build the label, audit it."""
from __future__ import annotations

import json

import polars as pl

from . import config as C


def load_zones() -> pl.DataFrame:
    z = pl.read_csv(C.ZONES_CSV)
    return z.select(
        pl.col("LocationID").cast(pl.Int32).alias("zone"),
        pl.col("Borough").fill_null("Unknown").alias("borough"),
        pl.col("Zone").fill_null("Unknown").alias("zone_name"),
    )


def load_trips(path_glob: str = C.TRIPS_GLOB) -> pl.DataFrame:
    """All Lyft trips with the label.

    wait_min = on_scene_datetime - request_datetime (driver arrival, boarding excluded).
    label_ok marks rows usable as training/eval labels: on_scene present and wait >= 0.
    The label is clipped to [0, 30] min; rows above 30 are counted in the audit.
    """
    df = pl.scan_parquet(path_glob).select(
        "request_datetime", "on_scene_datetime", "pickup_datetime", "dropoff_datetime",
        pl.col("PULocationID").cast(pl.Int32).alias("zone"),
        pl.col("DOLocationID").cast(pl.Int32).alias("do_zone"),
        (pl.col("wav_request_flag") == "Y").cast(pl.Int8).alias("wav"),
    ).collect()
    raw_wait = (pl.col("on_scene_datetime") - pl.col("request_datetime")).dt.total_seconds() / 60.0
    df = df.with_columns(raw_wait.alias("wait_raw"))
    df = df.with_columns(
        (pl.col("on_scene_datetime").is_not_null() & (pl.col("wait_raw") >= 0)).alias("label_ok"),
        pl.col("wait_raw").clip(0, C.WAIT_CLIP_MIN).alias("wait_min"),
    )
    return df.sort("request_datetime")


def audit(df: pl.DataFrame) -> dict:
    lab = df.filter(pl.col("request_datetime") >= C.LABEL_START)
    out = {
        "rows_total": df.height,
        "rows_from_label_start": lab.height,
        "on_scene_null_rate_before_label_start": float(
            df.filter(pl.col("request_datetime") < C.LABEL_START)["on_scene_datetime"].is_null().mean()),
        "on_scene_null_rate_from_label_start": float(lab["on_scene_datetime"].is_null().mean()),
        "negative_wait_rate": float((lab["wait_raw"] < 0).mean()),
        "wait_over_30_rate": float((lab["wait_raw"] > C.WAIT_CLIP_MIN).mean()),
        "label_rows": int(lab["label_ok"].sum()),
        "wait_quantiles_min": {q: float(lab.filter(pl.col("label_ok"))["wait_min"].quantile(q))
                               for q in (0.1, 0.5, 0.9, 0.99)},
        "boarding_median_min": float(
            ((lab["pickup_datetime"] - lab["on_scene_datetime"]).dt.total_seconds() / 60).median()),
    }
    return out


if __name__ == "__main__":
    a = audit(load_trips())
    (C.REPORTS / "data_audit.json").write_text(json.dumps(a, indent=2))
    print(json.dumps(a, indent=2))
