"""Monitoring stub: daily PSI of top features vs training, and daily P90 coverage drift."""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from . import config as C

TOP = ["wait_mean_30", "req_15", "drop_15", "ds_ratio_15", "req_60"]


def psi(ref, cur, bins=10):
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-6
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))


def run(train_df: pl.DataFrame, model="M1_lightgbm", cov_tol=0.03) -> dict:
    te = pl.read_parquet(C.REPORTS / "test_predictions.parquet").with_columns(
        pl.col("request_datetime").dt.date().alias("day"))
    days = []
    for (day,), g in te.group_by(["day"], maintain_order=True):
        cov = float((g["wait_min"] <= g[f"{model}_p90"]).mean())
        p = {f: psi(train_df[f].to_numpy().astype(float), g[f].to_numpy().astype(float)) for f in TOP}
        alerts = [f"psi:{f}" for f, v in p.items() if v > 0.2] + ([f"p90_coverage:{cov:.3f}"] if abs(cov - 0.9) > cov_tol else [])
        days.append({"day": str(day), "n": g.height, "p90_coverage": cov, "psi": p, "alerts": alerts})
    return {"model": model, "psi_alert_threshold": 0.2, "coverage_tolerance": cov_tol, "days": sorted(days, key=lambda d: d["day"])}


if __name__ == "__main__":
    from .train import build_dataset, split
    _, _, _, ds = build_dataset()
    tr, _, _ = split(ds)
    r = run(tr)
    (C.REPORTS / "monitoring.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
