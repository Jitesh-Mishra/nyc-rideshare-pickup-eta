"""End-to-end: audit -> point-in-time features -> B0 / LightGBM / PyTorch -> val -> one test run.

python -m eta.train            # full run (val selection + single locked test evaluation)
python -m eta.train --no-test  # tuning only, test untouched
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import date

import numpy as np
import polars as pl

from . import config as C
from .data import audit, load_trips, load_zones
from .features import FEATURES, ZoneMeta, build_events, offline_features
from .models import HistoricalMedian, LGBQuantile, TorchQuantile, metrics, torch_inputs

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def hour_bucket(h):
    return np.select([h < 6, h < 10, h < 16, h < 20], ["00-05 night", "06-09 am peak", "10-15 midday", "16-19 pm peak"],
                     "20-23 evening")


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT, text=True).strip()
    except Exception:
        return "uncommitted"


def build_dataset():
    trips = load_trips()
    zones = load_zones()
    meta = ZoneMeta.from_zones(zones)
    a = audit(trips)
    (C.REPORTS / "data_audit.json").write_text(json.dumps(a, indent=2))
    log(f"trips={trips.height:,} label_rows={a['label_rows']:,}")
    q = trips.filter(pl.col("label_ok") & (pl.col("request_datetime") >= C.LABEL_START)
                     & (pl.col("request_datetime") < C.TEST_END))
    ev = build_events(trips)
    feats = offline_features(trips, q, meta, ev)
    ds = pl.concat([q.select("request_datetime", "wait_min"), feats], how="horizontal_extend")
    ds = ds.join(zones.select("zone", "borough").with_columns(pl.col("zone").cast(pl.Int16)), on="zone", how="left")
    log(f"features built: {ds.height:,} rows x {len(FEATURES)} features")
    return trips, zones, meta, ds


def split(ds):
    tr = ds.filter(pl.col("request_datetime") < C.VAL_START)
    va = ds.filter((pl.col("request_datetime") >= C.VAL_START) & (pl.col("request_datetime") < C.TEST_START))
    te = ds.filter(pl.col("request_datetime") >= C.TEST_START)
    return tr, va, te


def X(df):
    return df.select(FEATURES).to_numpy().astype(np.float32)


def predict_all(models, df):
    b0, lgbm, tq = models
    y = df["wait_min"].to_numpy()
    b50, b90 = b0.predict(df["zone"].to_numpy(), df["how"].to_numpy())
    l50, l90 = lgbm.predict(X(df))
    num, cat = torch_inputs(df, b50.astype(np.float32))
    t50, t90 = tq.predict(num, cat)
    g50, g90 = b0.predict_global(len(y))
    return y, {"B0_global_median": (g50, g90), "B0_zone_how_median": (b50, b90),
               "M1_lightgbm": (l50, l90), "M2_torch_embed": (t50, t90)}


def sliced(df, y, preds):
    rows = []
    slices = {"borough": df["borough"].to_numpy(), "hour_bucket": hour_bucket(df["hour"].to_numpy()),
              "airport": np.where(df["is_airport"].to_numpy() == 1, "airport", "non-airport")}
    for sname, key in slices.items():
        for v in sorted(set(key)):
            m = key == v
            if m.sum() < 500:
                continue
            for name, (p50, p90) in preds.items():
                if name == "B0_global_median":
                    continue
                r = metrics(y[m], p50[m], p90[m])
                rows.append({"slice": sname, "value": v, "model": name, **r})
    return rows


def write_csv(path, rows, append=False):
    exists = path.exists() and append
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)


def main(run_test=True, lgb_rows=2_000_000):
    C.REPORTS.mkdir(exist_ok=True); C.MODEL_DIR.mkdir(exist_ok=True); C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    trips, zones, meta, ds = build_dataset()
    tr, va, te = split(ds)
    log(f"split train={tr.height:,} val={va.height:,} test={te.height:,}")

    ytr, yva = tr["wait_min"].to_numpy(), va["wait_min"].to_numpy()
    b0 = HistoricalMedian().fit(tr["zone"].to_numpy(), tr["how"].to_numpy(), ytr)
    b0.save(C.MODEL_DIR / "b0.npz")
    log("B0 fitted")

    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(tr.height, size=min(lgb_rows, tr.height), replace=False)
    t_l = time.time()
    lgbm = LGBQuantile().fit(X(tr)[idx], ytr[idx], X(va), yva, log=log)
    t_l = time.time() - t_l
    lgbm.save(str(C.MODEL_DIR / "lgb"))

    b50_tr, _ = b0.predict(tr["zone"].to_numpy(), tr["how"].to_numpy())
    b50_va, _ = b0.predict(va["zone"].to_numpy(), va["how"].to_numpy())
    ntr, ctr = torch_inputs(tr, b50_tr.astype(np.float32))
    nva, cva = torch_inputs(va, b50_va.astype(np.float32))
    t_t = time.time()
    tq = TorchQuantile().fit(ntr, ctr, ytr.astype(np.float32), nva, cva, yva, log=log)
    t_t = time.time() - t_t
    tq.save(C.MODEL_DIR / "torch_eta.pt")

    models = (b0, lgbm, tq)
    sha, today = git_sha(), date.today().isoformat()
    yv, pv = predict_all(models, va)
    exp_rows = []
    for name, (p50, p90) in pv.items():
        m = metrics(yv, p50, p90)
        cfg = {"M1_lightgbm": f"quantile; rows={len(idx)}; iters={lgbm.best_iter}",
               "M2_torch_embed": f"MLP256-128+emb; epochs={len(tq.history)}"}.get(name, "train-label quantiles")
        rt = {"M1_lightgbm": t_l, "M2_torch_embed": t_t}.get(name, 0.0)
        exp_rows.append({"date": today, "commit": sha, "approach": name, "config": cfg, "split": "val",
                         "val_mae_min": round(m["mae_min"], 4), "val_p90_coverage": round(m["p90_coverage"], 4),
                         "runtime_s": round(rt, 1), "notes": ""})
        log(f"VAL {name}: MAE={m['mae_min']:.3f} cov90={m['p90_coverage']:.3f}")
    write_csv(C.REPORTS / "experiments.csv", exp_rows, append=True)

    if not run_test:
        return
    runs_path = C.REPORTS / "test_runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else {"final_test_runs": 0, "runs": []}
    yt, pt = predict_all(models, te)
    res = []
    base = metrics(yt, *pt["B0_zone_how_median"])["mae_min"]
    for name, (p50, p90) in pt.items():
        m = metrics(yt, p50, p90)
        res.append({"model": name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
                    "mae_vs_B0_pct": round(100 * (m["mae_min"] - base) / base, 2)})
        log(f"TEST {name}: MAE={m['mae_min']:.3f} cov90={m['p90_coverage']:.3f}")
    write_csv(C.REPORTS / "results.csv", res)
    # Headroom diagnostic (not a model): median wait of the same zone in the same 10-min block,
    # computed from the labels themselves, i.e. uses the future. Upper bound for zone-time signals.
    o = te.with_columns(pl.col("request_datetime").dt.truncate("10m").alias("blk"))
    o = o.with_columns(pl.col("wait_min").median().over(["zone", "blk"]).alias("oracle"),
                       pl.len().over(["zone", "blk"]).alias("blk_n"))
    m = (o["blk_n"] >= 5).to_numpy()
    oracle = {"definition": "zone x 10-min block median of realised waits (uses future labels; ceiling, not a model)",
              "rows_with_block_n_ge_5": int(m.sum()),
              "oracle_mae": float(np.mean(np.abs(yt[m] - o["oracle"].to_numpy()[m])))}
    for name in ("B0_zone_how_median", "M1_lightgbm", "M2_torch_embed"):
        oracle[f"{name}_mae_same_rows"] = float(np.mean(np.abs(yt[m] - pt[name][0][m])))
    b, l = oracle["B0_zone_how_median_mae_same_rows"], oracle["M1_lightgbm_mae_same_rows"]
    oracle["lgb_share_of_headroom_captured"] = (b - l) / (b - oracle["oracle_mae"])
    (C.REPORTS / "headroom.json").write_text(json.dumps(oracle, indent=2))
    log(f"headroom: {oracle}")
    write_csv(C.REPORTS / "sliced_errors.csv", [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                                                for r in sliced(te, yt, pt)])
    out = te.select("request_datetime", "zone", "borough", "hour", "is_airport", "wait_min", *FEATURES[:8])
    for name, (p50, p90) in pt.items():
        out = out.with_columns(pl.Series(f"{name}_p50", p50), pl.Series(f"{name}_p90", p90))
    out.write_parquet(C.REPORTS / "test_predictions.parquet")
    runs["final_test_runs"] += 1
    runs["runs"].append({"date": today, "commit": sha, "test_rows": int(len(yt))})
    runs_path.write_text(json.dumps(runs, indent=2))
    log(f"done. final_test_runs={runs['final_test_runs']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--lgb-rows", type=int, default=2_000_000)
    a = ap.parse_args()
    main(run_test=not a.no_test, lgb_rows=a.lgb_rows)
