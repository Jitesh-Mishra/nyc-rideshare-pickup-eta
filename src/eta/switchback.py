"""Offline switchback-style replay: borough x 30-min blocks randomised to arms.

What this shows: prediction errors are correlated within a borough-time block (weather,
events, supply shocks), so treating requests as independent understates uncertainty.
We measure that with (1) A/A replays: same model in both arms, many random block
assignments, compare the empirical spread of the estimate with naive and cluster-robust
SEs, and (2) one A/B replay (B0 vs LightGBM) with both SEs.
What it does NOT show: interference. Predictions here don't change driver behaviour, so
there is no spillover to measure; that needs a dispatch simulator or live traffic.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from . import config as C


def diff_and_ses(y, arm, cluster):
    """Difference in means (B - A), naive iid SE, and cluster-robust (CR1) SE."""
    out = {}
    for a in (0, 1):
        m = arm == a
        ya, ca = y[m], cluster[m]
        mu, n = ya.mean(), len(ya)
        r = ya - mu
        _, inv = np.unique(ca, return_inverse=True)
        g_sums = np.bincount(inv, weights=r)
        G = len(g_sums)
        out[a] = (mu, ya.var(ddof=1) / n, (G / max(G - 1, 1)) * np.sum(g_sums ** 2) / n ** 2, G)
    d = out[1][0] - out[0][0]
    return d, float(np.sqrt(out[0][1] + out[1][1])), float(np.sqrt(out[0][2] + out[1][2])), out[0][3] + out[1][3]


def run(pred_path=C.REPORTS / "test_predictions.parquet", reps=500, seed=C.SEED) -> dict:
    df = pl.read_parquet(pred_path).filter(pl.col("borough").is_in(["Unknown", "N/A"]).not_())
    t = df["request_datetime"].dt.epoch("s").to_numpy()
    bor = df["borough"].to_numpy()
    block = t // 1800
    key = np.char.add(bor.astype(str), block.astype(str))
    _, cluster = np.unique(key, return_inverse=True)
    G = cluster.max() + 1
    y = df["wait_min"].to_numpy()
    err = {m: np.abs(y - df[f"{m}_p50"].to_numpy()) for m in ("B0_zone_how_median", "M1_lightgbm")}
    rng = np.random.default_rng(seed)

    # A/A: identical model in both arms -> true effect 0; spread across assignments is the real SE
    e = err["M1_lightgbm"]
    est_sw, naive_sw, cr_sw, est_req = [], [], [], []
    for _ in range(reps):
        arm = rng.integers(0, 2, G)[cluster]
        d, s_n, s_c, _ = diff_and_ses(e, arm, cluster)
        est_sw.append(d); naive_sw.append(s_n); cr_sw.append(s_c)
        arm_r = rng.integers(0, 2, len(e))
        est_req.append(e[arm_r == 1].mean() - e[arm_r == 0].mean())
    est_sw, naive_sw, cr_sw = map(np.array, (est_sw, naive_sw, cr_sw))
    aa = {
        "reps": reps,
        "empirical_sd_of_estimate_switchback": float(est_sw.std(ddof=1)),
        "mean_naive_se": float(naive_sw.mean()),
        "mean_cluster_robust_se": float(cr_sw.mean()),
        "ratio_cluster_to_naive_se": float(cr_sw.mean() / naive_sw.mean()),
        "ratio_empirical_to_naive_se": float(est_sw.std(ddof=1) / naive_sw.mean()),
        "false_positive_rate_naive_95": float(np.mean(np.abs(est_sw) > 1.96 * naive_sw)),
        "false_positive_rate_cluster_95": float(np.mean(np.abs(est_sw) > 1.96 * cr_sw)),
        "empirical_sd_request_level_randomisation": float(np.std(est_req, ddof=1)),
    }
    # A/B: arm 0 serves B0, arm 1 serves LightGBM, switching by borough x 30-min block
    arm = rng.integers(0, 2, G)[cluster]
    yab = np.where(arm == 1, err["M1_lightgbm"], err["B0_zone_how_median"])
    d, s_n, s_c, _ = diff_and_ses(yab, arm, cluster)
    ab = {"effect_mae_lgb_minus_b0_min": float(d), "naive_se": s_n, "cluster_robust_se": s_c,
          "ci95_naive": [float(d - 1.96 * s_n), float(d + 1.96 * s_n)],
          "ci95_cluster": [float(d - 1.96 * s_c), float(d + 1.96 * s_c)],
          "full_test_mae_difference_min": float(err["M1_lightgbm"].mean() - err["B0_zone_how_median"].mean())}
    return {"n_requests": int(len(y)), "n_blocks": int(G), "block": "borough x 30 min", "aa": aa, "ab": ab}


if __name__ == "__main__":
    r = run()
    (C.REPORTS / "switchback.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
