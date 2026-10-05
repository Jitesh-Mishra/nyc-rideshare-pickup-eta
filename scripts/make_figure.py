"""reports/figures/error_by_hour.png: holdout MAE and P90 coverage by hour of day."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
df = pl.read_parquet(ROOT / "reports" / "test_predictions.parquet")
MODELS = [("B0_zone_how_median", "B0 zone x hour-of-week median", "#2a78d6"),
          ("M1_lightgbm", "M1 LightGBM quantile", "#eb6834"),
          ("M2_torch_embed", "M2 PyTorch embedding", "#1baf7a")]
agg = df.group_by("hour").agg(
    *[(pl.col("wait_min") - pl.col(f"{m}_p50")).abs().mean().alias(f"{m}_mae") for m, _, _ in MODELS],
    *[(pl.col("wait_min") <= pl.col(f"{m}_p90")).mean().alias(f"{m}_cov") for m, _, _ in MODELS],
).sort("hour")

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfb")
for ax, suffix, title, ylab in [(axes[0], "mae", "P50 error by hour (holdout week)", "MAE, minutes"),
                                (axes[1], "cov", "P90 coverage by hour (target 0.90)", "share of waits at or below P90")]:
    ax.set_facecolor("#fcfcfb")
    for m, label, color in MODELS:
        ax.plot(agg["hour"], agg[f"{m}_{suffix}"], color=color, lw=2, label=label)
    if suffix == "cov":
        ax.axhline(0.9, color="#52514e", lw=1, ls="--")
    ax.set_title(title, loc="left", color="#0b0b0b")
    ax.set_xlabel("hour of request"); ax.set_ylabel(ylab)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_xticks(range(0, 24, 3))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].legend(frameon=False, loc="upper right")
fig.tight_layout()
out = ROOT / "reports" / "figures" / "error_by_hour.png"
fig.savefig(out, dpi=150)
print(out, file=sys.stderr)
