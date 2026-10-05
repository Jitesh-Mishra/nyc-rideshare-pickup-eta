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

THEMES = {"light": dict(bg="#ffffff", ink="#1f2328", ink2="#59636e", grid="#e6e9ed", ref="#59636e",
                       cols=["#2a78d6", "#eb6834", "#1baf7a"]),
          "dark": dict(bg="#0d1117", ink="#e6edf3", ink2="#9198a1", grid="#21262d", ref="#9198a1",
                       cols=["#3987e5", "#d95926", "#199e70"])}
for theme, th in THEMES.items():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": th["ink2"], "axes.labelcolor": th["ink"],
                         "xtick.color": th["ink2"], "ytick.color": th["ink2"], "text.color": th["ink"]})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=th["bg"])
    for ax, suffix, title, ylab in [(axes[0], "mae", "P50 error by hour (holdout week)", "MAE, minutes"),
                                    (axes[1], "cov", "P90 coverage by hour (target 0.90)", "share of waits at or below P90")]:
        ax.set_facecolor(th["bg"])
        for (m, label, _), color in zip(MODELS, th["cols"]):
            ax.plot(agg["hour"], agg[f"{m}_{suffix}"], color=color, lw=2, label=label)
        if suffix == "cov":
            ax.axhline(0.9, color=th["ref"], lw=1, ls="--")
        ax.set_title(title, loc="left", color=th["ink"])
        ax.set_xlabel("hour of request"); ax.set_ylabel(ylab)
        ax.grid(axis="y", color=th["grid"], lw=0.8); ax.set_xticks(range(0, 24, 3))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right", labelcolor=th["ink"])
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / f"error_by_hour-{theme}.png"
    fig.savefig(out, dpi=150, facecolor=th["bg"])
    print(out, file=sys.stderr)
