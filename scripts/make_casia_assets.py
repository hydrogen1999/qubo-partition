#!/usr/bin/env python3
"""Build committed CASIA result assets: IoU-distribution histogram + qualitative montage."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "casia_results_dense.csv"
FIG_DIR = ROOT / "results" / "figures" / "casia"
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)

rows = list(csv.DictReader(open(CSV)))
iou = np.array([float(r["iou"]) for r in rows])
by_name = {Path(r["image"]).stem: float(r["iou"]) for r in rows}

# --- 1. IoU distribution over the full tampered set -------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.hist(iou, bins=30, color="#4C72B0", edgecolor="white")
ax.axvline(iou.mean(), color="#C44E52", lw=2, label=f"mean = {iou.mean():.3f}")
ax.axvline(np.median(iou), color="#55A868", lw=2, ls="--", label=f"median = {np.median(iou):.3f}")
ax.set_xlabel("IoU")
ax.set_ylabel("number of images")
ax.set_title(f"CASIA v2.0 forgery localization — IoU over {len(iou)} tampered images")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "casia_iou_distribution.png", dpi=200)
plt.close(fig)
print("wrote assets/casia_iou_distribution.png")

# --- 2. Qualitative montage spanning the IoU range --------------------------
avail = [(p, by_name.get(p.stem)) for p in sorted(FIG_DIR.glob("*.png"))]
avail = [(p, v) for p, v in avail if v is not None]
avail.sort(key=lambda kv: kv[1], reverse=True)
if avail:
    # pick a spread: 3 high, 2 mid, 1 low (clamped to availability)
    n = len(avail)
    picks = [avail[0], avail[1], avail[2], avail[n // 2], avail[3 * n // 4], avail[-1]]
    seen, rows_sel = set(), []
    for p, v in picks:
        if p not in seen:
            seen.add(p)
            rows_sel.append((p, v))

    fig, axes = plt.subplots(len(rows_sel), 1, figsize=(11, 2.6 * len(rows_sel)))
    if len(rows_sel) == 1:
        axes = [axes]
    for ax, (p, v) in zip(axes, rows_sel):
        ax.imshow(mpimg.imread(p))
        ax.axis("off")
    fig.suptitle(
        "CASIA v2.0 — image+seeds | annealed | max-flow optimum | ground truth "
        "(rows span the IoU range)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / "casia_results_montage.png", dpi=150)
    plt.close(fig)
    print(f"wrote assets/casia_results_montage.png ({len(rows_sel)} rows)")
