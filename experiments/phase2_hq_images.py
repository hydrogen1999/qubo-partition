#!/usr/bin/env python3
"""Phase two on high-resolution images.

Run:  python experiments/phase2_hq_images.py --size 96 --num-reads 60
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from qubo_partition import viz
from qubo_partition.data.real import load_skimage_demo
from qubo_partition.evaluation.runner import run_segmentation
from qubo_partition.io_utils import write_csv, write_json, write_latex_table

RESULTS = Path("results")
FIGS = RESULTS / "figures" / "hq"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--data-weight", type=float, default=1.0)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    ap.add_argument("--num-reads", type=int, default=40)
    ap.add_argument("--num-sweeps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)

    images = load_skimage_demo(size=args.size, n_seeds=args.n_seeds, seed=args.seed)
    print(f"\n== High-res segmentation ({args.size}x{args.size}): anneal vs. max-flow ==")

    rows = []
    for seeded in images:
        t0 = time.perf_counter()
        rec = run_segmentation(
            seeded,
            lambda_smooth=args.lam,
            data_weight=args.data_weight,
            data_model="histogram",
            n_bins=args.n_bins,
            connectivity=args.connectivity,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
        )
        dt = time.perf_counter() - t0
        is_gt = bool(getattr(seeded, "truth_is_gt", False))
        row = rec.as_row()
        row["truth_is_gt"] = is_gt
        row["time_s"] = round(dt, 2)
        rows.append(row)
        tag = "GT" if is_gt else "Otsu"
        print(
            f"  {seeded.name:16s} gap={rec.gap.best_gap:8.3f} success={rec.gap.success_rate:.2f} "
            f"IoU(anneal)={rec.iou_annealed:.2f} IoU(opt)={rec.iou_optimal:.2f} "
            f"[{tag}] ({dt:.1f}s)"
        )
        viz.plot_segmentation(
            seeded.image,
            seeded.fg_seeds,
            seeded.bg_seeds,
            rec.annealed_labels,
            rec.optimal_labels,
            truth=seeded.truth,
            truth_label=("ground truth" if is_gt else "Otsu reference"),
            title=f"{seeded.name}: gap={rec.gap.best_gap:.2f}, "
            f"IoU(opt)={rec.iou_optimal:.2f} ({tag})",
            path=str(FIGS / f"{seeded.name}.png"),
        )

    write_csv(rows, RESULTS / "phase2_hq_images.csv")
    write_latex_table(
        rows,
        columns=[
            "name",
            "n_pixels",
            "best_gap",
            "success_rate",
            "iou_annealed",
            "iou_optimal",
            "truth_is_gt",
            "time_s",
        ],
        path=RESULTS / "tables" / "phase2_hq_images.tex",
        caption="High-resolution (96x96) seeded segmentation: simulated annealing "
        "vs. the maximum-flow optimum on real photographs and clean shapes.",
        label="tab:hq",
    )

    reached = sum(1 for r in rows if r["best_gap"] <= 1e-6)
    write_json(
        {
            "n_images": len(rows),
            "size": args.size,
            "reached_optimum": reached,
            "config": vars(args),
        },
        RESULTS / "phase2_hq_summary.json",
    )
    print(
        f"\nHigh-res phase done: {reached}/{len(rows)} images reached the "
        f"max-flow optimum. Figures in {FIGS}/"
    )


if __name__ == "__main__":
    main()
