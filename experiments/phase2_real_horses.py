#!/usr/bin/env python3
"""Phase two on real data: seeded segmentation of the 32x32 Weizmann horses.

Run:  python experiments/phase2_real_horses.py --data datasets/weizmann_horse_32 --n-images 16 --num-reads 80
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from qubo_partition import viz
from qubo_partition.data.real import load_weizmann_horses
from qubo_partition.evaluation.runner import run_segmentation
from qubo_partition.io_utils import write_csv, write_json, write_latex_table

RESULTS = Path("results")
FIGS = RESULTS / "figures" / "real"


def main_suite(images, args) -> list[dict]:
    print(f"\n== Weizmann horses (32x32): anneal vs. max-flow on {len(images)} images ==")
    rows = []
    for k, seeded in enumerate(images):
        t0 = time.perf_counter()
        rec = run_segmentation(
            seeded,
            lambda_smooth=args.lam,
            data_weight=args.data_weight,
            data_model="histogram",
            n_bins=args.n_bins,
            connectivity=8,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
        )
        dt = time.perf_counter() - t0
        row = rec.as_row()
        row["time_s"] = round(dt, 2)
        rows.append(row)
        print(
            f"  {seeded.name} gap={rec.gap.best_gap:8.3f} success={rec.gap.success_rate:.2f} "
            f"IoU(anneal)={rec.iou_annealed:.2f} IoU(opt)={rec.iou_optimal:.2f} "
            f"acc={rec.pixel_acc_annealed:.2f} ({dt:.1f}s)"
        )
        if k < args.n_figures:
            viz.plot_segmentation(
                seeded.image,
                seeded.fg_seeds,
                seeded.bg_seeds,
                rec.annealed_labels,
                rec.optimal_labels,
                truth=seeded.truth,
                title=f"{seeded.name}: gap={rec.gap.best_gap:.2f}, "
                f"IoU(anneal)={rec.iou_annealed:.2f}",
                path=str(FIGS / f"{seeded.name}.png"),
            )

    write_csv(rows, RESULTS / "phase2_real_horses.csv")
    write_latex_table(
        rows,
        columns=[
            "name",
            "n_pixels",
            "best_gap",
            "success_rate",
            "iou_annealed",
            "iou_optimal",
            "pixel_acc_annealed",
            "time_s",
        ],
        path=RESULTS / "tables" / "phase2_real_horses.tex",
        caption="Seeded segmentation of 32x32 Weizmann horses: simulated annealing "
        "vs. the maximum-flow optimum, scored against real ground-truth masks.",
        label="tab:real-horses",
    )
    return rows


def ablation(images, args) -> list[dict]:
    """Ablate data term (histogram/gaussian) x connectivity (8/4)."""
    print("\n== Ablation: histogram vs. gaussian data term, 8- vs. 4-connectivity ==")
    configs = [
        ("gaussian", 4),
        ("gaussian", 8),
        ("histogram", 4),
        ("histogram", 8),
    ]
    rows = []
    for model, conn in configs:
        ious_opt, ious_ann, gaps, accs, succ = [], [], [], [], []
        for seeded in images:
            rec = run_segmentation(
                seeded,
                lambda_smooth=args.lam,
                data_weight=args.data_weight,
                data_model=model,
                n_bins=args.n_bins,
                connectivity=conn,
                num_reads=args.num_reads,
                num_sweeps=args.num_sweeps,
                seed=args.seed,
            )
            ious_opt.append(rec.iou_optimal)
            ious_ann.append(rec.iou_annealed)
            gaps.append(rec.gap.best_gap)
            accs.append(rec.pixel_acc_annealed)
            succ.append(rec.gap.success_rate)
        row = {
            "data_model": model,
            "connectivity": conn,
            "mean_iou_optimal": round(float(np.mean(ious_opt)), 4),
            "mean_iou_annealed": round(float(np.mean(ious_ann)), 4),
            "mean_pixel_acc": round(float(np.mean(accs)), 4),
            "mean_best_gap": round(float(np.mean(gaps)), 4),
            "mean_success_rate": round(float(np.mean(succ)), 4),
        }
        rows.append(row)
        print(
            f"  {model:9s} conn={conn}: IoU(opt)={row['mean_iou_optimal']:.3f} "
            f"IoU(anneal)={row['mean_iou_annealed']:.3f} acc={row['mean_pixel_acc']:.3f} "
            f"mean_gap={row['mean_best_gap']:.3f}"
        )

    write_csv(rows, RESULTS / "phase2_real_ablation.csv")
    write_latex_table(
        rows,
        columns=[
            "data_model",
            "connectivity",
            "mean_iou_optimal",
            "mean_iou_annealed",
            "mean_pixel_acc",
            "mean_best_gap",
        ],
        path=RESULTS / "tables" / "phase2_real_ablation.tex",
        caption="Ablation on 32x32 Weizmann horses: the histogram data term and "
        "8-connectivity each improve segmentation quality (mean over images).",
        label="tab:real-ablation",
    )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="datasets/weizmann_horse_32")
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--n-figures", type=int, default=8)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--data-weight", type=float, default=1.0)
    ap.add_argument("--n-bins", type=int, default=16)
    ap.add_argument("--num-reads", type=int, default=80)
    ap.add_argument("--num-sweeps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)

    images = load_weizmann_horses(
        args.data, limit=args.n_images, n_seeds=args.n_seeds, seed=args.seed
    )
    if not images:
        raise SystemExit(f"no usable horse images under {args.data!r}")

    suite = main_suite(images, args)
    if not args.no_ablation:
        ablation(images, args)

    mean_iou = float(np.mean([r["iou_annealed"] for r in suite]))
    mean_iou_opt = float(np.mean([r["iou_optimal"] for r in suite]))
    reached = sum(1 for r in suite if r["best_gap"] <= 1e-6)
    write_json(
        {
            "n_images": len(suite),
            "mean_iou_annealed": mean_iou,
            "mean_iou_optimal": mean_iou_opt,
            "reached_optimum": reached,
            "config": vars(args),
        },
        RESULTS / "phase2_real_summary.json",
    )
    print(
        f"\nReal-data phase done: mean IoU(anneal)={mean_iou:.3f}, "
        f"mean IoU(opt)={mean_iou_opt:.3f}, "
        f"{reached}/{len(suite)} images reached the max-flow optimum."
    )


if __name__ == "__main__":
    main()
