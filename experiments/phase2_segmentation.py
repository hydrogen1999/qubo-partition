#!/usr/bin/env python3
"""Phase two experiments: image segmentation by graph cut.

Run:  python experiments/phase2_segmentation.py --size 16 --num-reads 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qubo_partition import viz
from qubo_partition.data.images import auto_seeds, benchmark_images, make_blob_image
from qubo_partition.evaluation.runner import run_segmentation
from qubo_partition.io_utils import write_csv, write_json, write_latex_table

RESULTS = Path("results")
FIGS = RESULTS / "figures"


def benchmark_suite(args) -> list[dict]:
    print("\n== Segmentation suite: anneal vs. maximum flow ==")
    rows = []
    for seeded in benchmark_images(size=args.size):
        rec = run_segmentation(
            seeded,
            lambda_smooth=args.lam,
            data_weight=args.data_weight,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
        )
        rows.append(rec.as_row())
        print(
            f"  {seeded.name:12s} {rec.shape[0]}x{rec.shape[1]} "
            f"E_anneal={rec.annealed_energy:.3f} E_opt={rec.optimal_energy:.3f} "
            f"best_gap={rec.gap.best_gap:.3f} success={rec.gap.success_rate:.2f} "
            f"IoU(anneal)={rec.iou_annealed:.2f} IoU(opt)={rec.iou_optimal:.2f}"
        )
        viz.plot_segmentation(
            seeded.image,
            seeded.fg_seeds,
            seeded.bg_seeds,
            rec.annealed_labels,
            rec.optimal_labels,
            truth=seeded.truth,
            title=f"{seeded.name}: gap={rec.gap.best_gap:.3f}, " f"IoU={rec.iou_annealed:.2f}",
            path=str(FIGS / f"seg_{seeded.name}.png"),
        )
    write_csv(rows, RESULTS / "phase2_benchmarks.csv")
    write_latex_table(
        rows,
        columns=[
            "name",
            "n_pixels",
            "annealed_energy",
            "optimal_energy",
            "best_gap",
            "mean_gap",
            "success_rate",
            "iou_annealed",
        ],
        path=RESULTS / "tables" / "phase2_benchmarks.tex",
        caption="Seeded segmentation: simulated annealing vs. maximum-flow optimum.",
        label="tab:seg-bench",
    )
    return rows


def lambda_study(args) -> list[dict]:
    print("\n== Smoothness-weight (lambda) study ==")
    seeded = make_blob_image(size=args.size, seed=0)
    rows = []
    succ, gaps = [], []
    lambdas = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    for lam in lambdas:
        rec = run_segmentation(
            seeded,
            lambda_smooth=lam,
            data_weight=args.data_weight,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
        )
        rows.append(
            {
                "lambda": lam,
                **rec.gap.as_dict(),
                "iou_annealed": rec.iou_annealed,
                "iou_optimal": rec.iou_optimal,
            }
        )
        succ.append(rec.gap.success_rate)
        gaps.append(rec.gap.best_gap)
        print(
            f"  lambda={lam:4.2f}: best_gap={rec.gap.best_gap:.3f} "
            f"success={rec.gap.success_rate:.2f} IoU(opt)={rec.iou_optimal:.2f}"
        )
    viz.plot_success_vs_param(
        lambdas,
        succ,
        xlabel="smoothness weight $\\lambda$",
        title="Segmentation: success rate vs. smoothness weight",
        path=str(FIGS / "seg_lambda_success.png"),
    )
    write_csv(rows, RESULTS / "phase2_lambda_study.csv")
    return rows


def seed_sensitivity(args) -> list[dict]:
    print("\n== Seed-sensitivity study ==")
    base = make_blob_image(size=args.size, seed=0)
    rows = []
    for n_each in [1, 2, 3, 5, 8]:
        ious, gaps, succ = [], [], []
        for trial in range(args.seed_trials):
            fg, bg = auto_seeds(base.truth, n_each=n_each, seed=100 + trial)
            seeded = base.__class__(base.image, base.truth, fg, bg, name=f"seeds{n_each}_t{trial}")
            rec = run_segmentation(
                seeded,
                lambda_smooth=args.lam,
                data_weight=args.data_weight,
                num_reads=args.num_reads,
                num_sweeps=args.num_sweeps,
                seed=args.seed,
            )
            ious.append(rec.iou_optimal)
            gaps.append(rec.gap.best_gap)
            succ.append(rec.gap.success_rate)
        row = {
            "seeds_per_class": n_each,
            "mean_iou_optimal": float(np.mean(ious)),
            "std_iou_optimal": float(np.std(ious)),
            "mean_best_gap": float(np.mean(gaps)),
            "mean_success_rate": float(np.mean(succ)),
        }
        rows.append(row)
        print(
            f"  seeds/class={n_each}: IoU(opt)={row['mean_iou_optimal']:.2f}"
            f"+/-{row['std_iou_optimal']:.2f} mean_gap={row['mean_best_gap']:.3f}"
        )
    write_csv(rows, RESULTS / "phase2_seed_sensitivity.csv")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=16, choices=[16, 32])
    ap.add_argument("--num-reads", type=int, default=200)
    ap.add_argument("--num-sweeps", type=int, default=2000)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--data-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-trials", type=int, default=4)
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)

    bench = benchmark_suite(args)
    lambda_study(args)
    seed_sensitivity(args)

    n_opt = sum(1 for r in bench if r["best_gap"] <= 1e-6)
    write_json(
        {"benchmarks": len(bench), "reached_optimum": n_opt, "config": vars(args)},
        RESULTS / "phase2_summary.json",
    )
    print(f"\nPhase 2 done: {n_opt}/{len(bench)} images reached the max-flow optimum.")


if __name__ == "__main__":
    main()
