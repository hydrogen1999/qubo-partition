#!/usr/bin/env python3
"""Run the seeded graph-cut method on the CASIA v2.0 tampering dataset.

For high segmentation quality we (1) use COLOR (RGB) data terms -- CASIA splices
differ in colour, not just grayscale intensity; (2) seed from the eroded
interior of the ground-truth mask (clean interactive seeds); and (3) report the
EXACT graph-cut optimum (maximum flow), decoupled from the annealer's gap.

Run (full dataset, exact graph cut, colour):
  CASIA_ROOT=datasets/casia_v2 python experiments/run_casia.py --size 96
Subset / annealer variant:
  python experiments/run_casia.py --limit 30 --solver sa
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from qubo_partition import viz
from qubo_partition.data.images import SeededImage
from qubo_partition.evaluation.runner import run_segmentation

DATASET_ROOT = Path(os.environ.get("CASIA_ROOT", "datasets/casia_v2"))
TAMPERED_DIR = DATASET_ROOT / "Tp"
MASK_DIR = next(
    (
        DATASET_ROOT / d
        for d in ("Gt", "CASIA 2 Groundtruth", "Groundtruth")
        if (DATASET_ROOT / d).is_dir()
    ),
    DATASET_ROOT / "Gt",
)
OUTPUT_DIR = Path("results/figures/casia")
RESULTS_CSV = Path("results/casia_results.csv")


def load_image(path: Path, size: int, color: bool) -> np.ndarray:
    mode = "RGB" if color else "L"
    img = Image.open(path).convert(mode).resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(path: Path, size: int) -> np.ndarray:
    # NEAREST + threshold: never interpolate a binary mask.
    mask = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return np.asarray(mask) > 127


def eroded_seeds(truth: np.ndarray, n_each: int, erode: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Sample interior fg/bg seeds from the eroded mask (away from the boundary)."""
    fg_in = ndimage.binary_erosion(truth, iterations=erode)
    bg_in = ndimage.binary_erosion(~truth, iterations=erode)
    if not fg_in.any():
        fg_in = truth
    if not bg_in.any():
        bg_in = ~truth

    def pick(region):
        idx = np.argwhere(region)
        out = np.zeros(truth.shape, dtype=bool)
        k = min(n_each, len(idx))
        for r, c in idx[rng.choice(len(idx), size=k, replace=False)]:
            out[r, c] = True
        return out

    return pick(fg_in), pick(bg_in)


def collect_pairs() -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in sorted(TAMPERED_DIR.iterdir()):
        if image_path.name == "Thumbs.db":
            continue
        mask_path = MASK_DIR / f"{image_path.stem}_gt.png"
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None, help="max images (default: all)")
    ap.add_argument("--n-seeds", type=int, default=15, help="seeds per class")
    ap.add_argument("--erode", type=int, default=1)
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--solver", choices=["maxflow", "sa"], default="maxflow")
    ap.add_argument("--color", choices=["rgb", "gray"], default="rgb")
    ap.add_argument("--num-reads", type=int, default=100)
    ap.add_argument("--num-sweeps", type=int, default=2000)
    ap.add_argument("--n-figures", type=int, default=12)
    ap.add_argument("--min-frac", type=float, default=0.01)
    ap.add_argument("--max-frac", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pairs = collect_pairs()
    print(f"Found {len(pairs)} image-mask pairs under {DATASET_ROOT}")
    print(
        f"Config: size={args.size} color={args.color} solver={args.solver} "
        f"lam={args.lam} n_seeds={args.n_seeds}\n"
    )

    rows = []
    processed = 0
    n_fig = 0
    t_start = time.time()

    for image_path, mask_path in pairs:
        if args.limit is not None and processed >= args.limit:
            break
        truth = load_mask(mask_path, args.size)
        frac = float(truth.mean())
        if frac < args.min_frac or frac > args.max_frac:
            continue

        image = load_image(image_path, args.size, color=(args.color == "rgb"))
        fg_seeds, bg_seeds = eroded_seeds(truth, args.n_seeds, args.erode, rng)
        if not fg_seeds.any() or not bg_seeds.any():
            continue

        seeded = SeededImage(
            image=image, truth=truth, fg_seeds=fg_seeds, bg_seeds=bg_seeds, name=image_path.stem
        )

        t0 = time.time()
        rec = run_segmentation(
            seeded,
            lambda_smooth=args.lam,
            data_model="histogram",
            connectivity=8,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
            solver=args.solver,
        )
        dt = time.time() - t0
        iou_val = rec.iou_optimal if args.solver == "maxflow" else rec.iou_annealed
        rows.append(
            {
                "image": image_path.name,
                "iou": round(iou_val, 4),
                "iou_optimal": round(rec.iou_optimal, 4),
                "gap": round(rec.gap.best_gap, 4),
                "runtime": round(dt, 3),
            }
        )
        processed += 1

        if n_fig < args.n_figures:
            disp = seeded.image if args.color == "gray" else seeded.image.mean(axis=2)
            viz.plot_segmentation(
                disp,
                seeded.fg_seeds,
                seeded.bg_seeds,
                rec.annealed_labels,
                rec.optimal_labels,
                truth=seeded.truth,
                title=f"{seeded.name}: IoU={iou_val:.2f}",
                path=str(OUTPUT_DIR / f"{seeded.name}.png"),
            )
            n_fig += 1

        if processed % 50 == 0:
            mean_iou = float(np.mean([r["iou"] for r in rows]))
            print(
                f"  [{processed}] {image_path.name[:40]:40s} IoU={iou_val:.3f} "
                f"(running mean {mean_iou:.3f}, {time.time()-t_start:.0f}s)"
            )
            _save_csv(rows)

    _save_csv(rows)
    if rows:
        ious = [r["iou"] for r in rows]
        print(f"\nDone. Processed {len(rows)} images in {time.time()-t_start:.0f}s.")
        print(
            f"mean IoU = {np.mean(ious):.4f} | median = {np.median(ious):.4f} | "
            f"max = {np.max(ious):.4f} | >=0.5: {np.mean(np.array(ious) >= 0.5):.1%}"
        )
    print(f"Saved: {RESULTS_CSV}  |  figures: {OUTPUT_DIR}")


def _save_csv(rows):
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "iou", "iou_optimal", "gap", "runtime"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
