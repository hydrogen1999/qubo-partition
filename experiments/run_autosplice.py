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
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.segmentation import slic

from qubo_partition import viz
from qubo_partition.data.images import SeededImage
from qubo_partition.evaluation.runner import run_segmentation

DATASET_ROOT = Path(
    os.environ.get("AUTOSPLICE_ROOT", "~/Downloads/AutoSplice")
).expanduser()

TAMPERED_DIR = DATASET_ROOT / "Forged_JPEG100"
MASK_DIR = DATASET_ROOT / "Mask"

OUTPUT_DIR = Path("results/figures/autosplice")
RESULTS_CSV = Path("results/autosplice_results.csv")


def _norm(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    return np.zeros_like(a) if hi - lo < 1e-9 else (a - lo) / (hi - lo)


def _ela_map(pil_rgb: Image.Image, quality: int = 90) -> np.ndarray:
    """Error Level Analysis: tampered regions recompress differently -> they light up."""
    buf = BytesIO()
    pil_rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)
    diff = np.abs(np.asarray(pil_rgb, dtype=np.float32) - recompressed).max(axis=2)
    return _norm(diff)


def _noise_map(gray: np.ndarray) -> np.ndarray:
    """High-frequency noise residual: spliced regions carry different sensor noise."""
    return _norm(np.abs(gray - ndimage.median_filter(gray, size=3)))


def build_features(path: Path, size: int, features: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (feature_stack HxWxC for the model, rgb HxWx3 for display)."""
    pil = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    rgb = np.asarray(pil, dtype=np.float32) / 255.0
    chans = []
    if "color" in features:
        chans.append(rgb)
    if "ela" in features:
        chans.append(_ela_map(pil)[..., None])
    if "noise" in features:
        chans.append(_noise_map(rgb.mean(axis=2))[..., None])
    if not chans:  # default to color
        chans.append(rgb)
    return np.concatenate(chans, axis=2).astype(np.float32), rgb


def load_mask(path: Path, size: int) -> np.ndarray:
    # NEAREST + threshold: never interpolate a binary mask.
    mask = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return np.asarray(mask) > 127


def eroded_seeds(
    truth: np.ndarray, n_each: int, erode: int, rng, frac: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sample interior fg/bg seeds from the eroded mask (away from the boundary).

    If ``frac`` is given, seed that fraction of each region's interior (dense
    interactive scribbles that scale with region size); otherwise seed ``n_each``
    pixels per region.
    """
    fg_in = ndimage.binary_erosion(truth, iterations=erode)
    bg_in = ndimage.binary_erosion(~truth, iterations=erode)
    if not fg_in.any():
        fg_in = truth
    if not bg_in.any():
        bg_in = ~truth

    def pick(region):
        idx = np.argwhere(region)
        out = np.zeros(truth.shape, dtype=bool)
        if len(idx) == 0:
            return out
        k = max(1, int(round(frac * len(idx)))) if frac else min(n_each, len(idx))
        k = min(k, len(idx))
        for r, c in idx[rng.choice(len(idx), size=k, replace=False)]:
            out[r, c] = True
        return out

    return pick(fg_in), pick(bg_in)

def automatic_seeds(
    image: np.ndarray,
    n_segments: int = 250,
    fg_percent: float = 0.02,
    bg_percent: float = 0.10,
):
    """
    Automatically generate foreground/background seeds using
    ELA + Noise anomaly maps and SLIC superpixels.
    """

    rgb = image[:, :, :3]
    ela = image[:, :, 3]
    noise = image[:, :, 4]

    anomaly = (ela + noise) / 2.0

    segments = slic(
        rgb,
        n_segments=n_segments,
        compactness=10,
        start_label=0,
    )

    fg = np.zeros(anomaly.shape, dtype=bool)
    bg = np.zeros(anomaly.shape, dtype=bool)

    scores = []

    for seg_id in np.unique(segments):
        region = segments == seg_id
        score = anomaly[region].mean()
        scores.append((seg_id, score))

    scores.sort(key=lambda x: x[1])

    n_fg = max(1, int(len(scores) * fg_percent))
    n_bg = max(1, int(len(scores) * bg_percent))

    for seg_id, _ in scores[-n_fg:]:
        fg[segments == seg_id] = True

    for seg_id, _ in scores[:n_bg]:
        bg[segments == seg_id] = True

    return fg, bg



def collect_pairs() -> list[tuple[Path, Path]]:
    pairs = []

    for image_path in sorted(TAMPERED_DIR.glob("*.jpg")):
        stem = image_path.stem.split("_")[0]
        mask_path = MASK_DIR / f"{stem}_mask.png"

        if mask_path.exists():
            pairs.append((image_path, mask_path))

    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None, help="max images (default: all)")
    ap.add_argument("--n-seeds", type=int, default=15, help="seeds per class (count mode)")
    ap.add_argument(
        "--seed-frac",
        type=float,
        default=None,
        help="dense interactive mode: fraction of each region's interior to seed",
    )
    ap.add_argument("--erode", type=int, default=1)
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument(
        "--solver",
        choices=["maxflow", "sa", "tabu", "greedy"],
        default="maxflow",
    )
    ap.add_argument(
        "--features",
        default="color,ela,noise",
        help="comma list of cues: color, ela (Error Level Analysis), noise residual",
    )
    ap.add_argument("--num-reads", type=int, default=100)
    ap.add_argument("--num-sweeps", type=int, default=2000)
    ap.add_argument("--n-figures", type=int, default=12)
    ap.add_argument("--min-frac", type=float, default=0.01)
    ap.add_argument("--max-frac", type=float, default=0.80)
    ap.add_argument("--sample", type=int, default=None, help="random representative subset size")
    ap.add_argument("--tag", default="", help="suffix for output CSV/figures (e.g. solver name)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    global OUTPUT_DIR, RESULTS_CSV
    if args.tag:
        OUTPUT_DIR = Path(f"results/figures/casia_{args.tag}")
        RESULTS_CSV = Path(f"results/casia_results_{args.tag}.csv")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    features = {f.strip() for f in args.features.split(",") if f.strip()}

    pairs = collect_pairs()
    print(f"Found {len(pairs)} image-mask pairs under {DATASET_ROOT}")
    if args.sample is not None and args.sample < len(pairs):
        # representative random subset (avoids the alphabetical easy-prefix bias)
        idx = rng.choice(len(pairs), size=args.sample, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
        print(f"Using a random representative sample of {len(pairs)} pairs")
    seed_desc = f"seed_frac={args.seed_frac}" if args.seed_frac else f"n_seeds={args.n_seeds}"
    print(
        f"Config: size={args.size} features={sorted(features)} solver={args.solver} "
        f"lam={args.lam} {seed_desc}\n"
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

        image, rgb = build_features(image_path, args.size, features)
        fg_seeds, bg_seeds = automatic_seeds(image)
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
            disp = rgb.mean(axis=2)  # grayscale view of the original for display
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
