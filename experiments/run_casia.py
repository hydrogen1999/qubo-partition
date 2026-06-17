import csv
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from qubo_partition import viz
from qubo_partition.data.images import SeededImage, auto_seeds
from qubo_partition.evaluation.runner import run_segmentation

# =========================
# Dataset paths (override the root with the CASIA_ROOT env var if needed)
# =========================
DATASET_ROOT = Path(os.environ.get("CASIA_ROOT", "datasets/casia_v2"))

TAMPERED_DIR = DATASET_ROOT / "Tp"
# the groundtruth folder is named "Gt" (corrected release) or "CASIA 2 Groundtruth" (Kaggle)
MASK_DIR = next(
    (DATASET_ROOT / d for d in ("Gt", "CASIA 2 Groundtruth", "Groundtruth") if (DATASET_ROOT / d).is_dir()),
    DATASET_ROOT / "Gt",
)

OUTPUT_DIR = Path("results/figures/casia")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = Path("results/casia_results.csv")

MAX_IMAGES = 20


# =========================
# Helper functions
# =========================


def load_image(path: Path, size: int = 64) -> np.ndarray:
    # bilinear is fine for a grayscale intensity image
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(path: Path, size: int = 64) -> np.ndarray:
    # NEAREST + threshold: never interpolate a binary mask (would blur its edges,
    # inflate the foreground, and corrupt IoU / seed placement).
    mask = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return np.asarray(mask) > 127


def mask_fraction(mask: np.ndarray) -> float:
    return float(mask.mean())


# =========================
# Collect image-mask pairs
# =========================

pairs = []

for image_path in sorted(TAMPERED_DIR.iterdir()):

    if image_path.name == "Thumbs.db":
        continue

    stem = image_path.stem
    mask_name = f"{stem}_gt.png"
    mask_path = MASK_DIR / mask_name

    if mask_path.exists():
        pairs.append((image_path, mask_path))

print(f"Found {len(pairs)} image-mask pairs")


# =========================
# Run experiment
# =========================

rows = []

print(f"\nRunning on {MAX_IMAGES} tampered images...\n")

processed = 0

for image_path, mask_path in pairs:

    if processed >= MAX_IMAGES:
        break

    image = load_image(image_path)
    truth = load_mask(mask_path)

    frac = mask_fraction(truth)

    # Skip outliers
    if frac < 0.01 or frac > 0.80:
        print(f"Skipping {image_path.name} (outlier)")
        continue

    fg_seeds, bg_seeds = auto_seeds(
        truth,
        n_each=5,
        seed=0,
    )

    seeded = SeededImage(
        image=image,
        truth=truth,
        fg_seeds=fg_seeds,
        bg_seeds=bg_seeds,
        name=image_path.stem,
    )

    print(f"Running: {image_path.name}")

    start = time.time()

    rec = run_segmentation(
        seeded,
        lambda_smooth=4.0,
        num_reads=200,
        num_sweeps=2000,
        seed=0,
        solver="sa",
    )

    elapsed = time.time() - start

    print(f"  IoU={rec.iou_annealed:.3f} " f"gap={rec.gap.best_gap:.3f} " f"time={elapsed:.2f}s")

    rows.append(
        {
            "image": image_path.name,
            "iou": round(rec.iou_annealed, 4),
            "gap": round(rec.gap.best_gap, 4),
            "runtime": round(elapsed, 2),
        }
    )

    viz.plot_segmentation(
        seeded.image,
        seeded.fg_seeds,
        seeded.bg_seeds,
        rec.annealed_labels,
        rec.optimal_labels,
        truth=seeded.truth,
        title=f"{seeded.name}: IoU={rec.iou_annealed:.2f}",
        path=str(OUTPUT_DIR / f"{seeded.name}.png"),
    )

    processed += 1


# =========================
# Save CSV
# =========================

with open(RESULTS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "iou", "gap", "runtime"])

    writer.writeheader()
    writer.writerows(rows)

print("\nDone.")
print(f"Saved figures to: {OUTPUT_DIR}")
print(f"Saved results to: {RESULTS_CSV}")
