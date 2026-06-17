from pathlib import Path
import csv
import time

import cv2
import numpy as np

from qubo_partition.data.images import SeededImage, auto_seeds
from qubo_partition.evaluation.runner import run_segmentation

DATASET_ROOT = Path("datasets/casia_v2/CASIA2")

TAMPERED_DIR = DATASET_ROOT / "Tp"
MASK_DIR = DATASET_ROOT / "CASIA 2 Groundtruth"

LAMBDA_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]

MAX_IMAGES = 20

RESULTS_PATH = Path("results/lambda_tuning.csv")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    image = cv2.resize(image, (64, 64))
    image = image.astype(np.float32) / 255.0

    return image


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return None

    mask = cv2.resize(mask, (64, 64))
    mask = mask > 0

    return mask


pairs = []

for image_path in sorted(TAMPERED_DIR.iterdir()):

    if image_path.name == "Thumbs.db":
        continue

    mask_path = MASK_DIR / f"{image_path.stem}_gt.png"

    if mask_path.exists():
        pairs.append((image_path, mask_path))

pairs = pairs[:MAX_IMAGES]

print(f"Running lambda tuning on {len(pairs)} images")

rows = []

for lam in LAMBDA_VALUES:

    print(f"\nTesting lambda = {lam}")

    ious = []
    gaps = []
    runtimes = []

    for image_path, mask_path in pairs:

        image = load_image(image_path)
        truth = load_mask(mask_path)

        if image is None or truth is None:
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

        start = time.time()

        rec = run_segmentation(
            seeded,
            lambda_smooth=lam,
            num_reads=200,
            num_sweeps=2000,
            seed=0,
        )

        runtime = time.time() - start

        ious.append(rec.iou_annealed)
        gaps.append(rec.gap.best_gap)
        runtimes.append(runtime)

    avg_iou = float(np.mean(ious))
    avg_gap = float(np.mean(gaps))
    avg_runtime = float(np.mean(runtimes))

    print(
        f"Average IoU = {avg_iou:.3f} | "
        f"Gap = {avg_gap:.3f} | "
        f"Runtime = {avg_runtime:.2f}s"
    )

    rows.append(
        {
            "lambda": lam,
            "avg_iou": round(avg_iou, 4),
            "avg_gap": round(avg_gap, 4),
            "avg_runtime": round(avg_runtime, 2),
        }
    )

with open(RESULTS_PATH, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "lambda",
            "avg_iou",
            "avg_gap",
            "avg_runtime",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved results to {RESULTS_PATH}")