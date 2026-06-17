import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import resize

from qubo_partition import viz
from qubo_partition.data.images import SeededImage, auto_seeds
from qubo_partition.evaluation.runner import run_segmentation

DATASET_ROOT = Path("datasets/oxford_pet")

IMAGE_DIR = DATASET_ROOT / "images"
MASK_DIR = DATASET_ROOT / "annotations" / "trimaps"

RESULTS_DIR = Path("results/figures/oxford_pet")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGES = 3
IMAGE_SIZE = 64

image_files = sorted(IMAGE_DIR.glob("*.jpg"))[:MAX_IMAGES]

print(f"Running on {len(image_files)} images...\n")

for image_path in image_files:
    mask_path = MASK_DIR / f"{image_path.stem}.png"

    if not mask_path.exists():
        continue

    print(f"Running: {image_path.name}")

    # Load and resize image
    image = np.array(Image.open(image_path).convert("L")) / 255.0
    image = resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        anti_aliasing=True,
    )

    # Load and resize trimap
    trimap = np.array(Image.open(mask_path))
    trimap = resize(
        trimap,
        (IMAGE_SIZE, IMAGE_SIZE),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.uint8)

    # Oxford trimap: 1=pet, 2=boundary, 3=background
    truth = trimap == 1

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

    start = time.perf_counter()

    rec = run_segmentation(
        seeded,
        lambda_smooth=3.0,
        num_reads=10,
        num_sweeps=1000,
        seed=0,
    )

    runtime = time.perf_counter() - start

    print(f"  IoU={rec.iou_annealed:.3f} " f"gap={rec.gap.best_gap:.3f} " f"time={runtime:.2f}s")

    viz.plot_segmentation(
        seeded.image,
        seeded.fg_seeds,
        seeded.bg_seeds,
        rec.annealed_labels,
        rec.optimal_labels,
        truth=seeded.truth,
        title=f"{seeded.name}: IoU={rec.iou_annealed:.2f}",
        path=str(RESULTS_DIR / f"{seeded.name}.png"),
    )

print("\nDone.")
print(f"Saved figures to: {RESULTS_DIR}")
