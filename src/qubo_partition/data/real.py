"""Real-image loaders (Weizmann horses, scikit-image demos) as seeded segmentation inputs."""

from __future__ import annotations

import glob
import os
import re

import numpy as np

from qubo_partition.data.images import SeededImage


def _to_gray(img: np.ndarray) -> np.ndarray:
    """RGB(A) or gray uint8 image -> float grayscale in [0, 1] (Rec. 601 luma)."""
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., :3].astype(float)
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    else:
        gray = img.astype(float)
    gray = gray / 255.0 if gray.max() > 1.0 else gray
    return np.clip(gray, 0.0, 1.0)


def _seeds_from_mask(
    truth: np.ndarray, n_each: int = 4, erode: int = 1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Sample interior foreground/background seeds from a binary GT mask."""
    from scipy import ndimage

    fg = truth.astype(bool)
    bg = ~fg
    fg_in = ndimage.binary_erosion(fg, iterations=erode)
    bg_in = ndimage.binary_erosion(bg, iterations=erode)
    if not fg_in.any():
        fg_in = fg
    if not bg_in.any():
        bg_in = bg

    rng = np.random.default_rng(seed)

    def _pick(region: np.ndarray, k: int) -> np.ndarray:
        idx = np.argwhere(region)
        out = np.zeros(truth.shape, dtype=bool)
        if len(idx) == 0:
            return out
        k = min(k, len(idx))
        for r, c in idx[rng.choice(len(idx), size=k, replace=False)]:
            out[r, c] = True
        return out

    return _pick(fg_in, n_each), _pick(bg_in, n_each)


def _min_max(img: np.ndarray) -> np.ndarray:
    img = img.astype(float)
    lo, hi = float(img.min()), float(img.max())
    return np.zeros_like(img) if hi - lo < 1e-9 else (img - lo) / (hi - lo)


def _otsu_foreground(gray: np.ndarray, otsu: float) -> np.ndarray:
    """Auto-detect the foreground Otsu class as the one NOT dominating the border."""
    high = gray > otsu
    border = np.zeros(gray.shape, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    frac_high_on_border = high[border].mean()
    return high if frac_high_on_border < 0.5 else ~high


_SKIMAGE_PHOTOS = ["camera", "coins", "astronaut", "clock", "cell", "coffee"]


def load_skimage_demo(
    size: int = 96,
    n_seeds: int = 6,
    erode: int = 2,
    seed: int = 0,
    photos: list[str] | None = None,
    include_synthetic: bool = True,
) -> list[SeededImage]:
    """Mixed demo images: real scikit-image photos (Otsu reference) plus synthetic shapes (exact GT)."""
    from skimage import color, data, filters, transform

    photos = list(_SKIMAGE_PHOTOS if photos is None else photos)
    items: list[SeededImage] = []
    rng = np.random.default_rng(seed)

    def _add(name: str, gray: np.ndarray, truth: np.ndarray, truth_is_gt: bool):
        fg, bg = _seeds_from_mask(truth, n_each=n_seeds, erode=erode, seed=seed)
        if fg.any() and bg.any():
            si = SeededImage(gray, truth, fg, bg, name=name)
            si.truth_is_gt = truth_is_gt  # type: ignore[attr-defined]
            items.append(si)

    def _resize(img, order=1):
        return transform.resize(
            img.astype(float),
            (size, size),
            order=order,
            anti_aliasing=(order > 0),
            preserve_range=False,
        )

    # real photographs: Otsu reference truth with auto foreground polarity
    for name in photos:
        try:
            raw = np.asarray(getattr(data, name)())
        except Exception:
            continue
        if raw.ndim == 3:
            raw = color.rgb2gray(raw[..., :3])
        gray = _min_max(_resize(raw))
        truth = _otsu_foreground(gray, filters.threshold_otsu(gray))
        _add(f"{name}_{size}", gray, truth, truth_is_gt=False)

    if not include_synthetic:
        return items

    # horse silhouette as smooth grayscale (exact GT)
    h = _resize(data.horse().astype(float), order=0) > 0.5
    horse_fg = h if h.mean() < 0.5 else ~h
    grad = np.linspace(0.0, 0.15, size)[None, :] * np.ones((size, 1))
    img_h = np.where(horse_fg, 0.25, 0.85) + grad + rng.normal(0, 0.03, (size, size))
    _add(f"horse_clean_{size}", _min_max(np.clip(img_h, 0, 1)), horse_fg, truth_is_gt=True)

    # synthetic disk (exact GT)
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= (0.32 * size) ** 2
    img_d = np.where(disk, 0.8, 0.2) + rng.normal(0, 0.04, (size, size))
    _add(f"disk_{size}", _min_max(np.clip(img_d, 0, 1)), disk, truth_is_gt=True)

    return items


def _numeric_id(path: str) -> int:
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_weizmann_horses(
    root: str,
    ids: list[int] | None = None,
    limit: int | None = None,
    size: int | None = None,
    n_seeds: int = 4,
    erode: int = 1,
    seed: int = 0,
) -> list[SeededImage]:
    """Load Weizmann horse image/mask pairs as :class:`SeededImage` objects."""
    from PIL import Image

    img_paths = {(_numeric_id(p)): p for p in glob.glob(os.path.join(root, "images", "*.bmp"))}
    msk_paths = {(_numeric_id(p)): p for p in glob.glob(os.path.join(root, "musks", "*.bmp"))}
    common = sorted(set(img_paths) & set(msk_paths))
    if not common:
        raise FileNotFoundError(
            f"no image/mask pairs under {root!r} (expected images/*.bmp and musks/*.bmp)"
        )
    if ids is not None:
        common = [i for i in ids if i in set(common)]
    elif limit is not None:
        common = common[:limit]

    out: list[SeededImage] = []
    for i in common:
        img = Image.open(img_paths[i])
        msk = Image.open(msk_paths[i])
        if size is not None and size != img.size[0]:
            img = img.resize((size, size), Image.BOX)
            msk = msk.resize((size, size), Image.NEAREST)
        gray = _to_gray(np.asarray(img))
        truth = np.asarray(msk)
        truth = (truth[..., 0] if truth.ndim == 3 else truth) > 127  # horse=255 -> True

        fg_seeds, bg_seeds = _seeds_from_mask(truth, n_each=n_seeds, erode=erode, seed=seed)
        # skip regions too small to seed after erosion
        if not fg_seeds.any() or not bg_seeds.any():
            continue
        out.append(SeededImage(gray, truth, fg_seeds, bg_seeds, name=f"horse-{i:03d}"))
    return out
