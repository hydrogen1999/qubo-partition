"""Synthetic grayscale images and seed masks for the segmentation phase."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SeededImage:
    """A grayscale image with its ground-truth region and user seeds."""

    image: np.ndarray  # HxW float in [0, 1]
    truth: np.ndarray  # HxW bool, the foreground region used to render
    fg_seeds: np.ndarray  # HxW bool
    bg_seeds: np.ndarray  # HxW bool
    name: str = "image"

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape  # type: ignore[return-value]


def _normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(float)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-9:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def make_blob_image(
    size: int = 16,
    fg_level: float = 0.8,
    bg_level: float = 0.2,
    noise: float = 0.08,
    radius_frac: float = 0.3,
    seed: int = 0,
) -> SeededImage:
    """A bright circular blob on a dark background, with additive Gaussian noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    r = radius_frac * size
    truth = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r

    img = np.where(truth, fg_level, bg_level).astype(float)
    img = img + rng.normal(0.0, noise, size=(size, size))
    img = _normalize(np.clip(img, 0.0, 1.0))

    fg_seeds, bg_seeds = auto_seeds(truth, n_each=max(2, size // 8), seed=seed)
    return SeededImage(img, truth, fg_seeds, bg_seeds, name=f"blob_{size}")


def make_two_region_image(
    size: int = 16,
    fg_level: float = 0.75,
    bg_level: float = 0.25,
    noise: float = 0.08,
    seed: int = 0,
) -> SeededImage:
    """A foreground square in one corner over a contrasting background."""
    rng = np.random.default_rng(seed)
    truth = np.zeros((size, size), dtype=bool)
    s0 = size // 4
    s1 = size - size // 4
    truth[s0:s1, s0:s1] = True

    img = np.where(truth, fg_level, bg_level).astype(float)
    img = img + rng.normal(0.0, noise, size=(size, size))
    img = _normalize(np.clip(img, 0.0, 1.0))

    fg_seeds, bg_seeds = auto_seeds(truth, n_each=max(2, size // 8), seed=seed)
    return SeededImage(img, truth, fg_seeds, bg_seeds, name=f"square_{size}")


def auto_seeds(truth: np.ndarray, n_each: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n_each`` foreground and background seed pixels from a region mask."""
    rng = np.random.default_rng(seed)
    fg_idx = np.argwhere(truth)
    bg_idx = np.argwhere(~truth)

    def _pick(idx, k):
        if len(idx) == 0:
            return np.zeros(truth.shape, dtype=bool)
        k = min(k, len(idx))
        chosen = idx[rng.choice(len(idx), size=k, replace=False)]
        m = np.zeros(truth.shape, dtype=bool)
        for r, c in chosen:
            m[r, c] = True
        return m

    return _pick(fg_idx, n_each), _pick(bg_idx, n_each)


def benchmark_images(size: int = 16) -> list[SeededImage]:
    """A small fixed suite of seeded images used across the experiments."""
    return [
        make_blob_image(size=size, seed=0),
        make_two_region_image(size=size, seed=1),
        make_blob_image(size=size, noise=0.15, seed=2),
    ]
