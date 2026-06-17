"""Seeded graph-cut segmentation QUBO (Boykov--Jolly): data term plus pairwise smoothness."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qubo_partition.qubo.base import QUBO

Pixel = tuple[int, int]  # (row, col)


def neighbor_offsets(connectivity: int) -> list[Pixel]:
    """Forward neighbor offsets (each undirected edge listed once)."""
    if connectivity == 4:
        return [(0, 1), (1, 0)]
    if connectivity == 8:
        return [(0, 1), (1, 0), (1, 1), (1, -1)]
    raise ValueError("connectivity must be 4 or 8")


def grid_edges(height: int, width: int, connectivity: int = 4) -> list[tuple[Pixel, Pixel]]:
    """Enumerate the undirected neighbor pairs of an ``height x width`` pixel grid."""
    edges: list[tuple[Pixel, Pixel]] = []
    offs = neighbor_offsets(connectivity)
    for r in range(height):
        for c in range(width):
            for dr, dc in offs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    edges.append(((r, c), (rr, cc)))
    return edges


def estimate_sigma(image: np.ndarray, connectivity: int = 4) -> float:
    """Robust contrast scale: RMS of neighboring (vector) intensity differences."""
    h, w = image.shape[:2]
    d2 = [
        float(np.sum((image[i] - image[j]) ** 2))
        for i, j in grid_edges(h, w, connectivity=connectivity)
    ]
    if not d2:
        return 1.0
    return max(float(np.sqrt(np.mean(d2))), 1e-3)


def boykov_jolly_weights(
    image: np.ndarray,
    lambda_smooth: float = 1.0,
    sigma: float | None = None,
    connectivity: int = 4,
) -> dict[tuple[Pixel, Pixel], float]:
    """Smoothness weights, large for similar neighbors and small for dissimilar ones.

    Works for grayscale (H x W) and color (H x W x C) images; the contrast is the
    squared Euclidean distance between neighboring pixel vectors.
    """
    if sigma is None:
        sigma = estimate_sigma(image, connectivity)
    inv = 1.0 / (2.0 * sigma * sigma)
    h, w = image.shape[:2]
    weights: dict[tuple[Pixel, Pixel], float] = {}
    for i, j in grid_edges(h, w, connectivity=connectivity):
        d2 = float(np.sum((image[i] - image[j]) ** 2))
        weights[(i, j)] = float(lambda_smooth * np.exp(-d2 * inv))
    return weights


def _gaussian_neglog(values: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Negative log-likelihood of a 1-D Gaussian, up to the additive 0.5*log(2pi)."""
    sigma = max(sigma, 1e-2)
    z = (values - mu) / sigma
    return 0.5 * z * z + np.log(sigma)


def _histogram_neglog(
    image: np.ndarray, seed_values: np.ndarray, n_bins: int = 16, smoothing: float = 1.0
) -> np.ndarray:
    """Neg-log-likelihood under an intensity histogram fit to seed pixels.

    Color images (H x W x C) are handled per channel under a naive-Bayes
    (channel-independence) assumption: the costs sum across channels.
    """
    if image.ndim == 3:
        cost = np.zeros(image.shape[:2])
        for ch in range(image.shape[2]):
            cost += _histogram_neglog(image[..., ch], seed_values[:, ch], n_bins, smoothing)
        return cost
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(np.clip(seed_values, 0.0, 1.0), bins=edges)
    probs = (counts + smoothing) / (counts.sum() + smoothing * n_bins)
    # bin index of every pixel (clip the rightmost edge into the last bin)
    idx = np.clip(np.digitize(np.clip(image, 0.0, 1.0), edges) - 1, 0, n_bins - 1)
    return -np.log(probs[idx])


def data_costs_from_seeds(
    image: np.ndarray,
    fg_seeds: np.ndarray,
    bg_seeds: np.ndarray,
    data_weight: float = 1.0,
    seed_penalty: float = 1.0e6,
    model: str = "histogram",
    n_bins: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel data costs D0 (background) and D1 (foreground), with seeds pinned."""
    if not fg_seeds.any() or not bg_seeds.any():
        raise ValueError(
            "Boykov--Jolly data costs need at least one foreground and one "
            "background seed to fit intensity models."
        )

    fg_vals = image[fg_seeds]
    bg_vals = image[bg_seeds]

    if model == "histogram":
        D1 = data_weight * _histogram_neglog(image, fg_vals, n_bins=n_bins)
        D0 = data_weight * _histogram_neglog(image, bg_vals, n_bins=n_bins)
    elif model == "gaussian":
        D1 = data_weight * _gaussian_neglog(image, float(fg_vals.mean()), float(fg_vals.std()))
        D0 = data_weight * _gaussian_neglog(image, float(bg_vals.mean()), float(bg_vals.std()))
    else:
        raise ValueError(f"unknown data model {model!r}; use 'histogram' or 'gaussian'")

    # Hard seed constraints overwrite the regional term.
    D0 = D0.copy()
    D1 = D1.copy()
    D0[fg_seeds] = seed_penalty  # fg seed: labeling it background is forbidden
    D1[fg_seeds] = 0.0
    D1[bg_seeds] = seed_penalty  # bg seed: labeling it foreground is forbidden
    D0[bg_seeds] = 0.0
    return D0, D1


@dataclass
class SegmentationModel:
    """A fully specified Boykov--Jolly energy over a fixed-size image."""

    image: np.ndarray
    fg_seeds: np.ndarray
    bg_seeds: np.ndarray
    D0: np.ndarray
    D1: np.ndarray
    weights: dict[tuple[Pixel, Pixel], float]
    connectivity: int = 4
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[:2]  # type: ignore[return-value]

    @property
    def pixels(self) -> list[Pixel]:
        h, w = self.shape
        return [(r, c) for r in range(h) for c in range(w)]

    def energy(self, labels: np.ndarray) -> float:
        """Energy of a full label image ``labels`` (same shape as ``image``)."""
        labels = labels.astype(int)
        e = float(np.where(labels == 1, self.D1, self.D0).sum())
        for (i, j), w in self.weights.items():
            if labels[i] != labels[j]:
                e += w
        return e

    def sample_to_labels(self, sample) -> np.ndarray:
        """Reshape a ``(r, c) -> bit`` QUBO sample into a label image."""
        h, w = self.shape
        out = np.zeros((h, w), dtype=np.int8)
        for (r, c), bit in sample.items():
            out[r, c] = int(bit)
        return out

    def to_qubo(self) -> QUBO:
        """Assemble the closed-form QUBO."""
        qubo = QUBO()
        h, w = self.shape

        # Unary part: a_i = (D1 - D0)_i, offset += sum D0.
        for r in range(h):
            for c in range(w):
                qubo.add_linear((r, c), float(self.D1[r, c] - self.D0[r, c]))
        qubo.add_offset(float(self.D0.sum()))

        # Pairwise part: w_ij (x_i + x_j - 2 x_i x_j).
        for (i, j), wij in self.weights.items():
            qubo.add_linear(i, wij)
            qubo.add_linear(j, wij)
            qubo.add_quadratic(i, j, -2.0 * wij)
        return qubo


def segmentation_qubo(
    image: np.ndarray,
    fg_seeds: np.ndarray,
    bg_seeds: np.ndarray,
    lambda_smooth: float = 1.0,
    sigma: float | None = None,
    data_weight: float = 1.0,
    seed_penalty: float = 1.0e6,
    connectivity: int = 4,
    data_model: str = "histogram",
    n_bins: int = 16,
) -> tuple[QUBO, SegmentationModel]:
    """Convenience builder: image + seeds -> (QUBO, SegmentationModel)."""
    D0, D1 = data_costs_from_seeds(
        image,
        fg_seeds,
        bg_seeds,
        data_weight=data_weight,
        seed_penalty=seed_penalty,
        model=data_model,
        n_bins=n_bins,
    )
    weights = boykov_jolly_weights(
        image, lambda_smooth=lambda_smooth, sigma=sigma, connectivity=connectivity
    )
    model = SegmentationModel(
        image=image,
        fg_seeds=fg_seeds,
        bg_seeds=bg_seeds,
        D0=D0,
        D1=D1,
        weights=weights,
        connectivity=connectivity,
        meta={
            "lambda_smooth": lambda_smooth,
            "sigma": sigma if sigma is not None else estimate_sigma(image, connectivity),
            "data_weight": data_weight,
            "seed_penalty": seed_penalty,
            "data_model": data_model,
            "n_bins": n_bins,
            "connectivity": connectivity,
        },
    )
    return model.to_qubo(), model
