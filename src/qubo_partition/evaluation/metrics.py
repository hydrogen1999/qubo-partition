"""Validity and optimality-gap metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def optimality_gap(annealed_energy: float, optimal_energy: float) -> float:
    """Absolute energy gap ``E_annealed - E_optimal`` (>= 0 for a true optimum)."""
    return float(annealed_energy - optimal_energy)


def relative_gap(annealed_energy: float, optimal_energy: float, eps: float = 1e-9) -> float:
    """Gap normalized by ``|E_optimal|``; 0 means the optimum was reached."""
    denom = abs(optimal_energy)
    if denom < eps:
        return float(annealed_energy - optimal_energy)
    return float((annealed_energy - optimal_energy) / denom)


@dataclass
class GapStats:
    """Best/mean/spread of the gap across reads, plus success rate."""

    best_gap: float
    mean_gap: float
    std_gap: float
    worst_gap: float
    success_rate: float
    n_reads: int

    @classmethod
    def from_energies(
        cls, read_energies: np.ndarray, optimal_energy: float, atol: float = 1e-6
    ) -> GapStats:
        gaps = np.asarray(read_energies, dtype=float) - optimal_energy
        # tiny gaps are float noise on the optimum
        gaps = np.where(np.abs(gaps) < atol, 0.0, gaps)
        success = float(np.mean(gaps <= atol))
        return cls(
            best_gap=float(gaps.min()),
            mean_gap=float(gaps.mean()),
            std_gap=float(gaps.std()),
            worst_gap=float(gaps.max()),
            success_rate=success,
            n_reads=int(len(gaps)),
        )

    def as_dict(self) -> dict:
        return {
            "best_gap": self.best_gap,
            "mean_gap": self.mean_gap,
            "std_gap": self.std_gap,
            "worst_gap": self.worst_gap,
            "success_rate": self.success_rate,
            "n_reads": self.n_reads,
        }


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    """Intersection-over-union between two boolean masks."""
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def pixel_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of pixels whose predicted label matches the ground-truth region."""
    return float((pred.astype(bool) == truth.astype(bool)).mean())
