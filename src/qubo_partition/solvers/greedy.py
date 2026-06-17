"""Greedy steepest-descent baseline (multi-restart) for a QUBO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qubo_partition.qubo.base import QUBO, Variable


@dataclass
class GreedyResult:
    sample: dict[Variable, int]
    energy: float
    energies: np.ndarray  # per-restart best energies (sorted) -- comparable to SA reads


def _descend(qubo: QUBO, sample: dict[Variable, int]) -> tuple[dict[Variable, int], float]:
    current_energy = qubo.energy(sample)
    improved = True
    while improved:
        improved = False
        for var in qubo.variables:
            candidate = sample.copy()
            candidate[var] = 1 - candidate[var]
            candidate_energy = qubo.energy(candidate)
            if candidate_energy < current_energy:
                sample, current_energy = candidate, candidate_energy
                improved = True
    return sample, current_energy


def greedy_solve(
    qubo: QUBO,
    num_reads: int = 20,
    seed: int | None = None,
) -> GreedyResult:
    """Steepest descent from ``num_reads`` independent random starts.

    Returns the best assignment found and the per-restart best energies, so that
    ``GapStats`` over ``energies`` is comparable to the simulated-annealing reads.
    """
    rng = np.random.default_rng(seed)
    best_sample: dict[Variable, int] | None = None
    best_energy = np.inf
    per_restart: list[float] = []

    for _ in range(max(1, num_reads)):
        start = {v: int(rng.integers(0, 2)) for v in qubo.variables}
        sample, energy = _descend(qubo, start)
        per_restart.append(energy)
        if energy < best_energy:
            best_energy, best_sample = energy, sample

    energies = np.sort(np.asarray(per_restart, dtype=float))
    return GreedyResult(sample=best_sample, energy=float(best_energy), energies=energies)
