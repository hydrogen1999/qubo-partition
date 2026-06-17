"""Tabu-search baseline (multi-restart) for a QUBO."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from qubo_partition.qubo.base import QUBO, Variable


@dataclass
class TabuResult:
    sample: dict[Variable, int]
    energy: float
    energies: np.ndarray  # per-restart best energies (sorted) -- comparable to SA reads


def _tabu_run(
    qubo: QUBO, start: dict[Variable, int], max_iters: int, tabu_size: int
) -> tuple[dict[Variable, int], float]:
    current = start
    current_energy = qubo.energy(current)
    best, best_energy = current.copy(), current_energy
    tabu_list: deque = deque(maxlen=tabu_size)

    for _ in range(max_iters):
        best_neighbor = None
        best_neighbor_energy = np.inf
        best_move = None
        for var in qubo.variables:
            if var in tabu_list:
                continue
            candidate = current.copy()
            candidate[var] = 1 - candidate[var]
            candidate_energy = qubo.energy(candidate)
            if candidate_energy < best_neighbor_energy:
                best_neighbor = candidate
                best_neighbor_energy = candidate_energy
                best_move = var
        if best_neighbor is None:
            break
        current, current_energy = best_neighbor, best_neighbor_energy
        tabu_list.append(best_move)
        if current_energy < best_energy:
            best, best_energy = current.copy(), current_energy
    return best, best_energy


def tabu_solve(
    qubo: QUBO,
    num_reads: int = 5,
    max_iters: int = 500,
    tabu_size: int = 20,
    seed: int | None = None,
) -> TabuResult:
    """Tabu search from ``num_reads`` independent random starts.

    Returns the best assignment and the per-restart best energies, so that
    ``GapStats`` over ``energies`` is comparable to the simulated-annealing reads.
    """
    rng = np.random.default_rng(seed)
    best_sample: dict[Variable, int] | None = None
    best_energy = np.inf
    per_restart: list[float] = []

    for _ in range(max(1, num_reads)):
        start = {v: int(rng.integers(0, 2)) for v in qubo.variables}
        sample, energy = _tabu_run(qubo, start, max_iters=max_iters, tabu_size=tabu_size)
        per_restart.append(energy)
        if energy < best_energy:
            best_energy, best_sample = energy, sample

    energies = np.sort(np.asarray(per_restart, dtype=float))
    return TabuResult(sample=best_sample, energy=float(best_energy), energies=energies)
