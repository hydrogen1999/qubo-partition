from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from qubo_partition.qubo.base import QUBO, Variable


@dataclass
class TabuResult:
    sample: dict[Variable, int]
    energy: float
    energies: np.ndarray


def tabu_solve(
    qubo: QUBO,
    max_iters: int = 500,
    tabu_size: int = 20,
    seed: int | None = None,
) -> TabuResult:

    rng = np.random.default_rng(seed)

    current = {
        v: int(rng.integers(0, 2))
        for v in qubo.variables
    }

    current_energy = qubo.energy(current)

    best = current.copy()
    best_energy = current_energy

    tabu_list = deque(maxlen=tabu_size)
    energies = [current_energy]

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

        current = best_neighbor
        current_energy = best_neighbor_energy

        tabu_list.append(best_move)
        energies.append(current_energy)

        if current_energy < best_energy:
            best = current.copy()
            best_energy = current_energy

    return TabuResult(
        sample=best,
        energy=float(best_energy),
        energies=np.asarray(energies),
    )