from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qubo_partition.qubo.base import QUBO, Variable


@dataclass
class GreedyResult:
    sample: dict[Variable, int]
    energy: float
    energies: np.ndarray


def greedy_solve(
    qubo: QUBO,
    seed: int | None = None,
) -> GreedyResult:

    rng = np.random.default_rng(seed)

    sample = {
        v: int(rng.integers(0, 2))
        for v in qubo.variables
    }

    current_energy = qubo.energy(sample)
    energies = [current_energy]

    improved = True

    while improved:
        improved = False

        for var in qubo.variables:

            candidate = sample.copy()
            candidate[var] = 1 - candidate[var]

            candidate_energy = qubo.energy(candidate)

            if candidate_energy < current_energy:
                sample = candidate
                current_energy = candidate_energy
                energies.append(current_energy)
                improved = True

    return GreedyResult(
        sample=sample,
        energy=float(current_energy),
        energies=np.asarray(energies),
    )