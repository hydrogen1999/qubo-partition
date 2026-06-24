"""Greedy steepest-descent baseline for a QUBO using D-Wave's optimized sampler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dwave.samplers import SteepestDescentSampler

from qubo_partition.qubo.base import QUBO, Variable


@dataclass
class GreedyResult:
    sample: dict[Variable, int]
    energy: float
    energies: np.ndarray


def greedy_solve(
    qubo: QUBO,
    num_reads: int = 20,
    seed: int | None = None,
) -> GreedyResult:
    bqm = qubo.to_bqm()

    sampler = SteepestDescentSampler()
    ss = sampler.sample(
        bqm,
        num_reads=max(1, num_reads),
        seed=seed,
    )

    energies = np.sort(np.asarray(ss.record.energy, dtype=float))

    return GreedyResult(
        sample=dict(ss.first.sample),
        energy=float(ss.first.energy),
        energies=energies,
    )
