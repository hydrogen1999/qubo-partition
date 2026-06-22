"""Tabu-search baseline for a QUBO using D-Wave's optimized sampler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dwave.samplers import TabuSampler

from qubo_partition.qubo.base import QUBO, Variable


@dataclass
class TabuResult:
    sample: dict[Variable, int]
    energy: float
    energies: np.ndarray


def tabu_solve(
    qubo: QUBO,
    num_reads: int = 5,
    max_iters: int = 500,  # kept for API compatibility
    tabu_size: int = 20,   # kept for API compatibility
    seed: int | None = None,
) -> TabuResult:
    bqm = qubo.to_bqm()

    sampler = TabuSampler()
    ss = sampler.sample(
        bqm,
        num_reads=max(1, num_reads),
        seed=seed,
    )

    energies = np.sort(np.asarray(ss.record.energy, dtype=float))

    return TabuResult(
        sample=dict(ss.first.sample),
        energy=float(ss.first.energy),
        energies=energies,
    )

