"""Simulated-annealing QUBO sampler (D-Wave Ocean SDK)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qubo_partition.qubo.base import QUBO, Variable


@dataclass
class SAResult:
    """Outcome of one ``anneal`` call."""

    sample: dict[Variable, int]
    energy: float
    energies: np.ndarray
    num_reads: int

    @property
    def best(self) -> float:
        return self.energy

    @property
    def mean(self) -> float:
        return float(np.mean(self.energies))

    @property
    def std(self) -> float:
        return float(np.std(self.energies))

    @property
    def worst(self) -> float:
        return float(np.max(self.energies))


def anneal(
    qubo: QUBO,
    num_reads: int = 100,
    num_sweeps: int = 1000,
    seed: int | None = None,
    beta_range: tuple | None = None,
) -> SAResult:
    """Minimize ``qubo`` with simulated annealing over ``num_reads`` runs."""
    from dwave.samplers import SimulatedAnnealingSampler

    bqm = qubo.to_bqm()
    sampler = SimulatedAnnealingSampler()
    kwargs = dict(num_reads=num_reads, num_sweeps=num_sweeps)
    if seed is not None:
        kwargs["seed"] = int(seed)
    if beta_range is not None:
        kwargs["beta_range"] = list(beta_range)

    sampleset = sampler.sample(bqm, **kwargs)

    record = sampleset.record
    energies = np.repeat(record.energy, record.num_occurrences).astype(float)
    energies.sort()

    best = sampleset.first
    sample = {v: int(best.sample[v]) for v in qubo.variables}
    return SAResult(
        sample=sample,
        energy=float(best.energy),
        energies=energies,
        num_reads=int(num_reads),
    )


def exact_qubo_min(qubo: QUBO) -> tuple[dict[Variable, int], float]:
    """Exact QUBO minimum via Ocean's ``ExactSolver`` (tiny instances only)."""
    import dimod

    n = qubo.num_variables
    if n > 22:
        raise ValueError(f"ExactSolver refuses {n} variables (> 22).")
    sampleset = dimod.ExactSolver().sample(qubo.to_bqm())
    best = sampleset.first
    sample = {v: int(best.sample[v]) for v in qubo.variables}
    return sample, float(best.energy)
