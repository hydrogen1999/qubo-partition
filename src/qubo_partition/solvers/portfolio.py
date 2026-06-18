"""Classical solver portfolio: benchmark several heuristics on the identical QUBO."""

from __future__ import annotations

import time
from dataclasses import dataclass

from qubo_partition.qubo.base import QUBO
from qubo_partition.solvers.gurobi import gurobi_solve


@dataclass
class SolverResult:
    name: str
    best_energy: float
    gap: float
    time_s: float
    success: bool


def run_portfolio(
    qubo: QUBO,
    optimal_energy: float,
    num_reads: int = 100,
    num_sweeps: int = 1000,
    seed: int = 0,
    rtol: float = 1e-6,
) -> list[SolverResult]:
    """Run SA, Tabu, Greedy, and Gurobi on one QUBO."""

    from dwave.samplers import (
        SimulatedAnnealingSampler,
        SteepestDescentSampler,
        TabuSampler,
    )

    bqm = qubo.to_bqm()
    atol = max(1.0, abs(optimal_energy)) * rtol

    specs = [
        (
            "SA",
            SimulatedAnnealingSampler(),
            {"num_reads": num_reads, "num_sweeps": num_sweeps, "seed": seed},
        ),
        ("Tabu", TabuSampler(), {"num_reads": num_reads, "seed": seed}),
        ("Greedy", SteepestDescentSampler(), {"num_reads": num_reads, "seed": seed}),
    ]

    results: list[SolverResult] = []

    for name, sampler, kw in specs:
        t0 = time.perf_counter()

        try:
            ss = sampler.sample(bqm, **kw)
        except TypeError:
            kw2 = {k: v for k, v in kw.items() if k not in ("seed", "num_sweeps")}
            ss = sampler.sample(bqm, **kw2)

        dt = time.perf_counter() - t0
        e = float(ss.first.energy)
        gap = e - optimal_energy

        results.append(
            SolverResult(
                name=name,
                best_energy=e,
                gap=gap,
                time_s=dt,
                success=gap <= atol,
            )
        )

    gurobi_res = gurobi_solve(qubo)
    gurobi_gap = gurobi_res.energy - optimal_energy

    results.append(
        SolverResult(
            name="Gurobi",
            best_energy=gurobi_res.energy,
            gap=gurobi_gap,
            time_s=gurobi_res.time_s,
            success=gurobi_gap <= atol,
        )
    )

    return results