"""Classical solver portfolio: benchmark several heuristics on the identical QUBO."""

from __future__ import annotations

import time
from dataclasses import dataclass

from qubo_partition.qubo.base import QUBO


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
    include_gurobi: str | bool = "auto",
) -> list[SolverResult]:
    """Run SA, Tabu, and steepest-descent (Greedy) on one QUBO; report gap and time.

    Gurobi (exact) is included only when ``include_gurobi`` is True, or "auto"
    (default) and gurobipy is importable; failures (missing package or license)
    are skipped rather than aborting the comparison.
    """
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
        except TypeError:  # some samplers reject seed/num_sweeps
            kw2 = {k: v for k, v in kw.items() if k not in ("seed", "num_sweeps")}
            ss = sampler.sample(bqm, **kw2)
        dt = time.perf_counter() - t0
        e = float(ss.first.energy)
        gap = e - optimal_energy
        results.append(SolverResult(name, e, gap, dt, gap <= atol))

    # Optional exact baseline (only if available); never abort the portfolio.
    from qubo_partition.solvers.gurobi import gurobi_available

    want_gurobi = include_gurobi is True or (include_gurobi == "auto" and gurobi_available())
    if want_gurobi:
        try:
            from qubo_partition.solvers.gurobi import gurobi_solve

            gres = gurobi_solve(qubo)
            ggap = gres.energy - optimal_energy
            results.append(SolverResult("Gurobi", gres.energy, ggap, gres.time_s, ggap <= atol))
        except Exception as exc:  # missing license, etc.
            print(f"[portfolio] Gurobi skipped: {exc}")

    return results
