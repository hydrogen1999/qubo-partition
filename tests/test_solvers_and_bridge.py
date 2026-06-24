"""QUBO container invariants, sampler sanity, and the reconstruction bridge."""

import networkx as nx
import numpy as np
import pytest

from qubo_partition.bridge.reconstruction import (
    exact_reconstruction,
    graph_reconstruction_qubo,
    run_reconstruction_demo,
    smoothed_features,
)
from qubo_partition.qubo.base import QUBO
from qubo_partition.qubo.vertex_cover import vertex_cover_qubo
from qubo_partition.solvers.annealer import anneal, exact_qubo_min


def test_qubo_offset_and_self_pair():
    q = QUBO()
    q.add_linear("a", 2.0)
    q.add_quadratic("a", "a", 3.0)  # self-pair folds into linear (x^2 = x)
    q.add_offset(1.0)
    assert q.energy({"a": 1}) == pytest.approx(2.0 + 3.0 + 1.0)
    assert q.energy({"a": 0}) == pytest.approx(1.0)


def test_to_bqm_preserves_energy():
    import dimod

    g = nx.cycle_graph(6)
    q = vertex_cover_qubo(g, penalty=2.0)
    bqm = q.to_bqm()
    rng = np.random.default_rng(0)
    for _ in range(10):
        sample = {v: int(rng.integers(0, 2)) for v in q.variables}
        assert dimod.BinaryQuadraticModel.energy(bqm, sample) == pytest.approx(
            q.energy(sample), abs=1e-9
        )


def test_annealer_reaches_optimum_on_small_instance():
    g = nx.petersen_graph()
    q = vertex_cover_qubo(g, penalty=2.0)
    _, opt = exact_qubo_min(q)
    res = anneal(q, num_reads=100, num_sweeps=1000, seed=0)
    assert res.energy == pytest.approx(opt, abs=1e-6)
    assert res.best <= res.mean + 1e-9  # best is no worse than the mean read


def test_reconstruction_exact_reference_is_optimal():
    """Brute force must not beat the analytic exact reconstruction optimum."""
    g = nx.cycle_graph(6)
    feats = smoothed_features(g, seed=0)
    qubo, costs = graph_reconstruction_qubo(feats, budget=g.number_of_edges(), penalty=5.0)
    _, opt_energy = exact_reconstruction(costs, budget=g.number_of_edges(), penalty=5.0)
    _, brute_energy = qubo.brute_force()  # 15 vars for n=6 -> ok
    assert opt_energy == pytest.approx(brute_energy, abs=1e-6)


def test_reconstruction_recovers_structure():
    rec = run_reconstruction_demo(nx.cycle_graph(8), num_reads=150, num_sweeps=1500, seed=0)
    assert rec.gap >= -1e-6  # annealer never beats the optimum
    assert rec.f1 > 0.5  # recovers most of the hidden edges


def test_solver_portfolio_runs_and_reaches_optimum():
    from qubo_partition.solvers.portfolio import run_portfolio

    g = nx.petersen_graph()
    q = vertex_cover_qubo(g, penalty=2.0)
    _, opt = exact_qubo_min(q)
    # include_gurobi=False must work even when gurobipy is absent (regression)
    res = run_portfolio(q, opt, num_reads=50, num_sweeps=500, seed=0, include_gurobi=False)
    names = {r.name for r in res}
    assert {"SA", "Tabu", "Greedy"} == names  # exactly the heuristics, no hard Gurobi dep
    for r in res:
        assert r.gap >= -1e-6  # no solver beats the exact optimum
        assert r.time_s >= 0.0


def test_greedy_tabu_return_per_restart_energies():
    """Multi-restart baselines must report independent per-restart energies."""
    from qubo_partition.solvers.greedy import greedy_solve
    from qubo_partition.solvers.tabu import tabu_solve

    g = nx.cycle_graph(8)
    q = vertex_cover_qubo(g, penalty=2.0)
    _, opt = exact_qubo_min(q)

    gres = greedy_solve(q, num_reads=10, seed=0)
    tres = tabu_solve(q, num_reads=4, seed=0)
    assert len(gres.energies) == 10 and len(tres.energies) == 4
    # energies are sorted ascending and the reported best matches the minimum
    assert gres.energy == pytest.approx(float(gres.energies.min()))
    assert tres.energy == pytest.approx(float(tres.energies.min()))
    assert gres.energy >= opt - 1e-9 and tres.energy >= opt - 1e-9
