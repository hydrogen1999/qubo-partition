"""Correctness of the minimum-vertex-cover QUBO."""

import networkx as nx
import pytest

from qubo_partition.qubo.vertex_cover import (
    cover_from_sample,
    is_vertex_cover,
    vertex_cover_qubo,
)
from qubo_partition.solvers.annealer import exact_qubo_min
from qubo_partition.solvers.exact_vc import (
    exact_min_vertex_cover,
    exact_min_vertex_cover_energy,
)


def test_penalty_must_exceed_one():
    with pytest.raises(ValueError):
        vertex_cover_qubo(nx.path_graph(4), penalty=1.0)


def test_qubo_energy_matches_handwritten_expression():
    # Single edge (0,1): H = x0 + x1 + P(1-x0)(1-x1), P=2.
    g = nx.Graph([(0, 1)])
    q = vertex_cover_qubo(g, penalty=2.0)
    # both unchosen -> 0 + 0 + 2*1 = 2
    assert q.energy({0: 0, 1: 0}) == pytest.approx(2.0)
    # one chosen -> 1 + 0 = 1 (penalty 0)
    assert q.energy({0: 1, 1: 0}) == pytest.approx(1.0)
    # both chosen -> 2
    assert q.energy({0: 1, 1: 1}) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "g",
    [
        nx.path_graph(6),
        nx.cycle_graph(7),
        nx.complete_graph(5),
        nx.star_graph(6),
        nx.petersen_graph(),
    ],
)
def test_exact_qubo_min_is_a_minimum_cover(g):
    q = vertex_cover_qubo(g, penalty=2.0)
    sample, energy = exact_qubo_min(q)
    cover = cover_from_sample(sample)
    # the QUBO minimizer is a valid cover...
    assert is_vertex_cover(g, cover)
    # ...and its size equals the true minimum cover size
    _, opt_size = exact_min_vertex_cover(g)
    assert len(cover) == opt_size
    # ...and its energy equals the reference energy
    _, ref_energy = exact_min_vertex_cover_energy(g, penalty=2.0)
    assert energy == pytest.approx(ref_energy)


def test_min_cover_energy_equals_cover_size():
    # For a valid minimum cover the penalty terms vanish, so energy == size.
    g = nx.cycle_graph(6)  # min cover size 3
    sample, energy = exact_min_vertex_cover_energy(g, penalty=2.0)
    assert sum(sample.values()) == 3
    assert energy == pytest.approx(3.0)
