"""Minimum-vertex-cover QUBO: unit node cost plus an edge-coverage penalty (P > 1)."""

from __future__ import annotations

from collections.abc import Hashable

import networkx as nx

from qubo_partition.qubo.base import QUBO


def vertex_cover_qubo(graph: nx.Graph, penalty: float = 2.0) -> QUBO:
    """Build the minimum-vertex-cover QUBO for ``graph``."""
    if penalty <= 1.0:
        raise ValueError(
            f"penalty P={penalty} must be > 1 so that covering an edge always "
            "beats saving a node; otherwise the minimizer need not be a cover."
        )

    qubo = QUBO()

    # Register every node so isolated nodes still appear as variables.
    for node in graph.nodes():
        qubo.add_linear(node, 1.0)

    n_edges = 0
    for u, v in graph.edges():
        if u == v:  # ignore self-loops
            continue
        n_edges += 1
        # (1 - x_u)(1 - x_v) = 1 - x_u - x_v + x_u x_v
        qubo.add_linear(u, -penalty)
        qubo.add_linear(v, -penalty)
        qubo.add_quadratic(u, v, penalty)

    qubo.add_offset(penalty * n_edges)
    return qubo


def is_vertex_cover(graph: nx.Graph, chosen: set[Hashable]) -> bool:
    """Return ``True`` iff ``chosen`` covers every edge of ``graph``."""
    for u, v in graph.edges():
        if u == v:
            continue
        if u not in chosen and v not in chosen:
            return False
    return True


def cover_from_sample(sample) -> set:
    """Extract the chosen-node set ``{i : x_i = 1}`` from a QUBO sample."""
    return {v for v, bit in sample.items() if bit}
