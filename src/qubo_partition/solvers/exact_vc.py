"""Exact minimum vertex cover by exhaustive search (phase-one reference)."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import combinations

import networkx as nx

from qubo_partition.qubo.vertex_cover import is_vertex_cover, vertex_cover_qubo


def exact_min_vertex_cover(graph: nx.Graph) -> tuple[set[Hashable], int]:
    """Return ``(cover, size)`` for a minimum vertex cover, by exhaustive search."""
    nodes: list[Hashable] = list(graph.nodes())
    n = len(nodes)
    if n > 25:
        raise ValueError(f"exhaustive search refuses {n} nodes (> 25).")

    if graph.number_of_edges() == 0:
        return set(), 0

    for k in range(1, n + 1):
        for subset in combinations(nodes, k):
            chosen = set(subset)
            if is_vertex_cover(graph, chosen):
                return chosen, k
    return set(nodes), n  # unreachable for a finite graph


def exact_min_vertex_cover_energy(graph: nx.Graph, penalty: float = 2.0) -> tuple[dict, float]:
    """Minimum-cover assignment and its QUBO energy under the given ``penalty``."""
    cover, _ = exact_min_vertex_cover(graph)
    qubo = vertex_cover_qubo(graph, penalty=penalty)
    sample = {v: (1 if v in cover else 0) for v in graph.nodes()}
    return sample, qubo.energy(sample)
