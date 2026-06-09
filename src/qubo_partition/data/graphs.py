"""Seeded graph generators for the vertex-cover phase."""

from __future__ import annotations

import networkx as nx


def erdos_renyi_graph(n: int, p: float = 0.3, seed: int = 0) -> nx.Graph:
    """A G(n, p) random graph, relabeled 0..n-1, with isolated nodes kept."""
    g = nx.gnp_random_graph(n, p, seed=seed)
    return nx.convert_node_labels_to_integers(g)


def random_geometric(n: int, radius: float = 0.4, seed: int = 0) -> nx.Graph:
    """A random geometric graph: nodes in the unit square, edges within ``radius``."""
    g = nx.random_geometric_graph(n, radius, seed=seed)
    return nx.convert_node_labels_to_integers(g)


def benchmark_graphs() -> dict[str, nx.Graph]:
    """Fixed suite of named instances used across the experiments."""
    suite: dict[str, nx.Graph] = {}

    suite["path_10"] = nx.path_graph(10)
    suite["cycle_12"] = nx.cycle_graph(12)
    suite["star_11"] = nx.star_graph(10)  # 11 nodes (center + 10 leaves)
    suite["complete_8"] = nx.complete_graph(8)
    suite["grid_3x4"] = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 4))
    suite["petersen"] = nx.petersen_graph()  # 10 nodes, 15 edges
    suite["wheel_11"] = nx.wheel_graph(10)  # hub + 10-cycle

    for n, p, s in [(12, 0.25, 1), (15, 0.3, 2), (18, 0.2, 3), (20, 0.25, 4)]:
        suite[f"er_n{n}_p{p}_s{s}"] = erdos_renyi_graph(n, p, seed=s)
    suite["rgg_n16"] = random_geometric(16, radius=0.45, seed=7)

    return suite


def size_sweep(
    sizes: list[int] = (8, 10, 12, 14, 16, 18, 20),
    p: float = 0.3,
    seeds: list[int] = (0, 1, 2, 3, 4),
) -> list[tuple[str, nx.Graph]]:
    """Instances for the optimality-gap-vs-size study (weeks 3--4 deliverable)."""
    out: list[tuple[str, nx.Graph]] = []
    for n in sizes:
        for s in seeds:
            out.append((f"er_n{n}_s{s}", erdos_renyi_graph(n, p, seed=s)))
    return out
