"""Graph reconstruction (GraphMI) cast as the same QUBO move as segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx
import numpy as np

from qubo_partition.qubo.base import QUBO

Edge = tuple[int, int]


def smoothed_features(graph: nx.Graph, dim: int = 4, steps: int = 3, seed: int = 0) -> np.ndarray:
    """Node features made homophilous by diffusing a random signal over ``graph``."""
    rng = np.random.default_rng(seed)
    n = graph.number_of_nodes()
    x = rng.normal(size=(n, dim))
    A = nx.to_numpy_array(graph, nodelist=range(n))
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    P = A / deg  # row-stochastic neighbor average
    alpha = 0.6
    for _ in range(steps):
        x = (1 - alpha) * x + alpha * (P @ x)
    return x


def _pair_costs(features: np.ndarray) -> dict[Edge, float]:
    """Squared feature distance ``||x_u - x_v||^2`` for every candidate pair."""
    n = features.shape[0]
    costs: dict[Edge, float] = {}
    for u, v in combinations(range(n), 2):
        d = features[u] - features[v]
        costs[(u, v)] = float(np.dot(d, d))
    return costs


def graph_reconstruction_qubo(
    features: np.ndarray, budget: int, penalty: float = 5.0
) -> tuple[QUBO, dict[Edge, float]]:
    """Build the reconstruction QUBO over edge-indicator variables."""
    costs = _pair_costs(features)
    qubo = QUBO()

    # Expand soft cardinality penalty P (sum e - m)^2 into QUBO coefficients.
    lin_budget = penalty * (1.0 - 2.0 * budget)
    edges = list(costs.keys())
    for e in edges:
        qubo.add_linear(e, costs[e] + lin_budget)
    for ea, eb in combinations(edges, 2):
        qubo.add_quadratic(ea, eb, 2.0 * penalty)
    qubo.add_offset(penalty * budget * budget)
    return qubo, costs


def exact_reconstruction(
    costs: dict[Edge, float], budget: int, penalty: float = 5.0
) -> tuple[set, float]:
    """Exact global optimum of the reconstruction QUBO (modular + cardinality)."""
    items = sorted(costs.items(), key=lambda kv: kv[1])  # (edge, cost) ascending
    sorted_costs = np.array([c for _, c in items], dtype=float)
    prefix = np.concatenate([[0.0], np.cumsum(sorted_costs)])
    M = len(items)

    best_k, best_energy = 0, np.inf
    for k in range(M + 1):
        energy = prefix[k] + penalty * (k - budget) ** 2
        if energy < best_energy:
            best_energy, best_k = energy, k
    chosen = {items[i][0] for i in range(best_k)}
    return chosen, float(best_energy)


@dataclass
class ReconstructionRecord:
    name: str
    n_nodes: int
    n_true_edges: int
    penalty: float
    annealed_energy: float
    optimal_energy: float
    gap: float
    precision: float
    recall: float
    f1: float
    edge_iou: float
    annealed_edges: set = field(repr=False, default_factory=set)
    true_edges: set = field(repr=False, default_factory=set)

    def as_row(self) -> dict:
        return {
            "name": self.name,
            "n_nodes": self.n_nodes,
            "n_true_edges": self.n_true_edges,
            "penalty": self.penalty,
            "annealed_energy": self.annealed_energy,
            "optimal_energy": self.optimal_energy,
            "gap": self.gap,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "edge_iou": self.edge_iou,
        }


def _prf(pred: set, truth: set) -> tuple[float, float, float, float]:
    if not pred and not truth:
        return 1.0, 1.0, 1.0, 1.0
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    union = len(pred | truth)
    edge_iou = tp / union if union else 1.0
    return precision, recall, f1, edge_iou


def run_reconstruction_demo(
    graph: nx.Graph,
    name: str = "hidden",
    feat_dim: int = 4,
    penalty: float = 5.0,
    num_reads: int = 200,
    num_sweeps: int = 2000,
    seed: int = 0,
) -> ReconstructionRecord:
    """Hide ``graph``, release smoothed features, reconstruct edges by annealing."""
    from qubo_partition.solvers.annealer import anneal

    graph = nx.convert_node_labels_to_integers(graph)
    true_edges = {(min(u, v), max(u, v)) for u, v in graph.edges() if u != v}
    budget = len(true_edges)

    features = smoothed_features(graph, dim=feat_dim, seed=seed)
    qubo, costs = graph_reconstruction_qubo(features, budget=budget, penalty=penalty)

    res = anneal(qubo, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    annealed_edges = {e for e, bit in res.sample.items() if bit}

    _, opt_energy = exact_reconstruction(costs, budget=budget, penalty=penalty)

    precision, recall, f1, edge_iou = _prf(annealed_edges, true_edges)
    return ReconstructionRecord(
        name=name,
        n_nodes=graph.number_of_nodes(),
        n_true_edges=len(true_edges),
        penalty=penalty,
        annealed_energy=res.energy,
        optimal_energy=opt_energy,
        gap=float(res.energy - opt_energy),
        precision=precision,
        recall=recall,
        f1=f1,
        edge_iou=edge_iou,
        annealed_edges=annealed_edges,
        true_edges=true_edges,
    )
