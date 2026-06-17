"""Shared experiment loop: build -> anneal (many reads) -> check -> measure."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from qubo_partition.data.images import SeededImage
from qubo_partition.evaluation.metrics import GapStats, iou, pixel_accuracy
from qubo_partition.qubo.segmentation import segmentation_qubo
from qubo_partition.qubo.vertex_cover import (
    cover_from_sample,
    is_vertex_cover,
    vertex_cover_qubo,
)
from qubo_partition.solvers.annealer import anneal
from qubo_partition.solvers.exact_vc import exact_min_vertex_cover_energy
from qubo_partition.solvers.greedy import greedy_solve
from qubo_partition.solvers.maxflow import min_cut_segmentation
from qubo_partition.solvers.tabu import tabu_solve


@dataclass
class VertexCoverRecord:
    name: str
    n_nodes: int
    n_edges: int
    penalty: float
    annealed_energy: float
    optimal_energy: float
    annealed_cover_size: int
    optimal_cover_size: int
    is_valid_cover: bool
    gap: GapStats
    annealed_sample: dict = field(repr=False, default_factory=dict)

    def as_row(self) -> dict:
        d = {
            "name": self.name,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "penalty": self.penalty,
            "annealed_energy": self.annealed_energy,
            "optimal_energy": self.optimal_energy,
            "annealed_cover_size": self.annealed_cover_size,
            "optimal_cover_size": self.optimal_cover_size,
            "is_valid_cover": self.is_valid_cover,
        }
        d.update(self.gap.as_dict())
        return d


def run_vertex_cover(
    graph: nx.Graph,
    name: str = "graph",
    penalty: float = 2.0,
    num_reads: int = 200,
    num_sweeps: int = 1000,
    seed: int | None = 0,
) -> VertexCoverRecord:
    """Build the MVC QUBO, anneal it, and check against exhaustive search."""
    qubo = vertex_cover_qubo(graph, penalty=penalty)
    res = anneal(qubo, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)

    opt_sample, opt_energy = exact_min_vertex_cover_energy(graph, penalty=penalty)
    opt_size = int(sum(opt_sample.values()))

    cover = cover_from_sample(res.sample)
    valid = is_vertex_cover(graph, cover)

    return VertexCoverRecord(
        name=name,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        penalty=penalty,
        annealed_energy=res.energy,
        optimal_energy=opt_energy,
        annealed_cover_size=len(cover),
        optimal_cover_size=opt_size,
        is_valid_cover=valid,
        gap=GapStats.from_energies(res.energies, opt_energy),
        annealed_sample=res.sample,
    )


@dataclass
class SegmentationRecord:
    name: str
    shape: tuple
    lambda_smooth: float
    annealed_energy: float
    optimal_energy: float
    gap: GapStats
    iou_annealed: float
    iou_optimal: float
    pixel_acc_annealed: float
    annealed_labels: np.ndarray = field(repr=False, default=None)
    optimal_labels: np.ndarray = field(repr=False, default=None)

    def as_row(self) -> dict:
        d = {
            "name": self.name,
            "height": self.shape[0],
            "width": self.shape[1],
            "n_pixels": self.shape[0] * self.shape[1],
            "lambda_smooth": self.lambda_smooth,
            "annealed_energy": self.annealed_energy,
            "optimal_energy": self.optimal_energy,
            "iou_annealed": self.iou_annealed,
            "iou_optimal": self.iou_optimal,
            "pixel_acc_annealed": self.pixel_acc_annealed,
        }
        d.update(self.gap.as_dict())
        return d


def run_segmentation(
    seeded: SeededImage,
    lambda_smooth: float = 1.0,
    sigma: float | None = None,
    data_weight: float = 1.0,
    num_reads: int = 200,
    num_sweeps: int = 2000,
    seed: int | None = 0,
    connectivity: int = 4,
    data_model: str = "histogram",
    n_bins: int = 16,
    solver: str = "sa",
) -> SegmentationRecord:
    """Build the Boykov--Jolly QUBO, solve it, and check against maximum flow."""

    qubo, model = segmentation_qubo(
        seeded.image,
        seeded.fg_seeds,
        seeded.bg_seeds,
        lambda_smooth=lambda_smooth,
        sigma=sigma,
        data_weight=data_weight,
        connectivity=connectivity,
        data_model=data_model,
        n_bins=n_bins,
    )

    opt_labels, opt_energy = min_cut_segmentation(model)

    if solver == "maxflow":
        # the exact graph-cut optimum is itself the reported result (gap 0)
        annealed_labels = opt_labels
        annealed_energy = opt_energy
        energies = np.asarray([opt_energy], dtype=float)
    elif solver == "sa":
        res = anneal(qubo, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
        annealed_labels = model.sample_to_labels(res.sample)
        annealed_energy, energies = res.energy, res.energies
    elif solver == "greedy":
        # pure-Python descent is O(n^2) per restart; cap restarts to stay tractable
        res = greedy_solve(qubo, num_reads=min(num_reads, 20), seed=seed)
        annealed_labels = model.sample_to_labels(res.sample)
        annealed_energy, energies = res.energy, res.energies
    elif solver == "tabu":
        res = tabu_solve(qubo, num_reads=min(num_reads, 5), seed=seed)
        annealed_labels = model.sample_to_labels(res.sample)
        annealed_energy, energies = res.energy, res.energies
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return SegmentationRecord(
        name=seeded.name,
        shape=seeded.shape,
        lambda_smooth=lambda_smooth,
        annealed_energy=annealed_energy,
        optimal_energy=opt_energy,
        gap=GapStats.from_energies(energies, opt_energy),
        iou_annealed=iou(annealed_labels, seeded.truth),
        iou_optimal=iou(opt_labels, seeded.truth),
        pixel_acc_annealed=pixel_accuracy(annealed_labels, seeded.truth),
        annealed_labels=annealed_labels,
        optimal_labels=opt_labels,
    )
