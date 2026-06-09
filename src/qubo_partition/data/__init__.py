"""Synthetic graph and image/seed generators (small, fixed-size by design)."""

from qubo_partition.data.graphs import (
    benchmark_graphs,
    erdos_renyi_graph,
    random_geometric,
)
from qubo_partition.data.images import (
    SeededImage,
    auto_seeds,
    benchmark_images,
    make_blob_image,
    make_two_region_image,
)
from qubo_partition.data.real import load_skimage_demo, load_weizmann_horses

__all__ = [
    "erdos_renyi_graph",
    "random_geometric",
    "benchmark_graphs",
    "make_blob_image",
    "make_two_region_image",
    "auto_seeds",
    "SeededImage",
    "benchmark_images",
    "load_weizmann_horses",
    "load_skimage_demo",
]
