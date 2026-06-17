"""Exact seeded segmentation by maximum flow / min s-t cut (phase-two reference)."""

from __future__ import annotations

import networkx as nx
import numpy as np

from qubo_partition.qubo.segmentation import SegmentationModel

SOURCE = "__source__"
SINK = "__sink__"


def _build_flow_graph(model: SegmentationModel, scale: float = 1.0e9) -> nx.DiGraph:
    """Build the s-t graph with integer, bounded-infinity capacities."""
    fg = model.fg_seeds
    bg = model.bg_seeds
    h, w = model.shape

    # INF must exceed any achievable finite cut so seed edges are never cut.
    finite = sum(max(0.0, float(wij)) for wij in model.weights.values())
    for r in range(h):
        for c in range(w):
            if fg[r, c] or bg[r, c]:
                continue
            d0, d1 = float(model.D0[r, c]), float(model.D1[r, c])
            finite += abs(d0) + abs(d1)
    INF = finite + 1.0

    def I(x: float) -> int:  # noqa: E743 - scale + round to integer capacity
        return int(round(x * scale))

    g = nx.DiGraph()
    g.add_node(SOURCE)
    g.add_node(SINK)

    for r in range(h):
        for c in range(w):
            node = (r, c)
            g.add_node(node)
            if fg[r, c]:
                g.add_edge(SOURCE, node, capacity=I(INF))
                continue
            if bg[r, c]:
                g.add_edge(node, SINK, capacity=I(INF))
                continue
            d0, d1 = float(model.D0[r, c]), float(model.D1[r, c])
            m = min(d0, d1)
            cap_s, cap_t = I(d0 - m), I(d1 - m)
            if cap_s > 0:
                g.add_edge(SOURCE, node, capacity=cap_s)
            if cap_t > 0:
                g.add_edge(node, SINK, capacity=cap_t)

    for (i, j), wij in model.weights.items():
        if wij < 0:
            # a negative smoothness weight is non-submodular; the min-cut would no
            # longer equal the energy minimum, so the "exact reference" would lie.
            raise ValueError(
                "min_cut_segmentation requires non-negative pairwise weights "
                f"(got {wij}); the energy is only submodular for w_ij >= 0."
            )
        cap = I(float(wij))
        if cap <= 0:
            continue
        g.add_edge(i, j, capacity=cap)
        g.add_edge(j, i, capacity=cap)
    return g


def min_cut_segmentation(
    model: SegmentationModel, scale: float = 1.0e9
) -> tuple[np.ndarray, float]:
    """Exact minimum-energy labeling and its (true) energy via maximum flow."""
    g = _build_flow_graph(model, scale=scale)
    _, (source_side, _sink_side) = nx.minimum_cut(g, SOURCE, SINK)

    h, w = model.shape
    labels = np.zeros((h, w), dtype=np.int8)
    for node in source_side:
        if node == SOURCE:
            continue
        r, c = node
        labels[r, c] = 1

    return labels, model.energy(labels)
