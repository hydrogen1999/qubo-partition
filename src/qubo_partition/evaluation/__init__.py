"""Validity + optimality-gap metrics and the multi-run experiment runner."""

from qubo_partition.evaluation.metrics import (
    GapStats,
    iou,
    optimality_gap,
    relative_gap,
)
from qubo_partition.evaluation.runner import (
    SegmentationRecord,
    VertexCoverRecord,
    run_segmentation,
    run_vertex_cover,
)

__all__ = [
    "optimality_gap",
    "relative_gap",
    "iou",
    "GapStats",
    "run_vertex_cover",
    "run_segmentation",
    "VertexCoverRecord",
    "SegmentationRecord",
]
