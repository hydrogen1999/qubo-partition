"""Correctness of the Boykov--Jolly segmentation QUBO and its max-flow check."""

import numpy as np
import pytest

from qubo_partition.data.images import make_blob_image, make_two_region_image
from qubo_partition.qubo.segmentation import segmentation_qubo
from qubo_partition.solvers.annealer import exact_qubo_min
from qubo_partition.solvers.maxflow import min_cut_segmentation


def test_qubo_energy_matches_model_energy():
    """The QUBO and the SegmentationModel must score every labeling identically."""
    seeded = make_two_region_image(size=6, seed=0)
    qubo, model = segmentation_qubo(seeded.image, seeded.fg_seeds, seeded.bg_seeds)
    rng = np.random.default_rng(0)
    for _ in range(20):
        labels = rng.integers(0, 2, size=seeded.shape)
        sample = {(r, c): int(labels[r, c]) for r in range(6) for c in range(6)}
        assert qubo.energy(sample) == pytest.approx(model.energy(labels), rel=1e-9, abs=1e-6)


def test_maxflow_matches_exact_qubo_min_on_tiny_image():
    """Max-flow optimum must equal the exact QUBO minimum (submodular energy)."""
    seeded = make_two_region_image(size=4, seed=1)  # 16 vars -> ExactSolver ok
    qubo, model = segmentation_qubo(seeded.image, seeded.fg_seeds, seeded.bg_seeds)
    _, qubo_min = exact_qubo_min(qubo)
    _, flow_energy = min_cut_segmentation(model)
    assert flow_energy == pytest.approx(qubo_min, rel=1e-6, abs=1e-4)


def test_data_term_breaks_the_trivial_solution():
    """With seeds, the optimum is not all-one-label (the degenerate solution)."""
    seeded = make_blob_image(size=8, seed=0)
    _, model = segmentation_qubo(seeded.image, seeded.fg_seeds, seeded.bg_seeds)
    labels, _ = min_cut_segmentation(model)
    assert labels.any() and not labels.all()  # both labels present


def test_seeds_are_respected_by_the_optimum():
    seeded = make_blob_image(size=8, seed=2)
    _, model = segmentation_qubo(seeded.image, seeded.fg_seeds, seeded.bg_seeds)
    labels, _ = min_cut_segmentation(model)
    assert labels[seeded.fg_seeds].all()  # fg seeds -> foreground
    assert not labels[seeded.bg_seeds].any()  # bg seeds -> background


def test_requires_both_seed_types():
    seeded = make_blob_image(size=6, seed=0)
    no_bg = np.zeros_like(seeded.bg_seeds)
    with pytest.raises(ValueError):
        segmentation_qubo(seeded.image, seeded.fg_seeds, no_bg)


@pytest.mark.parametrize("model", ["histogram", "gaussian"])
def test_both_data_models_match_maxflow(model):
    """QUBO and max-flow must agree regardless of the regional data model."""
    seeded = make_two_region_image(size=4, seed=3)
    qubo, mdl = segmentation_qubo(seeded.image, seeded.fg_seeds, seeded.bg_seeds, data_model=model)
    _, qubo_min = exact_qubo_min(qubo)
    _, flow_energy = min_cut_segmentation(mdl)
    assert flow_energy == pytest.approx(qubo_min, rel=1e-6, abs=1e-4)


def test_histogram_term_handles_8_connectivity():
    seeded = make_blob_image(size=6, seed=1)
    qubo, mdl = segmentation_qubo(
        seeded.image,
        seeded.fg_seeds,
        seeded.bg_seeds,
        data_model="histogram",
        connectivity=8,
    )
    labels, _ = min_cut_segmentation(mdl)
    assert labels[seeded.fg_seeds].all()
    assert not labels[seeded.bg_seeds].any()
