"""Tests for morphos.initialize: bio-inspired initial density fields.

Covers the Gray-Scott reaction-diffusion seeder, Murray's-law branching
network seeder, uniform-noise seeder, image-based seeder, and an end-to-end
check that a Gray-Scott field is a valid initial field for OCOptimizer.
"""

import numpy as np
import pytest

from morphos.field import Field
from morphos.initialize import GrayScottRD, murray_network, uniform_noise, from_image


# ---------------------------------------------------------------------------
# GrayScottRD
# ---------------------------------------------------------------------------

def test_gray_scott_generates_field_with_correct_shape():
    field = GrayScottRD(n_steps=50).generate((10, 20), 0.4)
    assert isinstance(field, Field)
    assert field.values.shape == (10, 20)


def test_gray_scott_volume_fraction_close_to_target():
    field = GrayScottRD(n_steps=200).generate((16, 16), 0.4)
    assert abs(field.values.mean() - 0.4) < 0.05


def test_gray_scott_preset_spots():
    field = GrayScottRD.preset("spots").generate((12, 12), 0.5)
    assert field.values.shape == (12, 12)


def test_gray_scott_preset_stripes():
    field = GrayScottRD.preset("stripes").generate((12, 12), 0.5)
    assert field.values.shape == (12, 12)


def test_gray_scott_different_seeds_give_different_patterns():
    f0 = GrayScottRD(n_steps=200, seed=0).generate((16, 16), 0.4)
    f1 = GrayScottRD(n_steps=200, seed=1).generate((16, 16), 0.4)
    assert np.max(np.abs(f0.values - f1.values)) > 0.01


def test_gray_scott_3d():
    field = GrayScottRD(n_steps=100).generate((4, 8, 8), 0.5)
    assert field.values.shape == (4, 8, 8)


# ---------------------------------------------------------------------------
# murray_network
# ---------------------------------------------------------------------------

def test_murray_network_shape_and_volume():
    field = murray_network((16, 32), 0.3)
    assert isinstance(field, Field)
    assert field.values.shape == (16, 32)
    assert abs(field.values.mean() - 0.3) < 0.1


# ---------------------------------------------------------------------------
# uniform_noise
# ---------------------------------------------------------------------------

def test_uniform_noise_shape_and_volume():
    field = uniform_noise((10, 10), 0.5)
    assert field.values.shape == (10, 10)
    assert np.all(field.values >= 1e-3)
    assert np.all(field.values <= 1.0)
    assert abs(field.values.mean() - 0.5) < 0.15


# ---------------------------------------------------------------------------
# Integration: Gray-Scott field as OCOptimizer initial field
# ---------------------------------------------------------------------------

def test_gray_scott_field_is_valid_initial_for_optimizer():
    from morphos.physics.elasticity import ElasticityOracle
    from morphos.objective.objective import MaximizeValue, PhysicalBound
    from morphos.optimize.oc import OCOptimizer

    shape = (8, 16)
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed = [(0, j, ax) for j in range(nny) for ax in ("x", "y")]
    loads = {(nnx - 1, nny - 1, "y"): -1.0}
    oracle = ElasticityOracle(shape=shape, fixed_dofs=fixed, loads=loads)
    obj = MaximizeValue(bound=PhysicalBound(value=0.0, name="rigid"))
    initial = GrayScottRD(n_steps=50).generate(shape, 0.4)
    opt = OCOptimizer(volume_fraction=0.4, max_iter=3)
    result = opt.run(initial, oracle, obj)
    assert np.isfinite(result.fom)


# ---------------------------------------------------------------------------
# from_image
# ---------------------------------------------------------------------------

def test_from_image_with_array_input():
    rng = np.random.default_rng(0)
    arr = (rng.random((10, 10)) > 0.5).astype(float)
    field = from_image(arr, (8, 8), 0.35)
    assert isinstance(field, Field)
    assert field.values.shape == (8, 8)
    assert abs(field.values.mean() - 0.35) < 0.15
