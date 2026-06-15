import numpy as np
import pytest

from morphos.field import Field
from morphos.manufacturing.constraints import (
    ManufacturabilityConstraint,
    MinFeatureSize,
    Connectivity,
)


def test_min_feature_is_a_constraint():
    assert isinstance(MinFeatureSize(radius=1), ManufacturabilityConstraint)


def test_min_feature_smooths_a_single_voxel_spike():
    vals = np.zeros((5, 5))
    vals[2, 2] = 1.0
    f = Field(vals, spacing=1.0)
    out = MinFeatureSize(radius=1).project(f)
    # the spike is spread out: center reduced, neighbours raised
    assert out.values[2, 2] < 1.0
    assert out.values[2, 3] > 0.0
    # total mass is preserved in the interior (box filter, far from edges)
    assert out.values.sum() == pytest.approx(1.0)


def test_project_preserves_grid():
    f = Field(np.zeros((4, 6)), spacing=(0.5, 2.0))
    out = MinFeatureSize(radius=1).project(f)
    assert out.shape == (4, 6)
    assert out.spacing == (0.5, 2.0)


def test_min_feature_vjp_is_self_adjoint():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(6, 6))
    b = rng.normal(size=(6, 6))
    c = MinFeatureSize(radius=2)
    fa = Field(a, spacing=1.0)
    # <project(a), b> == <a, vjp(b)>  for a self-adjoint linear filter
    lhs = np.sum(c.project(fa).values * b)
    rhs = np.sum(a * c.vjp(fa, b))
    assert lhs == pytest.approx(rhs)


def test_connectivity_reports_single_component():
    vals = np.zeros((5, 5))
    vals[1:4, 1:4] = 1.0
    f = Field(vals, spacing=1.0)
    rep = Connectivity(threshold=0.5).report(f)
    assert rep["num_components"] == 1
    assert rep["connected"] is True


def test_connectivity_reports_two_components():
    vals = np.zeros((5, 5))
    vals[0, 0] = 1.0
    vals[4, 4] = 1.0
    f = Field(vals, spacing=1.0)
    rep = Connectivity(threshold=0.5).report(f)
    assert rep["num_components"] == 2
    assert rep["connected"] is False


def test_connectivity_projection_and_vjp_are_identity():
    c = Connectivity()
    f = Field(np.ones((3, 3)), spacing=1.0)
    assert np.allclose(c.project(f).values, f.values)
    g = np.array([[1.0, 2.0, 3.0]] * 3)
    assert np.allclose(c.vjp(f, g), g)
