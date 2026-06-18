import numpy as np
import pytest

from fd_gate import fd_gate

from morphos.field import Field
from morphos.manufacturing.constraints import (
    ManufacturabilityConstraint,
    MinFeatureSize,
    Connectivity,
    Overhang,
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


# --- Overhang (Langelaar self-supporting AM filter) ------------------------


def test_overhang_is_a_constraint():
    assert isinstance(Overhang(angle_deg=45.0), ManufacturabilityConstraint)


def test_overhang_baseplate_layer_is_unconstrained():
    # build axis 0: layer 0 sits on the baseplate and is always fully supported,
    # so projecting must leave it unchanged regardless of overhang angle.
    rng = np.random.default_rng(0)
    vals = rng.uniform(size=(6, 6))
    f = Field(vals, spacing=1.0)
    out = Overhang(angle_deg=45.0, build_axis=0).project(f)
    assert np.allclose(out.values[0, :], vals[0, :])


def test_overhang_preserves_shape_and_spacing():
    f = Field(np.ones((5, 7)), spacing=(0.5, 2.0))
    out = Overhang(angle_deg=45.0, build_axis=0).project(f)
    assert out.shape == (5, 7)
    assert out.spacing == (0.5, 2.0)


def test_overhang_floating_island_is_unsupported():
    # a solid voxel with nothing but void beneath it, far from any neighbour
    # that could support it within the overhang cone, must be suppressed.
    vals = np.zeros((6, 6))
    vals[4, 3] = 1.0  # floating island, layers below are all void
    f = Field(vals, spacing=1.0)
    out = Overhang(angle_deg=45.0, build_axis=0).project(f)
    assert out.values[4, 3] < 0.1


def test_overhang_fully_supported_column_passes_through():
    # a solid column resting on the baseplate is self-supporting everywhere,
    # so projection should reproduce it (up to filter smoothing tolerance).
    vals = np.zeros((6, 6))
    vals[:, 3] = 1.0
    f = Field(vals, spacing=1.0)
    out = Overhang(angle_deg=45.0, build_axis=0).project(f)
    assert np.allclose(out.values[:, 3], 1.0, atol=1e-2)


def test_overhang_45_degree_diagonal_is_self_supporting():
    # a staircase at exactly the limiting 45 degree angle is the textbook
    # self-supporting case: each new layer is offset by one cell from the
    # one below, within the support footprint, so it should print through
    # mostly intact (allow small smoothing loss from the soft-max).
    n = 8
    vals = np.zeros((n, n))
    for layer in range(n):
        vals[layer, layer] = 1.0
    f = Field(vals, spacing=1.0)
    out = Overhang(angle_deg=45.0, build_axis=0).project(f)
    assert out.values.sum() > 0.5 * vals.sum()


def test_overhang_build_axis_1_transposes_behaviour():
    # building along axis 1 instead of axis 0 should give the transpose of
    # the axis-0 result on a transposed field.
    rng = np.random.default_rng(3)
    vals = rng.uniform(size=(6, 5))
    f0 = Field(vals, spacing=1.0)
    f1 = Field(vals.T.copy(), spacing=1.0)
    out0 = Overhang(angle_deg=45.0, build_axis=0).project(f0)
    out1 = Overhang(angle_deg=45.0, build_axis=1).project(f1)
    assert np.allclose(out0.values, out1.values.T, atol=1e-8)


def test_overhang_vjp_passes_directional_fd_gate():
    rng = np.random.default_rng(5)
    vals = rng.uniform(size=(6, 6))
    c = Overhang(angle_deg=45.0, build_axis=0)
    f = Field(vals, spacing=1.0)
    rng2 = np.random.default_rng(6)
    w = rng2.normal(size=(6, 6))

    def scalar_objective(x):
        out = c.project(Field(x, spacing=1.0)).values
        return float(np.sum(out * w))

    grad = c.vjp(f, w)
    fd_gate(scalar_objective, grad, vals, rel=1e-3, h=1e-5)


def test_overhang_report_keys():
    vals = np.zeros((5, 5))
    vals[4, 2] = 1.0  # floating, unsupported
    f = Field(vals, spacing=1.0)
    rep = Overhang(angle_deg=45.0, build_axis=0).report(f)
    assert "unsupported_volume_fraction" in rep
    assert "max_density_deficit" in rep
    assert rep["unsupported_volume_fraction"] > 0.0


def test_overhang_report_zero_for_baseplate_only():
    vals = np.zeros((5, 5))
    vals[0, :] = 1.0  # entirely on the baseplate: always supported
    f = Field(vals, spacing=1.0)
    rep = Overhang(angle_deg=45.0, build_axis=0).report(f)
    assert rep["unsupported_volume_fraction"] == pytest.approx(0.0, abs=1e-6)
