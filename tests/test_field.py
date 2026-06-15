import numpy as np
import pytest

from morphos.field import Field


def test_construct_2d_field_metadata():
    values = np.zeros((4, 6))
    f = Field(values, spacing=0.5)
    assert f.shape == (4, 6)
    assert f.ndim == 2
    assert f.spacing == (0.5, 0.5)


def test_scalar_spacing_broadcasts_per_axis():
    f = Field(np.zeros((2, 3, 4)), spacing=2.0)
    assert f.spacing == (2.0, 2.0, 2.0)


def test_per_axis_spacing_preserved():
    f = Field(np.zeros((2, 3)), spacing=(0.5, 1.5))
    assert f.spacing == (0.5, 1.5)


def test_voxel_volume_is_product_of_spacing():
    f = Field(np.zeros((2, 2, 2)), spacing=(0.5, 2.0, 3.0))
    assert f.voxel_volume == pytest.approx(0.5 * 2.0 * 3.0)


def test_physical_size_is_shape_times_spacing():
    f = Field(np.zeros((4, 6)), spacing=(0.5, 2.0))
    assert f.physical_size == pytest.approx((2.0, 12.0))


def test_non_positive_spacing_raises():
    with pytest.raises(ValueError):
        Field(np.zeros((2, 2)), spacing=0.0)
    with pytest.raises(ValueError):
        Field(np.zeros((2, 2)), spacing=(1.0, -1.0))


def test_spacing_length_mismatch_raises():
    with pytest.raises(ValueError):
        Field(np.zeros((2, 2)), spacing=(1.0, 1.0, 1.0))


def test_values_must_be_ndarray():
    with pytest.raises(TypeError):
        Field([[0, 0], [0, 0]], spacing=1.0)


def test_like_copies_grid_with_new_values():
    f = Field(np.zeros((3, 3)), spacing=(1.0, 2.0))
    g = f.like(np.ones((3, 3)))
    assert g.spacing == (1.0, 2.0)
    assert np.all(g.values == 1.0)
    # original untouched
    assert np.all(f.values == 0.0)


def test_like_rejects_shape_mismatch():
    f = Field(np.zeros((3, 3)), spacing=1.0)
    with pytest.raises(ValueError):
        f.like(np.ones((2, 2)))


def test_copy_is_independent():
    f = Field(np.zeros((2, 2)), spacing=1.0)
    g = f.copy()
    g.values[0, 0] = 9.0
    assert f.values[0, 0] == 0.0
