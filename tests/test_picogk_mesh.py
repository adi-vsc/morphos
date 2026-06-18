"""Unit tests for the box/cylinder mesh builders, with no native dependency.

PicoGK has no native box or cylinder primitive (only sphere and capsule); the
real C# API builds these by hand-assembling a ``Mesh`` (vertices + triangles)
and rasterizing it with ``Voxels_RenderMesh``. ``box_mesh`` / ``cylinder_mesh``
build that vertex/triangle data in plain Python, with no ctypes dependency, so
the geometry itself -- closed, correctly wound, correct volume -- is verified
everywhere, including hosts without the native PicoGK runtime.
"""

import numpy as np
import pytest

from morphos.geometry import _picogk_native as pk


def _mesh_signed_volume(vertices, triangles):
    """Signed volume of a closed triangle mesh via the divergence theorem.

    Sum of signed tetrahedron volumes (origin, A, B, C) over every triangle.
    Positive for outward-facing (right-handed / CCW from outside) winding.
    """
    total = 0.0
    for a, b, c in triangles:
        va, vb, vc = vertices[a], vertices[b], vertices[c]
        total += np.dot(va, np.cross(vb, vc)) / 6.0
    return total


def _mesh_is_closed(triangles, n_vertices):
    """Every undirected edge must be shared by exactly two triangles."""
    from collections import Counter

    edges = Counter()
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            edges[frozenset((u, v))] += 1
    return all(count == 2 for count in edges.values())


def test_box_mesh_has_eight_vertices_and_twelve_triangles():
    vertices, triangles = pk.box_mesh(center=(0.0, 0.0, 0.0), size=(2.0, 4.0, 6.0))
    assert len(vertices) == 8
    assert len(triangles) == 12


def test_box_mesh_vertices_sit_on_the_requested_bounds():
    center = (1.0, 2.0, 3.0)
    size = (2.0, 4.0, 6.0)
    vertices, _ = pk.box_mesh(center=center, size=size)
    vertices = np.asarray(vertices, dtype=float)
    lo = np.array(center) - np.array(size) / 2.0
    hi = np.array(center) + np.array(size) / 2.0
    assert np.allclose(vertices.min(axis=0), lo)
    assert np.allclose(vertices.max(axis=0), hi)


def test_box_mesh_is_closed_and_correctly_wound():
    vertices, triangles = pk.box_mesh(center=(0.0, 0.0, 0.0), size=(2.0, 3.0, 4.0))
    assert _mesh_is_closed(triangles, len(vertices))
    vol = _mesh_signed_volume(np.asarray(vertices, dtype=float), triangles)
    assert vol == pytest.approx(2.0 * 3.0 * 4.0)


def test_cylinder_mesh_is_closed_and_has_correct_volume():
    radius, height, segments = 2.0, 5.0, 32
    vertices, triangles = pk.cylinder_mesh(
        center=(0.0, 0.0, 0.0), axis=(0, 0, 1), radius=radius, height=height, segments=segments
    )
    assert _mesh_is_closed(triangles, len(vertices))
    vol = _mesh_signed_volume(np.asarray(vertices, dtype=float), triangles)
    expected = np.pi * radius**2 * height
    # an N-sided prism underestimates a true cylinder's area by sin(pi/N)/(pi/N)
    assert vol == pytest.approx(expected, rel=0.02)


def test_cylinder_mesh_along_arbitrary_axis_has_correct_volume():
    radius, height, segments = 1.5, 4.0, 32
    axis = (1.0, 1.0, 1.0)
    vertices, triangles = pk.cylinder_mesh(
        center=(1.0, -2.0, 3.0), axis=axis, radius=radius, height=height, segments=segments
    )
    assert _mesh_is_closed(triangles, len(vertices))
    vol = _mesh_signed_volume(np.asarray(vertices, dtype=float), triangles)
    expected = np.pi * radius**2 * height
    assert vol == pytest.approx(expected, rel=0.02)
