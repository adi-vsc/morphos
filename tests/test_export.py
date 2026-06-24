"""Tests for the manufacturing export pipeline (STL + voxel sidecar)."""

from collections import Counter

import numpy as np
import pytest

pytest.importorskip("skimage")

from morphos.field import Field
from morphos.spec import DesignResult
from morphos.manufacturing.export import (
    ManufacturingBundle,
    PrintParams,
    export_bundle,
)


def _solid_blob(n=16):
    """A 3D density field: a solid ball (density 1) in void (density 0)."""
    c = (n - 1) / 2.0
    z, y, x = np.mgrid[0:n, 0:n, 0:n]
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
    return np.clip(1.5 - 0.5 * r + 0.0, 0.0, 1.0) * (r < 5.0)


def _params():
    return PrintParams(
        material="IN625", layer_thickness_mm=0.04, laser_power_W=285.0,
        scan_speed_mm_s=960.0, hatch_spacing_mm=0.11, build_axis=2,
    )


def _stl_triangles(path):
    import struct

    with open(path, "rb") as fh:
        fh.read(80)
        (n,) = struct.unpack("<I", fh.read(4))
        tris = []
        for _ in range(n):
            fh.read(12)  # normal
            tri = [struct.unpack("<3f", fh.read(12)) for _ in range(3)]
            fh.read(2)
            tris.append(tri)
    return tris


def test_export_creates_stl_file(tmp_path):
    field = Field(_solid_blob(), spacing=1.0)
    bundle = export_bundle(field, _params(), tmp_path, iso_value=0.5)
    assert isinstance(bundle, ManufacturingBundle)
    assert bundle.stl_path.exists()
    assert bundle.stl_path.stat().st_size > 84  # header + count + at least one tri


def test_exported_stl_is_watertight(tmp_path):
    field = Field(_solid_blob(), spacing=1.0)
    bundle = export_bundle(field, _params(), tmp_path, iso_value=0.5)
    tris = _stl_triangles(bundle.stl_path)
    # Quantise vertices so shared corners match exactly, then check every edge
    # is shared by exactly two triangles (closed manifold).
    edges = Counter()
    for tri in tris:
        keys = [tuple(np.round(np.array(v), 4)) for v in tri]
        for u, w in ((0, 1), (1, 2), (2, 0)):
            edges[frozenset((keys[u], keys[w]))] += 1
    assert tris and all(c == 2 for c in edges.values())


def test_export_creates_voxel_volume_file(tmp_path):
    field = Field(_solid_blob(), spacing=1.0)
    bundle = export_bundle(field, _params(), tmp_path, iso_value=0.5)
    assert bundle.voxel_path.exists()
    data = np.load(bundle.voxel_path)
    assert "solid" in data and "density" in data
    assert data["solid"].sum() > 0


def test_bundle_carries_performance_report(tmp_path):
    field = Field(_solid_blob(), spacing=1.0)
    report = {"peak_temperature_K": 410.0}
    bundle = export_bundle(field, _params(), tmp_path, iso_value=0.5, report=report)
    assert bundle.report is report
    assert bundle.print_params.material == "IN625"


def test_export_populates_design_result(tmp_path):
    result = DesignResult(field=Field(_solid_blob(), spacing=1.0), figure_of_merit=0.0)
    assert result.is_exported is False
    bundle = export_bundle(result, _params(), tmp_path, iso_value=0.5)
    assert result.is_exported is True
    assert result.mesh_path == bundle.stl_path
    assert result.manufacturing_bundle is bundle


def test_export_rejects_invalid_iso(tmp_path):
    field = Field(_solid_blob(), spacing=1.0)
    with pytest.raises(ValueError):
        export_bundle(field, _params(), tmp_path, iso_value=1.5)


def test_export_rejects_2d_field(tmp_path):
    field = Field(np.ones((8, 8)) * 0.6, spacing=1.0)
    with pytest.raises(ValueError):
        export_bundle(field, _params(), tmp_path, iso_value=0.5)


def test_engine_run_with_export_populates_result(tmp_path):
    from morphos.engine import Engine
    from morphos.physics.analytic import AnalyticOracle
    from morphos.objective.objective import MaximizeValue
    from morphos.optimize.topopt import TopologyOptimizer
    from morphos.spec import DesignSpec

    target = _solid_blob(10)
    spec = DesignSpec(
        initial=Field(np.zeros((10, 10, 10)), spacing=1.0),
        oracle=AnalyticOracle(target=target),
        objective=MaximizeValue(),
        optimizer=TopologyOptimizer(step_size=0.5, max_iter=200, tol=1e-12,
                                    bounds=(0.0, 1.0)),
    )
    result = Engine().run(spec, export_dir=tmp_path, iso_value=0.5)
    assert result.is_exported
    assert result.manufacturing_bundle.stl_path.exists()
