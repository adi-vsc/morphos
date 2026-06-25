"""End-to-end regression test for the 3D counter-flow heat exchanger example.

This is the integration test for the whole stack: a real 3D conjugate-heat
topology optimization, a manufacturable STL export, and a serialisable
performance report. It runs the optimization once on a coarse grid (shared
across the assertions via a module-scoped fixture) to keep it fast.
"""

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from morphos import Engine, Field
from morphos.report import PerformanceReport, build_report


def _load_example():
    path = Path(__file__).resolve().parents[1] / "examples" / "heat_exchanger_3d.py"
    spec = importlib.util.spec_from_file_location("heat_exchanger_3d", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hx = _load_example()
SHAPE = (8, 8, 12)
VOLUME_FRACTION = 0.4


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """Run the heat-exchanger optimization once, exporting a bundle, and report
    the baseline (uniform-density) thermal resistance for comparison."""
    pytest.importorskip("skimage")
    out = tmp_path_factory.mktemp("hx_output")
    spec, oracle = hx.build_spec(
        shape=SHAPE, volume_fraction=VOLUME_FRACTION, max_iter=20
    )
    result = Engine().run(spec, export_dir=out, iso_value=0.5)
    uniform = Field(np.full(SHAPE, VOLUME_FRACTION), spacing=1.0)
    r_uniform = hx.thermal_resistance(oracle, uniform)
    r_opt = hx.thermal_resistance(oracle, result.field)
    return {
        "oracle": oracle,
        "result": result,
        "r_uniform": r_uniform,
        "r_opt": r_opt,
        "out": out,
    }


def test_heat_exchanger_objective_improves_over_uniform_init(solved):
    # After optimization the design conducts heat better than a uniform slab at
    # the same volume fraction: lower thermal resistance.
    assert solved["r_opt"] < solved["r_uniform"]
    # Volume fraction is held at the target by the VolumeConstraint.
    assert np.isclose(solved["result"].field.values.mean(), VOLUME_FRACTION, atol=1e-3)


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


def test_heat_exchanger_bundle_is_watertight(solved):
    bundle = solved["result"].manufacturing_bundle
    assert bundle is not None and bundle.stl_path.exists()
    tris = _stl_triangles(bundle.stl_path)
    edges = Counter()
    for tri in tris:
        keys = [tuple(np.round(np.array(v), 4)) for v in tri]
        for u, w in ((0, 1), (1, 2), (2, 0)):
            edges[frozenset((keys[u], keys[w]))] += 1
    assert tris and all(c == 2 for c in edges.values())


def test_heat_exchanger_report_serialises(solved, tmp_path):
    report = build_report(solved["result"], solved["oracle"])
    report_path = tmp_path / "report.json"
    report_path.write_text(report.to_json())

    restored = PerformanceReport.from_json(report_path.read_text())
    assert restored.quantities == report.quantities
    assert np.isclose(restored.figure_of_merit, report.figure_of_merit)
    assert np.isclose(restored.mass_fraction, report.mass_fraction)
    assert "thermal_resistance_K_per_W" in restored.quantities
