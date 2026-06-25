"""Tests for the public Python API: morphos.run()."""

import io
from contextlib import redirect_stdout

import pytest

pytest.importorskip("skimage")  # morphos.run writes an STL via export_bundle

import morphos
from morphos import CantileverIntent, MorphosResult


def _tiny_intent():
    """A cantilever small enough to run in a fraction of a second."""
    return CantileverIntent(
        span=8, height=8, load=-1.0, volume_fraction=0.5, max_iter=5
    )


def test_run_accepts_design_spec(tmp_path):
    spec = _tiny_intent().build()
    result = morphos.run(spec, output_dir=tmp_path, verbose=False)
    assert isinstance(result, MorphosResult)
    assert result.design_result is not None
    assert result.report is not None
    assert result.bundle is not None
    assert result.output_dir == tmp_path
    assert result.design_result.iterations > 0


def test_run_accepts_design_intent(tmp_path):
    result = morphos.run(_tiny_intent(), output_dir=tmp_path, verbose=False)
    assert isinstance(result, MorphosResult)
    assert result.report is not None
    assert result.bundle is not None


def test_run_writes_all_outputs(tmp_path):
    morphos.run(_tiny_intent(), output_dir=tmp_path, verbose=False)
    assert (tmp_path / "design.stl").exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "summary.txt").exists()


def test_run_returns_elapsed_seconds(tmp_path):
    result = morphos.run(_tiny_intent(), output_dir=tmp_path, verbose=False)
    assert isinstance(result.elapsed_seconds, float)
    assert result.elapsed_seconds > 0.0


def test_run_verbose_false_produces_no_stdout(tmp_path):
    buf = io.StringIO()
    with redirect_stdout(buf):
        morphos.run(_tiny_intent(), output_dir=tmp_path, verbose=False)
    assert buf.getvalue() == ""
