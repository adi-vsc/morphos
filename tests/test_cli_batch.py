"""Tests for the ``morphos batch`` subcommand."""

import json

import pytest

pytest.importorskip("skimage")  # batch runs the full output suite per spec

from morphos.cli.main import main


def _write_spec(path, max_iter=3):
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "intent": "CantileverIntent",
                "params": {
                    "span": 6,
                    "height": 6,
                    "load": -1.0,
                    "volume_fraction": 0.5,
                    "max_iter": max_iter,
                },
            }
        ),
        encoding="utf-8",
    )


def test_batch_runs_all_specs(tmp_path):
    specs = tmp_path / "specs"
    specs.mkdir()
    _write_spec(specs / "beam_a.json")
    _write_spec(specs / "beam_b.json")
    out = tmp_path / "batch_out"

    code = main(["batch", str(specs), "--output-dir", str(out), "--quiet"])

    assert code == 0
    assert (out / "beam_a" / "design.stl").exists()
    assert (out / "beam_b" / "design.stl").exists()


def test_batch_partial_failure_exits_3(tmp_path):
    specs = tmp_path / "specs"
    specs.mkdir()
    _write_spec(specs / "good.json")
    (specs / "broken.json").write_text("{ this is not valid json", encoding="utf-8")
    out = tmp_path / "batch_out"

    code = main(["batch", str(specs), "--output-dir", str(out), "--quiet"])

    assert code == 3
    assert (out / "good" / "design.stl").exists()
    assert (out / "broken" / "error.txt").exists()


def test_batch_jobs_flag_accepts_integer(tmp_path):
    specs = tmp_path / "specs"
    specs.mkdir()
    _write_spec(specs / "beam_a.json")
    _write_spec(specs / "beam_b.json")
    out = tmp_path / "batch_out"

    code = main(["batch", str(specs), "--output-dir", str(out), "--jobs", "2", "--quiet"])

    assert code == 0
    assert (out / "beam_a" / "design.stl").exists()
    assert (out / "beam_b" / "design.stl").exists()
