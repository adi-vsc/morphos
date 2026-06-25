"""Tests for the morphos CLI (run / info / validate)."""

import json
import struct

import numpy as np
import pytest

pytest.importorskip("skimage")  # run + watertight validate go through export_bundle

from morphos.cli.main import main
from morphos.field import Field
from morphos.manufacturing.export import PrintParams, export_bundle


def _write_cantilever_spec(path):
    spec = {
        "intent": "CantileverIntent",
        "params": {
            "span": 8,
            "height": 8,
            "load": -1.0,
            "volume_fraction": 0.5,
            "max_iter": 5,
        },
    }
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def _solid_blob(n=16):
    c = (n - 1) / 2.0
    z, y, x = np.mgrid[0:n, 0:n, 0:n]
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
    return np.clip(1.5 - 0.5 * r, 0.0, 1.0) * (r < 5.0)


def _watertight_stl(path):
    field = Field(_solid_blob(), spacing=1.0)
    params = PrintParams(
        material="IN625", layer_thickness_mm=0.04, laser_power_W=285.0,
        scan_speed_mm_s=960.0, hatch_spacing_mm=0.11,
    )
    bundle = export_bundle(field, params, path.parent, iso_value=0.5)
    return bundle.stl_path


def _broken_stl(path):
    """A single dangling triangle: edges are unshared, so not watertight."""
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", 1))
        fh.write(struct.pack("<3f", 0, 0, 1))
        for vert in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
            fh.write(struct.pack("<3f", *vert))
        fh.write(struct.pack("<H", 0))
    return path


def test_cli_run_cantilever(tmp_path):
    spec_file = _write_cantilever_spec(tmp_path / "spec.json")
    out = tmp_path / "out"
    code = main(["run", str(spec_file), "--output-dir", str(out)])
    assert code == 0
    assert (out / "design.stl").exists()
    assert (out / "report.json").exists()
    assert (out / "summary.txt").exists()


def test_cli_run_missing_file_exits_1(tmp_path):
    code = main(["run", str(tmp_path / "nope.json"), "--output-dir", str(tmp_path / "o")])
    assert code == 1


def test_cli_info_prints_grid_size(tmp_path, capsys):
    spec_file = _write_cantilever_spec(tmp_path / "spec.json")
    code = main(["info", str(spec_file)])
    out = capsys.readouterr().out
    assert code == 0
    assert "8 × 8" in out


def test_cli_validate_watertight_stl_exits_0(tmp_path):
    stl = _watertight_stl(tmp_path)
    assert main(["validate", str(stl)]) == 0


def test_cli_validate_bad_stl_exits_2(tmp_path):
    stl = _broken_stl(tmp_path / "bad.stl")
    assert main(["validate", str(stl)]) == 2
