"""Tests for the self-contained HTML run report (morphos.viz)."""

import json
import re

import numpy as np
import pytest

pytest.importorskip("skimage")  # report.html is written after a full run

import morphos
from morphos.intent import CantileverIntent
from morphos.viz import render_html_report

_SECTIONS = [
    "Run summary",
    "Convergence",
    "Density slices",
    "Manufacturing summary",
    "Downloads",
]


@pytest.fixture(scope="module")
def report_html(tmp_path_factory):
    out = tmp_path_factory.mktemp("viz")
    intent = CantileverIntent(
        span=8, height=8, load=-1.0, volume_fraction=0.5, max_iter=5
    )
    result = morphos.run(intent, output_dir=out, verbose=False)
    path = out / "report.html"
    render_html_report(result, path)
    return path.read_text(encoding="utf-8"), result


def test_html_report_is_self_contained(report_html):
    html, _ = report_html
    # No attribute may point at an external resource (http/https/protocol-rel).
    for attr in ("src", "href"):
        for m in re.finditer(rf'{attr}\s*=\s*"([^"]*)"', html):
            url = m.group(1)
            assert not url.startswith(("http://", "https://", "//")), url


def test_html_report_contains_all_sections(report_html):
    html, _ = report_html
    for heading in _SECTIONS:
        assert heading in html, f"missing section heading: {heading}"


def test_html_report_density_slice_matches_field(report_html):
    html, result = report_html
    m = re.search(r"const MORPHOS = (\{.*?\});", html, re.DOTALL)
    assert m, "embedded MORPHOS payload not found"
    payload = json.loads(m.group(1))
    grid = np.array(payload["slices"][0]["grid"], dtype=float)
    field = np.asarray(result.design_result.field.values, dtype=float)
    expected = np.round(field, 3)  # 2D cantilever -> single density map
    assert grid.shape == expected.shape
    assert np.allclose(grid, expected, atol=1e-3)


def test_html_report_under_150kb(report_html):
    html, _ = report_html
    assert len(html.encode("utf-8")) < 150 * 1024
