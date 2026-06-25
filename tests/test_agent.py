"""Agent layer: natural-language routing and end-to-end text -> STL.

These tests lock in two things permanently: (1) requests route to the correct
generator and unsupported requests are refused, and (2) the counter-flow gyroid
exchanger stays leak-tight with each fluid a single connected network -- the
quality properties we validated by hand. If a code change breaks the separation,
this test fails.
"""
import numpy as np
import pytest

from morphos.agent import interpret, design_from_text
from morphos.agent.router import extract_common


def test_routing_picks_the_right_generator():
    assert interpret("a counter-flow heat exchanger core").generator.name == "gyroid_heat_exchanger"
    assert interpret("a lightweight gyroid lattice bracket").generator.name == "gyroid_lattice_block"
    assert interpret("a pin-fin heat sink for a CPU").generator.name == "pin_fin_heat_sink"


def test_unsupported_request_is_refused_not_faked():
    plan = interpret("a turbopump impeller with curved blades")
    assert plan.generator is None
    assert "No generator matched" in plan.rationale


def test_parameter_extraction():
    c = extract_common("100 mm cylinder, wall thickness 1.4 mm, cell size 30 mm, 25%")
    assert c["size_mm"] == 100.0
    assert c["wall_mm"] == 1.4
    assert c["cell_mm"] == 30.0
    assert abs(c["vol_fraction"] - 0.25) < 1e-9
    assert c["shape"] == "cylinder"


def test_exchanger_is_built_sealed_and_single_network(tmp_path):
    r = design_from_text(
        "counter-flow heat exchanger, 56 mm cylinder, cell size 22 mm, wall thickness 1.4 mm",
        tmp_path,
    )
    assert r.generator == "gyroid_heat_exchanger"
    assert r.stl_path.exists()
    assert r.triangles > 0
    # The locked-in quality guarantees:
    assert r.metrics["leak_paths"] == 0.0
    assert r.metrics["hot_single_network_fraction"] > 0.99
    assert r.metrics["cold_single_network_fraction"] > 0.99
    assert 0.1 < r.metrics["wall_fraction"] < 0.6


def test_lattice_and_pinfin_build_valid_stls(tmp_path):
    for prompt, gen in [
        ("a lightweight gyroid lattice block at 30% volume fraction, 60 mm box", "gyroid_lattice_block"),
        ("a pin-fin heat sink, 60 mm, 6 mm pitch", "pin_fin_heat_sink"),
    ]:
        r = design_from_text(prompt, tmp_path)
        assert r.generator == gen
        assert r.stl_path.exists() and r.triangles > 0
        assert 0.02 < r.solid_fraction < 0.95
