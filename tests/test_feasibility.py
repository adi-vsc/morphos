"""Tests for the pre-run feasibility gate (morphos.agent.feasibility)."""

import pytest
from morphos.agent.feasibility import check, FeasibilityResult
from morphos.intent import CantileverIntent, ThermalSinkIntent


def test_valid_cantilever_passes():
    """A reasonable CantileverIntent passes check() with ok=True."""
    intent = CantileverIntent(span=50, height=25, load=-1.0, volume_fraction=0.3)
    result = check(intent)
    assert isinstance(result, FeasibilityResult)
    assert result.ok, f"Expected ok=True but got violations: {result.violations}"
    assert result.violations == []


def test_high_voxel_count_fails():
    """An intent with a very large 2-D domain (>5 M voxels) fails with a voxel violation."""
    # 2300 x 2300 = 5,290,000 voxels > 5,000,000 limit.
    intent = ThermalSinkIntent(
        nx=2300, ny=2300, heat_source_W_m3=1.0, k_solid=1.0
    )
    result = check(intent)
    assert not result.ok
    voxel_violations = [v for v in result.violations if "voxel count" in v]
    assert len(voxel_violations) >= 1, (
        f"Expected a voxel-count violation, got: {result.violations}"
    )


def test_thermal_flux_too_high_fails():
    """ThermalSinkIntent where volumetric heat source implies > 1 W/mm³ fails."""
    # heat_source = 2e10 W/m³ → 2e10 * 1e-9 = 20 W/mm³ >> 1 W/mm³ threshold.
    intent = ThermalSinkIntent(
        nx=50, ny=50, heat_source_W_m3=2e10, k_solid=1.0
    )
    result = check(intent)
    assert not result.ok
    thermal_violations = [v for v in result.violations if "heat flux" in v or "thermal" in v.lower()]
    assert len(thermal_violations) >= 1, (
        f"Expected a thermal-flux violation, got: {result.violations}"
    )


def test_out_of_range_param_fails():
    """CantileverIntent with volume_fraction below catalog minimum fails."""
    # volume_fraction=0.02 is below the 0.05 lower bound in several CantileverIntent
    # catalog entries (cross_rib, i_beam_rib, honeycomb_rib).
    intent = CantileverIntent(span=50, height=25, load=-1.0, volume_fraction=0.02)
    result = check(intent)
    assert not result.ok
    range_violations = [v for v in result.violations if "volume_fraction" in v]
    assert len(range_violations) >= 1, (
        f"Expected a volume_fraction range violation, got: {result.violations}"
    )


def test_high_cantilever_stress_fails():
    """CantileverIntent with very large load relative to section height fails."""
    # |load| / height = 600 / 1 = 600 MPa > 500 MPa threshold.
    intent = CantileverIntent(span=50, height=1, load=-600.0, volume_fraction=0.4)
    result = check(intent)
    assert not result.ok
    stress_violations = [v for v in result.violations if "stress" in v or "MPa" in v]
    assert len(stress_violations) >= 1, (
        f"Expected a stress violation, got: {result.violations}"
    )


def test_result_has_suggested_revision_on_failure():
    """A failing check provides a non-empty suggested_revision."""
    intent = CantileverIntent(span=50, height=25, load=-1.0, volume_fraction=0.02)
    result = check(intent)
    assert not result.ok
    assert result.suggested_revision is not None
    assert len(result.suggested_revision) > 0
