"""Step 6: build-orientation scoring for additive manufacturing.

Orientation on the build plate drives support volume, downward-facing surface
quality, and build time, so the export pipeline should recommend an orientation
rather than leaving the operator to guess. ``BuildOrientationScorer`` evaluates
a density Field against the 26 standard face/edge/corner directions of a unit
cube (the engineering candidate set for build-plate orientation) by running the
existing ``Overhang`` self-supporting filter in the given gravity direction, and
reports the orientation with the lowest weighted score.
"""

import numpy as np
import pytest

from fd_gate import fd_gate

from morphos.field import Field
from morphos.manufacturing.orientation import BuildOrientationScorer, OrientationScore


def _solid_cylinder(n=12, axis=0, radius=3.5):
    """A solid cylinder of ``radius`` whose axis runs along grid axis ``axis``."""
    coords = np.mgrid[0:n, 0:n, 0:n].astype(float)
    c = (n - 1) / 2.0
    other_axes = [a for a in range(3) if a != axis]
    d0 = coords[other_axes[0]] - c
    d1 = coords[other_axes[1]] - c
    r = np.sqrt(d0 ** 2 + d1 ** 2)
    return (r <= radius).astype(float)


def test_orientation_score_dataclass_fields():
    score = OrientationScore(
        orientation=(0.0, 0.0, 1.0),
        support_volume_fraction=0.1,
        overhang_area_fraction=0.2,
        projected_build_height=5.0,
        score=0.3,
    )
    assert score.orientation == (0.0, 0.0, 1.0)
    assert score.support_volume_fraction == pytest.approx(0.1)
    assert score.overhang_area_fraction == pytest.approx(0.2)
    assert score.projected_build_height == pytest.approx(5.0)
    assert score.score == pytest.approx(0.3)


def test_score_runs_for_an_axis_aligned_orientation():
    field = Field(_solid_cylinder(axis=2), spacing=1.0)
    scorer = BuildOrientationScorer()
    result = scorer.score(field, (0.0, 0.0, 1.0))
    assert isinstance(result, OrientationScore)
    assert 0.0 <= result.support_volume_fraction <= 1.0
    assert 0.0 <= result.overhang_area_fraction <= 1.0
    assert result.projected_build_height > 0.0


def test_score_runs_for_a_diagonal_orientation():
    # A corner-diagonal direction exercises the non-axis-aligned path.
    field = Field(_solid_cylinder(axis=2), spacing=1.0)
    scorer = BuildOrientationScorer()
    d = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    result = scorer.score(field, tuple(d))
    assert isinstance(result, OrientationScore)
    assert np.isfinite(result.score)
    assert 0.0 <= result.support_volume_fraction <= 1.0 + 1e-6


def test_vertical_cylinder_scores_lower_than_horizontal():
    """A cylinder printed standing up (build axis along its own axis) needs much
    less support than the same cylinder lying flat (build axis perpendicular to
    its axis), since the flat cylinder's round side is a continuous overhang."""
    scorer = BuildOrientationScorer()

    vertical = Field(_solid_cylinder(axis=2), spacing=1.0)
    horizontal = Field(_solid_cylinder(axis=0), spacing=1.0)

    vertical_score = scorer.score(vertical, (0.0, 0.0, 1.0))
    horizontal_score = scorer.score(horizontal, (0.0, 0.0, 1.0))

    assert vertical_score.support_volume_fraction < horizontal_score.support_volume_fraction
    assert vertical_score.score < horizontal_score.score


def test_best_orientation_returns_minimum_score():
    field = Field(_solid_cylinder(axis=2), spacing=1.0)
    scorer = BuildOrientationScorer()
    best = scorer.best_orientation(field)
    assert isinstance(best, OrientationScore)

    all_scores = [
        scorer.score(field, cand).score
        for cand in scorer.candidate_orientations()
    ]
    assert best.score <= min(all_scores) + 1e-9
    assert best.score == pytest.approx(min(all_scores), abs=1e-6)


def test_candidate_orientations_has_26_unit_vectors():
    scorer = BuildOrientationScorer()
    candidates = scorer.candidate_orientations()
    assert len(candidates) == 26
    for c in candidates:
        norm = np.linalg.norm(np.array(c))
        assert norm == pytest.approx(1.0, abs=1e-6)
    # all distinct
    arr = np.array(candidates)
    rounded = np.round(arr, 6)
    unique_rows = {tuple(r) for r in rounded}
    assert len(unique_rows) == 26


def test_orientation_scorer_is_differentiable():
    """The VJP of support_volume_fraction w.r.t. the field passes the
    directional FD gate, so orientation can eventually be co-optimised
    alongside topology (the spec's stated reason this must be differentiable)."""
    rng = np.random.default_rng(7)
    vals = rng.uniform(size=(6, 6, 6))
    scorer = BuildOrientationScorer()
    orientation = (0.0, 0.0, 1.0)

    f = Field(vals, spacing=1.0)
    grad = scorer.support_volume_fraction_vjp(f, orientation)

    def scalar_objective(x):
        return scorer.support_volume_fraction(Field(x, spacing=1.0), orientation)

    fd_gate(scalar_objective, grad, vals, rel=1e-3, h=1e-5)


def test_export_bundle_populates_recommended_orientation(tmp_path):
    pytest.importorskip("skimage")
    from morphos.manufacturing.export import PrintParams, export_bundle

    field = Field(_solid_cylinder(axis=2, n=14), spacing=1.0)
    params = PrintParams(
        material="IN625", layer_thickness_mm=0.04, laser_power_W=285.0,
        scan_speed_mm_s=960.0, hatch_spacing_mm=0.11, build_axis=2,
    )
    bundle = export_bundle(field, params, tmp_path, iso_value=0.5)
    assert bundle.recommended_orientation is not None
    assert isinstance(bundle.recommended_orientation, OrientationScore)
    assert len(bundle.recommended_orientation.orientation) == 3
