"""Build-orientation scoring for additive manufacturing.

Orientation on the build plate is a free design decision the export pipeline
otherwise leaves to the operator's guesswork, yet it drives support volume,
downward-facing surface quality, and build time (proportional to height along
the build axis). ``BuildOrientationScorer`` scores a density Field against the
26 standard face/edge/corner directions of a unit cube -- the usual engineering
candidate set for build-plate orientation studies -- and recommends the one
with the lowest weighted score.

Support volume is computed by running the existing Langelaar ``Overhang``
filter (see ``constraints.py``) in the candidate gravity direction, exactly as
specified: ``Overhang`` only understands an integer, grid-aligned build axis,
so an axis-aligned candidate (one of the 6 face directions) runs directly on
the field, while an off-axis candidate (one of the 12 edge or 8 corner
diagonals) runs on a copy of the field resampled into a frame where the
candidate direction becomes the grid's axis-0. The resampling is a fixed
trilinear interpolation (an explicit linear map of the input voxels), so the
vector-Jacobian product threads through it as the same resampling applied to
the incoming gradient with the inverse rotation -- this is what makes
``support_volume_fraction`` differentiable for every one of the 26 directions,
including the axis-aligned ones where the "resampling" is the identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import ndimage

from morphos.field import Field
from morphos.manufacturing.constraints import Overhang


@dataclass
class OrientationScore:
    """Result of scoring a Field against one candidate build orientation."""

    orientation: Tuple[float, float, float]
    support_volume_fraction: float
    overhang_area_fraction: float
    projected_build_height: float
    score: float


def _normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("orientation vector must be non-zero")
    return v / n


def _rotation_to_axis0(direction: np.ndarray) -> np.ndarray:
    """A rotation matrix R such that ``R @ direction`` is aligned with
    grid axis 0, i.e. ``R @ direction = (1, 0, 0)`` (up to sign/orthogonal
    freedom about that axis, which does not matter for a layer-walking
    overhang filter that only cares about the axis-0 component).

    Built from an orthonormal basis with ``direction`` as the first row:
    a Householder-free Gram-Schmidt construction that is exact, cheap, and
    well-defined for every unit vector (including the axis-aligned ones,
    where it reduces to a signed permutation matrix).
    """
    e0 = direction
    # Pick a helper vector not parallel to e0 to seed Gram-Schmidt.
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, e0)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = helper - np.dot(helper, e0) * e0
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(e0, e1)
    return np.stack([e0, e1, e2], axis=0)


def _axis_aligned_index(direction: np.ndarray, atol: float = 1e-8):
    """If ``direction`` is (within tolerance) +/- a grid axis, return
    ``(axis, sign)``; otherwise ``None``. Drives the exact, trivial-VJP
    fast path for the 6 face directions."""
    for axis in range(3):
        unit = np.zeros(3)
        unit[axis] = 1.0
        if np.allclose(direction, unit, atol=atol):
            return axis, 1
        if np.allclose(direction, -unit, atol=atol):
            return axis, -1
    return None


def _to_build_frame(values: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Resample ``values`` into a frame where ``direction`` is grid axis 0.

    Axis-aligned directions use an exact moveaxis/flip (no interpolation,
    no information loss); off-axis directions use a linear trilinear
    resampling via ``scipy.ndimage.affine_transform``, recentred on the
    grid so the rotation pivots about the volume's centre.
    """
    aligned = _axis_aligned_index(direction)
    if aligned is not None:
        axis, sign = aligned
        out = np.moveaxis(values, axis, 0)
        if sign < 0:
            out = np.flip(out, axis=0)
        return np.ascontiguousarray(out)

    R = _rotation_to_axis0(direction)
    shape = np.asarray(values.shape, dtype=float)
    center = (shape - 1.0) / 2.0
    # affine_transform is "pull" resampling: out[o] = in[matrix @ o + offset].
    # We want out frame coordinates o to map back to input coordinates via
    # R^-1 = R.T (R is orthonormal), pivoting about the volume centre.
    matrix = R.T
    offset = center - matrix @ center
    out = ndimage.affine_transform(
        values, matrix, offset=offset, order=1, mode="constant", cval=0.0,
    )
    return out


def _from_build_frame(values: np.ndarray, direction: np.ndarray, shape) -> np.ndarray:
    """Inverse of :func:`_to_build_frame`, used to map a gradient living in
    the build frame back onto the original grid (the VJP of the resampling).
    """
    aligned = _axis_aligned_index(direction)
    if aligned is not None:
        axis, sign = aligned
        out = values
        if sign < 0:
            out = np.flip(out, axis=0)
        out = np.moveaxis(out, 0, axis)
        return np.ascontiguousarray(out)

    R = _rotation_to_axis0(direction)
    shape = np.asarray(shape, dtype=float)
    center = (shape - 1.0) / 2.0
    # Forward map this time: input (original) coordinates pull from the
    # build-frame array via matrix = R, the inverse of R.T used above.
    matrix = R
    offset = center - matrix @ center
    out = ndimage.affine_transform(
        values, matrix, offset=offset, order=1, mode="constant", cval=0.0,
    )
    return out


def _candidate_orientations() -> List[Tuple[float, float, float]]:
    """The 26 standard face/edge/corner directions of a unit cube."""
    candidates = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if x == 0 and y == 0 and z == 0:
                    continue
                v = _normalize((x, y, z))
                candidates.append(tuple(v.tolist()))
    return candidates


class BuildOrientationScorer:
    """Score a density Field against a set of candidate build orientations.

    ``score()`` evaluates one orientation; ``best_orientation()`` evaluates the
    26 standard candidates (plus any user-supplied extras) and returns the one
    with the lowest weighted ``score``. The weights trade off support volume,
    unsupported surface area, and build height (a proxy for build time); all
    three terms are normalised fractions/ratios so the default equal weights
    are dimensionally sensible.
    """

    def __init__(
        self,
        angle_deg: float = 45.0,
        w_support: float = 1.0,
        w_overhang_area: float = 1.0,
        w_height: float = 0.1,
    ) -> None:
        self.angle_deg = float(angle_deg)
        self.w_support = float(w_support)
        self.w_overhang_area = float(w_overhang_area)
        self.w_height = float(w_height)

    def candidate_orientations(self) -> List[Tuple[float, float, float]]:
        return _candidate_orientations()

    def _overhang_for(self, direction: np.ndarray) -> Overhang:
        return Overhang(angle_deg=self.angle_deg, build_axis=0)

    def support_volume_fraction(self, field: Field, orientation) -> float:
        """Fraction of the field's total density that ``Overhang`` (run with
        the candidate direction as gravity) finds unsupported."""
        direction = _normalize(orientation)
        values = np.clip(np.asarray(field.values, dtype=float), 0.0, None)
        build_values = _to_build_frame(values, direction)
        build_field = Field(build_values, spacing=field.spacing[0])
        overhang = self._overhang_for(direction)
        printed = np.clip(overhang.project(build_field).values, 0.0, None)
        deficit = np.clip(build_values - printed, 0.0, None)
        total = values.sum()
        return float(deficit.sum() / total) if total > 0.0 else 0.0

    def support_volume_fraction_vjp(self, field: Field, orientation) -> np.ndarray:
        """d(support_volume_fraction)/d(field.values), exact for axis-aligned
        orientations (moveaxis/flip is its own trivial-Jacobian inverse) and
        the linear-resampling chain rule for off-axis ones."""
        direction = _normalize(orientation)
        values = np.clip(np.asarray(field.values, dtype=float), 0.0, None)
        rho_positive = (np.asarray(field.values, dtype=float) > 0.0).astype(float)
        build_values = _to_build_frame(values, direction)
        build_field = Field(build_values, spacing=field.spacing[0])
        overhang = self._overhang_for(direction)
        printed = overhang.project(build_field).values
        deficit_raw = build_values - printed
        deficit = np.clip(deficit_raw, 0.0, None)
        total = values.sum()
        if total <= 0.0:
            return np.zeros_like(values)

        active = (deficit_raw > 0.0).astype(float)
        sum_deficit = float(deficit.sum())

        # d(sum_deficit)/d(build_values) = active * (1 - d(printed)/d(build_values)),
        # where the second term is exactly Overhang's own VJP (it IS that
        # Jacobian-vector product, applied to the "active" mask as the
        # incoming gradient).
        d_printed = overhang.vjp(build_field, active)
        d_build_values = active - d_printed

        # Chain back through the (linear) resampling into build frame.
        d_values_from_build = _from_build_frame(d_build_values, direction, values.shape)

        # Quotient rule for sum_deficit / total, where total = sum(clip(rho,0,None)).
        d_total = rho_positive
        grad = (d_values_from_build * total - sum_deficit * d_total) / (total * total)
        # The clip at deficit/build_values' lower bound and rho's lower bound
        # only zero out gradient where the unclipped quantity was already
        # negative; rho_positive already accounts for the density clip, and
        # build_values was clipped before resampling/Overhang so its
        # zero-gradient region is carried through d_values_from_build itself.
        return grad

    def _overhang_area_fraction(self, field: Field, direction: np.ndarray) -> float:
        values = np.clip(np.asarray(field.values, dtype=float), 0.0, None)
        build_values = _to_build_frame(values, direction)
        build_field = Field(build_values, spacing=field.spacing[0])
        overhang = self._overhang_for(direction)
        printed = np.clip(overhang.project(build_field).values, 0.0, None)
        # A voxel's surface is "unsupported" if its own deficit is material
        # relative to a unit reference area (one voxel face), proxying area
        # fraction by the count of voxels with a non-trivial deficit over the
        # count of solid voxels (a discretisation of overhang surface area).
        deficit = np.clip(build_values - printed, 0.0, None)
        solid_mask = build_values > 1e-6
        n_solid = float(solid_mask.sum())
        if n_solid <= 0.0:
            return 0.0
        unsupported_mask = deficit > 1e-3
        return float(np.sum(unsupported_mask & solid_mask) / n_solid)

    def _projected_build_height(self, field: Field, direction: np.ndarray) -> float:
        """Physical extent of the field's solid voxels along ``direction``,
        a proxy for build time (more layers along the build axis -> longer
        print). Uses the voxel grid's physical coordinates dotted with the
        (unit) build direction."""
        values = np.asarray(field.values, dtype=float)
        spacing = np.asarray(field.spacing, dtype=float)
        solid = np.argwhere(values > 0.5)
        if solid.size == 0:
            return 0.0
        coords = solid * spacing[np.newaxis, :]
        projected = coords @ direction
        return float(projected.max() - projected.min())

    def score(self, field: Field, orientation) -> OrientationScore:
        direction = _normalize(orientation)
        support = self.support_volume_fraction(field, direction)
        overhang_area = self._overhang_area_fraction(field, direction)
        height = self._projected_build_height(field, direction)
        weighted = (
            self.w_support * support
            + self.w_overhang_area * overhang_area
            + self.w_height * height
        )
        return OrientationScore(
            orientation=tuple(direction.tolist()),
            support_volume_fraction=support,
            overhang_area_fraction=overhang_area,
            projected_build_height=height,
            score=float(weighted),
        )

    def best_orientation(
        self, field: Field, n_candidates: int = 26, extra_candidates=None
    ) -> OrientationScore:
        candidates = self.candidate_orientations()
        if extra_candidates:
            candidates = list(candidates) + [tuple(_normalize(c).tolist()) for c in extra_candidates]
        scored = [self.score(field, c) for c in candidates]
        return min(scored, key=lambda s: s.score)
