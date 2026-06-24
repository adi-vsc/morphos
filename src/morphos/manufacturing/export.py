"""Manufacturing output pipeline: turn an optimized density Field into a
watertight surface mesh plus a machine/material parameter record.

This wires the latent geometry in a :class:`~morphos.spec.DesignResult` to files
a slicer or AM/SLM build prep can consume:

1. Threshold the density field at ``iso_value`` and extract the iso-surface with
   marching cubes (the ``mfg`` extra, scikit-image), zero padded so the surface
   closes into a watertight manifold.
2. Write a binary STL (hand-rolled, no extra dependency) -- the manufacturable
   artifact.
3. Save the thresholded signed-distance volume as a portable ``.npz`` sidecar.
   A true OpenVDB write requires the PicoGK/OpenVDB native runtime; when that is
   present ``vdb_path`` is populated, otherwise it stays ``None`` and the portable
   voxel sidecar carries the volume (kept explicit rather than writing a fake VDB).
4. Score the field against the 26 standard build-plate orientations and attach
   the lowest-scoring one as ``recommended_orientation``, so the operator gets
   an orientation suggestion (support volume, overhang area, build height)
   alongside the geometry rather than having to guess.

The result is a :class:`ManufacturingBundle` and, when ``export_bundle`` is given
the ``DesignResult``, the bundle and its STL/voxel paths are written back onto it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from morphos.field import Field
from morphos.manufacturing.orientation import BuildOrientationScorer, OrientationScore
from morphos.spec import DesignResult


@dataclass
class PrintParams:
    """Machine and material parameters for a single SLM/DMLS build."""

    material: str
    layer_thickness_mm: float
    laser_power_W: float
    scan_speed_mm_s: float
    hatch_spacing_mm: float
    build_axis: int = 2


@dataclass
class ManufacturingBundle:
    """Complete output package for a manufacturable design."""

    stl_path: Path
    voxel_path: Path
    print_params: PrintParams
    field: Field
    report: Optional[object] = None
    vdb_path: Optional[Path] = None
    recommended_orientation: Optional[OrientationScore] = None


def _write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a binary STL from vertex coordinates and triangle indices."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)  # 80-byte header
        fh.write(struct.pack("<I", len(f)))
        for tri in f:
            a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            n = n / norm if norm > 0 else n
            fh.write(struct.pack("<3f", *n))
            for vert in (a, b, c):
                fh.write(struct.pack("<3f", *vert))
            fh.write(struct.pack("<H", 0))  # attribute byte count


def _iso_surface(values: np.ndarray, iso_value: float, spacing):
    """Marching-cubes iso-surface of a 3D density field, zero padded so the
    surface closes into a watertight mesh. Returns (vertices, faces)."""
    try:
        from skimage import measure
    except ImportError as exc:  # pragma: no cover - only without the mfg extra
        raise ImportError(
            "export_bundle needs scikit-image; install the 'mfg' extra "
            "(pip install morphos[mfg])"
        ) from exc

    padded = np.pad(values, 1, mode="constant", constant_values=0.0)
    verts, faces, _normals, _vals = measure.marching_cubes(
        padded, level=iso_value, spacing=tuple(spacing)
    )
    # Undo the one-voxel pad shift so coordinates sit in the original frame.
    verts = verts - np.asarray(spacing, dtype=float)
    return verts, faces


def export_bundle(
    result_or_field,
    params: PrintParams,
    output_dir,
    iso_value: float = 0.5,
    report: Optional[object] = None,
) -> ManufacturingBundle:
    """Export a manufacturable bundle (STL + voxel sidecar + parameters).

    ``result_or_field`` may be a :class:`DesignResult` or a bare :class:`Field`.
    When a ``DesignResult`` is passed, its ``mesh_path`` / ``manufacturing_bundle``
    are populated in place. The density field must be 3D (an STL is a 3D surface)
    and ``iso_value`` must lie strictly inside ``(0, 1)``.
    """
    if not 0.0 < iso_value < 1.0:
        raise ValueError(f"iso_value must be in (0, 1), got {iso_value}")

    if isinstance(result_or_field, DesignResult):
        result = result_or_field
        field = result.field
    else:
        result = None
        field = result_or_field

    values = np.asarray(field.values, dtype=float)
    if values.ndim != 3:
        raise ValueError(
            f"export_bundle needs a 3D density field for STL extraction, got "
            f"{values.ndim}D"
        )
    if values.min() < -1e-9 or values.max() > 1 + 1e-9:
        raise ValueError("density field must lie in [0, 1] for iso thresholding")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stl_path = output_dir / "design.stl"
    voxel_path = output_dir / "design_voxels.npz"

    verts, faces = _iso_surface(values, iso_value, field.spacing)
    _write_binary_stl(stl_path, verts, faces)

    solid = (values >= iso_value).astype(np.int8)
    np.savez_compressed(
        voxel_path, density=values, solid=solid,
        spacing=np.asarray(field.spacing, dtype=float), iso_value=iso_value,
    )

    recommended_orientation = BuildOrientationScorer().best_orientation(field)

    bundle = ManufacturingBundle(
        stl_path=stl_path,
        voxel_path=voxel_path,
        print_params=params,
        field=field,
        report=report,
        vdb_path=None,  # populated only when an OpenVDB/PicoGK writer is available
        recommended_orientation=recommended_orientation,
    )

    if result is not None:
        result.mesh_path = stl_path
        result.manufacturing_bundle = bundle

    return bundle
