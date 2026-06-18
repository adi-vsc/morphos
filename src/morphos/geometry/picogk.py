"""PicoGK geometry backend: the production implicit kernel, wired and verified.

PicoGK (Leap71) is the intended production geometry kernel. It represents
geometry as implicit fields on a voxel grid, the same representation Morphos
uses, so it plugs in behind :class:`morphos.geometry.kernel.GeometryKernel`
without changing the engine core.

Interop runs through ctypes directly to the native PicoGK runtime -- no .NET
layer (see ``docs/design/2026-06-18-picogk-interop-spike.md`` and the binding in
:mod:`morphos.geometry._picogk_native`). PicoGK is narrow-band: it stores signed
distance only near the surface, so the deep interior reads as the positive
background. :func:`morphos.geometry._picogk_native.solidify` floods the exterior
to recover a sign-correct solid, which :meth:`build` then samples onto the
kernel grid. The result is verified against an analytic sphere in
``tests/test_picogk_kernel.py`` (the discipline every other backend follows).
"""

from __future__ import annotations

import numpy as np

from morphos.field import Field
from morphos.geometry import _picogk_native as _native
from morphos.geometry.kernel import GeometryKernel


class PicoGKKernel(GeometryKernel):
    """Production PicoGK backend: builds a solid signed-distance Field."""

    def build(self, spec: dict, picogk_voxel_mm: float | None = None) -> Field:
        if not _native.picogk_available():
            raise RuntimeError(
                "native PicoGK runtime is not available; set $PICOGK_NATIVE_DIR "
                "or install the runtime (see _picogk_native), or use VoxelKernel"
            )
        if len(self.grid_shape) != 3:
            raise ValueError("PicoGKKernel builds on a 3D grid")
        if max(self.spacing) - min(self.spacing) > 1e-12:
            raise ValueError("PicoGKKernel assumes isotropic voxel spacing")

        primitive = spec.get("primitive")
        voxel_mm = self.spacing[0]
        # PicoGK can build at a different (still isotropic) voxel size than the
        # kernel grid requests; default to matching exactly (no resample needed).
        build_voxel_mm = float(picogk_voxel_mm) if picogk_voxel_mm is not None else voxel_mm

        if primitive == "sphere":
            vf = _native.sphere_sdf_volume(
                center_mm=spec["center"], radius_mm=float(spec["radius"]), voxel_mm=build_voxel_mm
            )
        elif primitive == "box":
            vertices, triangles = _native.box_mesh(center=spec["center"], size=spec["size"])
            vf = _native.mesh_sdf_volume(vertices, triangles, voxel_mm=build_voxel_mm)
        elif primitive == "cylinder":
            vertices, triangles = _native.cylinder_mesh(
                center=spec["center"],
                axis=spec["axis"],
                radius=float(spec["radius"]),
                height=float(spec["height"]),
                segments=int(spec.get("segments", 32)),
            )
            vf = _native.mesh_sdf_volume(vertices, triangles, voxel_mm=build_voxel_mm)
        else:
            raise ValueError(f"unsupported PicoGK primitive: {primitive!r}")

        solid = _native.solidify(vf.volume, vf.background)  # voxel units, (nz, ny, nx)

        zoom_factor = build_voxel_mm / voxel_mm
        if abs(zoom_factor - 1.0) > 1e-9:
            # Resample PicoGK's own block onto the kernel's requested spacing
            # before placement: the array gets denser/sparser by zoom_factor,
            # and the distance values (still in build_voxel_mm units) need the
            # same rescale to read correctly in voxel_mm units afterwards.
            solid = _native.resample_volume(solid, zoom_factor) * zoom_factor
            origin = _native.resample_origin(vf.origin, zoom_factor)
            background = vf.background * zoom_factor
        else:
            origin = vf.origin
            background = vf.background

        values = self._place_on_grid(solid, origin, background) * voxel_mm
        return Field(values, self.spacing)

    def _place_on_grid(self, solid, origin, background) -> np.ndarray:
        """Drop PicoGK's active block into the kernel grid at its voxel origin.

        PicoGK sizes its volume to the active region; everything outside that box
        is exterior, so the kernel grid starts at ``+background`` and the solid
        block (axes reordered from PicoGK's (z, y, x) to the kernel's (x, y, z))
        is copied in at the origin offset, clipped to the grid bounds.
        """
        out = np.full(self.grid_shape, float(background), dtype=float)
        block = np.ascontiguousarray(solid.transpose(2, 1, 0))  # (nx, ny, nz)

        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        for axis in range(3):
            start = origin[axis]
            n_src = block.shape[axis]
            n_dst = self.grid_shape[axis]
            d0 = max(start, 0)
            d1 = min(start + n_src, n_dst)
            if d1 <= d0:
                return out  # active block falls entirely outside the grid
            dst[axis] = slice(d0, d1)
            src[axis] = slice(d0 - start, d1 - start)
        out[tuple(dst)] = block[tuple(src)]
        return out
