"""Stress-graded gyroid lattice: material follows the load, like bone.

This is the bridge between Morphos's two halves. The physics-optimization leg
solves an elasticity problem and hands its strain-energy field straight to the
geometry-generation leg, which grades a gyroid lattice so wall thickness tracks
local stress: dense at the clamped root, thin at the unloaded corner. The result
is a single watertight, self-supporting SDF, exported to STL.

Run:  python -m examples.graded_lattice
"""

from __future__ import annotations

import numpy as np

from morphos.field import Field
from morphos.geometry.graded import graded_lattice, physics_demand_field
from morphos.geometry.tpms import gyroid_sdf
from morphos.physics.elasticity import ElasticityOracle


def main() -> None:
    # 1. A tip-loaded cantilever: clamp the left edge, pull down at the far
    #    bottom corner. Solve once on a solid probe to read the stress field.
    ny, nx = 24, 48
    nny, nnx = ny + 1, nx + 1
    fixed = [(0, j, d) for j in range(nny) for d in ("x", "y")]
    loads = {(nnx - 1, nny - 1, "y"): -1.0}
    oracle = ElasticityOracle(shape=(ny, nx), fixed_dofs=fixed, loads=loads, penalty=1.0)

    solid = Field(np.ones((ny, nx)), spacing=1.0)
    result = oracle.solve(solid)
    demand2d = physics_demand_field(result.gradient)  # strain-energy density

    # 2. Extrude the 2D stress field into a 3D slab and grade a gyroid to it.
    nz = 12
    shape = (nz, ny, nx)
    demand3d = np.broadcast_to(demand2d, shape).copy()
    grid = Field(np.zeros(shape), spacing=1.0)
    tpms = gyroid_sdf(grid, period=6.0, thickness=0.0)
    envelope = Field(np.full(shape, -1.0), spacing=1.0)  # solid block envelope

    lattice = graded_lattice(envelope, tpms, demand3d, vf_min=0.2, vf_max=0.7)

    # 3. Report the grading: solid fraction at the loaded root vs the free corner.
    root = float(np.mean(lattice.values[:, :, :6] <= 0.0))
    far = float(np.mean(lattice.values[:, : ny // 2, nx - 6 :] <= 0.0))
    overall = float(np.mean(lattice.values <= 0.0))
    print(f"overall solid fraction : {overall:.3f}")
    print(f"root (high stress)     : {root:.3f}")
    print(f"corner (low stress)    : {far:.3f}")
    print(f"grading ratio root/corner: {root / max(far, 1e-9):.2f}x")

    # 4. Export the watertight STL (needs the 'mfg' extra: pip install morphos[mfg]).
    try:
        from morphos.manufacturing.export import _iso_surface, _write_binary_stl
        from pathlib import Path

        density = np.clip(0.5 - lattice.values / 2.0, 0.0, 1.0)
        verts, faces = _iso_surface(density, 0.5, grid.spacing)
        out = Path("graded_lattice_out")
        out.mkdir(exist_ok=True)
        _write_binary_stl(out / "graded_lattice.stl", verts, faces)
        print(f"wrote {out / 'graded_lattice.stl'} ({len(faces)} triangles)")
    except ImportError:
        print("(install morphos[mfg] to export the STL)")


if __name__ == "__main__":
    main()
