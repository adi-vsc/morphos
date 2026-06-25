"""End-to-end 3D counter-flow plate heat exchanger.

This is the integration demo for the whole Morphos stack: it minimises the
thermal resistance of a counter-flow heat exchanger on a 3D voxel grid, subject
to a volume-fraction constraint, then exports a manufacturable bundle.

Pipeline exercised here:

  * physics: ``ConjugateHeatOracle`` in 3D (SIMP conductivity plus a frozen
    counter-flow advection field, SUPG stabilised);
  * initial design: the ``tree_manifold`` component from the geometry library,
    thresholded to a binary occupancy that the optimizer then redistributes;
  * optimization: ``TopologyOptimizer`` with SIMP p-continuation, a density
    filter, and a ``VolumeConstraint`` that holds the volume fraction at the
    target throughout;
  * manufacturing: ``MinFeatureSize`` / ``MinWallThickness`` / ``Overhang``
    (build axis z) manufacturability reports, plus a recommended build
    orientation attached to the exported bundle;
  * reporting: a ``PerformanceReport`` serialised to ``report.json``.

Run from the repository root:

    python examples/heat_exchanger_3d.py

It writes ``./output/design.stl`` and ``./output/report.json`` and checkpoints
to ``./checkpoints/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from morphos import Field, Engine, DesignSpec, MaximizeValue
from morphos.optimize.topopt import TopologyOptimizer
from morphos.physics.conjugate_heat import ConjugateHeatOracle
from morphos.manufacturing.constraints import (
    MinFeatureSize,
    MinWallThickness,
    Overhang,
    VolumeConstraint,
)
from morphos.library.manifolds import tree_manifold


def counterflow_velocity(shape: Tuple[int, int, int], speed: float = 0.5) -> np.ndarray:
    """A frozen counter-flow velocity field on the node grid of ``shape``.

    The upper half of the domain (in z) flows in +x, the lower half flows in
    -x: two streams crossing the solid in opposite directions, the defining
    feature of a counter-flow plate exchanger. Returns an
    ``(nnz, nny, nnx, 3)`` array.
    """
    nelz, nely, nelx = shape
    nnz, nny, nnx = nelz + 1, nely + 1, nelx + 1
    vel = np.zeros((nnz, nny, nnx, 3))
    upper = np.arange(nnz)[:, None, None] >= (nnz / 2.0)
    vx = np.where(upper, speed, -speed)
    vel[..., 0] = np.broadcast_to(vx, (nnz, nny, nnx))
    return vel


def inlet_face_nodes(shape: Tuple[int, int, int]) -> List[int]:
    """Flat node indices on the x=0 inlet face, matching the 3D node order."""
    nelz, nely, nelx = shape
    nnz, nny, nnx = nelz + 1, nely + 1, nelx + 1
    return [(iz * nny + iy) * nnx + 0 for iz in range(nnz) for iy in range(nny)]


def tree_initial_density(shape: Tuple[int, int, int]) -> np.ndarray:
    """Binary-ish occupancy from a ``tree_manifold`` SDF, used as the initial
    design. Solid (inside the manifold) starts denser than void; the
    ``VolumeConstraint`` then pulls the mean to the target volume fraction while
    preserving this spatial pattern as the optimization seed.
    """
    grid = Field(np.zeros(shape), spacing=1.0)
    sdf = tree_manifold({"depth": 2, "inlet_radius": 2.0}, grid).values
    occ = (sdf <= 0.0).astype(float)
    return 0.2 + 0.5 * occ


def build_spec(
    shape: Tuple[int, int, int] = (10, 10, 20),
    volume_fraction: float = 0.4,
    max_iter: int = 20,
    step_size: float = 5.0,
    speed: float = 0.5,
) -> Tuple[DesignSpec, ConjugateHeatOracle]:
    """Assemble the heat-exchanger ``DesignSpec`` and its oracle.

    ``shape`` is ``(nelz, nely, nelx)`` with the flow (and the long axis) along
    x. Returns the spec and the oracle so callers can re-solve the oracle for
    reporting and baselining.
    """
    velocity = counterflow_velocity(shape, speed=speed)
    fixed = inlet_face_nodes(shape)
    oracle = ConjugateHeatOracle(
        shape=shape,
        velocity=velocity,
        source=np.ones(shape),
        fixed_nodes=fixed,
        fixed_values=[0.0] * len(fixed),
        k_solid=1.0,
        k_fluid=0.1,
        rho_cp=1.0,
        p_simp=3.0,
    )
    spec = DesignSpec(
        initial=Field(tree_initial_density(shape), spacing=1.0),
        oracle=oracle,
        objective=MaximizeValue(),
        optimizer=TopologyOptimizer(
            step_size=step_size,
            max_iter=max_iter,
            bounds=(1e-3, 1.0),
            p_start=1.0,
            p_end=3.0,
            filter_radius=1,
            tol=1e-15,
        ),
        constraint=VolumeConstraint(volume_fraction),
        name="heat-exchanger-3d",
    )
    return spec, oracle


def thermal_resistance(oracle: ConjugateHeatOracle, field: Field) -> float:
    """Thermal resistance proxy: the oracle minimises thermal compliance, and
    its ``value`` is the negated compliance, so resistance is ``-value``.
    """
    return float(-oracle.solve(field).value)


def main() -> None:
    output_dir = Path("./output")
    checkpoint_dir = Path("./checkpoints")
    shape = (10, 10, 20)
    volume_fraction = 0.4

    spec, oracle = build_spec(shape=shape, volume_fraction=volume_fraction)

    uniform = Field(np.full(shape, volume_fraction), spacing=1.0)
    r_uniform = thermal_resistance(oracle, uniform)

    result = Engine().run(
        spec,
        export_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=10,
        iso_value=0.5,
    )

    from morphos.report import build_report

    report = build_report(result, oracle)
    report.manufacturability = {
        "min_feature_size": MinFeatureSize(radius=1).report(result.field),
        "min_wall_thickness": MinWallThickness(min_thickness_voxels=2).report(result.field),
        "overhang": Overhang(build_axis=2).report(result.field),
        "volume": VolumeConstraint(volume_fraction).report(result.field),
    }
    (output_dir / "report.json").write_text(report.to_json())

    r_opt = thermal_resistance(oracle, result.field)
    orient = result.manufacturing_bundle.recommended_orientation
    achieved_vf = float(result.field.values.mean())
    print(
        f"heat exchanger {shape}: thermal resistance {r_uniform:.4f} -> {r_opt:.4f} "
        f"({r_uniform / max(r_opt, 1e-12):.2f}x), volume fraction {achieved_vf:.3f}, "
        f"recommended build orientation {tuple(round(c, 2) for c in orient.orientation)}"
    )
    print(f"wrote {output_dir / 'design.stl'} and {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
