"""Cantilever SIMP demo: 2D compliance minimization, the classic topology
optimization test case.

ElasticityOracle solves plane-stress static equilibrium ``K(rho) u = F`` with a
Q4 finite-element discretization, density-penalized (SIMP) stiffness, and a
self-adjoint compliance sensitivity. This script:

  1. builds a cantilever (left edge fixed, downward point load at the free tip)
     at a fixed uniform "gray" density and reports its baseline compliance;
  2. runs the engine's TopologyOptimizer (gradient ascent on -compliance) from
     a uniform density at the target volume fraction and confirms compliance
     drops substantially while the figure of merit never exceeds the rigid-body
     ceiling of 0;
  3. prints an ASCII density map of the optimized design -- recognizable as
     a strut-and-tie topology, the qualitative signature of SIMP cantilevers
     in the literature, rather than image output.

Needs no extra dependencies beyond scipy. Run:
    python examples/cantilever_simp.py
"""

import numpy as np

from morphos import Field, Engine, DesignSpec, MaximizeValue, PhysicalBound
from morphos.optimize.topopt import TopologyOptimizer
from morphos.physics.elasticity import ElasticityOracle


def cantilever_bcs(shape, load=-1.0):
    """Fix the entire left edge; apply a downward point load at the bottom
    corner of the free (right) edge -- a standard 2D cantilever benchmark.
    """
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed = []
    for j in range(nny):
        fixed.append((0, j, "x"))
        fixed.append((0, j, "y"))
    loads = {(nnx - 1, nny - 1, "y"): load}
    return fixed, loads


def ascii_density_map(rho: np.ndarray) -> str:
    """A coarse text rendering of the density field, solid voxels as '#'."""
    levels = " .:-=+*#%@"
    scaled = np.clip(rho, 0.0, 1.0)
    idx = np.clip((scaled * (len(levels) - 1)).astype(int), 0, len(levels) - 1)
    rows = ["".join(levels[i] for i in row) for row in idx[::-1]]  # y-up
    return "\n".join(rows)


def main() -> None:
    shape = (20, 40)  # (ny, nx) elements
    volume_frac = 0.4
    fixed, loads = cantilever_bcs(shape, load=-1.0)
    oracle = ElasticityOracle(shape=shape, fixed_dofs=fixed, loads=loads)

    print(f"cantilever {shape[0]}x{shape[1]} elements, volume fraction {volume_frac}")

    baseline = oracle.solve(Field(np.full(shape, volume_frac), spacing=1.0))
    print(f"uniform-density compliance:   {baseline.aux['compliance']:.6f}")

    spec = DesignSpec(
        initial=Field(np.full(shape, volume_frac), spacing=1.0),
        oracle=oracle,
        objective=MaximizeValue(bound=PhysicalBound(value=0.0, name="rigid-limit")),
        optimizer=TopologyOptimizer(
            step_size=2e-3, max_iter=150, tol=1e-12, bounds=(1e-3, 1.0)
        ),
        name="cantilever-simp",
    )
    res = Engine().run(spec)
    final_compliance = -res.figure_of_merit

    print(f"optimized compliance:         {final_compliance:.6f}")
    print(f"compliance reduction:         {baseline.aux['compliance'] / final_compliance:.2f}x")
    print(f"figure of merit (-compliance): {res.figure_of_merit:.6f}")
    print(f"margin to rigid-body bound:    {res.margin:.6f}")
    print(f"iterations:                   {res.iterations}")
    print(f"used finite differences:      {res.used_finite_differences}")
    print()
    print("optimized density (# = solid, space = void):")
    print(ascii_density_map(res.field.values))


if __name__ == "__main__":
    main()
