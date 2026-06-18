"""Linear-solve scaling: direct LU vs preconditioned conjugate gradient.

The heat-conduction backend assembles the symmetric positive-definite interior
Laplacian and can solve it three ways:

    direct       sparse LU factorization (scipy spsolve)
    cg/jacobi    diagonally preconditioned conjugate gradient
    cg/amg       algebraic-multigrid preconditioned conjugate gradient

Direct LU is near optimal on small 2D grids, so the iterative paths only earn
their keep at scale, where LU fill-in grows faster than the grid. This script
times all three across grid sizes and prints the crossover. The honest takeaway:
plain Jacobi CG loses to direct LU (its iteration count grows with the grid),
while AMG CG pulls ahead and the lead widens as the grid grows. In 3D, where
direct fill-in is far worse, the gap is larger still.

Needs the optional iterative-solver dependency:
    pip install pyamg
Run:
    python examples/solver_scaling.py
"""

import time

import numpy as np

from morphos import Field
from morphos.physics.heat import HeatConductionOracle


def _time_solve(oracle: HeatConductionOracle, source: Field) -> tuple[float, float]:
    oracle.solve(source)  # warm up assembly + preconditioner build
    t0 = time.perf_counter()
    result = oracle.solve(source)
    return time.perf_counter() - t0, result.value


def main() -> None:
    print(f"{'N':>5} {'dof':>9} {'direct':>10} {'jacobiCG':>10} {'amgCG':>10} {'amg/direct':>11}")
    for n in (128, 256, 384):
        rng = np.random.default_rng(0)
        target = rng.normal(size=(n, n)) * 0.05
        source = Field(rng.normal(size=(n, n)), spacing=1.0)

        t_direct, v_direct = _time_solve(
            HeatConductionOracle(target=target, solver="direct"), source
        )
        t_jac, _ = _time_solve(
            HeatConductionOracle(target=target, solver="cg", preconditioner="jacobi"), source
        )
        t_amg, v_amg = _time_solve(
            HeatConductionOracle(target=target, solver="cg", preconditioner="amg"), source
        )

        # sanity: the iterative answer agrees with the direct one
        assert abs(v_amg - v_direct) / max(abs(v_direct), 1.0) < 1e-6

        print(
            f"{n:>5} {n * n:>9} {t_direct * 1e3:>9.1f}m {t_jac * 1e3:>9.1f}m "
            f"{t_amg * 1e3:>9.1f}m {t_direct / t_amg:>10.2f}x"
        )


if __name__ == "__main__":
    main()
