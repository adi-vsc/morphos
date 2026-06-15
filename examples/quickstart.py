"""Quickstart: run the engine end to end on the reference problem.

This uses the dependency-light analytic oracle, whose optimum is known, to show
the whole pipeline running: geometry as a Field, an inverse-design loop, a
manufacturability report, and the margin to the physical limit. Swap the oracle
for the electromagnetic backend and the same engine designs a real device.

Run:
    python examples/quickstart.py
"""

import numpy as np

from morphos import (
    Field,
    AnalyticOracle,
    MaximizeValue,
    PhysicalBound,
    TopologyOptimizer,
    Connectivity,
    DesignSpec,
    Engine,
)


def main() -> None:
    # A target geometry the engine should recover (a connected solid block).
    target = np.zeros((9, 9))
    target[2:7, 2:7] = 1.0

    spec = DesignSpec(
        initial=Field(np.zeros((9, 9)), spacing=1.0),
        oracle=AnalyticOracle(target=target),
        objective=MaximizeValue(bound=PhysicalBound(value=0.0, name="target-match")),
        optimizer=TopologyOptimizer(step_size=0.2, max_iter=5000, tol=1e-12),
        constraint=Connectivity(threshold=0.5),
        name="quickstart-target-match",
    )

    result = Engine().run(spec)

    print(f"design:               {spec.name}")
    print(f"iterations:           {result.iterations}")
    print(f"converged:            {result.converged}")
    print(f"used finite diff:     {result.used_finite_differences}")
    print(f"figure of merit:      {result.figure_of_merit:.3e}")
    print(f"physical ceiling:     {result.bound.value:.3e} ({result.bound.name})")
    print(f"margin to ceiling:    {result.margin:.3e}")
    print(f"manufacturability:    {result.manufacturability}")
    recovered = np.allclose(result.field.values, target, atol=1e-3)
    print(f"recovered target:     {recovered}")


if __name__ == "__main__":
    main()
