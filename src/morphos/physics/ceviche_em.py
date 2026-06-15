"""Electromagnetic physics backend: documented integration point.

The electromagnetic path is the first real target for Morphos. The intended
backend is an FDFD solver (for example ceviche), which is a natural fit: it
discretizes space on the same regular Cartesian grid the Field uses, and it
supplies adjoint gradients, so the optimizer gets the gradient of a figure of
merit with respect to every voxel from one extra solve.

This backend is declared as an optional dependency (``pip install morphos[em]``)
and is not wired up in the skeleton. Rather than ship a silent stub that returns
a wrong field, construction fails loudly if the dependency is missing, and the
solve raises until the real FDFD setup (sources, boundaries, mode overlap figure
of merit) is implemented here.
"""

from __future__ import annotations

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle, PhysicsResult


class CevicheEMOracle(PhysicsOracle):
    """Placeholder for the FDFD electromagnetic backend."""

    provides_gradient = True

    def __init__(self, *args, **kwargs) -> None:
        try:
            import ceviche  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only with the extra
            raise ImportError(
                "CevicheEMOracle needs the optional electromagnetic dependency. "
                "Install it with: pip install morphos[em]"
            ) from exc
        self._args = args
        self._kwargs = kwargs

    def solve(self, field: Field) -> PhysicsResult:
        raise NotImplementedError(
            "The FDFD electromagnetic solve is the next gate and is not yet wired "
            "up. Implement source and boundary setup, the forward solve, the "
            "figure of merit (for example mode overlap or transmission), and the "
            "adjoint gradient here, returning a PhysicsResult."
        )
