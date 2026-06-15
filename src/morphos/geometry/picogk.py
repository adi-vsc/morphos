"""PicoGK geometry backend: documented integration point.

PicoGK is the intended production geometry kernel. It represents geometry as
implicit fields on a voxel grid, the same representation Morphos uses, so it
plugs in behind :class:`morphos.geometry.kernel.GeometryKernel` without changing
the engine core.

It is not wired up in the skeleton. PicoGK runs on .NET, so a real integration
crosses a language boundary (for example through a .NET interop layer), reading
PicoGK voxel fields into the numpy grid the rest of the engine speaks. Rather
than ship a silent stub that returns wrong geometry, :meth:`build` fails loudly
so the integration point is explicit.
"""

from __future__ import annotations

from morphos.field import Field
from morphos.geometry.kernel import GeometryKernel


class PicoGKKernel(GeometryKernel):
    """Placeholder for the production PicoGK backend."""

    def build(self, spec: dict) -> Field:
        raise NotImplementedError(
            "PicoGKKernel is a documented integration point and is not wired up "
            "in the skeleton. Use VoxelKernel for the reference pipeline, or "
            "implement the .NET interop that reads PicoGK voxel fields into a "
            "numpy grid here."
        )
