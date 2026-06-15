# Morphos

A physics-driven engine that generates manufacturable geometry by optimizing it
toward the physical limit of performance, and reports how close to that limit it
reached.

Morphos is deterministic and physics-in-the-loop. It does not use a learned model
as its source of truth. The asset is the method: one engine that computes the
optimal geometry for a given problem and proves the margin to the physical bound,
rather than any single part.

## Why it is built this way

The engine separates the parts that never change from the parts that change per
product, so a new product is a new backend rather than a rewrite:

- Core (invariant): the optimization loop, the geometry-to-physics handoff, the
  constraint framework, and the margin-to-limit reporting.
- Backends (variant): the physics being solved, the geometry kernel, the figure
  of merit, the manufacturing process.

Every layer speaks one data type, a `Field`: a scalar on a regular voxel grid.
That representation is shared by implicit geometry kernels and by grid based
electromagnetic solvers, so geometry passes to physics with no meshing step.

## Status

Skeleton under construction. The core interfaces and a dependency-light reference
backend run a full inverse-design loop end to end and are covered by tests. The
real electromagnetic backend and the production geometry kernel plug in behind the
same interfaces.

## Layout

```
src/morphos/
  field.py            the Field type, shared by every layer
  geometry/           GeometryKernel interface and backends
  physics/            PhysicsOracle interface and backends
  objective/          Objective and physical-bound interfaces
  optimize/           Optimizer interface and topology optimization
  manufacturing/      manufacturability constraints and projections
  engine.py           orchestration: DesignSpec to DesignResult
  spec.py             DesignSpec and DesignResult
tests/                test-driven coverage of every interface
docs/design/          architecture and design decisions
```

## Develop

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

The real electromagnetic backend is optional:

```
.venv/Scripts/python -m pip install -e ".[em]"
```
