# Morphos engine skeleton: design

Date: 2026-06-15. Status: approved for build (full authority handoff).

## What this is

Morphos is a physics-driven generative geometry engine. Given a design
specification (an objective, a domain, manufacturing constraints), it generates
a geometry that is optimized toward the physical limit of performance, and it
reports how close to that limit it got.

It is deterministic and physics-in-the-loop. Machine learning is not part of the
core; if it ever enters, it is only a surrogate trained on the physics, never the
source of truth. The headline is the method, not any single part: a single engine
that computes the optimal geometry for a problem and proves the margin to the
physical bound.

## Non-goals (YAGNI)

- Not a CAD replacement. Geometry is represented as implicit fields, not B-rep.
- Not a multi-physics solver suite on day one. The core is physics-agnostic, but
  only the electromagnetic path is implemented first.
- Not a manufacturing pipeline. Manufacturability enters as constraints on the
  optimization, not as a slicer or printer driver.

## Core principle: separate the invariant from the variant

The reason a one-product codebase gets scrapped at product two is that it fuses
the parts that never change with the parts that always change. Morphos splits
them with explicit interfaces:

- Invariant (the engine core): the optimization loop, the geometry-to-physics
  handoff, the constraint framework, the margin-to-limit reporting.
- Variant (the backends): which physics is solved, which device grammar is used,
  which figure of merit, which manufacturing process.

A new product implements new backends behind the same interfaces. The core does
not change.

## The lingua franca: the Field

Every layer speaks one data type: a `Field`, a scalar value sampled on a regular
Cartesian grid (a voxel grid). It can carry a signed distance (geometry) or a
material density in [0, 1] (topology optimization), or a physical quantity.

This choice is deliberate. PicoGK represents geometry as implicit fields on a
voxel grid. FDTD and FDFD electromagnetic solvers discretize space on a regular
Cartesian grid. The two share the same representation, so geometry passes to
physics with no meshing step, which is the usual fragile and slow part of an
automated design loop.

## Architecture (six interfaces, plus orchestration)

```
DesignSpec ---> Engine ---> DesignResult
                  |
   +--------------+-------------------------------------------+
   |              |              |             |              |
GeometryKernel  PhysicsOracle  Objective   Optimizer   Manufacturability
(produce a      (solve fields, (figure of  (inverse    (constraints and
 Field from      return value   merit and   design      projections on
 parameters)     and gradient)  bound)      loop)       the Field)
```

1. `GeometryKernel` (geometry/kernel.py): turns design parameters into a `Field`.
   Backends: `numpy_voxel` (reference, used for tests and bring-up), `picogk`
   (documented integration point for the production kernel, not rebuilt here).

2. `PhysicsOracle` (physics/oracle.py): given a `Field`, returns a
   `PhysicsResult` carrying the raw solved quantity, and where available a
   gradient with respect to the field (the adjoint). Backends: `analytic` (a
   closed-form oracle with exact gradients, dependency-light, for tests and for
   bringing the whole pipeline up green), `ceviche_em` (the electromagnetic
   backend integration point, FDFD via ceviche, an optional dependency that is
   declared but not yet wired up, so it fails loudly rather than faking a solve).

3. `Objective` (objective/objective.py): maps a `PhysicsResult` to a scalar
   figure of merit to be maximized, and optionally exposes a `PhysicalBound`,
   the best value physics allows, so the result can report margin to the limit.

4. `Optimizer` (optimize/optimizer.py): runs the inverse-design loop, calling the
   oracle and objective, using gradients when present and finite differences
   otherwise. Two concrete methods: density-based topology optimization (every
   voxel is a design variable, driven by the adjoint), and parametric design (a
   small parameter vector built into geometry by a kernel, optimized with a
   bounded quasi-Newton method since a single parameter accumulates sensitivity
   over the whole grid and is therefore scale sensitive).

5. `Manufacturability` (manufacturing/constraints.py): constraints and
   projections applied to the `Field` inside the loop. First: minimum feature
   size (via a density filter plus projection) and a connectivity check.

6. `Engine` (engine.py): orchestration. Takes a `DesignSpec`, runs the optimizer
   against the oracle and objective under the constraints, returns a
   `DesignResult` with the final field, achieved figure of merit, margin to the
   physical bound, optimization history, and a manufacturability report.

## Data flow per iteration

1. Optimizer proposes design parameters.
2. GeometryKernel turns them into a `Field`.
3. Manufacturability projects the `Field` (enforce min feature size, etc).
4. PhysicsOracle solves on the projected `Field`, returns value and gradient.
5. Objective turns the value into a figure of merit; the chain rule carries the
   gradient back through the projection to the parameters.
6. Optimizer updates parameters. Repeat until converged or budget spent.
7. Engine assembles the `DesignResult`, including margin to `PhysicalBound`.

## Error handling

- Backends validate `Field` shape and grid spacing on entry and raise clear
  errors, never silently reshape.
- An optimizer step that does not improve the objective is recorded; the loop
  stops on a no-improvement budget rather than running forever.
- `PhysicsOracle` backends that cannot supply a gradient declare it; the
  optimizer falls back to finite differences and records that it did so.
- The `picogk` backend raises `NotImplementedError` with guidance, so the
  integration point is explicit and never a silent stub returning garbage.

## Testing strategy

Test-driven. Each interface has a reference backend with a known closed-form
answer so the whole pipeline can be verified without heavy dependencies:

- Field: construction, grid metadata, invariants.
- numpy_voxel kernel: known shapes (a sphere SDF, a slab) match analytic values.
- analytic oracle: value and gradient verified against finite differences.
- objective and bound: figure of merit and margin computed correctly.
- topology optimizer: recovers the optimum of a known convex test problem.
- manufacturability: min-feature projection removes sub-threshold features.
- engine end to end: on the analytic problem, converges to within tolerance of
  the known optimum and reports a correct margin to the bound.

The electromagnetic backend integration point is covered by a separate, skipped
test that runs only when the optional dependency is installed, so core CI stays
fast and dependency-light.

## What this proves and does not prove

Proves: the architecture runs a full inverse-design loop end to end, the layers
are swappable behind interfaces, and the geometry-to-physics handoff needs no
meshing. Does not prove: any electromagnetic result (needs the real backend and,
ultimately, one printed part measured on a vector network analyzer), nor that a
generated device beats an incumbent. Those are the next gates, not this one.
