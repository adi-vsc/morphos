# Morphos

Morphos is an open-source engine for physics-driven design and manufacturable
geometry generation. It turns a design request into a printable part, and reports
how that part performs against the physics.

It has two complementary halves:

1. **Optimization against physics.** A topology / parametric optimizer steps a
   design's density field using gradients from a swappable **physics oracle**
   (elasticity, heat, flow, modal, electromagnetics), under manufacturing
   constraints, and reports the gap to a reference performance bound.
2. **Implicit geometry generation.** A library of signed-distance-field (SDF)
   generators (gyroid / TPMS exchangers, lattices, fin arrays, manifolds) that
   produce crisp, watertight, self-supporting parts *by construction* — no gray
   voxels, no marching-cubes mush.

A thin **agent layer** sits on top: it maps a plain-English request to the right
generator and parameters and exports a validated STL — or refuses, honestly, when
no generator fits. Morphos is not a universal "describe anything → geometry"
synthesizer; it builds the part families it has generators for, and grows by
adding generators.

> **Status: early research preview (v0.1.0).** Every physics oracle is
> cross-checked against an **independent** reference — closed-form solutions, a
> published topology-optimization benchmark, and an external FEM solver
> (scikit-fem), in 2D and 3D — by the [validation suite](#validation) (14/14
> cases pass). But Morphos has **not** been validated against commercial FEA/CFD
> packages or physical hardware, and the agent builds only a small catalog of
> part families. Treat it as a research and educational tool, not production
> engineering sign-off. See [Status & Limitations](#status--limitations).

## How it works

```
                    natural-language request
                              |
                      [ agent: router ]            <- picks a generator + params,
                              |                        or refuses if none fit
            +-----------------+------------------+
            |                                    |
   implicit geometry                     physics optimization
   (SDF generators)                      (DesignIntent -> Engine)
            |                                    |
        occupancy Field  <------ shared ------>  density Field
            |                                    |
            +-----------------+------------------+
                              |
              [ manufacturing: mesh + constraints ]
                              |
                    [ STL | JSON | HTML ]   + performance report
```

Everything speaks one data type, the `Field` (a scalar grid). A physics oracle
maps a `Field` to a value and a gradient; the optimizer steps the field; an SDF
generator builds a `Field` directly from a level set. The same export pipeline
meshes either one into a watertight STL with a manufacturability report.

## Install

There is no PyPI release yet, so install editable from a clone:

```
pip install -e .
```

Core (NumPy + SciPy) covers most oracles and all geometry generators. Optional
extras enable specific backends:

```
pip install -e ".[mfg]"        # STL export + build-orientation study (scikit-image)
pip install -e ".[cfd]"        # viscous-flow (Stokes/Brinkman) oracle (scikit-fem)
pip install -e ".[iterative]"  # AMG-preconditioned solver for large 3D (pyamg)
pip install -e ".[em]"         # electromagnetic FDFD adjoint backend (ceviche)
pip install -e ".[dev]"        # everything, for running the full test suite
```

## Quickstart 1 — describe a part in English (agent)

```python
from morphos.agent import design_from_text

r = design_from_text(
    "a counter-flow heat exchanger, 100 mm cylindrical core, 1.4 mm walls",
    output_dir="out",
)
print(r.generator)        # 'gyroid_heat_exchanger'
print(r.stl_path)         # out/gyroid_heat_exchanger.stl  (watertight)
print(r.metrics)          # {'leak_paths': 0, 'hot_single_network_fraction': 1.0, ...}
```

The request is routed to a generator from the [catalog](#geometry-generators),
its parameters are parsed from the text, and a validated STL is written. A
request with no matching generator raises rather than inventing geometry:

```python
design_from_text("a turbopump impeller with curved blades", "out")
# ValueError: No generator matched this request. Available: gyroid_heat_exchanger, ...
```

## Quickstart 2 — optimize a design against physics (Python API)

```python
import morphos
from morphos.intent import CantileverIntent

# A cantilever beam on a 60 x 30 grid: fixed left edge, downward tip load.
intent = CantileverIntent(span=60, height=30, load=-1.0,
                          volume_fraction=0.4, max_iter=40)

result = morphos.run(intent, output_dir="morphos_out")
print("objective (fom):", round(result.report.figure_of_merit, 4))
print("volume fraction:", round(result.report.mass_fraction, 3))
print("converged:", result.design_result.converged)
```

This writes `design.stl`, `report.json`, `summary.txt`, and `report.html`, and
returns a `MorphosResult` with the `DesignResult`, the `PerformanceReport`, the
`ManufacturingBundle`, and the wall-clock time.

## Quickstart 3 — CLI

```
morphos run spec.json --output-dir ./out      # run a spec
morphos info spec.json                         # inspect a spec without optimizing
morphos validate ./out/design.stl             # check a mesh is watertight
morphos batch ./specs --output-dir ./b --jobs 4
```

A spec file names an intent and its parameters; pass a spec to `morphos info`
to print the resolved intent, parameters, and the oracle it will run.

## Outputs

Every optimization run writes four files to the output directory:

| File          | Contents                                                              |
| ------------- | -------------------------------------------------------------------- |
| `design.stl`  | The watertight surface mesh of the optimised geometry (binary STL).  |
| `report.json` | The full performance report: figure of merit, engineering-unit quantities, gap to the reference bound, and manufacturability metrics. |
| `summary.txt` | A five-line human-readable summary: objective, volume fraction, converged flag, recommended build orientation, wall-clock time. |
| `report.html` | A self-contained interactive report (convergence chart, density slices, manufacturing summary, download links). |

## Supported physics

Each physics backend is an oracle the engine optimises against. They are
swappable behind one interface, so a new physics is a new backend, not a rewrite.

| Oracle                 | Physics                                          | 2D | 3D | Notes                                            |
| ---------------------- | ------------------------------------------------ | -- | -- | ------------------------------------------------ |
| `ElasticityOracle`     | Linear elasticity, compliance minimisation       | y  | y  | SIMP, Q4 plane-stress or Hex8 solid.             |
| `ThermoElasticOracle`  | Coupled thermo-elasticity                         | y  | y  | Mechanical load plus steady thermal gradient.    |
| `HeatConductionOracle` | Steady heat conduction                            | y  | y  | Conductive material layout.                      |
| `ConjugateHeatOracle`  | Advection-diffusion conjugate heat                | y  | y  | Consumes a solved velocity field.                |
| `DarcyFlowOracle`      | Darcy / potential flow dissipation                | y  | y  | Channel-routing surrogate.                       |
| `StokesFlowOracle`     | Stokes / Brinkman viscous flow                    | y  | y  | Taylor-Hood elements (needs `[cfd]`).            |
| `ModalOracle`          | Modal eigenfrequency                              | y  | y  | Fundamental-frequency objectives.                |
| `FDTD3DOracle`         | Electromagnetics (time-domain FDTD)               |    | y  | 3D electromagnetic topology optimisation.        |

## Geometry generators

The implicit-geometry generators the agent can build today. Each is a
parametric SDF that produces a watertight, self-supporting part:

| Generator               | Builds                                                        | Quality check |
| ----------------------- | ------------------------------------------------------------ | ------------- |
| `gyroid_heat_exchanger` | Counter-flow gyroid (TPMS) core: a sealed wall separating two interpenetrating fluid networks. | leak-tightness + single-network per fluid, verified |
| `gyroid_lattice_block`  | An envelope filled with a gyroid lattice at a target volume fraction (lightweighting). | target volume fraction |
| `pin_fin_heat_sink`     | A base plate with a periodic array of square pins (convective cooling). | — |

Adding a capability is adding a generator plus a `GeneratorSpec` in
`morphos/agent/catalog.py`. That is the unit of progress: breadth comes from
banking generators, not from a universal synthesizer.

## Validation

Each physics oracle is cross-checked against an **independent** reference — a
closed-form solution, a published topology-optimization benchmark, or an external
FEM solver. Every reference is assembled from first principles or by an external
library; none reuses the oracle under test. The suite lives in `benchmarks/` and
regenerates with `python -m benchmarks`; full results, sources, and
mesh-refinement studies are in [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

All 14 cases pass their documented tolerance (2D and 3D):

| Case | Reference | QoI | Rel. error | Tol |
| --- | --- | --- | ---: | ---: |
| heat conduction | analytic (manufactured solution) | L2 temperature, 129x129 | 5.0e-05 | 1e-3 |
| modal membrane | analytic (separation of variables) | fundamental eigenvalue | 2.0e-04 | 1e-2 |
| elasticity (uniaxial bar) | analytic (Hooke's law) | tip displacement | 1.4e-14 | 1e-9 |
| elasticity (cantilever) | analytic (Euler-Bernoulli) | tip deflection | 8.9e-04 | 2e-2 |
| elasticity (vs scikit-fem) | independent FEM | compliance | 2.4e-09 | 1e-6 |
| darcy (porous column) | analytic (Darcy's law) | pressure drop | 1.7e-14 | 1e-9 |
| stokes (Poiseuille) | analytic (Poiseuille flow) | L2 x-velocity | 8.1e-15 | 1e-6 |
| topology MBB | literature (88-line SIMP) | converged compliance | 1.3e-02 | 1e-1 |
| topology Michell | theoretical bound (full-solid) | compliance ratio | see note | - |
| heat conduction (3D) | analytic (manufactured solution) | L2 temperature, 33^3 | 8.0e-04 | 1e-3 |
| elasticity 3D bar (Hex8) | analytic (Hooke's law) | tip displacement | 1.7e-14 | 1e-9 |
| elasticity 3D (vs scikit-fem) | independent FEM | compliance, Hex8 | 4.6e-12 | 1e-6 |

Mesh-refinement studies confirm second-order convergence for the heat (2.00 in
2D and 3D), modal (2.00), and elasticity (1.94) solvers. **These are
*verification* benchmarks** (analytic and code-to-code): they show the math is
implemented correctly. They are **not** validation against commercial FEA/CFD or
physical hardware — see [Status & Limitations](#status--limitations).

## Architecture

A `DesignIntent` builds a `DesignSpec` from engineering-unit parameters. The
`Engine` runs the optimiser, which steps the density `Field` using gradients from
the `PhysicsOracle` under manufacturing constraints. The final `Field` (from the
optimizer, or straight from an SDF generator) is meshed by the export pipeline
into a `ManufacturingBundle`, serialised as STL, JSON, and HTML.

## Contributing

Run the test suite with `pytest`; every change ships with a test, written first.
Run the benchmarks with `python -m benchmarks`. Source modules open with a
rationale docstring explaining why the code exists and the engineering choices
behind it, not just what it does; follow that convention.

## Status & Limitations

Morphos is an early-stage research project. It is built carefully and tested
thoroughly, but it is not a substitute for a validated commercial engineering
tool. Please read this before relying on its output.

**Verified, not yet validated.** Every solver is *verified* — checked against
analytic solutions, an independent FEM code (scikit-fem) in 2D and 3D, and
finite-difference gradient gates. That establishes the math is implemented
correctly. It does **not** establish that the results match physical reality.
Morphos has not been cross-checked against commercial FEA/CFD (Abaqus, Ansys,
COMSOL) or physical test data. Treat all outputs as indicative.

**The agent is a router over a small catalog, not a universal designer.** It
builds the part families it has generators for (currently three) and refuses the
rest. It does not synthesize arbitrary geometry from a description. Breadth grows
by adding generators, each of which is hand-written and tested.

**Generated parts are geometry, not yet finished products.** The generators
produce a manufacturable core (e.g. a leak-tight gyroid exchanger), but not the
inlet/outlet headers, ports, or sealing shells a real assembly needs, and the
geometry is a uniform lattice — not yet *graded* or physics-optimized to a
specific duty. A performance figure (heat-transfer coefficient, pressure drop)
requires running the relevant oracle on the part, which is a separate step.

**The two legs differ in maturity.** The implicit-geometry leg produces clean,
crisp parts. The topology-optimization leg is younger: at low resolution or too
few iterations it can produce gray, blobby results. Use enough resolution,
iterations, and a projection/continuation scheme for binary designs.

**"Reference bound" is a ceiling you supply, not a proven physical optimum.**
The bounds wired into the built-in intents are simple theoretical ceilings (e.g.
zero compliance — an infinitely stiff, unattainable reference), useful as a fixed
yardstick but *not* tight problem-specific optimality bounds. Read the reported
gap accordingly.

**Manufacturability is partly checked, partly approximate.** STL export is
watertight, and some properties are genuinely verified (the gyroid exchanger's
leak-tightness and single-network-per-fluid are computed, not assumed). But the
overhang / minimum-feature constraints are density-field surrogates, and there is
no slicer/toolpath stage or per-process qualification.

**Scale and maturity.** 3D solves use direct factorisation by default
(memory-bound); the `[iterative]` AMG path extends reach but realistic
high-resolution 3D still hits practical ceilings. Single-author, pre-1.0, no PyPI
release; API and file formats may change. Intended for research, experimentation,
and education.
