# Morphos

Morphos is a physics-driven engine that generates manufacturable geometry by
optimising a design toward its physical limit and reporting the margin to that
limit. It is an open-source alternative to the closed computational-engineering
models used for topology and parametric design.

## Install

There is no PyPI release yet, so install editable from a clone:

```
pip install -e .
```

The geometry export (STL and the build-orientation study) needs scikit-image.
Install it with the manufacturing extra:

```
pip install -e ".[mfg]"
```

## Quickstart: Python API

```python
import morphos
from morphos.intent import CantileverIntent

# A cantilever beam on a 60 x 30 grid: fixed left edge, downward tip load.
intent = CantileverIntent(
    span=60,
    height=30,
    load=-1.0,
    volume_fraction=0.4,
    max_iter=40,
)

result = morphos.run(intent, output_dir="morphos_out")

print("objective (fom):", round(result.report.figure_of_merit, 4))
print("volume fraction:", round(result.report.mass_fraction, 3))
print("converged:", result.design_result.converged)
print("outputs written to:", result.output_dir)
```

Running this writes `design.stl`, `report.json`, `summary.txt`, and `report.html`
to `morphos_out/`, and returns a `MorphosResult` with the raw `DesignResult`, the
`PerformanceReport`, the `ManufacturingBundle`, and the wall-clock time.

## Quickstart: CLI

After installing, the same run is one command. Write a spec file:

```json
{
  "version": "1.0",
  "intent": "CantileverIntent",
  "params": {
    "span": 60,
    "height": 30,
    "load": -1.0,
    "volume_fraction": 0.4,
    "max_iter": 40
  }
}
```

Then run it, inspect a spec without optimising, validate a mesh, or run a whole
directory of specs:

```
morphos run spec.json --output-dir ./out
morphos info spec.json
morphos validate ./out/design.stl
morphos batch ./specs --output-dir ./batch_out --jobs 4
```

The spec file schema (both the intent form above and the raw-spec form) is
documented in [docs/spec_schema.md](docs/spec_schema.md).

## Outputs

Every run writes four files to the output directory:

| File          | Contents                                                              |
| ------------- | -------------------------------------------------------------------- |
| `design.stl`  | The watertight surface mesh of the optimised geometry (binary STL).  |
| `report.json` | The full performance report: figure of merit, engineering-unit quantities, margin to the physical bound, and manufacturability metrics. |
| `summary.txt` | A five-line human-readable summary: objective, volume fraction, converged flag, recommended build orientation, wall-clock time. |
| `report.html` | A self-contained interactive report (convergence chart, density slices, manufacturing summary, download links) that opens in any browser with no external dependencies. |

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
| `StokesFlowOracle`     | Stokes / Brinkman viscous flow                    | y  | y  | Taylor-Hood elements, viscous dissipation.       |
| `ModalOracle`          | Modal eigenfrequency                              | y  | y  | Fundamental-frequency objectives.                |
| `FDTD3DOracle`         | Electromagnetics (time-domain FDTD)               |    | y  | 3D electromagnetic topology optimisation.        |

## Architecture overview

```
DesignIntent
    -> Engine
        -> [ Optimizer + PhysicsOracle + MfgConstraints ]
        -> Field
        -> GeometryKernel
        -> ManufacturingBundle
        -> [ STL | JSON | HTML ]
```

A `DesignIntent` builds a `DesignSpec` from engineering-unit parameters. The
`Engine` runs the optimiser, which steps the density `Field` using gradients from
the `PhysicsOracle` under the manufacturing constraints. The final `Field` is
meshed by the `GeometryKernel` into a `ManufacturingBundle`, which the output
suite serialises as STL, JSON, and HTML.

## Contributing

Run the test suite with `pytest`; every change ships with a test, written first.
Source modules open with a rationale docstring explaining why the code exists and
the engineering choices behind it, not just what it does; follow that convention.
