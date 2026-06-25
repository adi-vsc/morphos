"""The public entry point: ``morphos.run()``.

One function takes a design (a :class:`~morphos.spec.DesignSpec` or a
:class:`~morphos.intent.DesignIntent`), runs the engine, and writes the full
output suite (STL, JSON report, text summary, and -- via
:mod:`morphos.viz` -- an interactive HTML report) to ``output_dir``. It
returns a :class:`MorphosResult` bundling everything a caller might want to
inspect programmatically, so a script needs no knowledge of the internal
Engine/Optimizer/Field machinery.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from morphos.engine import Engine
from morphos.intent import DesignIntent
from morphos.manufacturing.export import PrintParams, export_bundle
from morphos.report import PerformanceReport, build_report
from morphos.spec import CoupledSpec, DesignResult, DesignSpec, ParametricSpec

_DEFAULT_PRINT_PARAMS = PrintParams(
    material="SS316L",
    layer_thickness_mm=0.04,
    laser_power_W=200.0,
    scan_speed_mm_s=800.0,
    hatch_spacing_mm=0.1,
)


@dataclass
class MorphosResult:
    """Everything a single ``morphos.run()`` produced.

    Attributes
    ----------
    design_result:
        The raw :class:`~morphos.spec.DesignResult` from the engine.
    report:
        The :class:`~morphos.report.PerformanceReport`.
    bundle:
        The :class:`~morphos.manufacturing.export.ManufacturingBundle` (STL,
        voxel sidecar, recommended build orientation).
    output_dir:
        The directory all outputs were written to.
    elapsed_seconds:
        Wall-clock time of the engine run.
    """

    design_result: DesignResult
    report: PerformanceReport
    bundle: Any
    output_dir: Path
    elapsed_seconds: float


def _safe_write(text: str) -> None:
    """Write to stdout without ever raising on a console whose encoding cannot
    represent the progress glyphs (Windows code pages choke on the Greek
    delta/beta and block characters); unrepresentable characters degrade to a
    placeholder instead of crashing the run."""
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(enc, errors="replace").decode(enc))
    sys.stdout.flush()


def _simple_progress(total: int):
    """A minimal ``\\r``-overwriting per-iteration progress line. Replaced by the
    richer :class:`morphos.cli.progress.ProgressBar` when one is wired in, but
    kept as the zero-dependency default ``verbose`` display."""
    width = len(str(total))

    def on_iteration(iteration, fom, delta, p, beta):
        _safe_write(
            f"\r  iter {iteration:0{width}d}/{total}  fom={fom:.4f}  "
            f"Δ={delta:.4f}  p={p:.2f}  β={beta:.2f}"
        )

    return on_iteration


def _format_orientation(orientation) -> str:
    """Format a unit build-orientation vector as e.g. ``[0, 0, 1]``."""
    parts = []
    for x in orientation:
        if abs(x - round(x)) < 1e-6:
            parts.append(str(int(round(x))))
        else:
            parts.append(f"{x:.2f}")
    return "[" + ", ".join(parts) + "]"


def _extrude_to_3d(field):
    """Extrude a 2D density field into a thin 3D slab so the manufacturing
    export (which needs a 3D surface) can run on a 2D topology result. The
    extrusion is a prismatic sweep along a new leading axis, which is the
    honest physical interpretation of a 2D plane-stress design."""
    from morphos.field import Field

    if field.ndim == 3:
        return field
    depth = 4
    values = np.repeat(field.values[np.newaxis, :, :], depth, axis=0)
    return Field(values, spacing=field.spacing[0])


def _write_summary(path: Path, result: DesignResult, bundle, elapsed: float) -> None:
    orientation = (
        _format_orientation(bundle.recommended_orientation.orientation)
        if bundle.recommended_orientation is not None
        else "n/a"
    )
    volume_fraction = float(np.mean(result.field.values))
    lines = [
        f"objective (fom): {result.figure_of_merit:.4f}",
        f"volume fraction: {volume_fraction:.3f}",
        f"converged: {'yes' if result.converged else 'no'}",
        f"recommended build orientation: {orientation}",
        f"wall-clock time: {elapsed:.1f} s",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    spec,
    output_dir="./morphos_out",
    checkpoint_dir=None,
    resume_from=None,
    solver="auto",
    verbose=True,
):
    """Run a design through the engine and write its full output suite.

    Parameters
    ----------
    spec:
        A :class:`~morphos.spec.DesignSpec` (or :class:`ParametricSpec`) or a
        :class:`~morphos.intent.DesignIntent` (``.build()`` is called for you).
    output_dir:
        Directory to write ``design.stl``, ``report.json``, ``summary.txt`` and
        ``report.html`` to. Created if it does not exist.
    checkpoint_dir, resume_from:
        Passed through to the optimizer for checkpoint/resume.
    solver:
        ``"auto"`` (default), ``"direct"`` or ``"iterative"``; set on the
        oracle's linear-solver selection when it exposes one.
    verbose:
        When ``True`` (default), print a live progress line per iteration.

    Returns
    -------
    MorphosResult
    """
    if isinstance(spec, DesignIntent):
        spec = spec.build()

    if isinstance(spec, CoupledSpec):
        raise NotImplementedError(
            "morphos.run() handles single-stage DesignSpec/ParametricSpec runs; "
            f"a multi-stage CoupledSpec (here {spec.name!r}) must be driven via "
            "morphos.CoupledEngine, which returns one result per stage."
        )

    oracle = spec.oracle
    if hasattr(oracle, "solver"):
        oracle.solver = solver

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    on_iteration = _simple_progress(spec.optimizer.max_iter) if verbose else None

    engine = Engine()
    t0 = time.perf_counter()
    design_result = engine.run(
        spec,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        on_iteration=on_iteration,
    )
    elapsed = time.perf_counter() - t0
    if verbose:
        _safe_write("\n")

    report = build_report(design_result, oracle)

    export_field = _extrude_to_3d(design_result.field)
    bundle = export_bundle(
        export_field,
        _DEFAULT_PRINT_PARAMS,
        output_dir,
        iso_value=0.5,
        report=report,
    )

    (output_dir / "report.json").write_text(report.to_json(), encoding="utf-8")
    _write_summary(output_dir / "summary.txt", design_result, bundle, elapsed)

    result = MorphosResult(
        design_result=design_result,
        report=report,
        bundle=bundle,
        output_dir=output_dir,
        elapsed_seconds=elapsed,
    )

    # Interactive HTML report (Step 4). Imported lazily so the core run path has
    # no hard dependency on the viz layer.
    try:
        from morphos.viz import render_html_report

        render_html_report(result, output_dir / "report.html")
    except ImportError:
        pass

    return result
