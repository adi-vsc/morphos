"""Benchmark harness: result records, error metrics, and output emitters.

This module is deliberately free of any Morphos physics. It only defines the
data structures every benchmark case fills in, the quantitative error metrics
the cases assert against, and the JSON / Markdown / convergence-plot emitters
that turn a list of results into the artifacts the README links to.

The split is the whole point of the validation effort: a case (in ``cases.py``)
computes a Morphos quantity of interest, ``references.py`` computes the same
quantity *independently* (closed form or an external solver), and the harness
here only compares the two and reports. No reference ever touches the oracle
under test.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Error metrics (the only place a "rel. error" is ever defined)
# --------------------------------------------------------------------------

def rel_error(value: float, reference: float) -> float:
    """Relative error of a scalar QoI against a (nonzero) reference.

    ``|value - reference| / |reference|``; falls back to the absolute error
    when the reference is (near) zero so the metric never divides by zero.
    """
    denom = abs(reference)
    if denom < 1e-300:
        return abs(value - reference)
    return abs(value - reference) / denom


def rel_l2(field: np.ndarray, reference: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Relative discrete L2 error of a field against a reference field.

    ``||field - reference||_2 / ||reference||_2`` over either the whole array
    or, when ``mask`` is given, the masked subset (used to exclude Dirichlet
    boundary nodes that are identically zero on both sides).
    """
    f = np.asarray(field, dtype=float)
    r = np.asarray(reference, dtype=float)
    if mask is not None:
        f = f[mask]
        r = r[mask]
    num = float(np.sqrt(np.sum((f - r) ** 2)))
    den = float(np.sqrt(np.sum(r ** 2)))
    if den < 1e-300:
        return num
    return num / den


def observed_order(hs: Sequence[float], errors: Sequence[float]) -> float:
    """Least-squares slope of ``log(error)`` vs ``log(h)``: the observed
    convergence order ``p`` in ``error ~ C h^p``. Uses every refinement level,
    so it is robust to a single noisy point.
    """
    logh = np.log(np.asarray(hs, dtype=float))
    loge = np.log(np.asarray(errors, dtype=float))
    A = np.vstack([logh, np.ones_like(logh)]).T
    slope, _ = np.linalg.lstsq(A, loge, rcond=None)[0]
    return float(slope)


def richardson_order(values: Sequence[float]) -> float:
    """Observed order from three QoI values on grids refined by 2x each.

    ``p = log2( |u0 - u1| / |u1 - u2| )`` for ``values = [u(h), u(h/2),
    u(h/4)]``. Used for QoIs whose exact continuum limit is not closed-form
    (the FE elasticity tip deflection), where convergence is measured by
    self-consistency rather than against an analytic value.
    """
    u0, u1, u2 = values[-3], values[-2], values[-1]
    d01 = abs(u0 - u1)
    d12 = abs(u1 - u2)
    if d12 < 1e-300:
        return float("nan")
    return math.log2(d01 / d12)


def richardson_limit(values: Sequence[float], p: float) -> float:
    """Richardson-extrapolated continuum limit from the two finest grids
    (refined 2x) given the observed order ``p``."""
    u1, u2 = values[-2], values[-1]
    return u2 + (u2 - u1) / (2.0 ** p - 1.0)


# --------------------------------------------------------------------------
# Result records
# --------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """One row of the results table: a Morphos QoI vs an independent reference."""

    case: str
    reference_type: str          # "analytic" | "independent FEM (...)" | "literature (...)"
    qoi: str                     # human description of the quantity compared
    morphos: float
    reference: float
    rel_error: float
    tol: float
    passed: bool
    wall_clock_s: float
    source: str                  # equation / paper / solver+version the reference came from
    notes: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class ConvergenceStudy:
    """A mesh-refinement study: error (or QoI) vs grid spacing, plus the
    observed convergence rate and the order theory predicts."""

    case: str
    hs: List[float]
    errors: List[float]
    observed_order: float
    expected_order: float
    qoi: str
    notes: str = ""
    plot_path: Optional[str] = None


@contextmanager
def timed():
    """Context manager yielding a one-element list whose entry is filled with
    the wall-clock seconds elapsed inside the ``with`` block."""
    holder = [0.0]
    t0 = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - t0


def seed_everything(seed: int = 0) -> None:
    """Make every case deterministic. Cases use ``numpy`` only; seeding its
    global RNG (and the legacy one) is enough."""
    np.random.seed(seed)


# --------------------------------------------------------------------------
# Emitters
# --------------------------------------------------------------------------

def to_json(
    results: Sequence[BenchmarkResult],
    studies: Sequence[ConvergenceStudy],
    path: Path,
    meta: Optional[dict] = None,
) -> None:
    payload = {
        "meta": meta or {},
        "results": [asdict(r) for r in results],
        "convergence_studies": [asdict(s) for s in studies],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fmt(x: float) -> str:
    """Compact numeric formatting for the Markdown table."""
    if x == 0:
        return "0"
    ax = abs(x)
    if ax < 1e-3 or ax >= 1e5:
        return f"{x:.3e}"
    if ax < 1:
        return f"{x:.5g}"
    return f"{x:.5g}"


def to_markdown(
    results: Sequence[BenchmarkResult],
    studies: Sequence[ConvergenceStudy],
    path: Path,
    meta: Optional[dict] = None,
) -> None:
    meta = meta or {}
    lines: List[str] = []
    lines.append("# Morphos validation results")
    lines.append("")
    lines.append(
        "Each row compares a Morphos physics oracle against an **independent** "
        "reference (a closed-form solution, a published benchmark, or an external "
        "FEM solver). Every reference is assembled from first principles or by an "
        "external library; none reuses the oracle under test. Regenerate with "
        "`python -m benchmarks`."
    )
    lines.append("")
    if meta:
        bits = ", ".join(f"{k}: {v}" for k, v in meta.items())
        lines.append(f"_Run metadata — {bits}._")
        lines.append("")

    n_pass = sum(1 for r in results if r.passed)
    lines.append(f"**{n_pass}/{len(results)} cases pass their documented tolerance.**")
    lines.append("")

    # Main table.
    lines.append(
        "| Case | Reference type | QoI | Morphos | Reference | Rel. error | Tol | Pass | Time (s) |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | :---: | ---: |")
    for r in results:
        check = "✅" if r.passed else "❌"
        lines.append(
            f"| {r.case} | {r.reference_type} | {r.qoi} | {_fmt(r.morphos)} | "
            f"{_fmt(r.reference)} | {_fmt(r.rel_error)} | {_fmt(r.tol)} | {check} | "
            f"{r.wall_clock_s:.2f} |"
        )
    lines.append("")

    # Sources.
    lines.append("## Reference sources")
    lines.append("")
    for r in results:
        lines.append(f"- **{r.case}** — {r.source}")
        if r.notes:
            lines.append(f"  - {r.notes}")
    lines.append("")

    # Convergence studies.
    if studies:
        lines.append("## Mesh-refinement / convergence studies")
        lines.append("")
        lines.append(
            "| Study | QoI | Observed order | Expected order | Levels (h) | Errors |"
        )
        lines.append("| --- | --- | ---: | ---: | --- | --- |")
        for s in studies:
            hs = ", ".join(f"{h:.4g}" for h in s.hs)
            es = ", ".join(f"{e:.2e}" for e in s.errors)
            lines.append(
                f"| {s.case} | {s.qoi} | {s.observed_order:.3f} | {s.expected_order:.2f} | "
                f"{hs} | {es} |"
            )
        lines.append("")
        for s in studies:
            if s.plot_path:
                lines.append(f"### {s.case}")
                lines.append("")
                lines.append(f"![{s.case} convergence]({s.plot_path})")
                if s.notes:
                    lines.append("")
                    lines.append(s.notes)
                lines.append("")

    # Known discrepancies.
    fails = [r for r in results if not r.passed]
    if fails:
        lines.append("## Known discrepancies")
        lines.append("")
        for r in fails:
            lines.append(
                f"- **{r.case}**: Morphos {_fmt(r.morphos)} vs reference "
                f"{_fmt(r.reference)} (rel. error {_fmt(r.rel_error)} > tol "
                f"{_fmt(r.tol)})."
            )
            if r.notes:
                lines.append(f"  - Hypothesis: {r.notes}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def plot_convergence(study: ConvergenceStudy, path: Path) -> Optional[str]:
    """Render a log-log convergence plot with a reference slope triangle.

    Returns the file name written (relative, for embedding) or ``None`` if
    matplotlib is unavailable. Never raises on a missing backend.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    hs = np.asarray(study.hs, dtype=float)
    es = np.asarray(study.errors, dtype=float)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.loglog(hs, es, "o-", label="observed error")

    # Reference slope line at the expected order, anchored at the finest point.
    p = study.expected_order
    c = es[-1] / (hs[-1] ** p)
    ax.loglog(hs, c * hs ** p, "k--", alpha=0.6, label=f"O(h^{p:g}) reference")

    ax.set_xlabel("grid spacing h")
    ax.set_ylabel("error")
    ax.set_title(f"{study.case}\nobserved order {study.observed_order:.2f}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name
