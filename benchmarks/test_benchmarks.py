"""Validation benchmarks as pytest assertions.

These are marked ``benchmark`` and are deselected from the normal suite by the
``-m "not benchmark"`` default in ``pyproject.toml``. Run them explicitly:

    pytest -m benchmark benchmarks

Each case asserts its Morphos quantity of interest is within the documented
tolerance of an independent reference. Cases needing an optional dependency are
skipped (not failed) when it is absent, so this file is safe to collect even in
a minimal install.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from benchmarks import cases as C
from benchmarks.harness import seed_everything


def _ids():
    return [e["name"] for e in C.CASES]


@pytest.fixture(autouse=True)
def _seeded():
    seed_everything(0)


@pytest.mark.benchmark
@pytest.mark.parametrize("entry", C.CASES, ids=_ids())
def test_benchmark_case(entry):
    missing = C.missing_deps(entry["needs"])
    if missing:
        pytest.skip(f"missing optional dependency: {', '.join(missing)}")

    if entry["plots"]:
        with tempfile.TemporaryDirectory() as d:
            results, _studies = entry["fn"](Path(d))
    else:
        results, _studies = entry["fn"]()

    assert results, f"case {entry['name']} produced no results"
    failures = [r for r in results if not r.passed]
    msg = "; ".join(
        f"{r.case}: morphos={r.morphos:.6g} ref={r.reference:.6g} "
        f"rel_err={r.rel_error:.3e} > tol={r.tol:.1e} ({r.notes})"
        for r in failures
    )
    assert not failures, msg


@pytest.mark.benchmark
def test_convergence_orders():
    """Smooth-problem oracles must converge at their expected (~2nd) order."""
    seed_everything(0)
    with tempfile.TemporaryDirectory() as d:
        plots = Path(d)
        _, heat_studies = C.case_heat(plots)
        _, modal_studies = C.case_modal(plots)
        _, elas_studies = C.case_elasticity_cantilever(plots)

    for study in heat_studies + modal_studies:
        assert 1.8 <= study.observed_order <= 2.2, (
            f"{study.case} observed order {study.observed_order:.3f} not ~2"
        )
    # The clamped-cantilever QoI carries a re-entrant-corner singularity, so a
    # looser band is documented and expected.
    for study in elas_studies:
        assert 1.5 <= study.observed_order <= 2.5, (
            f"{study.case} observed order {study.observed_order:.3f} out of band"
        )
