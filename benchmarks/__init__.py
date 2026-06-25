"""Morphos validation benchmark suite.

Cross-checks each physics oracle against an independent reference (closed-form
solutions, a published topology-optimization benchmark, or an external FEM
solver), computes a quantitative error metric per case, and emits a results
table for the README.

This package does NOT run as part of the normal test suite. Run it explicitly:

    python -m benchmarks            # full run, writes results.json + RESULTS.md
    pytest -m benchmark             # the same checks as assertions

Cases that need an optional dependency (scikit-fem) are skipped, not failed,
when it is absent. See ``RESULTS.md`` for the rendered table.
"""

__all__ = ["harness", "references", "cases"]
