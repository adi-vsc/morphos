"""Run the full validation suite and write the artifacts.

    python -m benchmarks [--quick]

Writes ``benchmarks/results.json``, ``benchmarks/RESULTS.md`` and one
convergence plot per refinement study under ``benchmarks/plots/``. Cases whose
optional dependency is missing are skipped (and noted), never failed.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks import cases as C
from benchmarks import references as R
from benchmarks.harness import seed_everything, to_json, to_markdown


HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    seed_everything(0)

    plots_dir = HERE / "plots"
    plots_dir.mkdir(exist_ok=True)

    all_results = []
    all_studies = []
    skipped = []

    for entry in C.CASES:
        name = entry["name"]
        missing = C.missing_deps(entry["needs"])
        if missing:
            skipped.append((name, missing))
            print(f"[skip] {name}: missing {', '.join(missing)}")
            continue
        print(f"[run ] {name} ...", flush=True)
        if entry["plots"]:
            results, studies = entry["fn"](plots_dir)
        else:
            results, studies = entry["fn"]()
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"        {status}  {r.case}: rel_err={r.rel_error:.3e} "
                  f"tol={r.tol:.1e} ({r.wall_clock_s:.2f}s)")
        all_results.extend(results)
        all_studies.extend(studies)

    import scipy
    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "seed": 0,
        "skipped": [f"{n} (missing {', '.join(m)})" for n, m in skipped],
    }
    if R.skfem_available():
        meta["scikit-fem"] = R.skfem_version()

    to_json(all_results, all_studies, HERE / "results.json", meta=meta)
    to_markdown(all_results, all_studies, HERE / "RESULTS.md", meta=meta)

    n_pass = sum(1 for r in all_results if r.passed)
    print(f"\n{n_pass}/{len(all_results)} cases passed. "
          f"Wrote results.json and RESULTS.md to {HERE}.")
    if skipped:
        print(f"Skipped (missing deps): {', '.join(n for n, _ in skipped)}")

    # A failing benchmark is a reported discrepancy, not a runner error; exit 0
    # so the artifacts are always produced. Use pytest -m benchmark for asserts.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
