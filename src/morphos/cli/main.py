"""The ``morphos`` command-line interface.

Three subcommands, all built on the standard-library ``argparse`` (no Click,
no Typer): ``run`` (optimise a spec and write the output suite), ``info`` (a
dry-run summary of a spec without optimising), and ``validate`` (check an STL
is watertight). Every command takes ``--quiet`` to suppress stdout except
errors, and ``--debug`` to surface a full traceback instead of a clean message.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

import morphos
from morphos.intent import DesignIntent
from morphos.spec import CoupledSpec, DesignSpec


def _safe_print(text: str) -> None:
    """Print a line through an encoding that never raises on a console that
    cannot represent the box-drawing/arrow glyphs (Windows code pages)."""
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(enc, errors="replace").decode(enc) + "\n")


def _error(message: str) -> None:
    """Report a clean, single-line error on stderr (shown even under --quiet)."""
    sys.stderr.write(f"error: {message}\n")


# --- spec loading -------------------------------------------------------------


def _intent_registry() -> dict:
    """All concrete ``DesignIntent`` subclasses keyed by class name."""
    registry: dict = {}

    def collect(cls):
        for sub in cls.__subclasses__():
            registry[sub.__name__] = sub
            collect(sub)

    collect(DesignIntent)
    return registry


def _load_spec(path):
    """Load a spec JSON file into a ``DesignIntent`` (shape A) or a
    ``DesignSpec`` (shape B). Raises with a message stating what was expected
    and what was found."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"spec file not found: expected a readable JSON file, found nothing at {p}"
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"spec file is not valid JSON: expected a JSON object, found a parse "
            f"error in {p} ({exc})"
        ) from exc

    if "intent" in data:
        name = data["intent"]
        registry = _intent_registry()
        if name not in registry:
            raise ValueError(
                f"unknown intent {name!r}: expected one of {sorted(registry)}"
            )
        params = data.get("params", {})
        return registry[name](**params)

    if "spec" in data:
        return DesignSpec.from_dict(data["spec"])

    raise ValueError(
        "spec file has no design: expected an 'intent' key (shape A) or a "
        "'spec' key (shape B), found neither"
    )


# --- run ----------------------------------------------------------------------


def _format_orientation(orientation) -> str:
    parts = []
    for x in orientation:
        if abs(x - round(x)) < 1e-6:
            parts.append(str(int(round(x))))
        else:
            parts.append(f"{x:.2f}")
    return "[" + ", ".join(parts) + "]"


def _print_run_summary(result) -> None:
    rule = "─" * 40
    orient = (
        _format_orientation(result.bundle.recommended_orientation.orientation)
        if result.bundle.recommended_orientation is not None
        else "n/a"
    )
    converged = "yes" if result.design_result.converged else "no"
    lines = [
        rule,
        "  Morphos run complete",
        rule,
        f"  Objective (FOM)     {result.report.figure_of_merit:.4f}",
        f"  Volume fraction     {result.report.mass_fraction:.3f}",
        f"  Converged           {converged}",
        f"  Build orientation   {orient}",
        f"  Wall-clock time     {result.elapsed_seconds:.1f} s",
        f"  Outputs → {result.output_dir}/",
        "    design.stl",
        "    report.json",
        "    summary.txt",
        "    report.html",
        rule,
    ]
    for line in lines:
        _safe_print(line)


def _cmd_run(args) -> int:
    try:
        spec = _load_spec(args.spec_file)
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the user
        if args.debug:
            traceback.print_exc()
        _error(str(exc))
        return 1

    try:
        result = morphos.run(
            spec,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from,
            solver=args.solver,
            verbose=not args.quiet,
        )
    except Exception as exc:  # noqa: BLE001
        if args.debug:
            traceback.print_exc()
        _error(str(exc))
        return 1

    if not args.quiet:
        _print_run_summary(result)
    return 0


# --- info ---------------------------------------------------------------------


def _dof_per_node(oracle, ndim: int) -> int:
    """Estimated degrees of freedom per grid node for an oracle: a vector field
    (elasticity, thermo-elasticity) carries one DOF per spatial dimension; a
    scalar field (heat, pressure) carries one."""
    return ndim if "Elastic" in type(oracle).__name__ else 1


def _cmd_info(args) -> int:
    try:
        spec = _load_spec(args.spec_file)
    except Exception as exc:  # noqa: BLE001
        if args.debug:
            traceback.print_exc()
        _error(str(exc))
        return 1

    intent_name = type(spec).__name__ if isinstance(spec, DesignIntent) else "(raw spec)"
    built = spec.build() if isinstance(spec, DesignIntent) else spec

    if isinstance(built, CoupledSpec):
        field = built.initial_field
        oracles = [stage[0] for stage in built.stages]
        constraint = None
    else:
        field = built.initial
        oracles = [built.oracle]
        constraint = built.constraint

    ny, nx = field.shape[-2], field.shape[-1]
    n_elements = int(np.prod(field.shape))
    oracle_names = " · ".join(type(o).__name__ for o in oracles)
    if constraint is None:
        constraint_names = "none"
    else:
        constraint_names = type(constraint).__name__

    volume_fraction = getattr(spec, "volume_fraction", float(np.mean(field.values)))
    max_iter = getattr(built.optimizer, "max_iter", "?")
    n_nodes = int(np.prod([s + 1 for s in field.shape]))
    dofs = n_nodes * _dof_per_node(oracles[0], field.ndim)

    if not args.quiet:
        _safe_print(f"  Intent              {intent_name}")
        _safe_print(f"  Grid                {nx} × {ny}  ({n_elements} elements)")
        _safe_print(f"  Oracles             {oracle_names}")
        _safe_print(f"  Constraints         {constraint_names}")
        _safe_print(f"  Volume fraction     {volume_fraction:.2f}")
        _safe_print(f"  Max iterations      {max_iter}")
        _safe_print(f"  Estimated DOFs      {dofs}")
    return 0


# --- validate -----------------------------------------------------------------


def _read_stl_triangles(path):
    with open(path, "rb") as fh:
        fh.read(80)
        (n,) = struct.unpack("<I", fh.read(4))
        tris = []
        for _ in range(n):
            fh.read(12)  # normal
            tri = [struct.unpack("<3f", fh.read(12)) for _ in range(3)]
            fh.read(2)  # attribute byte count
            tris.append(tri)
    return tris


def _is_watertight(tris) -> bool:
    """A closed manifold: every quantised edge is shared by exactly two
    triangles. This is the same check used in tests/test_export.py."""
    edges = Counter()
    for tri in tris:
        keys = [tuple(np.round(np.array(v), 4)) for v in tri]
        for u, w in ((0, 1), (1, 2), (2, 0)):
            edges[frozenset((keys[u], keys[w]))] += 1
    return bool(tris) and all(c == 2 for c in edges.values())


def _cmd_validate(args) -> int:
    p = Path(args.stl_file)
    if not p.exists():
        _error(f"STL file not found: expected a readable .stl file, found nothing at {p}")
        return 1
    try:
        tris = _read_stl_triangles(p)
    except Exception as exc:  # noqa: BLE001
        if args.debug:
            traceback.print_exc()
        _error(f"could not read STL: expected a binary STL, found a read error ({exc})")
        return 1

    watertight = _is_watertight(tris)
    verts = np.array([v for tri in tris for v in tri], dtype=float) if tris else np.zeros((0, 3))
    if verts.size:
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        bbox = f"[{lo[0]:.3g}, {lo[1]:.3g}, {lo[2]:.3g}] to [{hi[0]:.3g}, {hi[1]:.3g}, {hi[2]:.3g}]"
    else:
        bbox = "empty"

    if not args.quiet:
        _safe_print(f"  Watertight          {'yes' if watertight else 'no'}")
        _safe_print(f"  Triangles           {len(tris)}")
        _safe_print(f"  Bounding box        {bbox}")
    return 0 if watertight else 2


# --- entry point --------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="morphos",
        description="Physics-driven generation of manufacturable geometry.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="optimise a spec and write the output suite")
    p_run.add_argument("spec_file", help="path to a spec JSON file")
    p_run.add_argument("--output-dir", default="./morphos_out")
    p_run.add_argument("--checkpoint-dir", default=None)
    p_run.add_argument("--resume-from", default=None)
    p_run.add_argument("--solver", choices=["auto", "direct", "iterative"], default="auto")
    p_run.add_argument("--quiet", action="store_true")
    p_run.add_argument("--debug", action="store_true")
    p_run.set_defaults(func=_cmd_run)

    p_info = sub.add_parser("info", help="dry-run summary of a spec without optimising")
    p_info.add_argument("spec_file", help="path to a spec JSON file")
    p_info.add_argument("--quiet", action="store_true")
    p_info.add_argument("--debug", action="store_true")
    p_info.set_defaults(func=_cmd_info)

    p_validate = sub.add_parser("validate", help="check an STL is watertight")
    p_validate.add_argument("stl_file", help="path to a binary STL file")
    p_validate.add_argument("--quiet", action="store_true")
    p_validate.add_argument("--debug", action="store_true")
    p_validate.set_defaults(func=_cmd_validate)

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
