"""Optimizer checkpoint save/load: warm-start state for long-running runs.

Long topology/parametric optimization runs (hundreds of iterations on large
3-D grids) can take hours; without checkpointing, a crash or a deliberate
pause loses all of that work. A checkpoint captures everything needed to
resume exactly where a run left off: the current design field, the iteration
index, the current SIMP penalty ``p`` and Heaviside sharpness ``beta`` (both
products of a continuation schedule, not recoverable from the field alone),
and the running objective-value history. Stored as a single compressed
``.npz`` so the array payload (the field, possibly large) and the small
scalar/history metadata travel together as one file.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from morphos.field import Field


def save_checkpoint(
    path,
    field: Field,
    iteration: int,
    p: float,
    beta: float,
    history: List[float],
) -> None:
    """Write a compressed checkpoint of the current optimizer state.

    ``path``'s parent directory is created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        field_values=np.asarray(field.values, dtype=float),
        spacing=np.asarray(field.spacing, dtype=float),
        iteration=np.int64(iteration),
        p=np.float64(p),
        beta=np.float64(beta),
        history=np.asarray(history, dtype=float),
    )


def load_checkpoint(path) -> Tuple[Field, int, float, float, List[float]]:
    """Read a checkpoint written by :func:`save_checkpoint`.

    Returns ``(field, iteration, p, beta, history)``.
    """
    data = np.load(Path(path), allow_pickle=False)
    field = Field(data["field_values"], tuple(data["spacing"].tolist()))
    iteration = int(data["iteration"])
    p = float(data["p"])
    beta = float(data["beta"])
    history = data["history"].tolist()
    return field, iteration, p, beta, history


def checkpoint_path(checkpoint_dir, iteration: Optional[int] = None) -> Path:
    """Default checkpoint filename inside ``checkpoint_dir``.

    A fixed name (``checkpoint.npz``) is used when ``iteration`` is omitted,
    so each periodic save overwrites the previous one (only the latest state
    is ever needed to resume); pass ``iteration`` for a per-iteration archive.
    """
    checkpoint_dir = Path(checkpoint_dir)
    name = "checkpoint.npz" if iteration is None else f"checkpoint_{iteration:08d}.npz"
    return checkpoint_dir / name
