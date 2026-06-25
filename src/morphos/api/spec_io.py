"""Load and save Morphos spec files: ``morphos.from_json`` / ``morphos.to_json``.

A spec file is a small JSON document in one of two shapes (both documented in
``docs/spec_schema.md``):

* **Shape A (intent)** -- ``{"version": "1.0", "intent": "CantileverIntent",
  "params": {...}}``. The human and pipeline path, and the only lossless one: a
  :class:`~morphos.intent.DesignIntent` captures its constructor arguments, so it
  round-trips exactly.
* **Shape B (raw spec)** -- ``{"version": "1.0", "spec": {...}}``. Structural
  metadata for an already-built :class:`~morphos.spec.DesignSpec`; not lossless,
  because an oracle consumes its boundary conditions at construction (see
  :meth:`morphos.spec.DesignSpec.to_dict`).
"""

from __future__ import annotations

import json
from pathlib import Path

from morphos.intent import DesignIntent
from morphos.spec import DesignSpec

_VERSION = "1.0"


def from_json(path):
    """Load a spec JSON file into a :class:`DesignIntent` (Shape A) or a
    :class:`DesignSpec` (Shape B).

    Raises with a message naming what was expected and what was found when the
    file is missing, is not valid JSON, or has neither an ``intent`` nor a
    ``spec`` key.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"spec file not found: expected a readable JSON file, found nothing at {p}"
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "spec file is not valid JSON: expected a JSON object, found a parse "
            f"error in {p} ({exc})"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"spec file has the wrong shape: expected a JSON object, found a "
            f"{type(data).__name__} in {p}"
        )

    if "intent" in data:
        return DesignIntent.from_dict(data)
    if "spec" in data:
        return DesignSpec.from_dict(data["spec"])
    raise ValueError(
        "spec file has no design: expected an 'intent' key (shape A) or a "
        f"'spec' key (shape B), found keys {sorted(data)}"
    )


def to_json(spec, path) -> None:
    """Serialise a :class:`DesignIntent` (Shape A) or :class:`DesignSpec`
    (Shape B) to a spec JSON file at ``path``."""
    if isinstance(spec, DesignIntent):
        payload = spec.to_dict()
    elif isinstance(spec, DesignSpec):
        payload = {"version": _VERSION, "spec": spec.to_dict()}
    else:
        raise TypeError(
            "to_json expects a DesignIntent or DesignSpec, found a "
            f"{type(spec).__name__}"
        )
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
