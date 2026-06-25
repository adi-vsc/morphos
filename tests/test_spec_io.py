"""Tests for spec serialisation: morphos.from_json / morphos.to_json."""

from pathlib import Path

import pytest

import morphos
from morphos.intent import CantileverIntent, HeatExchangerIntent

_DOC = Path(__file__).resolve().parents[1] / "docs" / "spec_schema.md"


def _normalise(params: dict) -> dict:
    """Treat tuples and lists alike (JSON has no tuple type)."""
    out = {}
    for k, v in params.items():
        out[k] = list(v) if isinstance(v, (list, tuple)) else v
    return out


def test_cantilever_intent_round_trips_json(tmp_path):
    intent = CantileverIntent(
        span=60, height=30, load=-1.0, volume_fraction=0.4, max_iter=120
    )
    path = tmp_path / "cantilever.json"
    morphos.to_json(intent, path)
    restored = morphos.from_json(path)

    assert isinstance(restored, CantileverIntent)
    assert _normalise(restored._init_params) == _normalise(intent._init_params)


def test_heat_exchanger_intent_round_trips_json(tmp_path):
    intent = HeatExchangerIntent(
        nx=12, ny=8, mu=2.0, u_max=1.5, heat_source_W_m3=3.0, k_solid=4.0
    )
    path = tmp_path / "hx.json"
    morphos.to_json(intent, path)
    restored = morphos.from_json(path)

    assert isinstance(restored, HeatExchangerIntent)
    assert _normalise(restored._init_params) == _normalise(intent._init_params)


def test_from_json_raises_on_unknown_intent(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"version": "1.0", "intent": "NoSuchIntent", "params": {}}')
    with pytest.raises(ValueError) as exc:
        morphos.from_json(path)
    assert "NoSuchIntent" in str(exc.value)


def test_spec_schema_doc_exists():
    assert _DOC.exists(), f"missing schema doc at {_DOC}"
    text = _DOC.read_text(encoding="utf-8")
    assert '"1.0"' in text
