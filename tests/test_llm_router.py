"""Tests for LLMInterpreter — all Anthropic API calls are mocked."""

import pytest
from unittest.mock import MagicMock, patch


def _make_tool_use_block(name: str, input_dict: dict):
    """Build a mock content block that looks like an Anthropic tool_use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_dict
    return block


def _make_mock_response(content_blocks: list):
    """Build a mock Anthropic Messages response."""
    response = MagicMock()
    response.content = content_blocks
    return response


def test_llm_interpreter_successful_extraction(monkeypatch):
    """Mock the anthropic client to return a CantileverIntent tool call.

    Asserts that LLMInterpreter returns a Plan with a non-None generator
    whose intent_class is CantileverIntent.
    """
    pytest.importorskip("anthropic")
    from morphos.agent.llm_router import LLMInterpreter
    from morphos.intent import CantileverIntent

    # Build a fake tool-use response.
    fake_input = {"span": 50, "height": 25, "load": -1.0, "volume_fraction": 0.3}
    fake_block = _make_tool_use_block("CantileverIntent", fake_input)
    fake_response = _make_mock_response([fake_block])

    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        interp = LLMInterpreter(api_key="sk-test-fake-key")
        plan = interp("a cantilever beam, 50 mm span, 25 mm height, 1 N tip load")

    assert plan.generator is not None, "Plan.generator should not be None for a matched intent"
    assert plan.generator.intent_class is CantileverIntent, (
        f"Expected CantileverIntent but got {plan.generator.intent_class}"
    )
    assert plan.params == fake_input


def test_llm_interpreter_unsupported_raises(monkeypatch):
    """Mock the anthropic client to return an 'unsupported' tool call.

    Asserts that LLMInterpreter raises ValueError with the model's reason.
    """
    pytest.importorskip("anthropic")
    from morphos.agent.llm_router import LLMInterpreter

    reason = "No intent matches a turbopump impeller with variable-pitch blades."
    fake_block = _make_tool_use_block("unsupported", {"reason": reason})
    fake_response = _make_mock_response([fake_block])

    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        interp = LLMInterpreter(api_key="sk-test-fake-key")
        with pytest.raises(ValueError, match=reason[:30]):
            interp("design a turbopump impeller with variable-pitch blades")


def test_llm_interpreter_raises_without_api_key(monkeypatch):
    """LLMInterpreter raises ValueError when ANTHROPIC_API_KEY is absent."""
    pytest.importorskip("anthropic")
    from morphos.agent.llm_router import LLMInterpreter

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch("anthropic.Anthropic"):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            LLMInterpreter()


def test_llm_interpreter_raises_without_anthropic_package(monkeypatch):
    """LLMInterpreter raises ImportError when 'anthropic' is not installed."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    from morphos.agent import llm_router as lr
    import importlib
    importlib.reload(lr)
    with pytest.raises(ImportError, match="pip install anthropic"):
        lr.LLMInterpreter(api_key="key")
