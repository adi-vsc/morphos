"""LLM-backed interpreter using the Anthropic API.

Drop-in replacement for KeywordInterpreter satisfying the Interpreter protocol.
When running inside Claude Code, ANTHROPIC_API_KEY is set automatically.
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Dict, Optional

from morphos.agent.router import Plan, Interpreter
from morphos.agent.catalog import CATALOG, GeneratorSpec


def _build_tool_schemas() -> list:
    """Derive JSON tool schemas from _intent_registry() __init__ signatures."""
    from morphos.intent import _intent_registry
    registry = _intent_registry()

    tools = []
    for class_name, intent_cls in registry.items():
        sig = inspect.signature(intent_cls.__init__)
        properties: Dict[str, Any] = {}
        required = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            # Infer JSON type from annotation or default.
            annotation = param.annotation
            default = param.default
            if annotation is inspect.Parameter.empty:
                # Guess from default value type.
                if isinstance(default, bool):
                    json_type = "boolean"
                elif isinstance(default, int):
                    json_type = "integer"
                elif isinstance(default, float):
                    json_type = "number"
                elif isinstance(default, str):
                    json_type = "string"
                else:
                    json_type = "number"
            elif annotation in (int,):
                json_type = "integer"
            elif annotation in (float,):
                json_type = "number"
            elif annotation in (str,):
                json_type = "string"
            elif annotation in (bool,):
                json_type = "boolean"
            else:
                json_type = "number"

            prop: Dict[str, Any] = {"type": json_type}
            if default is not inspect.Parameter.empty and not isinstance(default, tuple):
                prop["default"] = default
            properties[param_name] = prop
            if default is inspect.Parameter.empty:
                required.append(param_name)

        tools.append({
            "name": class_name,
            "description": (
                f"Populate a {class_name} design intent. "
                f"Units: mm (lengths), N (forces), W (power), Pa (pressure). "
                f"Default volume_fraction=0.3, mesh resolution ~50 voxels/axis."
            ),
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        })

    # Always include the refusal sentinel.
    tools.append({
        "name": "unsupported",
        "description": (
            "Call when no available intent fits the request. "
            "Explain concisely why in the 'reason' field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    })
    return tools


def _build_system_prompt() -> str:
    catalog_lines = []
    for entry in CATALOG:
        tags = ", ".join(entry.physics_tags) if entry.physics_tags else "general"
        catalog_lines.append(f"  - {entry.name} [{tags}]: {entry.description}")
    catalog_block = "\n".join(catalog_lines)
    return (
        "You are the Morphos design-intent router. Your job is to map the user's "
        "natural-language engineering request to exactly one JSON tool call that "
        "populates the best-matching DesignIntent.\n\n"
        "RULES:\n"
        "1. Always respond with exactly one tool call — never prose.\n"
        "2. Use SI-adjacent units: mm for lengths, N for forces, W for power, Pa for pressure.\n"
        "3. Default volume_fraction = 0.3 unless the user specifies otherwise.\n"
        "4. Default mesh resolution = 50 voxels per axis (nx=50, ny=50 or span=50, height=50).\n"
        "5. If no intent fits, call the 'unsupported' tool with a clear reason.\n\n"
        "AVAILABLE GENERATORS:\n"
        f"{catalog_block}"
    )


class LLMInterpreter:
    """Route a request using claude-sonnet-4-6 with forced tool use.

    Derives JSON tool schemas from _intent_registry() by inspecting __init__
    signatures. Never accepts prose-only responses (tool_choice='any').
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic to use LLMInterpreter")
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. If running inside Claude Code, "
                "this is set automatically by Claude Code's internal authentication."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._tools = _build_tool_schemas()
        self._system_prompt = _build_system_prompt()

    def __call__(self, request: str) -> Plan:
        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=self._system_prompt,
            tools=self._tools,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": request}],
        )

        # Extract the first tool-use block.
        tool_use = next(
            (block for block in response.content if block.type == "tool_use"),
            None,
        )
        if tool_use is None:
            raise ValueError("Model returned no tool call; expected forced tool use.")

        tool_name = tool_use.name
        tool_input = tool_use.input

        if tool_name == "unsupported":
            reason = tool_input.get("reason", "No matching intent found.")
            raise ValueError(f"Request unsupported: {reason}")

        # Find the matching GeneratorSpec by intent class name.
        generator = next(
            (
                entry
                for entry in CATALOG
                if entry.intent_class is not None
                and entry.intent_class.__name__ == tool_name
            ),
            None,
        )

        return Plan(
            request=request,
            generator=generator,
            params=dict(tool_input),
            confidence=1.0,
            rationale=f"LLMInterpreter routed to {tool_name} via tool call.",
        )
