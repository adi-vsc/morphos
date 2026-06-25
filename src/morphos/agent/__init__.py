"""Agent layer: natural-language request -> generator -> validated STL.

A thin, honest orchestration layer over the geometry generators. It does not
synthesise arbitrary geometry; it maps a request to one of a known catalog of
parametric generators and builds it. Breadth grows by adding generators.
"""

from morphos.agent.catalog import CATALOG, GeneratorSpec
from morphos.agent.router import Plan, interpret, KeywordInterpreter, Interpreter
from morphos.agent.build import design_from_text, DesignResultFromText

__all__ = [
    "CATALOG",
    "GeneratorSpec",
    "Plan",
    "interpret",
    "KeywordInterpreter",
    "Interpreter",
    "design_from_text",
    "DesignResultFromText",
]
