"""Schema completeness tests for the generator catalog.

Ensures every GeneratorSpec carries the capability-metadata fields added in the
Task-2 hardening pass: physics_tags, description, and param_ranges.
"""

import pytest
from morphos.agent.catalog import CATALOG


def test_all_entries_have_physics_tags():
    for entry in CATALOG:
        assert isinstance(entry.physics_tags, list), f"{entry.name} missing physics_tags"
        assert len(entry.physics_tags) > 0, f"{entry.name} has empty physics_tags"


def test_all_entries_have_description():
    for entry in CATALOG:
        assert isinstance(entry.description, str), f"{entry.name} missing description"
        assert len(entry.description) > 0, f"{entry.name} has empty description"


def test_all_entries_have_param_ranges():
    for entry in CATALOG:
        assert isinstance(entry.param_ranges, dict), f"{entry.name} missing param_ranges"
        assert len(entry.param_ranges) > 0, f"{entry.name} has empty param_ranges"


def test_intent_class_in_registry():
    from morphos.intent import _intent_registry
    registry = _intent_registry()
    for entry in CATALOG:
        if entry.intent_class is not None:
            assert entry.intent_class.__name__ in registry, (
                f"{entry.name}: {entry.intent_class.__name__} not in registry"
            )


def test_catalog_has_thirteen_entries():
    assert len(CATALOG) == 13, f"Expected 13 catalog entries, got {len(CATALOG)}"


def test_all_entries_have_unique_names():
    names = [e.name for e in CATALOG]
    assert len(names) == len(set(names)), "Duplicate names found in CATALOG"
