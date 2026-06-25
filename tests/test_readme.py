"""Test that the README's Python quickstart actually runs."""

import os
import re
from pathlib import Path

import pytest

pytest.importorskip("skimage")  # the quickstart writes design.stl via export

_README = Path(__file__).resolve().parents[1] / "README.md"


def _first_python_block(text: str) -> str:
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    assert m, "no ```python block found in README"
    return m.group(1)


def test_readme_quickstart_is_executable(tmp_path, monkeypatch):
    snippet = _first_python_block(_README.read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__readme__"}
    exec(compile(snippet, "<readme-quickstart>", "exec"), namespace)
    # The agent quickstart writes its STL under out/ named after the generator.
    assert (tmp_path / "out" / "gyroid_heat_exchanger.stl").exists()
