"""Tests for the terminal progress bar."""

import io

from morphos.cli.progress import ProgressBar


class _FakeTTY(io.StringIO):
    """A StringIO that claims to be a TTY, for exercising the bar path."""

    def isatty(self):
        return True


def test_progress_bar_renders_without_tty():
    # A plain StringIO is not a TTY, so the bar falls back to one timestamped
    # line per 10 iterations. Ten updates at multiples of 10 -> ten lines.
    buf = io.StringIO()
    bar = ProgressBar(total=100, stream=buf)
    for i in range(1, 11):
        bar.update(i * 10, fom=0.5 + i * 0.01, delta=0.001, p=2.0, beta=4.0)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 10
    assert all(":" in ln for ln in lines)  # each carries a HH:MM:SS timestamp


def test_progress_bar_close_writes_newline():
    buf = io.StringIO()
    bar = ProgressBar(total=10, stream=buf)
    bar.update(5, fom=0.5, delta=0.01, p=2.0, beta=4.0)
    bar.close(converged=True)
    assert buf.getvalue().endswith("\n")


def test_progress_bar_width_clamps_to_minimum(monkeypatch):
    import os
    import shutil

    monkeypatch.setattr(
        shutil, "get_terminal_size", lambda *a, **k: os.terminal_size((10, 24))
    )
    buf = _FakeTTY()
    bar = ProgressBar(total=200, stream=buf)
    # Must render without error even when the terminal is narrower than the bar.
    bar.update(52, fom=0.8312, delta=0.0021, p=2.40, beta=4.00)
    assert buf.getvalue() != ""
