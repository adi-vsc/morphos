"""A standard-library terminal progress display for ``morphos.run()``.

No ``rich``, no ``tqdm``: a single updating block drawn with ``sys``, ``shutil``
and ``time`` only. On a TTY it draws an adaptive bar that overwrites one line;
when stdout is redirected or piped (not a TTY) it falls back to one timestamped
line every ten iterations, so a captured log stays readable.
"""

from __future__ import annotations

import shutil
import sys
import time

_BLOCK_FULL = "█"
_BLOCK_EMPTY = "░"
_MIN_BAR_WIDTH = 20
_LOG_EVERY = 10


class ProgressBar:
    """Render optimisation progress to a terminal.

    Parameters
    ----------
    total:
        Total iteration count (drives the bar fraction and the field width).
    stream:
        Output stream; defaults to ``sys.stdout``. Its ``isatty()`` selects the
        bar vs the line-log fallback.
    """

    def __init__(self, total: int, stream=None) -> None:
        self.total = max(1, int(total))
        self.stream = stream if stream is not None else sys.stdout
        try:
            self.is_tty = bool(self.stream.isatty())
        except (AttributeError, ValueError):
            self.is_tty = False
        self.start = time.time()
        self._width = len(str(self.total))

    def _emit(self, text: str) -> None:
        """Write ``text`` without ever raising on a console whose encoding
        cannot represent the bar glyphs (Windows code pages)."""
        enc = getattr(self.stream, "encoding", None) or "utf-8"
        try:
            self.stream.write(text.encode(enc, errors="replace").decode(enc))
        except (UnicodeError, TypeError):
            self.stream.write(text)
        self.stream.flush()

    def update(self, iteration, fom, delta, p, beta) -> None:
        elapsed = int(time.time() - self.start)
        if self.is_tty:
            self._draw_bar(iteration, fom, delta, p, beta, elapsed)
        elif iteration % _LOG_EVERY == 0 or iteration >= self.total:
            stamp = time.strftime("%H:%M:%S")
            self._emit(
                f"[{stamp}] iter {iteration:0{self._width}d}/{self.total}  "
                f"fom {fom:.4f}  Δ {delta:.4f}  p {p:.2f}  β {beta:.2f}  {elapsed}s\n"
            )

    def _draw_bar(self, iteration, fom, delta, p, beta, elapsed) -> None:
        suffix = (
            f" {iteration:0{self._width}d}/{self.total}  fom {fom:.4f}  "
            f"Δ {delta:.4f}  p {p:.2f}  β {beta:.2f}  {elapsed}s"
        )
        columns = shutil.get_terminal_size((80, 24)).columns
        bar_width = max(_MIN_BAR_WIDTH, columns - len(suffix) - 6)
        frac = min(1.0, iteration / self.total)
        filled = int(round(bar_width * frac))
        bar = _BLOCK_FULL * filled + _BLOCK_EMPTY * (bar_width - filled)
        self._emit(f"\r  [{bar}]{suffix}")

    def close(self, converged: bool) -> None:
        """Print a final status line and a trailing newline so the shell prompt
        lands cleanly on the next line."""
        elapsed = int(time.time() - self.start)
        status = "converged" if converged else "stopped at max iterations"
        if self.is_tty:
            self._emit(f"\n  {status} in {elapsed}s\n")
        else:
            stamp = time.strftime("%H:%M:%S")
            self._emit(f"[{stamp}] {status} in {elapsed}s\n")
