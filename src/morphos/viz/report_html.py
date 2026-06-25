"""Self-contained interactive HTML report for a ``morphos.run()`` result.

``render_html_report`` writes a single ``report.html`` with every byte of CSS
and JavaScript inlined: no Chart.js, no D3, no external fonts, no image files. It
opens straight from disk in any browser. The convergence chart is a hand-rolled
SVG polyline; the density slices are drawn into ``<canvas>`` elements by a few
lines of vanilla JS over an embedded numeric array; the viridis colour map is a
256-stop lookup table generated here (a polynomial approximation, so we keep the
no-matplotlib-at-runtime promise) and embedded as a plain JS array.

The layout is five sections: run summary, convergence chart, density slice
viewer, manufacturing summary, and download links. Styling is a clean monochrome
palette with the Morphos purple (#534AB7) as the single accent.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

_ACCENT = "#534AB7"

# Polynomial approximation of the viridis colormap (matte-black to yellow,
# perceptually uniform). Coefficients evaluate a degree-6 polynomial in t in
# [0, 1] per channel; generating the 256-stop table here means the browser ships
# zero colour-map code and the package needs no matplotlib at runtime.
_VIRIDIS_COEFFS = np.array(
    [
        [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
        [0.1050930431085774, 1.404613529898575, 1.384590162594685],
        [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
        [-4.634230498983486, -5.799100973351585, -19.33244095627987],
        [6.228269936347081, 14.17993336680509, 56.69055260068105],
        [4.776384997670288, -13.74514537774601, -65.35303263337234],
        [-5.435455855934631, 4.645852612178535, 26.3124352495832],
    ]
)


def _viridis_lut(n: int = 256) -> list:
    """A 256-stop viridis lookup table as a list of ``[r, g, b]`` 0-255 ints."""
    t = np.linspace(0.0, 1.0, n)
    powers = np.vstack([t ** k for k in range(len(_VIRIDIS_COEFFS))])  # (7, n)
    rgb = _VIRIDIS_COEFFS.T @ powers  # (3, n)
    rgb = np.clip(rgb.T, 0.0, 1.0)
    return (rgb * 255.0).round().astype(int).tolist()


def _slices(field) -> dict:
    """Density slices to display. A 2D field gives a single ``density`` map; a
    3D field gives the three orthogonal mid-planes (XY, XZ, YZ)."""
    v = np.asarray(field.values, dtype=float)
    if v.ndim == 2:
        return {"density": v}
    nz, ny, nx = v.shape
    return {
        "XY mid-plane": v[nz // 2, :, :],
        "XZ mid-plane": v[:, ny // 2, :],
        "YZ mid-plane": v[:, :, nx // 2],
    }


def _round_grid(arr: np.ndarray) -> list:
    """A 2D array as a nested list rounded to 3 decimals (compact, lossless
    enough for a [0, 1] density and small enough to keep the file tiny)."""
    return np.round(np.asarray(arr, dtype=float), 3).tolist()


def _ramp_series(optimizer, n: int):
    """Reconstruct the per-iteration SIMP p-ramp and Heaviside beta-ramp from a
    topology optimizer's continuation schedule, or ``(None, None)`` for an
    optimizer (e.g. parametric L-BFGS-B) that has no such schedule."""
    if n <= 0 or not hasattr(optimizer, "_current_p"):
        return None, None
    p = [float(optimizer._current_p(i)) for i in range(n)]
    beta = [float(optimizer._current_beta(i)) for i in range(n)]
    # A flat ramp (p_start == p_end) carries no information worth a second axis.
    if len(set(p)) == 1 and len(set(beta)) == 1:
        return None, None
    return p, beta


def _polyline(xs, ys, x0, x1, y0, y1, lo, hi) -> str:
    """Map data ``(xs, ys)`` into the plot box ``[x0,x1] x [y0,y1]`` (y down)
    and return an SVG ``points`` string. ``lo``/``hi`` bound the y data range."""
    n = len(xs)
    span = (hi - lo) or 1.0
    xspan = (xs[-1] - xs[0]) or 1.0
    pts = []
    for i in range(n):
        px = x0 + (xs[i] - xs[0]) / xspan * (x1 - x0)
        py = y1 - (ys[i] - lo) / span * (y1 - y0)
        pts.append(f"{px:.1f},{py:.1f}")
    return " ".join(pts)


def _convergence_svg(history, p_series, beta_series) -> str:
    """A hand-rolled SVG line chart of FOM vs iteration, auto-scaled with five
    y-axis ticks, plus optional p-ramp and beta-ramp on a right axis."""
    if not history:
        return "<p class='muted'>No convergence history was recorded.</p>"

    w, h = 720, 320
    x0, x1, y0, y1 = 60, w - 60, 20, h - 40
    xs = list(range(1, len(history) + 1))
    lo, hi = min(history), max(history)
    if hi == lo:
        hi = lo + 1.0

    parts = [f"<svg viewBox='0 0 {w} {h}' class='chart' role='img' "
             "aria-label='convergence chart'>"]
    # Axes.
    parts.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' class='axis'/>")
    parts.append(f"<line x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}' class='axis'/>")
    # Five y ticks + gridlines + labels (left axis: FOM).
    for k in range(5):
        val = lo + (hi - lo) * k / 4
        py = y1 - (val - lo) / (hi - lo) * (y1 - y0)
        parts.append(f"<line x1='{x0}' y1='{py:.1f}' x2='{x1}' y2='{py:.1f}' "
                     "class='grid'/>")
        parts.append(f"<text x='{x0 - 6}' y='{py + 3:.1f}' class='tick' "
                     f"text-anchor='end'>{val:.3g}</text>")
    # X labels (first and last iteration).
    parts.append(f"<text x='{x0}' y='{y1 + 18}' class='tick'>1</text>")
    parts.append(f"<text x='{x1}' y='{y1 + 18}' class='tick' "
                 f"text-anchor='end'>{xs[-1]}</text>")
    parts.append(f"<text x='{(x0 + x1) / 2:.0f}' y='{h - 6}' class='axislabel' "
                 "text-anchor='middle'>iteration</text>")
    parts.append(f"<text x='16' y='{(y0 + y1) / 2:.0f}' class='axislabel' "
                 f"transform='rotate(-90 16 {(y0 + y1) / 2:.0f})' "
                 "text-anchor='middle'>fom</text>")

    # FOM polyline (accent).
    pts = _polyline(xs, history, x0, x1, y0, y1, lo, hi)
    parts.append(f"<polyline points='{pts}' class='fom'/>")

    # Secondary right axis: p-ramp and beta-ramp, shared 0..max scale.
    if p_series is not None and beta_series is not None:
        rlo = 0.0
        rhi = max(max(p_series), max(beta_series)) or 1.0
        for series, cls in ((p_series, "pramp"), (beta_series, "bramp")):
            rpts = _polyline(xs, series, x0, x1, y0, y1, rlo, rhi)
            parts.append(f"<polyline points='{rpts}' class='{cls}'/>")
        for k in range(5):
            val = rlo + (rhi - rlo) * k / 4
            py = y1 - (val - rlo) / (rhi - rlo) * (y1 - y0)
            parts.append(f"<text x='{x1 + 6}' y='{py + 3:.1f}' class='tick' "
                         f"text-anchor='start'>{val:.2g}</text>")
        parts.append(
            f"<g class='legend'>"
            f"<rect x='{x1 - 150}' y='{y0}' width='10' height='3' class='fom-sw'/>"
            f"<text x='{x1 - 135}' y='{y0 + 4}' class='tick'>fom</text>"
            f"<rect x='{x1 - 100}' y='{y0}' width='10' height='3' class='p-sw'/>"
            f"<text x='{x1 - 85}' y='{y0 + 4}' class='tick'>p</text>"
            f"<rect x='{x1 - 60}' y='{y0}' width='10' height='3' class='b-sw'/>"
            f"<text x='{x1 - 45}' y='{y0 + 4}' class='tick'>beta</text>"
            f"</g>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _orientation_svg(orientation) -> str:
    """A small isometric unit-cube SVG with the recommended build axis drawn as
    a highlighted arrow from the cube centre."""
    if orientation is None:
        return "<p class='muted'>No recommended orientation.</p>"
    ox, oy, oz = (float(c) for c in orientation)

    # Simple isometric projection of a 3D point to 2D screen coordinates.
    def proj(x, y, z):
        sx = 90 + (x - y) * 38
        sy = 80 + (x + y) * 20 - z * 44
        return sx, sy

    # Cube corners.
    c = {(i, j, k): proj(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    edges = [
        ((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, 0, 1)),
        ((1, 1, 1), (0, 1, 1)), ((1, 1, 1), (1, 0, 1)), ((1, 1, 1), (1, 1, 0)),
        ((1, 0, 0), (1, 1, 0)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (1, 1, 0)),
        ((0, 1, 0), (0, 1, 1)), ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1)),
    ]
    parts = ["<svg viewBox='0 0 180 160' class='cube' role='img' "
             "aria-label='recommended build orientation'>"]
    for a, b in edges:
        (ax, ay), (bx, by) = c[a], c[b]
        parts.append(f"<line x1='{ax:.1f}' y1='{ay:.1f}' x2='{bx:.1f}' "
                     f"y2='{by:.1f}' class='edge'/>")
    cx, cy = proj(0.5, 0.5, 0.5)
    tx, ty = proj(0.5 + ox, 0.5 + oy, 0.5 + oz)
    parts.append(f"<line x1='{cx:.1f}' y1='{cy:.1f}' x2='{tx:.1f}' y2='{ty:.1f}' "
                 "class='buildaxis'/>")
    parts.append(f"<circle cx='{tx:.1f}' cy='{ty:.1f}' r='4' class='buildtip'/>")
    parts.append("</svg>")
    return "".join(parts)


def _status_for(value) -> str:
    """Best-effort pass/warn/fail classification of a manufacturability metric."""
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, (int, float)):
        if value <= 0.05:
            return "pass"
        if value <= 0.2:
            return "warn"
        return "fail"
    return "info"


def _manufacturing_rows(manufacturability: dict) -> str:
    """HTML table rows from a flat (or one-level-nested) manufacturability dict.

    Each leaf metric becomes a row: constraint/metric name, a status badge, and
    the metric value. Booleans map to pass/fail, fractions to pass/warn/fail."""
    if not manufacturability:
        return ("<tr><td colspan='3' class='muted'>No manufacturing "
                "constraints were applied.</td></tr>")
    rows = []

    def emit(name, value):
        status = _status_for(value)
        if isinstance(value, bool):
            shown = "yes" if value else "no"
        elif isinstance(value, float):
            shown = f"{value:.4g}"
        else:
            shown = html.escape(str(value))
        rows.append(
            f"<tr><td>{html.escape(str(name))}</td>"
            f"<td><span class='badge {status}'>{status}</span></td>"
            f"<td>{shown}</td></tr>"
        )

    for key, val in manufacturability.items():
        if isinstance(val, dict):
            for sub, subval in val.items():
                emit(f"{key} / {sub}", subval)
        else:
            emit(key, val)
    return "".join(rows)


def _summary_card(result) -> str:
    dr = result.design_result
    rep = result.report
    intent = getattr(result, "intent_name", None) or "design"
    field = dr.field
    if field.values.ndim == 2:
        ny, nx = field.shape
        dims = f"{nx} x {ny}"
    else:
        nz, ny, nx = field.shape
        dims = f"{nx} x {ny} x {nz}"
    if dr.converged:
        badge = "<span class='badge pass'>converged</span>"
    else:
        badge = "<span class='badge fail'>stopped at max iterations</span>"
    return (
        "<div class='cards'>"
        f"<div class='card'><div class='k'>intent</div>"
        f"<div class='v'>{html.escape(str(intent))}</div></div>"
        f"<div class='card'><div class='k'>grid</div><div class='v'>{dims}</div></div>"
        f"<div class='card'><div class='k'>wall-clock time</div>"
        f"<div class='v'>{result.elapsed_seconds:.1f} s</div></div>"
        f"<div class='card'><div class='k'>status</div><div class='v'>{badge}</div></div>"
        f"<div class='card'><div class='k'>final fom</div>"
        f"<div class='v'>{rep.figure_of_merit:.4f}</div></div>"
        f"<div class='card'><div class='k'>volume fraction</div>"
        f"<div class='v'>{rep.mass_fraction:.3f}</div></div>"
        "</div>"
    )


_STYLE = """
:root { --accent: #534AB7; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #111; background: #fff; margin: 0; padding: 0 24px 64px; }
header { padding: 32px 0 8px; border-bottom: 2px solid var(--accent); margin-bottom: 24px; }
h1 { font-size: 22px; margin: 0; font-weight: 600; }
h1 .dot { color: var(--accent); }
h2 { font-size: 15px; text-transform: none; letter-spacing: 0; margin: 32px 0 12px;
  color: var(--accent); font-weight: 600; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; }
.card { border: 1px solid #e5e5e5; border-radius: 8px; padding: 12px 16px; min-width: 150px; }
.card .k { font-size: 11px; color: #666; }
.card .v { font-size: 18px; font-weight: 600; margin-top: 2px; }
.badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 600; }
.badge.pass { background: #e3f5e9; color: #1a7f3c; }
.badge.warn { background: #fdf3df; color: #9a6b00; }
.badge.fail { background: #fde4e4; color: #b32020; }
.badge.info { background: #eee; color: #444; }
.chart { width: 100%; max-width: 720px; height: auto; }
.axis { stroke: #999; stroke-width: 1; }
.grid { stroke: #eee; stroke-width: 1; }
.tick { font-size: 10px; fill: #666; }
.axislabel { font-size: 11px; fill: #444; }
.fom { fill: none; stroke: var(--accent); stroke-width: 2; }
.pramp { fill: none; stroke: #d08a00; stroke-width: 1.5; stroke-dasharray: 4 3; }
.bramp { fill: none; stroke: #1a7f3c; stroke-width: 1.5; stroke-dasharray: 1 3; }
.fom-sw { fill: var(--accent); } .p-sw { fill: #d08a00; } .b-sw { fill: #1a7f3c; }
.slices { display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }
.slice { text-align: center; }
.slice canvas { border: 1px solid #ddd; image-rendering: pixelated; width: 220px; height: auto; }
.slice .lbl { font-size: 12px; color: #444; margin-top: 6px; }
.toggle { margin: 8px 0 4px; }
button { font: inherit; font-size: 13px; border: 1px solid var(--accent); color: var(--accent);
  background: #fff; border-radius: 6px; padding: 5px 12px; cursor: pointer; }
button:hover { background: var(--accent); color: #fff; }
table { border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 6px 14px; border-bottom: 1px solid #eee; }
th { color: #666; font-weight: 600; }
.cube { width: 180px; height: 160px; }
.edge { stroke: #bbb; stroke-width: 1.2; }
.buildaxis { stroke: var(--accent); stroke-width: 3; }
.buildtip { fill: var(--accent); }
.downloads a { display: inline-block; margin-right: 16px; color: var(--accent);
  text-decoration: none; font-weight: 600; border: 1px solid var(--accent);
  border-radius: 6px; padding: 6px 14px; }
.downloads a:hover { background: var(--accent); color: #fff; }
.muted { color: #999; }
footer { margin-top: 48px; font-size: 11px; color: #aaa; }
"""


def _canvas_script(slices: dict, lut: list) -> str:
    """The vanilla-JS block that paints each density slice into its canvas and
    wires the greyscale / false-colour toggle."""
    data = {name: _round_grid(arr) for name, arr in slices.items()}
    payload = {
        "slices": [
            {"id": f"slice{i}", "name": name, "grid": data[name]}
            for i, name in enumerate(data)
        ],
        "viridis": lut,
    }
    return (
        "<script>\n"
        f"const MORPHOS = {json.dumps(payload, separators=(',', ':'))};\n"
        "let useColour = false;\n"
        "function paint() {\n"
        "  for (const s of MORPHOS.slices) {\n"
        "    const cv = document.getElementById(s.id); if (!cv) continue;\n"
        "    const g = s.grid, h = g.length, w = g[0].length;\n"
        "    cv.width = w; cv.height = h;\n"
        "    const ctx = cv.getContext('2d');\n"
        "    const img = ctx.createImageData(w, h);\n"
        "    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {\n"
        "      let v = g[y][x]; v = v < 0 ? 0 : v > 1 ? 1 : v;\n"
        "      let r, gg, b;\n"
        "      if (useColour) { const c = MORPHOS.viridis[Math.round(v*255)]; r=c[0]; gg=c[1]; b=c[2]; }\n"
        "      else { const t = Math.round((1-v)*255); r=t; gg=t; b=t; }\n"
        "      const o = (y*w+x)*4; img.data[o]=r; img.data[o+1]=gg; img.data[o+2]=b; img.data[o+3]=255;\n"
        "    }\n"
        "    ctx.putImageData(img, 0, 0);\n"
        "  }\n"
        "}\n"
        "function toggleColour() {\n"
        "  useColour = !useColour;\n"
        "  document.getElementById('cmap-label').textContent = useColour ? 'false colour (viridis)' : 'greyscale';\n"
        "  paint();\n"
        "}\n"
        "window.addEventListener('DOMContentLoaded', paint);\n"
        "</script>\n"
    )


def render_html_report(result, path) -> None:
    """Write a self-contained ``report.html`` for ``result`` to ``path``."""
    path = Path(path)
    dr = result.design_result
    slices = _slices(dr.field)
    lut = _viridis_lut()

    optimizer = getattr(result, "optimizer", None)
    p_series, beta_series = _ramp_series(optimizer, len(dr.history))
    chart = _convergence_svg(dr.history, p_series, beta_series)

    slice_blocks = []
    for i, name in enumerate(slices):
        slice_blocks.append(
            f"<div class='slice'><canvas id='slice{i}'></canvas>"
            f"<div class='lbl'>{html.escape(name)}</div></div>"
        )

    orient = None
    if result.bundle is not None and result.bundle.recommended_orientation is not None:
        orient = result.bundle.recommended_orientation.orientation
    cube = _orientation_svg(orient)

    mfg_rows = _manufacturing_rows(result.report.manufacturability)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Morphos run report</title>
<style>{_STYLE}</style>
</head>
<body>
<header><h1>Morphos<span class="dot">.</span> run report</h1></header>

<section>
<h2>Run summary</h2>
{_summary_card(result)}
</section>

<section>
<h2>Convergence</h2>
{chart}
</section>

<section>
<h2>Density slices</h2>
<div class="toggle">
  <button onclick="toggleColour()">toggle colour map</button>
  <span class="muted">current: <span id="cmap-label">greyscale</span></span>
</div>
<div class="slices">{''.join(slice_blocks)}</div>
</section>

<section>
<h2>Manufacturing summary</h2>
<div class="slices">
  <table>
    <tr><th>constraint / metric</th><th>status</th><th>value</th></tr>
    {mfg_rows}
  </table>
  <div class="slice">{cube}<div class="lbl">recommended build orientation</div></div>
</div>
</section>

<section>
<h2>Downloads</h2>
<div class="downloads">
  <a href="design.stl" download>design.stl</a>
  <a href="report.json" download>report.json</a>
</div>
</section>

<footer>Generated by morphos. Self-contained report, no external dependencies.</footer>
{_canvas_script(slices, lut)}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
