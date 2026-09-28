"""Inline-SVG chart primitives for the dashboard.

Colours come from CSS custom properties defined once on ``.viz-root``, so the
same markup renders correctly in light and dark without regenerating anything.
Every chart carries a hover layer and every chart has a table fallback.
"""
from __future__ import annotations

import html
from typing import Sequence

from .normalize import inr


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def compact(value: float) -> str:
    """Indian short form: 1.2 Cr / 3.4 L / 12,500."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return ""
    sign = "-" if n < 0 else ""
    a = abs(n)
    if a >= 1e7:
        return f"{sign}{a / 1e7:,.2f} Cr"
    if a >= 1e5:
        return f"{sign}{a / 1e5:,.2f} L"
    if a >= 1000:
        return f"{sign}{a:,.0f}"
    return f"{sign}{a:,.0f}"


def _nice_max(value: float) -> float:
    """Round an axis maximum up to a readable step."""
    if value <= 0:
        return 1.0
    import math
    exp = math.floor(math.log10(value))
    base = 10 ** exp
    for mult in (1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10):
        if value <= mult * base:
            return mult * base
    return 10 * base


# ---------------------------------------------------------------- h-bars

def hbar(items: Sequence[tuple[str, float]], *, unit: str = "Rs.",
         color: str = "var(--series-1)", height: int = 22,
         label_w: int = 150, value_fmt=compact, max_bars: int = 12) -> str:
    """Horizontal bars, one series. Label on the left, value direct-labelled at
    the bar end -- no legend needed for a single series."""
    items = [(str(k), float(v or 0)) for k, v in items][:max_bars]
    if not items:
        return '<p class="empty">Nothing to show.</p>'
    top = _nice_max(max(abs(v) for _, v in items) or 1)
    pad_r = 92
    width = 640
    plot_w = width - label_w - pad_r
    rows = []
    for i, (label, value) in enumerate(items):
        y = i * height
        w = max(2.0, abs(value) / top * plot_w)
        rows.append(
            f'<g class="mk" tabindex="0" role="listitem" '
            f'data-tip="{esc(label)} {esc(unit)} {esc(inr(value))}">'
            f'<rect class="hit" x="0" y="{y}" width="{width}" height="{height}"/>'
            f'<text class="lbl" x="{label_w - 8}" y="{y + height / 2}" '
            f'text-anchor="end" dominant-baseline="central">{esc(label[:26])}</text>'
            f'<rect class="bar" x="{label_w}" y="{y + 4}" width="{w:.1f}" '
            f'height="{height - 9}" rx="4" fill="{color}"/>'
            f'<text class="val" x="{label_w + w + 8:.1f}" y="{y + height / 2}" '
            f'dominant-baseline="central">{esc(value_fmt(value))}</text>'
            f"</g>")
    total_h = len(items) * height + 6
    return (f'<svg class="chart" viewBox="0 0 {width} {total_h}" role="list" '
            f'preserveAspectRatio="xMidYMin meet">'
            f'<line class="axis" x1="{label_w}" y1="0" x2="{label_w}" y2="{total_h - 6}"/>'
            + "".join(rows) + "</svg>")


# ------------------------------------------------------------ stacked bar

def stacked_bar(segments: Sequence[tuple[str, float, str]], *,
                unit: str = "") -> str:
    """One horizontal bar split into named segments, 2px surface gaps between
    them so adjacent fills never touch."""
    segments = [(n, float(v or 0), c) for n, v, c in segments if float(v or 0) > 0]
    if not segments:
        return '<p class="empty">Nothing to show.</p>'
    total = sum(v for _, v, _ in segments)
    width, height = 640, 34
    x = 0.0
    parts = []
    for name, value, color in segments:
        w = value / total * width
        draw = max(w - 2, 1.5)
        parts.append(
            f'<g class="mk" tabindex="0" '
            f'data-tip="{esc(name)} {esc(str(int(value)))} {esc(unit)} '
            f'({value / total * 100:.0f}%)">'
            f'<rect x="{x:.1f}" y="0" width="{draw:.1f}" height="{height}" rx="4" '
            f'fill="{color}"/>'
            + (f'<text class="seg" x="{x + draw / 2:.1f}" y="{height / 2}" '
               f'text-anchor="middle" dominant-baseline="central">{int(value)}</text>'
               if draw > 26 else "")
            + "</g>")
        x += w
    return (f'<svg class="chart" viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="none" style="height:34px">' + "".join(parts) + "</svg>")


# ---------------------------------------------------------------- lines

def line_chart(x_labels: Sequence[str], series: Sequence[tuple[str, Sequence[float], str]],
               *, unit: str = "Rs.", height: int = 260) -> str:
    """Multi-series line chart with a shared y-axis, direct end labels and a
    crosshair tooltip. One axis only -- never two scales."""
    if not x_labels or not series:
        return '<p class="empty">Nothing to show.</p>'
    width = 720
    pad_l, pad_r, pad_t, pad_b = 66, 108, 14, 34
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    values = [v for _, vals, _ in series for v in vals if v is not None]
    top = _nice_max(max(values) if values else 1)
    n = max(len(x_labels) - 1, 1)

    def X(i: int) -> float:
        return pad_l + i * plot_w / n

    def Y(v: float) -> float:
        return pad_t + plot_h - (v / top * plot_h if top else 0)

    grid, ticks = [], 4
    for k in range(ticks + 1):
        v = top * k / ticks
        y = Y(v)
        grid.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" '
                    f'x2="{pad_l + plot_w}" y2="{y:.1f}"/>')
        grid.append(f'<text class="axl" x="{pad_l - 10}" y="{y:.1f}" text-anchor="end" '
                    f'dominant-baseline="central">{esc(compact(v))}</text>')

    step = max(1, len(x_labels) // 12)
    for i, lab in enumerate(x_labels):
        if i % step:
            continue
        grid.append(f'<text class="axl" x="{X(i):.1f}" y="{pad_t + plot_h + 18}" '
                    f'text-anchor="middle">{esc(lab)}</text>')

    paths = []
    for name, vals, color in series:
        pts = [(X(i), Y(float(v or 0))) for i, v in enumerate(vals)]
        d = " ".join(("M" if i == 0 else "L") + f"{x:.1f} {y:.1f}"
                     for i, (x, y) in enumerate(pts))
        paths.append(f'<path class="ln" d="{d}" stroke="{color}"/>')
        lx, ly = pts[-1]
        paths.append(f'<text class="end" x="{lx + 10:.1f}" y="{ly:.1f}" fill="{color}" '
                     f'dominant-baseline="central">{esc(name)}</text>')
        for i, (x, y) in enumerate(pts):
            paths.append(
                f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" '
                f'tabindex="0" data-tip="{esc(name)} · {esc(x_labels[i])}'
                f' {esc(unit)} {esc(inr(vals[i]))}"/>')

    return (f'<svg class="chart line" viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="xMidYMid meet">'
            + "".join(grid)
            + f'<line class="axis" x1="{pad_l}" y1="{pad_t + plot_h}" '
              f'x2="{pad_l + plot_w}" y2="{pad_t + plot_h}"/>'
            + "".join(paths) + "</svg>")


def legend(entries: Sequence[tuple[str, str]]) -> str:
    """A legend is always present for two or more series."""
    if len(entries) < 2:
        return ""
    items = "".join(
        f'<span class="lg"><i style="background:{c}"></i>{esc(n)}</span>'
        for n, c in entries)
    return f'<div class="legend">{items}</div>'


def table(headers: Sequence[str], rows: Sequence[Sequence], *,
          numeric: Sequence[int] = ()) -> str:
    """The table view every chart falls back to."""
    th = "".join(f'<th{" class=n" if i in numeric else ""}>{esc(h)}</th>'
                 for i, h in enumerate(headers))
    body = []
    for r in rows:
        tds = "".join(f'<td{" class=n" if i in numeric else ""}>{esc(c)}</td>'
                      for i, c in enumerate(r))
        body.append(f"<tr>{tds}</tr>")
    return (f'<table class="tbl"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')
