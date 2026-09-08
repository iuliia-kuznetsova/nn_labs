"""Shared drawing primitives for the lecture-note figures.

Every lecture-note figure set imports from here, so the documents stay
visually consistent.
"""

from __future__ import annotations

FONT = "Segoe UI, system-ui, -apple-system, Helvetica, Arial, sans-serif"

INK = "#111827"
MUTED = "#6b7280"
LINE = "#9ca3af"

# fill / stroke pairs, keyed by layer role.
#
# One palette across every architecture diagram:
#   gray input · green convolution · blue pooling · yellow flatten
#   orange fully connected · red output (softmax / ReLU / final volume)
#
# Convolutions use a green ramp so filter sizes stay distinguishable without
# leaving the green family.
C = {
    "input": ("#e5e7eb", "#6b7280"),    # gray
    "conv1": ("#dcfce7", "#22c55e"),    # light green  — 1×1 convolution
    "conv": ("#bbf7d0", "#16a34a"),     # green        — standard convolution
    "conv5": ("#86efac", "#15803d"),    # dark green   — large or depthwise conv
    "block": ("#a7f3d0", "#059669"),    # teal green   — a repeated conv block
    "pool": ("#bfdbfe", "#2563eb"),     # blue
    "flat": ("#fef08a", "#ca8a04"),     # yellow
    "fc": ("#fed7aa", "#ea580c"),       # orange
    "out": ("#fecaca", "#dc2626"),      # red
    "concat": ("#e5e7eb", "#4b5563"),   # neutral bar that groups other blocks
    "pad": ("#fef9c3", "#ca8a04"),      # zero padding or bias, not a layer block
}

# aliases, so each figure can name the thing it is actually drawing
C["one"] = C["conv1"]                   # a 1×1 convolution
C["point"] = C["conv1"]                 # a pointwise conv is a 1×1 conv
C["depth"] = C["conv5"]                 # a depthwise conv
C["mpool"] = C["pool"]
C["g1"], C["g2"], C["g3"] = C["conv1"], C["conv"], C["conv5"]

# accent colours used for highlights and annotations
RED = "#dc2626"
BLUE = "#2563eb"
GREEN = "#16a34a"
AMBER = "#ca8a04"
PURPLE = "#7c3aed"
PINK = "#db2777"


# ----------------------------------------------------------------- primitives


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(w: float, h: float, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" '
        f'width="{w:g}" height="{h:g}">\n'
        '<defs>\n'
        '  <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse">\n'
        f'    <path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/>\n'
        '  </marker>\n'
        '</defs>\n'
        f'<rect width="{w:g}" height="{h:g}" fill="#ffffff"/>\n'
        f'<g font-family="{FONT}" fill="{INK}" font-size="12">\n'
        f"{body}\n"
        "</g>\n"
        "</svg>\n"
    )


def write(out, fname: str, w: float, h: float, body) -> None:
    if isinstance(body, list):
        body = "\n".join(body)
    (out / fname).write_text(svg(w, h, body), encoding="utf-8")


def rect(x, y, w, h, role="conv", rx=3, sw=1.4, fill=None, stroke=None, dash=None):
    f, s = C.get(role, C["conv"])
    f, s = fill or f, stroke or s
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx:g}" '
        f'fill="{f}" stroke="{s}" stroke-width="{sw:g}"{d}/>'
    )


def txt(x, y, s, size=12, anchor="middle", fill=INK, weight="normal", rot=None):
    tr = f' transform="rotate({rot:g} {x:g} {y:g})"' if rot is not None else ""
    return (
        f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}"{tr}>{esc(s)}</text>'
    )


def lines(x, y, rows, size=11, anchor="middle", fill=INK, weight="normal", lh=13):
    return "\n".join(
        txt(x, y + i * lh, r, size, anchor, fill, weight) for i, r in enumerate(rows)
    )


def arrow(x1, y1, x2, y2, color=LINE, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color}" '
        f'stroke-width="{sw:g}" marker-end="url(#ah)"{d}/>'
    )


def seg(x1, y1, x2, y2, color=LINE, sw=1.0, dash=None):
    """A plain line, no arrowhead."""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color}" '
        f'stroke-width="{sw:g}"{d}/>'
    )


def path(d, stroke=LINE, sw=1.4, fill="none", marker=True, dash=None):
    m = ' marker-end="url(#ah)"' if marker else ""
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}"{m}{da}/>'
    )


def poly(pts, fill="#f3f4f6", stroke="none", sw=1.0):
    p = " ".join(f"{x:g},{y:g}" for x, y in pts)
    return f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}"/>'


def circle(cx, cy, r, fill="#ffffff", stroke=LINE, sw=1.4):
    return (
        f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw:g}"/>'
    )


def title(w, s, sub=None):
    out = [txt(w / 2, 24, s, 15, weight="600")]
    if sub:
        out.append(txt(w / 2, 42, sub, 11.5, fill=MUTED))
    return "\n".join(out)


# the colour key used on every end-to-end architecture diagram
ARCH_LEGEND = [
    ("input", "input"),
    ("conv", "CONV"),
    ("pool", "POOL"),
    ("flat", "flatten"),
    ("fc", "FC"),
    ("out", "softmax / ReLU"),
]


def legend(x, y, items, size=10.5):
    """items: [(role, label), ...] laid out horizontally."""
    out, cx = [], x
    for role, label in items:
        out.append(rect(cx, y - 9, 11, 11, role, rx=2, sw=1.1))
        out.append(txt(cx + 16, y, label, size, anchor="start", fill=MUTED))
        cx += 16 + len(label) * size * 0.62 + 24
    return "\n".join(out)


def panel(x, y, w, h, fill="#f9fafb", stroke=LINE, label=None, label_size=12.5):
    """A soft background box, optionally with a heading above its top-left corner."""
    out = [rect(x, y, w, h, "input", rx=6, sw=1.2, fill=fill, stroke=stroke)]
    if label:
        out.append(txt(x + 8, y - 10, label, label_size, anchor="start", weight="600"))
    return "\n".join(out)


# -------------------------------------------------------------- colour scales


def _hex(r, g, b):
    return f"#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"


def _mix(c1, c2, t):
    """Linear blend between two #rrggbb colours."""
    a = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i : i + 2], 16) for i in (1, 3, 5)]
    return _hex(*[a[k] + (b[k] - a[k]) * t for k in range(3)])


def clamp01(t):
    return 0.0 if t < 0 else (1.0 if t > 1 else t)


def gray(v, lo=0.0, hi=1.0):
    """Grayscale ramp: lo maps to near-black, hi to near-white."""
    t = 0.0 if hi == lo else clamp01((v - lo) / (hi - lo))
    g = 34 + t * 216
    return _hex(g, g, g)


def diverge(v, mx, neg=BLUE, pos=RED):
    """Blue for negative, white at zero, red for positive."""
    if mx == 0:
        return "#ffffff"
    t = clamp01(abs(v) / mx)
    return _mix("#ffffff", pos if v > 0 else neg, t * 0.85)


def contrast(fill):
    """Pick black or white text so it stays readable on the given fill."""
    if not fill.startswith("#") or len(fill) != 7:
        return INK
    r, g, b = (int(fill[i : i + 2], 16) for i in (1, 3, 5))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return INK if lum > 140 else "#ffffff"


# --------------------------------------------------------------- value grids


def _num(v):
    if isinstance(v, str):
        return v
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}"


def matrix(x, y, vals, cell=26, fills=None, fmt=None, fs=11, stroke=LINE, sw=1.0,
           tfill=None):
    """Draw a 2-D grid of values.

    fills: a callable f(v, i, j) -> colour, a 2-D list of colours, or None.
    tfill: a callable, a fixed colour, or None to auto-contrast against the fill.
    """
    out = []
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            cx, cy = x + j * cell, y + i * cell
            if callable(fills):
                f = fills(v, i, j)
            elif fills is not None:
                f = fills[i][j]
            else:
                f = "#ffffff"
            out.append(
                f'<rect x="{cx:g}" y="{cy:g}" width="{cell:g}" height="{cell:g}" '
                f'fill="{f}" stroke="{stroke}" stroke-width="{sw:g}"/>'
            )
            s = fmt(v) if callable(fmt) else (fmt.format(v) if fmt else _num(v))
            if s != "":
                tc = tfill(v, i, j) if callable(tfill) else (tfill or contrast(f))
                out.append(txt(cx + cell / 2, cy + cell / 2 + fs * 0.36, s, fs, fill=tc))
    return "\n".join(out)


def mat_w(vals, cell=26):
    return len(vals[0]) * cell


def mat_h(vals, cell=26):
    return len(vals) * cell


def outline(x, y, i, j, rows, cols, cell=26, color=RED, sw=2.4, dash=None):
    """Highlight a sub-region of a matrix drawn at (x, y)."""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x + j * cell:g}" y="{y + i * cell:g}" width="{cols * cell:g}" '
        f'height="{rows * cell:g}" fill="none" stroke="{color}" '
        f'stroke-width="{sw:g}"{d}/>'
    )


def caption(x, y, rows, size=10.5, anchor="middle", fill=MUTED, lh=16):
    return lines(x, y, rows, size, anchor, fill, lh=lh)


# ------------------------------------------------------- volume pipeline plot


def pipeline(out, fname, head, stages, ops, width=920, sub=None, note=None, leg=None):
    """Horizontal 'shrinking volumes' diagram.

    stages: dicts with either kind='vol' (s, c) or kind='vec' (u), plus
            name, shape, role.
    ops:    len(stages)-1 strings, '\\n' splits lines.
    """
    HMIN, HMAX, WMIN, WMAX = 17.0, 72.0, 11.0, 46.0
    smax = max((st["s"] for st in stages if st["kind"] == "vol"), default=1)
    cmax = max((st["c"] for st in stages if st["kind"] == "vol"), default=1)
    umax = max((st["u"] for st in stages if st["kind"] == "vec"), default=1)

    dims = []
    for st in stages:
        if st["kind"] == "vol":
            h = HMIN + (st["s"] / smax) ** 0.85 * (HMAX - HMIN)
            w = WMIN + (st["c"] / cmax) ** 0.6 * (WMAX - WMIN)
        else:
            h = HMIN + (st["u"] / umax) ** 0.5 * (HMAX - HMIN - 6)
            w = 13.0
        dims.append((w, h))

    margin = 30.0
    gap = (width - 2 * margin - sum(w for w, _ in dims)) / (len(stages) - 1)
    cy = 128.0

    body = [title(width, head, sub)]
    xs, x = [], margin
    for w, _ in dims:
        xs.append(x)
        x += w + gap

    for i, (st, (w, h)) in enumerate(zip(stages, dims)):
        x0 = xs[i]
        body.append(rect(x0, cy - h / 2, w, h, st.get("role", "conv")))
        body.append(txt(x0 + w / 2, cy + HMAX / 2 + 22, st["shape"], 11, weight="600"))
        body.append(txt(x0 + w / 2, cy + HMAX / 2 + 36, st["name"], 10, fill=MUTED))

    for i, op in enumerate(ops):
        x1, x2 = xs[i] + dims[i][0], xs[i + 1]
        body.append(arrow(x1 + 4, cy, x2 - 4, cy))
        rows = op.split("\n")
        body.append(
            lines((x1 + x2) / 2, 62 - (len(rows) - 1) * 6, rows, 10, fill=MUTED, lh=12)
        )

    y = cy + HMAX / 2 + 36  # baseline of the stage-name row
    if leg:
        y += 28
        body.append(legend(margin, y, leg))
    if note:
        y += 26
        body.append(txt(width / 2, y, note, 10.5, fill=MUTED))

    write(out, fname, width, y + 16, body)
