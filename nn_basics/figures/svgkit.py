"""Shared drawing primitives for the lecture-note figures.

Every lecture-note figure set imports from here, so the documents stay
visually consistent.
"""

from __future__ import annotations

import re

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


def _marker_id(color: str) -> str:
    return "ah-" + "".join(c for c in color.lower() if c.isalnum())


def _marker_def(color: str, mid: str) -> str:
    return (
        f'  <marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">\n'
        f'    <path d="M0,0 L10,5 L0,10 z" fill="{color}"/>\n'
        f'  </marker>\n'
    )


def svg(w: float, h: float, body: str) -> str:
    colors = {LINE, INK, RED, GREEN, BLUE, AMBER, PURPLE, PINK, MUTED}
    for m in re.findall(r'stroke="(#[0-9A-Fa-f]{3,8})"', body):
        colors.add(m)
    markers = [_marker_def(LINE, "ah")]  # default grey head, used by older figures
    seen = {"ah"}
    for c in colors:
        mid = _marker_id(c)
        if mid not in seen:
            markers.append(_marker_def(c, mid))
            seen.add(mid)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" '
        f'width="{w:g}" height="{h:g}">\n'
        '<defs>\n'
        + "".join(markers)
        + '</defs>\n'
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
        f'stroke-width="{sw:g}" marker-end="url(#{_marker_id(color)})"{d}/>'
    )


def seg(x1, y1, x2, y2, color=LINE, sw=1.0, dash=None):
    """A plain line, no arrowhead."""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color}" '
        f'stroke-width="{sw:g}"{d}/>'
    )


def path(d, stroke=LINE, sw=1.4, fill="none", marker=True, dash=None):
    m = f' marker-end="url(#{_marker_id(stroke)})"' if marker else ""
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


# ------------------------------------------------------- isometric volumes


ISO_X, ISO_Y = 0.62, 0.38  # depth → right/up offset


def volume3d(x, y, fw, fh, d, role="conv", label=None, fs=11):
    """Isometric prism. (x, y) is the top-left of the front face.

    fw, fh — front-face width and height (spatial size).
    d      — isometric depth (channel count).
    label  — short layer name drawn on the front face (Input, CONV, …).
    Returns (svg, info) where info has right/top/bottom/cx/cy for layout.
    """
    fill, stroke = C.get(role, C["conv"])
    ox, oy = max(d, 6) * ISO_X, max(d, 6) * ISO_Y
    top_f = _mix(fill, "#ffffff", 0.28)
    side_f = _mix(fill, "#111827", 0.16)
    out = [
        poly(
            [(x, y), (x + ox, y - oy), (x + fw + ox, y - oy), (x + fw, y)],
            fill=top_f, stroke=stroke, sw=1.15,
        ),
        poly(
            [(x + fw, y), (x + fw + ox, y - oy),
             (x + fw + ox, y + fh - oy), (x + fw, y + fh)],
            fill=side_f, stroke=stroke, sw=1.15,
        ),
        poly(
            [(x, y), (x + fw, y), (x + fw, y + fh), (x, y + fh)],
            fill=fill, stroke=stroke, sw=1.3,
        ),
    ]
    if label:
        size = fs if fw >= 48 else max(8.5, fs - 2)
        out.append(txt(x + fw / 2, y + fh / 2 + size * 0.36, label, size, weight="600"))
    info = dict(
        x=x, y=y, fw=fw, fh=fh, ox=ox, oy=oy,
        left=x, right=x + fw + ox, top=y - oy, bottom=y + fh,
        cx=x + fw / 2, cy=y + fh / 2,
        mid_right=x + fw + ox * 0.35,
        mid_left=x,
    )
    return "\n".join(out), info


# ------------------------------------------------------- volume pipeline plot


def pipeline(out, fname, head, stages, ops, width=920, sub=None, note=None, leg=None):
    """Horizontal shrinking-volume diagram, drawn as isometric cubes.

    stages: dicts with either kind='vol' (s, c) or kind='vec' (u), plus
            name (inside the cube), shape (under the cube), role.
    ops:    len(stages)-1 strings drawn just above each arrow; '\\n' splits lines.
    """
    FACE_MIN, FACE_MAX = 36.0, 86.0
    DEPTH_MIN, DEPTH_MAX = 10.0, 40.0
    VEC_W = 46.0

    smax = max((st["s"] for st in stages if st["kind"] == "vol"), default=1)
    cmax = max((st["c"] for st in stages if st["kind"] == "vol"), default=1)
    umax = max((st["u"] for st in stages if st["kind"] == "vec"), default=1)

    dims = []
    for st in stages:
        if st["kind"] == "vol":
            face = FACE_MIN + (st["s"] / smax) ** 0.7 * (FACE_MAX - FACE_MIN)
            depth = DEPTH_MIN + (st["c"] / cmax) ** 0.45 * (DEPTH_MAX - DEPTH_MIN)
            dims.append((face, face, depth))
        else:
            h = FACE_MIN + (st["u"] / umax) ** 0.45 * (FACE_MAX - FACE_MIN - 8)
            dims.append((VEC_W, h, 12.0))

    occ = [fw + d * ISO_X for fw, _, d in dims]
    margin = 28.0
    min_gap = 26.0
    n = len(stages)
    need = 2 * margin + sum(occ) + min_gap * max(n - 1, 0)
    if need > width:
        scale = (width - 2 * margin - min_gap * max(n - 1, 0)) / max(sum(occ), 1)
        dims = [(fw * scale, fh * scale, d * scale) for fw, fh, d in dims]
        occ = [fw + d * ISO_X for fw, _, d in dims]
    gap = (width - 2 * margin - sum(occ)) / max(n - 1, 1)
    gap = max(gap, min_gap)
    cy = 158.0

    body = [title(width, head, sub)]
    drawn = []
    x = margin
    for st, (fw, fh, d) in zip(stages, dims):
        y0 = cy - fh / 2
        svg, info = volume3d(
            x, y0, fw, fh, d, st.get("role", "conv"), label=st.get("name", ""),
        )
        body.append(svg)
        drawn.append(info)
        body.append(txt(
            info["cx"], info["bottom"] + 16, st["shape"],
            10 if len(st["shape"]) > 12 else 11, weight="600",
        ))
        x += (fw + d * ISO_X) + gap

    for i, op in enumerate(ops):
        x1 = drawn[i]["right"]
        x2 = drawn[i + 1]["left"]
        midy = cy
        body.append(arrow(x1 + 2, midy, x2 - 4, midy))
        rows = [r for r in op.split("\n") if r]
        lh = 11
        # sit in the gap, a few pixels above the arrow (not above the cubes)
        y_last = midy - 8
        y0 = y_last - (len(rows) - 1) * lh
        body.append(lines((x1 + x2) / 2, y0, rows, 9.5, fill=MUTED, lh=lh))

    y = cy + FACE_MAX / 2 + 28
    if leg:
        y += 22
        body.append(legend(margin, y, leg))
    if note:
        y += 24
        body.append(txt(width / 2, y, note, 10.5, fill=MUTED))

    write(out, fname, width, y + 16, body)
