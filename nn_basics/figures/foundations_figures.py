"""Generate the diagrams used by convolutional_nn_foundations.md.

Driven by make_figures.py; can also be run directly.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from svgkit import (  # noqa: E402
    AMBER, ARCH_LEGEND, BLUE, C, GREEN, INK, LINE, MUTED, PURPLE, RED,
    arrow, caption, circle, contrast, diverge, gray, legend, lines, mat_h, mat_w,
    matrix, outline, panel, path, pipeline as _pipeline, poly, rect, seg, title,
    txt, volume3d, write, _mix,
)

OUT = pathlib.Path(__file__).parent


# ------------------------------------------------------------- shared numbers

IMG = [
    [3, 0, 1, 2, 7, 4],
    [1, 5, 8, 9, 3, 1],
    [2, 7, 2, 5, 1, 3],
    [0, 1, 3, 1, 7, 8],
    [4, 2, 1, 6, 2, 8],
    [2, 4, 5, 2, 3, 9],
]
VFILT = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
HFILT = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]

BRIGHT_LEFT = [[10, 10, 10, 0, 0, 0] for _ in range(6)]
BRIGHT_RIGHT = [[0, 0, 0, 10, 10, 10] for _ in range(6)]
CHECKER = [[10, 10, 10, 0, 0, 0] for _ in range(3)] + [
    [0, 0, 0, 10, 10, 10] for _ in range(3)
]
POOL_IN = [[1, 3, 2, 1], [2, 9, 1, 1], [1, 3, 2, 3], [5, 6, 1, 2]]


def pad_with_zeros(img, p):
    n = len(img[0])
    row = [[0] * (n + 2 * p) for _ in range(p)]
    mid = [[0] * p + list(r) + [0] * p for r in img]
    return row + mid + [[0] * (n + 2 * p) for _ in range(p)]


def conv2d(img, filt, stride=1, pad=0):
    """Cross-correlation, i.e. what deep learning calls convolution."""
    if pad:
        img = pad_with_zeros(img, pad)
    n, f = len(img), len(filt)
    size = (n - f) // stride + 1
    out = []
    for i in range(size):
        row = []
        for j in range(size):
            acc = 0
            for a in range(f):
                for b in range(f):
                    acc += img[i * stride + a][j * stride + b] * filt[a][b]
            row.append(acc)
        out.append(row)
    return out


# Every matrix printed in the notes is recomputed here, so a wrong number in
# either place shows up as a failed assertion rather than a wrong picture.
CONV_OUT = conv2d(IMG, VFILT)
assert CONV_OUT == [[-5, -4, 0, 8], [-10, -2, 2, 3], [0, -2, -4, -7], [-3, -2, -3, -16]]
assert conv2d(BRIGHT_LEFT, VFILT) == [[0, 30, 30, 0]] * 4
assert conv2d(BRIGHT_RIGHT, VFILT) == [[0, -30, -30, 0]] * 4
assert conv2d(CHECKER, HFILT) == [
    [0, 0, 0, 0],
    [30, 10, -10, -30],
    [30, 10, -10, -30],
    [0, 0, 0, 0],
]


def seq_blue(v, lo, hi):
    t = 0.0 if hi == lo else (v - lo) / (hi - lo)
    return _mix("#eff6ff", "#1d4ed8", max(0.0, min(1.0, t)))


def op(x, y, s, size=20):
    """A big operator glyph such as * or = between two grids."""
    return txt(x, y, s, size, fill=MUTED)


# ------------------------------------------------- 2.1 the convolution itself


def fig_conv_operation():
    W, H = 920, 470
    b = [title(W, "The convolution operation",
               "slide the filter over the input, multiply element-wise, and sum into one number")]

    cell = 28
    iw, fw, ow = mat_w(IMG, cell), mat_w(VFILT, cell), mat_w(CONV_OUT, cell)
    total = iw + 44 + fw + 44 + ow
    x_img = (W - total) / 2
    x_f = x_img + iw + 44
    x_o = x_f + fw + 44
    y_img, y_f, y_o = 78, 78 + 42, 78 + 28

    b.append(matrix(x_img, y_img, IMG, cell))
    b.append(op(x_img + iw + 22, y_img + 84 + 6, "∗"))
    b.append(matrix(x_f, y_f, VFILT, cell, fills=lambda v, i, j: diverge(v, 1)))
    b.append(op(x_f + fw + 22, y_img + 84 + 6, "="))
    b.append(matrix(x_o, y_o, CONV_OUT, cell, fills=lambda v, i, j: diverge(v, 16)))

    for cx, lab in [(x_img + iw / 2, "6×6 input"), (x_f + fw / 2, "3×3 filter"),
                    (x_o + ow / 2, "4×4 output")]:
        b.append(txt(cx, 268, lab, 11.5, weight="600"))
    b.append(txt(x_o + ow / 2, 284, "one value per filter position", 10, fill=MUTED))

    # tie the highlighted window to the output element it produces, routing the
    # connector above the grids so it does not cross any numbers
    b.append(outline(x_img, y_img, 0, 0, 3, 3, cell, RED))
    b.append(outline(x_o, y_o, 0, 0, 1, 1, cell, RED))
    b.append(path(f"M {x_img + 3 * cell},{y_img + 6} "
                  f"C {x_img + 3 * cell},{y_img - 26} {x_o + cell / 2},{y_o - 46} "
                  f"{x_o + cell / 2},{y_o - 4}", stroke=RED, sw=1.3, dash="4 3"))

    # the same window, expanded
    b.append(txt(70, 322, "The top-left window, step by step", 12.5, anchor="start", weight="600"))
    c2 = 26
    patch = [r[0:3] for r in IMG[0:3]]
    prod = [[patch[i][j] * VFILT[i][j] for j in range(3)] for i in range(3)]
    assert sum(sum(r) for r in prod) == CONV_OUT[0][0]

    xp, yp = 90, 344
    pw = mat_w(patch, c2)
    b.append(matrix(xp, yp, patch, c2))
    b.append(op(xp + pw + 17, yp + 45, "⊙", 17))
    xf2 = xp + pw + 34
    b.append(matrix(xf2, yp, VFILT, c2, fills=lambda v, i, j: diverge(v, 1)))
    b.append(op(xf2 + pw + 17, yp + 45, "=", 17))
    xr = xf2 + pw + 34
    b.append(matrix(xr, yp, prod, c2, fills=lambda v, i, j: diverge(v, 8)))
    for cx, lab in [(xp + pw / 2, "window"), (xf2 + pw / 2, "filter"), (xr + pw / 2, "products")]:
        b.append(txt(cx, yp + 96, lab, 10, fill=MUTED))

    tx = xr + pw + 46
    b.append(txt(tx, yp + 16, "left column:    3 + 1 + 2 = +6", 11.5, anchor="start"))
    b.append(txt(tx, yp + 36, "middle column: weights are 0, so it adds 0", 11.5, anchor="start"))
    b.append(txt(tx, yp + 56, "right column:  −(1 + 8 + 2) = −11", 11.5, anchor="start"))
    b.append(seg(tx, yp + 66, tx + 250, yp + 66, INK, 1.0))
    b.append(txt(tx, yp + 86, "sum = 6 + 0 − 11 = −5", 12.5, anchor="start", weight="600", fill=RED))

    b.append(txt(W / 2, H - 12,
                 "A 3×3 filter has exactly 4×4 positions that fit inside a 6×6 input, which is why the output is 4×4:  n − f + 1 = 6 − 3 + 1 = 4.",
                 10.5, fill=MUTED))
    write(OUT, "conv-operation.svg", W, H, b)


# ------------------------------------------------------- 2.2 vertical edges


def fig_edge_vertical():
    W, H = 920, 476
    out = conv2d(BRIGHT_LEFT, VFILT)
    b = [title(W, "Why the 1, 0, −1 filter detects a vertical edge",
               "cells are shaded by value, so each grid can be read as a picture")]

    cell = 28
    iw, fw, ow = 6 * cell, 3 * cell, 4 * cell
    total = iw + 44 + fw + 44 + ow
    x_img = (W - total) / 2
    x_f = x_img + iw + 44
    x_o = x_f + fw + 44
    y_img, y_f, y_o = 78, 120, 106

    b.append(matrix(x_img, y_img, BRIGHT_LEFT, cell, fills=lambda v, i, j: gray(v, 0, 10)))
    b.append(op(x_img + iw + 22, y_img + 90, "∗"))
    b.append(matrix(x_f, y_f, VFILT, cell, fills=lambda v, i, j: diverge(v, 1)))
    b.append(op(x_f + fw + 22, y_img + 90, "="))
    b.append(matrix(x_o, y_o, out, cell, fills=lambda v, i, j: gray(v, 0, 30)))

    b.append(txt(x_img + iw / 2, 268, "bright left half, dark right half", 11.5, weight="600"))
    b.append(txt(x_f + fw / 2, 268, "vertical edge filter", 11.5, weight="600"))
    b.append(txt(x_o + ow / 2, 268, "bright band on the edge", 11.5, weight="600"))

    # the two window cases, marked on the input in matching colours
    b.append(outline(x_img, y_img, 0, 0, 3, 3, cell, GREEN))
    b.append(outline(x_img, y_img, 3, 1, 3, 3, cell, RED))

    b.append(txt(60, 318, "Case A — window entirely inside the bright region", 12,
                 anchor="start", weight="600", fill=GREEN))
    b.append(txt(500, 318, "Case B — window straddling the edge", 12,
                 anchor="start", weight="600", fill=RED))

    c2 = 26
    a_patch = [r[0:3] for r in BRIGHT_LEFT[0:3]]
    b_patch = [r[1:4] for r in BRIGHT_LEFT[3:6]]
    for x0, pch, col, sums in [
        (72, a_patch, GREEN, ["left column  ×(+1):  +30", "right column ×(−1):  −30", "output = 0"]),
        (512, b_patch, RED, ["left column  ×(+1):  +30", "right column ×(−1):   0", "output = 30"]),
    ]:
        b.append(matrix(x0, 336, pch, c2, fills=lambda v, i, j: gray(v, 0, 10)))
        b.append(outline(x0, 336, 0, 0, 3, 3, c2, col, sw=2.0))
        b.append(arrow(x0 + 84, 375, x0 + 116, 375))
        for k, s in enumerate(sums):
            weight = "600" if k == 2 else "normal"
            fill = col if k == 2 else INK
            b.append(txt(x0 + 128, 355 + k * 20, s, 11.5, anchor="start", weight=weight, fill=fill))

    b.append(caption(W / 2, H - 32, [
        "Inside a uniform region the +1 and −1 columns cancel, so the output is 0. Only where brightness changes across the filter does a large value survive.",
        "The band looks thick only because the image is 6×6; on a 1000×1000 image it is a thin line.",
    ]))
    write(OUT, "conv-edge-vertical.svg", W, H, b)


# ------------------------------------------------------- 2.3 sign of the edge


def fig_edge_signs():
    W, H = 920, 384
    b = [title(W, "The sign of the output encodes the direction of the transition",
               "the same filter, with the image flipped left to right")]

    cell = 22
    iw, fw, ow = 6 * cell, 3 * cell, 4 * cell
    for x0, img, head, note, col in [
        (44, BRIGHT_LEFT, "Light on the left, dark on the right",
         "positive output: light → dark", RED),
        (516, BRIGHT_RIGHT, "Dark on the left, light on the right",
         "negative output: dark → light", BLUE),
    ]:
        out = conv2d(img, VFILT)
        xf = x0 + iw + 30
        xo = xf + fw + 30
        b.append(txt(x0, 76, head, 12, anchor="start", weight="600"))
        b.append(matrix(x0, 92, img, cell, fills=lambda v, i, j: gray(v, 0, 10)))
        b.append(op(x0 + iw + 15, 92 + 72, "∗", 17))
        b.append(matrix(xf, 92 + 33, VFILT, cell, fills=lambda v, i, j: diverge(v, 1)))
        b.append(op(xf + fw + 15, 92 + 72, "=", 17))
        b.append(matrix(xo, 92 + 22, out, cell, fills=lambda v, i, j: diverge(v, 30)))
        b.append(txt(x0 + iw / 2, 246, "input", 10, fill=MUTED))
        b.append(txt(xf + fw / 2, 246, "filter", 10, fill=MUTED))
        b.append(txt(xo + ow / 2, 246, "output", 10, fill=MUTED))
        b.append(txt(x0 + (iw + 30 + fw + 30 + ow) / 2, 276, note, 12, weight="600", fill=col))

    b.append(seg(460, 70, 460, 300, LINE, 1.0, dash="4 4"))

    # two separate text runs: SVG collapses the whitespace in a single string
    b.append(rect(228, 298, 464, 44, "input", fill="#f9fafb", stroke=LINE))
    b.append(txt(342, 318, "positive = bright on the left", 11, fill=RED, weight="600"))
    b.append(seg(460, 306, 460, 328, LINE, 1.0))
    b.append(txt(578, 318, "negative = bright on the right", 11, fill=BLUE, weight="600"))
    b.append(txt(460, 336, "take absolute values if only the presence of an edge matters", 10.5, fill=MUTED))
    b.append(txt(W / 2, H - 8, "Only the two middle columns respond, because those are the filter positions that span the boundary.", 10.5, fill=MUTED))
    write(OUT, "conv-edge-signs.svg", W, H, b)


# ---------------------------------------------------- 2.4 horizontal edges


def fig_edge_horizontal():
    W, H = 920, 400
    out = conv2d(CHECKER, HFILT)
    b = [title(W, "The horizontal edge filter",
               "rotate the filter 90° and it responds to a bright-above, dark-below region instead")]

    cell = 28
    iw, fw, ow = 6 * cell, 3 * cell, 4 * cell
    total = iw + 44 + fw + 44 + ow
    x_img = (W - total) / 2
    x_f = x_img + iw + 44
    x_o = x_f + fw + 44
    y_img = 80

    b.append(matrix(x_img, y_img, CHECKER, cell, fills=lambda v, i, j: gray(v, 0, 10)))
    b.append(op(x_img + iw + 22, y_img + 90, "∗"))
    b.append(matrix(x_f, y_img + 42, HFILT, cell, fills=lambda v, i, j: diverge(v, 1)))
    b.append(op(x_f + fw + 22, y_img + 90, "="))
    b.append(matrix(x_o, y_img + 28, out, cell, fills=lambda v, i, j: diverge(v, 30)))

    b.append(txt(x_img + iw / 2, 272, "bright upper-left and lower-right blocks", 11.5, weight="600"))
    b.append(txt(x_f + fw / 2, 272, "horizontal filter", 11.5, weight="600"))
    b.append(txt(x_o + ow / 2, 272, "signed horizontal edges", 11.5, weight="600"))

    # the output cells are already fill-coded by sign, so the legend below reads
    # them by value rather than by an outline that would vanish into the fill
    rows = [
        (RED, "±30", "the window sits fully on one edge: bright above and dark below, or the reverse"),
        (AMBER, "±10", "the window spans a positive and a negative edge at once, so the two contributions partly cancel"),
        (BLUE, "0", "the window is inside a uniform block, so the +1 and −1 rows cancel exactly (the top and bottom output rows)"),
    ]
    for k, (col, lab, expl) in enumerate(rows):
        y = 316 + k * 22
        b.append(rect(64, y - 11, 40, 17, "input", rx=3, sw=1.4, fill="#ffffff", stroke=col))
        b.append(txt(84, y + 1, lab, 10.5, weight="600", fill=col))
        b.append(txt(116, y + 1, expl, 11, anchor="start"))

    b.append(txt(W / 2, H - 8, "The ±10 values are an artifact of a 6×6 image; on a large image these transition cells are negligible.", 10.5, fill=MUTED))
    write(OUT, "conv-edge-horizontal.svg", W, H, b)


# ------------------------------------------- 2.7 cross-correlation vs convolution


def fig_cross_correlation():
    W, H = 820, 330
    K = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    lr = [list(reversed(r)) for r in K]
    both = list(reversed(lr))
    b = [title(W, "Cross-correlation versus true convolution",
               "the textbook definition flips the filter on both axes first")]

    cell = 30
    gw = 3 * cell
    total = gw * 3 + 2 * 84
    x0 = (W - total) / 2
    ys = 100
    xs = [x0, x0 + gw + 84, x0 + 2 * (gw + 84)]

    for x, m in zip(xs, [K, lr, both]):
        b.append(matrix(x, ys, m, cell, fills=lambda v, i, j: seq_blue(v, 1, 9)))
    for k in range(2):
        xa = xs[k] + gw + 12
        b.append(arrow(xa, ys + 45, xa + 60, ys + 45))
        b.append(txt(xa + 30, ys + 32, ["flip left–right", "flip top–bottom"][k], 10, fill=MUTED))

    b.append(txt(xs[0] + gw / 2, ys + 112, "the filter as stored", 11, weight="600"))
    b.append(txt(xs[2] + gw / 2, ys + 112, "flipped on both axes", 11, weight="600"))

    b.append(rect(70, 234, 340, 56, "input", fill="#f0fdf4", stroke=GREEN))
    b.append(txt(240, 254, "Deep learning: no flip", 12, weight="600", fill=GREEN))
    b.append(txt(240, 274, "uses the filter as stored — strictly, cross-correlation", 10.5))

    b.append(rect(430, 234, 320, 56, "input", fill="#eff6ff", stroke=BLUE))
    b.append(txt(590, 254, "Signal processing: flip first", 12, weight="600", fill=BLUE))
    b.append(txt(590, 274, "the flip is what makes convolution associative", 10.5))

    b.append(txt(W / 2, H - 10, "For learned weights the distinction does not matter: backprop simply learns the flipped filter instead.", 10.5, fill=MUTED))
    write(OUT, "conv-cross-correlation.svg", W, H, b)


# --------------------------------------------------------------- 1.2 dense cost


def fig_fc_explosion():
    W, H = 880, 356
    b = [title(W, "Why a fully connected first layer does not scale to images",
               "the input dimension grows with the square of the image side")]

    rows = [
        ("64 × 64 × 3", 12_288, 44, "12,288", "1,000 × 12,288", "12.3 million", MUTED),
        ("1000 × 1000 × 3", 3_000_000, 92, "3,000,000", "1,000 × 3,000,000", "3 billion", RED),
    ]
    for k, (shape, _nx, side, nx_s, wshape, params, col) in enumerate(rows):
        cy = 108 + k * 108
        b.append(rect(60 - side / 2 + 40, cy - side / 2, side, side, "input", rx=3, sw=1.3))
        b.append(txt(100, cy + side / 2 + 18, shape, 10.5, fill=MUTED))
        b.append(arrow(150, cy, 196, cy))
        b.append(txt(173, cy - 12, "flatten", 9.5, fill=MUTED))
        b.append(rect(200, cy - 18, 116, 36, "flat", rx=3))
        b.append(txt(258, cy + 5, nx_s, 12, weight="600"))
        b.append(txt(258, cy + 34, "input features", 9.5, fill=MUTED))
        b.append(arrow(322, cy, 368, cy))
        b.append(txt(345, cy - 12, "dense", 9.5, fill=MUTED))
        b.append(rect(372, cy - 18, 168, 36, "fc", rx=3))
        b.append(txt(456, cy + 5, wshape, 11.5, weight="600"))
        b.append(txt(456, cy + 34, "shape of W", 9.5, fill=MUTED))
        b.append(arrow(546, cy, 592, cy))
        b.append(rect(596, cy - 20, 168, 40, "input", rx=4, sw=1.5, fill="#ffffff", stroke=col))
        b.append(txt(680, cy + 6, params, 14, weight="600", fill=col))
        b.append(txt(680, cy + 34, "parameters in one layer", 9.5, fill=MUTED))

    b.append(txt(W / 2, H - 46, "Three billion parameters in a single layer is infeasible to regularize, to store, and to compute with —", 10.5, fill=MUTED))
    b.append(txt(W / 2, H - 30, "and that is for a one-megapixel image with only 1,000 hidden units.", 10.5, fill=MUTED))
    b.append(txt(W / 2, H - 8, "A convolutional layer replaces W entirely with a small filter reused at every position.", 11, weight="600"))
    write(OUT, "fc-parameter-explosion.svg", W, H, b)


# -------------------------------------------------------------------- padding


def fig_padding():
    W, H = 920, 496
    padded = pad_with_zeros(IMG, 1)
    out_same = conv2d(IMG, VFILT, pad=1)
    assert len(out_same) == 6

    b = [title(W, "Padding", "a border of zeros keeps the output the same size as the input")]

    cell = 20
    xg = 62.0

    def row(y, img, out, head, tags, ring=0):
        """ring: width of the zero border to tint, 0 for an unpadded input."""
        o = [txt(xg, y - 12, head, 12.5, anchor="start", weight="600")]
        iw, fw, ow = mat_w(img, cell), 3 * cell, mat_w(out, cell)
        xf = xg + iw + 30
        xo = xf + fw + 30
        ch = mat_h(img, cell)
        n, m = len(img), len(img[0])

        def cellfill(v, i, j):
            on_ring = i < ring or j < ring or i >= n - ring or j >= m - ring
            return C["pad"][0] if (ring and on_ring) else "#ffffff"

        o.append(matrix(xg, y, img, cell, fills=cellfill))
        o.append(op(xg + iw + 15, y + ch / 2 + 6, "∗", 17))
        o.append(matrix(xf, y + ch / 2 - fw / 2, VFILT, cell,
                        fills=lambda v, i, j: diverge(v, 1)))
        o.append(op(xf + fw + 15, y + ch / 2 + 6, "=", 17))
        o.append(matrix(xo, y + ch / 2 - mat_h(out, cell) / 2, out, cell,
                        fills=lambda v, i, j: diverge(v, 16)))
        for cx, t in zip([xg + iw / 2, xf + fw / 2, xo + ow / 2], tags):
            o.append(txt(cx, y + ch + 20, t, 11, weight="600"))
        return "\n".join(o)

    b.append(row(88, IMG, CONV_OUT, "Valid convolution, p = 0",
                 ["6×6 input", "3×3", "4×4 — two smaller"]))
    b.append(row(258, padded, out_same, "Same convolution, p = 1",
                 ["8×8 padded", "3×3", "6×6 — size preserved"], ring=1))
    b.append(rect(xg - 6, 252, 8 * cell + 12, 8 * cell + 8, "input", rx=3, sw=1.3,
                  fill="none", stroke=C["pad"][1], dash="4 3"))
    b.append(txt(xg + 8 * cell + 26, 268, "the yellow ring is the", 10, anchor="start", fill=MUTED))
    b.append(txt(xg + 8 * cell + 26, 282, "added zero border", 10, anchor="start", fill=MUTED))

    px = 540
    b.append(panel(px, 76, 340, 200, label="Output size"))
    b.append(txt(px + 20, 108, "valid:", 11.5, anchor="start", fill=MUTED))
    b.append(txt(px + 90, 108, "n − f + 1", 12.5, anchor="start", weight="600"))
    b.append(txt(px + 20, 138, "padded:", 11.5, anchor="start", fill=MUTED))
    b.append(txt(px + 90, 138, "n + 2p − f + 1", 12.5, anchor="start", weight="600"))
    b.append(txt(px + 20, 168, "same:", 11.5, anchor="start", fill=MUTED))
    b.append(txt(px + 90, 168, "p = (f − 1) / 2", 12.5, anchor="start", weight="600"))
    b.append(seg(px + 20, 186, px + 320, 186, LINE, 1.0))
    for k, (f, p) in enumerate([(3, 1), (5, 2), (7, 3)]):
        b.append(txt(px + 40 + k * 100, 210, f"f = {f}", 11, anchor="start", fill=MUTED))
        b.append(txt(px + 40 + k * 100, 230, f"p = {p}", 12, anchor="start", weight="600"))
    b.append(txt(px + 170, 260, "odd f keeps the padding symmetric", 10.5, fill=MUTED))

    b.append(panel(px, 320, 340, 96, fill="#fffbeb", stroke=C["pad"][1],
                   label="Two things padding fixes"))
    b.append(txt(px + 20, 350, "1.  the output no longer shrinks at every layer", 11, anchor="start"))
    b.append(txt(px + 20, 374, "2.  border pixels stop being under-used", 11, anchor="start"))
    b.append(txt(px + 20, 400, "\"same\" only preserves size when s = 1", 10.5, anchor="start", fill=MUTED))

    b.append(txt(W / 2, H - 10, "Without padding, a 100-layer stack of 3×3 valid convolutions would shrink the representation by 200 pixels and collapse it.", 10.5, fill=MUTED))
    write(OUT, "padding.svg", W, H, b)


def fig_padding_coverage():
    W, H = 780, 412
    b = [title(W, "How many filter windows contain each pixel",
               "a 6×6 input with a 3×3 filter — the corners are the problem padding solves")]

    def coverage(n, f, p):
        size = n + 2 * p - f + 1
        counts = []
        for i in range(n):
            ip = i + p
            lo, hi = max(0, ip - f + 1), min(size - 1, ip)
            counts.append(hi - lo + 1)
        return [[counts[i] * counts[j] for j in range(n)] for i in range(n)]

    no_pad = coverage(6, 3, 0)
    with_pad = coverage(6, 3, 1)
    assert no_pad[0][0] == 1 and no_pad[2][2] == 9
    assert with_pad[0][0] == 4

    cell = 34
    gw = 6 * cell
    for x0, m, head, corner, col in [
        (70, no_pad, "No padding  (p = 0)", "corner pixel: 1 window", RED),
        (430, with_pad, "One-pixel border  (p = 1)", "corner pixel: 4 windows", GREEN),
    ]:
        b.append(txt(x0 + gw / 2, 80, head, 12.5, weight="600"))
        b.append(matrix(x0, 94, m, cell, fills=lambda v, i, j: seq_blue(v, 1, 9), fs=12))
        b.append(outline(x0, 94, 0, 0, 1, 1, cell, col, sw=2.6))
        b.append(txt(x0 + gw / 2, 94 + gw + 26, corner, 12, weight="600", fill=col))
        b.append(txt(x0 + gw / 2, 94 + gw + 44, "centre pixel: 9 windows", 10.5, fill=MUTED))

    b.append(caption(W / 2, H - 42, [
        "Each cell counts the filter positions that see that pixel. Without padding a corner is read once while a central pixel is read nine times,",
        "so information near the border is effectively discarded. A one-pixel border raises the corner from 1 to 4.",
    ]))
    write(OUT, "padding-coverage.svg", W, H, b)


# ------------------------------------------------------------------- striding


def fig_strided():
    W, H = 920, 442
    out = [[91, 100, 83], [69, 91, 127], [44, 72, 74]]
    b = [title(W, "Strided convolution",
               "stride 2 moves the window two positions at a time, so only every other placement is used")]

    cell = 34
    gx, gy = 66, 92
    blank = [["" for _ in range(7)] for _ in range(7)]
    b.append(matrix(gx, gy, blank, cell, fills=lambda v, i, j: "#f9fafb"))

    cols = [RED, AMBER, GREEN]
    for k, c0 in enumerate([0, 2, 4]):
        b.append(outline(gx, gy, 0, c0, 3, 3, cell, cols[k], sw=2.6))
    # every stride-2 window centre, i.e. one dot per output element
    for r0 in (0, 2, 4):
        for c0 in (0, 2, 4):
            b.append(circle(gx + (c0 + 1.5) * cell, gy + (r0 + 1.5) * cell, 3.2,
                            fill=LINE, stroke="none", sw=0))
    b.append(txt(gx + 3.5 * cell, gy - 14, "7×7 input", 11.5, weight="600"))
    for k, c0 in enumerate([0, 2, 4]):
        b.append(txt(gx + (c0 + 1.5) * cell, gy + 7 * cell + 20,
                     ["1st window", "2nd", "3rd"][k], 10, weight="600", fill=cols[k]))

    xa = gx + 7 * cell + 26
    b.append(arrow(xa, gy + 3.5 * cell, xa + 74, gy + 3.5 * cell))
    b.append(lines(xa + 37, gy + 3.5 * cell - 26, ["3×3 filter", "s = 2"], 10.5, fill=MUTED))

    xo = xa + 96
    b.append(matrix(xo, gy + 2 * cell, out, cell, fills=lambda v, i, j: seq_blue(v, 40, 130), fs=12))
    for k in range(3):
        b.append(outline(xo, gy + 2 * cell, 0, k, 1, 1, cell, cols[k], sw=2.6))
    b.append(txt(xo + 1.5 * cell, gy + 2 * cell - 14, "3×3 output", 11.5, weight="600"))
    b.append(txt(xo + 1.5 * cell, gy + 5 * cell + 22, "one value per window position", 10, fill=MUTED))

    px = 640
    b.append(panel(px, 92, 250, 128, label="Output size"))
    b.append(txt(px + 125, 126, "⌊ (n + 2p − f) / s ⌋ + 1", 13, weight="600"))
    b.append(seg(px + 20, 142, px + 230, 142, LINE, 1.0))
    b.append(txt(px + 125, 166, "⌊ (7 + 0 − 3) / 2 ⌋ + 1 = 3", 11.5))
    b.append(txt(px + 125, 194, "with s = 1 the same filter", 10.5, fill=MUTED))
    b.append(txt(px + 125, 209, "would give a 5×5 output", 10.5, fill=MUTED))

    b.append(caption(W / 2, H - 46, [
        "The floor matters: a window is only computed when it fits entirely inside the input plus padding, and a window hanging off the edge is skipped.",
        "Stride 2 roughly halves the height and width, which is one of the two standard ways to downsample — pooling is the other.",
    ]))
    write(OUT, "strided-conv.svg", W, H, b)


# ---------------------------------------------------------- volumes & filters


def _slices(x, y, n, w, h, role, dx=8, dy=-8, sw=1.3):
    """A stack of offset squares, drawn back to front."""
    return "\n".join(
        rect(x + i * dx, y + i * dy, w, h, role, rx=2, sw=sw)
        for i in reversed(range(n))
    )


def fig_conv_volumes():
    W, H = 920, 438
    b = [title(W, "Convolution over a volume",
               "the filter must match the input's channel count, and the output's channel count is the number of filters")]

    # ---- one filter
    b.append(panel(30, 72, 860, 152, label="One 3×3×3 filter"))
    b.append(_slices(74, 118, 3, 62, 62, "input"))
    b.append(txt(105, 210, "6×6×3", 11.5, weight="600"))
    b.append(op(196, 156, "∗"))
    b.append(_slices(226, 132, 3, 34, 34, "conv", dx=6, dy=-6))
    b.append(txt(243, 210, "3×3×3", 11.5, weight="600"))
    b.append(op(310, 156, "="))
    b.append(rect(338, 122, 46, 46, "conv", rx=2, sw=1.3))
    b.append(txt(361, 210, "4×4×1", 11.5, weight="600"))
    b.append(txt(440, 128, "The filter has 3 · 3 · 3 = 27 numbers.", 11, anchor="start"))
    b.append(txt(440, 150, "At each position all 27 products are added into a single value,", 11, anchor="start"))
    b.append(txt(440, 172, "so the channel axis collapses: the output is 4×4×1, not 4×4×3.", 11, anchor="start"))
    b.append(txt(440, 200, "One filter detects one feature.", 11, anchor="start", weight="600"))

    # ---- two filters
    b.append(panel(30, 258, 860, 152, label="Two 3×3×3 filters"))
    b.append(_slices(74, 304, 3, 62, 62, "input"))
    b.append(txt(105, 396, "6×6×3", 11.5, weight="600"))
    b.append(op(196, 342, "∗"))
    b.append(_slices(222, 318, 3, 34, 34, "conv", dx=6, dy=-6))
    b.append(_slices(268, 318, 3, 34, 34, "conv5", dx=6, dy=-6))
    b.append(txt(258, 396, "2 filters of 3×3×3", 11.5, weight="600"))
    b.append(op(346, 342, "="))
    b.append(_slices(374, 314, 2, 46, 46, "conv", dx=9, dy=-9))
    b.append(txt(400, 396, "4×4×2", 11.5, weight="600"))
    b.append(txt(470, 314, "Each filter produces its own 4×4 map; stacking the two maps", 11, anchor="start"))
    b.append(txt(470, 336, "along the channel axis gives 4×4×2.", 11, anchor="start"))
    b.append(txt(470, 364, "The output's last dimension is the number of filters —", 11, anchor="start", weight="600"))
    b.append(txt(470, 384, "not the number of input channels. 128 filters give 4×4×128.", 11, anchor="start", weight="600"))

    b.append(txt(W / 2, H - 10, "n × n × nC  ∗  {nC′ filters of f × f × nC}  =  (n − f + 1) × (n − f + 1) × nC′", 12, weight="600"))
    write(OUT, "conv-volumes.svg", W, H, b)


# ------------------------------------------------------ one convolutional layer


def fig_conv_layer():
    W, H = 920, 406
    b = [title(W, "One layer of a convolutional network",
               "convolve, add a bias, apply a non-linearity, then stack the maps")]

    b.append(_slices(52, 168, 3, 58, 58, "input"))
    b.append(txt(81, 254, "6×6×3", 11.5, weight="600"))
    b.append(txt(81, 270, "a[l−1]", 10, fill=MUTED))

    for k, (cy, role, fname, bname) in enumerate([
        (128, "conv", "filter 1", "b₁"),
        (250, "conv5", "filter 2", "b₂"),
    ]):
        b.append(arrow(132, 190, 184, cy))
        b.append(rect(190, cy - 20, 72, 40, role, rx=3))
        b.append(txt(226, cy + 4, fname, 10.5))
        b.append(txt(226, cy - 28, "3×3×3", 9.5, fill=MUTED))
        b.append(arrow(266, cy, 300, cy))
        b.append(rect(304, cy - 20, 44, 40, "input", rx=3))
        b.append(txt(326, cy + 4, "4×4", 10.5))
        b.append(txt(326, cy - 28, f"z{k + 1}", 9.5, fill=MUTED))
        b.append(arrow(352, cy, 386, cy))
        b.append(rect(390, cy - 20, 56, 40, "pad", rx=3))
        b.append(txt(418, cy + 4, f"+ {bname}", 10.5))
        b.append(arrow(450, cy, 484, cy))
        b.append(rect(488, cy - 20, 62, 40, "out", rx=3))
        b.append(txt(519, cy + 4, "ReLU", 10.5))
        b.append(arrow(554, cy, 588, cy))
        b.append(rect(592, cy - 22, 44, 44, "conv" if k == 0 else "conv5", rx=3))
        b.append(txt(614, cy + 3, "4×4", 10.5))
        b.append(txt(614, cy - 30, f"a{k + 1}", 9.5, fill=MUTED))
        b.append(arrow(640, cy, 686, 178 + k * 16))

    # this is the layer's activation, i.e. a1 and a2 stacked, not a network output
    b.append(_slices(690, 166, 2, 48, 48, "conv", dx=9, dy=-9))
    b.append(txt(723, 254, "4×4×2", 11.5, weight="600"))
    b.append(txt(723, 270, "a[l]", 10, fill=MUTED))
    b.append(txt(723, 138, "stack", 10.5, fill=MUTED))

    b.append(panel(52, 300, 400, 76, label=None, fill="#f9fafb"))
    b.append(txt(252, 326, "z[l] = W[l] ∗ a[l−1] + b[l]", 13, weight="600"))
    b.append(txt(252, 352, "a[l] = g( z[l] )", 13, weight="600"))
    b.append(txt(252, 370, "exactly the dense-layer equations, with ∗ in place of a matrix product", 9.5, fill=MUTED))

    b.append(panel(486, 300, 382, 76, label=None, fill="#f9fafb"))
    b.append(txt(500, 324, "One bias per filter, broadcast to all 16 positions of that map.", 10.5, anchor="start"))
    b.append(txt(500, 344, "Parameters here: (3 · 3 · 3 + 1) × 2 = 56 — independent of", 10.5, anchor="start"))
    b.append(txt(500, 364, "the input's height and width.", 10.5, anchor="start", weight="600"))
    write(OUT, "conv-layer.svg", W, H, b)


# -------------------------------------------------------------------- pooling


def fig_pooling():
    W, H = 900, 396
    b = [title(W, "Max and average pooling",
               "split the input into regions and reduce each one to a single number")]

    cell = 34
    gx, gy = 84, 152
    tints = ["#fee2e2", "#fef3c7", "#dcfce7", "#dbeafe"]
    edges = [RED, AMBER, GREEN, BLUE]

    def region(i, j):
        return (i // 2) * 2 + (j // 2)

    b.append(matrix(gx, gy, POOL_IN, cell,
                    fills=lambda v, i, j: tints[region(i, j)], fs=12.5))
    for k in range(4):
        b.append(outline(gx, gy, (k // 2) * 2, (k % 2) * 2, 2, 2, cell, edges[k], sw=2.2))
    b.append(txt(gx + 2 * cell, gy - 14, "4×4 input", 11.5, weight="600"))
    b.append(txt(gx + 2 * cell, gy + 4 * cell + 20, "four 2×2 regions,  f = 2,  s = 2", 10.5, fill=MUTED))

    mx = [[max(POOL_IN[2 * i + a][2 * j + b_] for a in (0, 1) for b_ in (0, 1))
           for j in range(2)] for i in range(2)]
    av = [[sum(POOL_IN[2 * i + a][2 * j + b_] for a in (0, 1) for b_ in (0, 1)) / 4
           for j in range(2)] for i in range(2)]
    assert mx == [[9, 2], [6, 3]]
    assert av == [[3.75, 1.25], [3.75, 2.0]]

    xo = 400
    for yo, m, head, note in [
        (108, mx, "Max pooling", "keeps the strongest response in each region"),
        (256, av, "Average pooling", "keeps the mean of each region"),
    ]:
        b.append(arrow(gx + 4 * cell + 18, gy + 2 * cell, xo - 22, yo + cell))
        b.append(matrix(xo, yo, m, cell, fills=lambda v, i, j: tints[i * 2 + j],
                        fmt=None if m is mx else "{:.2f}", fs=12.5))
        for k in range(4):
            b.append(outline(xo, yo, k // 2, k % 2, 1, 1, cell, edges[k], sw=2.2))
        b.append(txt(xo + cell, yo - 14, head, 11.5, weight="600"))
        b.append(txt(xo + cell, yo + 2 * cell + 20, note, 10, fill=MUTED))

    b.append(txt(300, 168, "max", 10.5, fill=MUTED))
    b.append(txt(300, 268, "mean", 10.5, fill=MUTED))

    px = 552
    b.append(panel(px, 92, 320, 108, label="Hyperparameters"))
    b.append(txt(px + 18, 122, "f  — region size (2, sometimes 3)", 11, anchor="start"))
    b.append(txt(px + 18, 144, "s  — stride (usually 2)", 11, anchor="start"))
    b.append(txt(px + 18, 166, "max or average;  p = 0 almost always", 11, anchor="start"))
    b.append(txt(px + 18, 188, "f = s = 2 roughly halves height and width", 10.5, anchor="start", fill=MUTED))

    b.append(panel(px, 236, 320, 124, label="Parameters: none", fill="#eff6ff", stroke=BLUE))
    b.append(txt(px + 18, 266, "Nothing here is learned by gradient descent.", 11, anchor="start"))
    b.append(txt(px + 18, 292, "Backprop still routes gradients through it:", 10.5, anchor="start", fill=MUTED))
    b.append(txt(px + 18, 312, "max sends the gradient to the argmax only,", 10.5, anchor="start"))
    b.append(txt(px + 18, 330, "average spreads it evenly over the region.", 10.5, anchor="start"))
    b.append(txt(px + 18, 352, "Applied per channel, so nC is unchanged.", 10.5, anchor="start", weight="600"))

    b.append(txt(W / 2, H - 10, "Max pooling reports whether a feature appeared in the region, discarding exactly where it appeared.", 10.5, fill=MUTED))
    write(OUT, "pooling.svg", W, H, b)


# --------------------------------------------------------- complete ConvNets


def fig_convnet_conv_only():
    _pipeline(
        OUT,
        "convnet-conv-only.svg",
        "A convolution-only ConvNet  ·  cat / not cat",
        [
            dict(kind="vol", s=39, c=3, shape="39×39×3", name="Input", role="input"),
            dict(kind="vol", s=37, c=10, shape="37×37×10", name="CONV", role="conv"),
            dict(kind="vol", s=17, c=20, shape="17×17×20", name="CONV", role="conv"),
            dict(kind="vol", s=7, c=40, shape="7×7×40", name="CONV", role="conv"),
            dict(kind="vec", u=1960, shape="1,960", name="Flatten", role="flat"),
            dict(kind="vec", u=1, shape="1", name="Output", role="out"),
        ],
        [
            "convolution 3×3×3,\n10 filters, s=1",
            "convolution 5×5×10,\n20 filters, s=2",
            "convolution 5×5×20,\n40 filters, s=2",
            "flatten\n7·7·40",
            "logistic /\nsoftmax",
        ],
        sub="no pooling and no hidden fully connected layers — spatial size 39 → 37 → 17 → 7, channels 3 → 10 → 20 → 40",
        leg=ARCH_LEGEND,
        note="Stride 2 shrinks the representation far faster than stride 1: 37 → 17 → 7 in two layers.",
    )


def fig_convnet_lenet_style():
    _pipeline(
        OUT,
        "convnet-lenet-style.svg",
        "A LeNet-5-inspired ConvNet  ·  10 digit classes  ·  ~62 K parameters",
        [
            dict(kind="vol", s=32, c=3, shape="32×32×3", name="Input", role="input"),
            dict(kind="vol", s=28, c=6, shape="28×28×6", name="CONV", role="conv"),
            dict(kind="vol", s=14, c=6, shape="14×14×6", name="POOL", role="pool"),
            dict(kind="vol", s=10, c=16, shape="10×10×16", name="CONV", role="conv"),
            dict(kind="vol", s=5, c=16, shape="5×5×16", name="POOL", role="pool"),
            dict(kind="vec", u=400, shape="400", name="Flatten", role="flat"),
            dict(kind="vec", u=120, shape="120", name="FC", role="fc"),
            dict(kind="vec", u=84, shape="84", name="FC", role="fc"),
            dict(kind="vec", u=10, shape="10", name="Output", role="out"),
        ],
        [
            "convolution 5×5×3,\n6 filters",
            "max pool\nf=2, s=2",
            "convolution 5×5×6,\n16 filters",
            "max pool\nf=2, s=2",
            "flatten\n5·5·16",
            "fully\nconnected",
            "fully\nconnected",
            "softmax,\n10 classes",
        ],
        sub="CONV → POOL repeated, then fully connected layers, then softmax — the standard template",
        leg=ARCH_LEGEND,
        note="Counting only layers with weights, CONV1+POOL1 together are layer 1 and CONV2+POOL2 are layer 2.",
    )


def fig_parameter_distribution():
    W, H = 920, 456
    b = [title(W, "Where the parameters and the activations live",
               "the LeNet-5-inspired network of section 8.2")]

    rows = [
        ("Input", 3072, 0, "input"),
        ("CONV1", 4704, 456, "conv"),
        ("POOL1", 1176, 0, "pool"),
        ("CONV2", 1600, 2416, "conv"),
        ("POOL2", 400, 0, "pool"),
        ("Flatten", 400, 0, "flat"),
        ("FC3", 120, 48120, "fc"),
        ("FC4", 84, 10164, "fc"),
        ("Softmax", 10, 850, "out"),
    ]
    assert sum(r[2] for r in rows) == 62006

    y0, pitch, bh = 100.0, 32.0, 21.0
    lx = 128.0
    ax0, ax1 = 138.0, 452.0
    px0, px1 = 594.0, 890.0
    amax = max(r[1] for r in rows)
    pmax = max(r[2] for r in rows)

    b.append(txt((ax0 + ax1) / 2, 78, "activation size  (numbers to hold)", 11.5, weight="600"))
    b.append(txt((px0 + px1) / 2, 78, "parameters  (weights to learn)", 11.5, weight="600"))

    for k, (name, act, par, role) in enumerate(rows):
        y = y0 + k * pitch
        b.append(txt(lx, y + 15, name, 11.5, anchor="end", weight="600"))
        wa = (act / amax) * (ax1 - ax0 - 74)
        b.append(rect(ax0, y, max(wa, 2), bh, role, rx=2, sw=1.2))
        b.append(txt(ax0 + wa + 8, y + 15, f"{act:,}", 10.5, anchor="start", fill=MUTED))
        wp = (par / pmax) * (px1 - px0 - 70)
        if par:
            b.append(rect(px0, y, max(wp, 2), bh, role, rx=2, sw=1.2))
        b.append(txt(px0 + wp + 8, y + 15, f"{par:,}" if par else "0", 10.5,
                     anchor="start", fill=MUTED if par else GREEN,
                     weight="600" if par == 48120 else "normal"))

    b.append(seg(520, 92, 520, y0 + 9 * pitch, LINE, 1.0, dash="4 4"))

    b.append(caption(W / 2, H - 58, [
        "Pooling layers hold no parameters at all. Convolutional layers hold remarkably few relative to the activations they produce,",
        "while the fully connected layers hold about 59,000 of the roughly 62,000 total — FC3 alone has more than every convolutional layer combined.",
        "Activation size falls gradually, 4,704 → 1,600 → 400 → 84; dropping it too abruptly usually costs accuracy.",
    ]))
    write(OUT, "parameter-distribution.svg", W, H, b)


# ------------------------------------------- parameter sharing & sparse links


def fig_sharing_sparsity():
    W, H = 920, 440
    b = [title(W, "Parameter sharing and sparsity of connections",
               "the two mechanisms that make a convolutional layer cheap")]

    cell = 26

    # ---- dense
    b.append(panel(30, 76, 420, 240, label="Fully connected"))
    gx, gy = 74, 116
    dense_in = [["" for _ in range(4)] for _ in range(4)]
    b.append(matrix(gx, gy, dense_in, cell, fills=lambda v, i, j: "#f3f4f6"))
    b.append(txt(gx + 2 * cell, gy + 4 * cell + 18, "16 inputs", 10.5, fill=MUTED))
    outs = [(330, 158), (330, 236)]
    for ox, oy in outs:
        for i in range(4):
            for j in range(4):
                b.append(seg(gx + (j + 0.5) * cell, gy + (i + 0.5) * cell, ox - 15, oy,
                             "#c7cbd1", 0.6))
    for k, (ox, oy) in enumerate(outs):
        b.append(circle(ox, oy, 15, fill=C["fc"][0], stroke=C["fc"][1], sw=1.4))
        b.append(txt(ox, oy + 4, f"o{k + 1}", 10))
    b.append(txt(240, 294, "every output has its own weight for every input", 10.5, fill=MUTED))

    # ---- convolution
    b.append(panel(478, 76, 412, 240, label="Convolution, 3×3 filter"))
    cx0, cy0 = 512, 108
    conv_in = [["" for _ in range(5)] for _ in range(5)]
    b.append(matrix(cx0, cy0, conv_in, cell, fills=lambda v, i, j: "#f3f4f6"))
    ox0, oy0 = 782, 134
    conv_out = [["" for _ in range(3)] for _ in range(3)]
    b.append(matrix(ox0, oy0, conv_out, cell, fills=lambda v, i, j: "#f3f4f6"))

    for (oi, oj), col in [((0, 0), RED), ((2, 2), BLUE)]:
        faint = _mix(col, "#ffffff", 0.55)
        for a in range(3):
            for c in range(3):
                b.append(seg(cx0 + (oj + c + 0.5) * cell, cy0 + (oi + a + 0.5) * cell,
                             ox0 + (oj + 0.5) * cell, oy0 + (oi + 0.5) * cell,
                             faint, 0.8))
        b.append(outline(cx0, cy0, oi, oj, 3, 3, cell, col, sw=2.2))
        b.append(outline(ox0, oy0, oi, oj, 1, 1, cell, col, sw=2.4))
    b.append(txt(cx0 + 2.5 * cell, cy0 + 5 * cell + 18, "25 inputs", 10.5, fill=MUTED))
    b.append(txt(ox0 + 1.5 * cell, oy0 + 3 * cell + 18, "9 outputs", 10.5, fill=MUTED))
    b.append(txt(684, 294, "each output reads 9 inputs, using the same 9 weights", 10.5, fill=MUTED))

    # ---- the payoff, using the numbers from section 9.1
    b.append(panel(30, 352, 860, 62, fill="#f9fafb"))
    b.append(txt(52, 376, "32 × 32 × 3  →  28 × 28 × 6", 11.5, anchor="start", weight="600"))
    b.append(txt(52, 398, "the same layer, computed two ways", 10, anchor="start", fill=MUTED))
    b.append(txt(340, 376, "fully connected:  3,072 × 4,704  ≈  14,000,000 weights", 12, anchor="start", fill=RED))
    b.append(txt(340, 398, "convolutional:  (5 · 5 · 3 + 1) × 6  =  456 parameters", 12, anchor="start", fill=GREEN, weight="600"))

    # both mechanisms belong to the convolution panel; the dense layer has neither
    b.append(txt(240, 334, "no sharing and no sparsity: 32 independent weights", 10.5, fill=RED))
    b.append(txt(684, 334, "sparsity — each output reads only its own patch;   sharing — the same 9 weights everywhere",
                 10.5, fill=GREEN))
    write(OUT, "param-sharing-sparsity.svg", W, H, b)


# -------------------------------------------------------------- receptive field


def fig_receptive_field():
    W, H = 920, 372
    b = [title(W, "Receptive field: two 3×3 convolutions see a 5×5 region",
               "which is why small filters stacked deep replaced large filters")]

    cell = 30
    cyc = 186.0
    x_in, x_mid, x_out = 66.0, 300.0, 512.0
    in_g = [["" for _ in range(5)] for _ in range(5)]
    mid_g = [["" for _ in range(3)] for _ in range(3)]
    out_g = [[""]]

    y_in = cyc - 2.5 * cell
    y_mid = cyc - 1.5 * cell
    y_out = cyc - 0.5 * cell

    # dependency cones, drawn under the grids
    b.append(poly([(x_in + 5 * cell, y_in), (x_in + 5 * cell, y_in + 5 * cell),
                   (x_mid, y_mid + 3 * cell), (x_mid, y_mid)], fill="#eafaf0"))
    b.append(poly([(x_mid + 3 * cell, y_mid), (x_mid + 3 * cell, y_mid + 3 * cell),
                   (x_out, y_out + cell), (x_out, y_out)], fill="#eafaf0"))

    b.append(matrix(x_in, y_in, in_g, cell, fills=lambda v, i, j: C["conv1"][0]))
    b.append(matrix(x_mid, y_mid, mid_g, cell, fills=lambda v, i, j: C["conv"][0]))
    b.append(matrix(x_out, y_out, out_g, cell, fills=lambda v, i, j: C["conv5"][1]))

    b.append(arrow(x_in + 5 * cell + 12, cyc, x_mid - 12, cyc))
    b.append(lines((x_in + 5 * cell + x_mid) / 2, cyc - 28, ["3×3 conv", "s = 1"], 10.5, fill=MUTED))
    b.append(arrow(x_mid + 3 * cell + 12, cyc, x_out - 12, cyc))
    b.append(lines((x_mid + 3 * cell + x_out) / 2, cyc - 28, ["3×3 conv", "s = 1"], 10.5, fill=MUTED))

    b.append(txt(x_in + 2.5 * cell, y_in - 16, "5×5 input region", 11.5, weight="600"))
    b.append(txt(x_in + 2.5 * cell, y_in + 5 * cell + 22, "everything this unit can see", 10, fill=MUTED))
    b.append(txt(x_mid + 1.5 * cell, y_mid - 16, "after one conv", 11.5, weight="600"))
    b.append(txt(x_out + 0.5 * cell, y_out - 16, "one unit", 11.5, weight="600"))

    px = 600
    b.append(panel(px, 108, 290, 156, label="Same field, fewer weights"))
    hdr = [("stack", 0), ("field", 128), ("weights", 208)]
    for lab, dx in hdr:
        b.append(txt(px + 20 + dx, 138, lab, 10, anchor="start", fill=MUTED))
    b.append(seg(px + 16, 146, px + 274, 146, LINE, 1.0))
    table = [
        ("two 3×3", "5×5", "18", GREEN),
        ("one 5×5", "5×5", "25", MUTED),
        ("three 3×3", "7×7", "27", GREEN),
        ("one 7×7", "7×7", "49", MUTED),
    ]
    for k, (s, f, wgt, col) in enumerate(table):
        y = 168 + k * 22
        b.append(txt(px + 20, y, s, 11, anchor="start", fill=INK))
        b.append(txt(px + 148, y, f, 11, anchor="start", fill=INK))
        b.append(txt(px + 228, y, wgt, 11.5, anchor="start", fill=col, weight="600"))

    b.append(caption(W / 2, H - 44, [
        "Two stacked 3×3 convolutions cover the same 5×5 region as a single 5×5 convolution, with 18 weights instead of 25 and an extra non-linearity in between.",
        "This is why 3×3 filters dominate modern architectures. Striding and pooling enlarge the receptive field much faster than stacking alone.",
    ]))
    write(OUT, "receptive-field.svg", W, H, b)


# ------------------------------------------------------------ dilated variant


def fig_dilated():
    W, H = 740, 432
    b = [title(W, "Dilated (atrous) convolution",
               "spreading the nine taps apart enlarges the receptive field at no extra parameter cost")]

    cell = 30
    gw = 7 * cell
    total = gw * 2 + 100
    x0 = (W - total) / 2
    xs = [x0, x0 + gw + 100]
    gy = 104.0
    blank = [["" for _ in range(7)] for _ in range(7)]

    for xg, rate, head in [(xs[0], 1, "dilation 1  (standard)"), (xs[1], 2, "dilation 2")]:
        # a 3x3 filter with dilation r spans (3-1)*r + 1 cells and taps every r-th one
        field = 2 * rate + 1
        start = (7 - field) // 2
        taps = [(start + a * rate, start + c * rate) for a in range(3) for c in range(3)]
        tapset = set(taps)
        b.append(matrix(xg, gy, blank, cell,
                        fills=lambda v, i, j, ts=tapset: (C["conv"][0] if (i, j) in ts else "#f9fafb")))
        for (i, j) in taps:
            b.append(circle(xg + (j + 0.5) * cell, gy + (i + 0.5) * cell, 4.2,
                            fill=C["conv"][1], stroke="none", sw=0))
        b.append(outline(xg, gy, start, start, field, field, cell, RED, sw=2.4, dash="5 3"))
        b.append(txt(xg + gw / 2, 88, head, 12.5, weight="600"))
        b.append(txt(xg + gw / 2, gy + gw + 24, f"{field}×{field} receptive field", 11.5, weight="600"))
        b.append(txt(xg + gw / 2, gy + gw + 42, "9 weights", 11, fill=GREEN, weight="600"))

    b.append(caption(W / 2, H - 46, [
        "Both filters have exactly nine learnable weights. The dilated one reaches a 5×5 region because its taps skip a cell in each direction,",
        "which is useful for segmentation, where you want a large field without losing resolution to pooling.",
    ]))
    write(OUT, "dilated-conv.svg", W, H, b)


# ---------------------------------------------------------- 1D and 3D conv


def fig_1d_conv():
    W, H = 920, 340
    b = [title(W, "1-D convolution on a time series",
               "the same 5-tap filter slides along an EKG; 14 ∗ 5 = 10, just as 14×14 ∗ 5×5 = 10×10")]

    def bar(x, y, n, w, h, role, label, sub=None):
        o = [rect(x, y, n * w, h, role, rx=3)]
        for i in range(1, n):
            o.append(seg(x + i * w, y, x + i * w, y + h, LINE, 0.7))
        o.append(txt(x + n * w / 2, y + h / 2 + 4, f"{n}", 12, weight="600"))
        o.append(txt(x + n * w / 2, y + h + 16, label, 11, weight="600"))
        if sub:
            o.append(txt(x + n * w / 2, y + h + 32, sub, 10.5, fill=MUTED))
        return "\n".join(o)

    b.append(txt(40, 78, "2-D (images)", 12.5, anchor="start", weight="600"))
    b.append(rect(40, 96, 70, 70, "input", rx=3))
    b.append(txt(75, 134, "14×14", 11, weight="600"))
    b.append(txt(130, 134, "∗", 18, fill=MUTED))
    b.append(rect(150, 110, 42, 42, "conv", rx=3))
    b.append(txt(171, 134, "5×5", 11, weight="600"))
    b.append(txt(210, 134, "=", 18, fill=MUTED))
    b.append(rect(230, 104, 54, 54, "out", rx=3))
    b.append(txt(257, 134, "10×10", 11, weight="600"))
    b.append(txt(400, 134, "16 filters  →  10×10×16", 12, anchor="start", weight="600"))

    b.append(seg(40, 186, 880, 186, LINE, 1.0, dash="4 4"))

    b.append(txt(40, 214, "1-D (EKG, audio, other sequences)", 12.5, anchor="start", weight="600"))
    b.append(bar(40, 232, 14, 16, 36, "input", "length 14", "one lead"))
    b.append(txt(280, 254, "∗", 18, fill=MUTED))
    b.append(bar(304, 232, 5, 16, 36, "conv", "length 5"))
    b.append(txt(404, 254, "=", 18, fill=MUTED))
    b.append(bar(428, 232, 10, 16, 36, "out", "length 10"))
    b.append(txt(605, 248, "16 filters → 10×16", 12, anchor="start", weight="600"))
    b.append(txt(605, 268, "then ∗ 5, 32 filters → 6×32", 11, anchor="start", fill=MUTED))

    b.append(caption(W / 2, H - 18, [
        "Same idea as 2-D: one feature detector, reused at every position. RNNs are the usual tool for sequences; 1-D ConvNets are a competitive alternative.",
    ]))
    write(OUT, "conv-1d.svg", W, H, b)


def fig_3d_conv():
    W, H = 920, 360
    b = [title(W, "3-D convolution on a volume",
               "height, width, and depth are all spatial; channels are a fourth axis, just as in 2-D")]

    svg, info = volume3d(48, 130, 86, 86, 28, "input", label="14³")
    b.append(svg)
    b.append(txt(info["cx"], info["bottom"] + 18, "CT / video", 10.5, fill=MUTED))

    b.append(txt(168, 168, "∗", 20, fill=MUTED))
    svg, info = volume3d(188, 148, 44, 44, 18, "conv", label="5³")
    b.append(svg)
    b.append(txt(info["cx"], info["bottom"] + 18, "3-D filter", 10.5, fill=MUTED))

    b.append(txt(286, 168, "=", 20, fill=MUTED))
    svg, info = volume3d(308, 140, 62, 62, 22, "out", label="10³")
    b.append(svg)
    b.append(txt(info["cx"], info["bottom"] + 18, "one filter", 10.5, fill=MUTED))

    b.append(arrow(408, 168, 448, 168, LINE, 1.5))
    b.append(txt(560, 100, "16 filters of 5×5×5×1", 12.5, weight="600"))
    b.append(txt(560, 122, "→  10×10×10×16", 14, weight="600"))
    b.append(txt(560, 154, "next layer: 5×5×5×16, 32 filters", 12, fill=MUTED))
    b.append(txt(560, 176, "→  6×6×6×32", 14, weight="600"))

    b.append(rect(448, 210, 424, 88, "input", rx=8, fill="#f9fafb", stroke=LINE))
    b.append(txt(660, 236, "Not the same as a 2-D RGB convolution.", 12, weight="600"))
    b.append(txt(660, 258, "An image is 14×14×3: the 3 is channels, not depth.", 11.5, fill=MUTED))
    b.append(txt(660, 278, "A CT scan is 14×14×14×1: the third 14 is a spatial axis.", 11.5, fill=MUTED))

    b.append(caption(W / 2, H - 18, [
        "Movies are 3-D too: two spatial axes plus time. The same output-size formula applies once per spatial axis, including depth.",
    ]))
    write(OUT, "conv-3d.svg", W, H, b)


# ---------------------------------------------------------------------- main

FIGURES = [
    fig_fc_explosion,
    fig_conv_operation,
    fig_edge_vertical,
    fig_edge_signs,
    fig_edge_horizontal,
    fig_cross_correlation,
    fig_padding,
    fig_padding_coverage,
    fig_strided,
    fig_conv_volumes,
    fig_conv_layer,
    fig_pooling,
    fig_convnet_conv_only,
    fig_convnet_lenet_style,
    fig_parameter_distribution,
    fig_sharing_sparsity,
    fig_receptive_field,
    fig_dilated,
    fig_1d_conv,
    fig_3d_conv,
]


def build():
    for f in FIGURES:
        f()


if __name__ == "__main__":
    build()
    print(f"{len(FIGURES)} foundations figures written")
