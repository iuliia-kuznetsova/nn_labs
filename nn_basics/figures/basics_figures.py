"""Diagrams for the non-CNN lecture notes.

Run via  python figures/make_figures.py  (or this file directly).
Palette and primitives come from svgkit.py so these match the CNN figures.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from svgkit import (  # noqa: E402
    AMBER, BLUE, C, GREEN, INK, LINE, MUTED, PURPLE, RED,
    arrow, caption, circle, legend, lines, panel, path, poly, rect, seg,
    title, txt, write,
)

OUT = pathlib.Path(__file__).parent


# ----------------------------------------------------------------- helpers


def ellipse(cx, cy, rx, ry, fill="none", stroke=LINE, sw=1.2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<ellipse cx="{cx:g}" cy="{cy:g}" rx="{rx:g}" ry="{ry:g}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}"{d}/>'
    )


def polyline(pts, color, sw=2.2, dash=None, fill="none"):
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{p}" fill="{fill}" stroke="{color}" '
        f'stroke-width="{sw:g}"{d}/>'
    )


def axes(x0, y0, x1, y1, xlabel="", ylabel=""):
    o = [
        f'<line x1="{x0:g}" y1="{y1:g}" x2="{x1:g}" y2="{y1:g}" '
        f'stroke="{INK}" stroke-width="1.4"/>',
        f'<line x1="{x0:g}" y1="{y1:g}" x2="{x0:g}" y2="{y0:g}" '
        f'stroke="{INK}" stroke-width="1.4"/>',
    ]
    if xlabel:
        o.append(txt((x0 + x1) / 2, y1 + 28, xlabel, 11.5))
    if ylabel:
        o.append(txt(x0 - 40, (y0 + y1) / 2, ylabel, 11.5, rot=-90))
    return "\n".join(o)


def mapxy(t, vmin, vmax, a, b):
    if vmax == vmin:
        return a
    return a + (t - vmin) / (vmax - vmin) * (b - a)


def node(cx, cy, r, role, label=None, size=11):
    fill, stroke = C[role]
    o = [circle(cx, cy, r, fill=fill, stroke=stroke, sw=1.5)]
    if label is not None:
        o.append(txt(cx, cy + 4, label, size, weight="600"))
    return "\n".join(o)


def wires(src, dst, color=LINE, sw=0.85):
    return "\n".join(seg(x1, y1, x2, y2, color, sw) for x1, y1 in src for x2, y2 in dst)


def col_nodes(cx, n, ymid, r=12, vgap=28, role="conv", labels=None):
    total = (n - 1) * vgap
    ys = [ymid - total / 2 + i * vgap for i in range(n)]
    pts = [(cx, y) for y in ys]
    body = []
    for i, (x, y) in enumerate(pts):
        lab = None if labels is None else labels[i]
        body.append(node(x, y, r, role, lab))
    return pts, "\n".join(body)


def box(x, y, w, h, role, *rows, size=11):
    o = [rect(x, y, w, h, role)]
    cy = y + h / 2 + 4 - (len(rows) - 1) * 6.5
    o.append(lines(x + w / 2, cy, rows, size, lh=13))
    return "\n".join(o)


def nest_ellipses(cx, cy, n, rx, ry, color=LINE):
    o = []
    for i in range(1, n + 1):
        t = i / n
        o.append(ellipse(cx, cy, rx * t, ry * t, stroke=color, sw=1.05))
    return "\n".join(o)


# =========================================================== network graphs


def fig_log_neuron():
    W, H = 720, 280
    b = [title(W, "Logistic regression is a one-neuron network",
               "linear score, then sigmoid, then a probability")]
    xs = [90, 210, 370, 520, 640]
    y = 150
    b.append(box(xs[0] - 40, y - 28, 80, 56, "input", "x", size=14))
    b.append(box(xs[1] - 40, y - 28, 80, 56, "fc", "w, b", size=13))
    b.append(box(xs[2] - 44, y - 28, 88, 56, "conv", "z = w⊤x + b", size=11))
    b.append(box(xs[3] - 40, y - 28, 80, 56, "out", "a = σ(z)", size=12))
    b.append(box(xs[4] - 36, y - 28, 72, 56, "out", "ŷ", size=14))
    for i in range(4):
        b.append(arrow(xs[i] + 44, y, xs[i + 1] - 44, y))
    b.append(caption(W / 2, H - 36, [
        "No hidden layer: one linear unit plus a sigmoid. The decision boundary is linear in the original features.",
    ]))
    b.append(legend(70, H - 18, [
        ("input", "input"), ("fc", "parameters"), ("conv", "logit"), ("out", "sigmoid / output"),
    ]))
    write(OUT, "log-neuron.svg", W, H, b)


def fig_log_compgraph():
    W, H = 760, 320
    b = [title(W, "Computation graph for one example",
               "forward left to right; backprop walks the same edges in reverse")]
    y = 155
    b.append(box(40, 88, 80, 50, "input", "x"))
    b.append(box(40, 172, 80, 50, "fc", "w, b"))
    b.append(box(200, y - 28, 90, 56, "conv", "z"))
    b.append(box(360, y - 28, 110, 56, "out", "a = σ(z)"))
    b.append(box(540, y - 28, 110, 56, "out", "ℒ(a, y)"))
    b.append(arrow(128, 113, 192, y - 8))
    b.append(arrow(128, 197, 192, y + 8))
    b.append(arrow(298, y, 352, y))
    b.append(arrow(478, y, 532, y))
    b.append(arrow(532, y + 28, 478, y + 28, color=RED))
    b.append(arrow(352, y + 28, 298, y + 28, color=RED))
    b.append(arrow(192, y + 36, 128, 210, color=RED))
    b.append(txt(W / 2, 72, "forward  →", 12, fill=MUTED, weight="600"))
    b.append(txt(W / 2, y + 56, "←  backward  (chain rule)", 12, fill=RED, weight="600"))
    b.append(caption(W / 2, H - 28, [
        "Each node is a tiny operation. Local derivatives multiply along the path; merging paths add.",
    ]))
    write(OUT, "log-compgraph.svg", W, H, b)


def fig_log_vectorized():
    W, H = 720, 300
    b = [title(W, "Design matrix X is nₓ × m",
               "each column is one example; one matrix multiply scores the whole batch")]
    cell = 28
    vals = [[1, 0, 2, 1], [3, 1, 0, 2], [0, 4, 1, 3]]
    mx, my = 80, 86
    from svgkit import matrix
    b.append(matrix(mx, my, vals, cell=cell, fills=lambda v, i, j: C["input"][0]))
    b.append(txt(mx + 2 * cell, my - 12, "m examples", 11.5, fill=MUTED))
    b.append(txt(mx - 18, my + 1.5 * cell + 4, "nₓ features", 11.5, fill=MUTED, rot=-90))
    for j, lab in enumerate(["x⁽¹⁾", "x⁽²⁾", "x⁽³⁾", "x⁽⁴⁾"]):
        b.append(txt(mx + (j + 0.5) * cell, my + 3 * cell + 18, lab, 11, fill=MUTED))
    b.append(arrow(mx + 4 * cell + 16, my + 1.5 * cell, 430, my + 1.5 * cell))
    b.append(box(440, 100, 100, 44, "fc", "w⊤", size=13))
    b.append(txt(490, 160, "×", 16, fill=MUTED))
    b.append(box(440, 176, 100, 44, "input", "X", size=13))
    b.append(arrow(552, 148, 610, 148))
    b.append(box(618, 120, 70, 56, "conv", "Z", "1 × m", size=12))
    write(OUT, "log-vectorized.svg", W, H, b)


def fig_log_gd():
    W, H = 680, 340
    b = [title(W, "Gradient descent on a convex bowl",
               "subtract the gradient so each step goes downhill")]
    cx, cy = 340, 200
    b.append(nest_ellipses(cx, cy, 6, 210, 110, LINE))
    # a path of shrinking steps toward the centre
    pts = [(160, 250), (230, 230), (280, 214), (310, 206), (328, 202), (338, 200)]
    b.append(polyline(pts, RED, 2.4))
    for x, y in pts:
        b.append(circle(x, y, 3.4, fill=RED, stroke=RED))
    b.append(circle(cx, cy, 5, fill=GREEN, stroke=GREEN))
    b.append(txt(cx + 18, cy + 4, "minimum", 11, anchor="start", fill=GREEN, weight="600"))
    b.append(txt(160, 268, "start", 11, fill=RED, weight="600"))
    b.append(caption(W / 2, H - 28, [
        "Too large α overshoots; too small α crawls. For logistic regression the bowl is convex.",
    ]))
    write(OUT, "log-gd.svg", W, H, b)


def fig_sha_arch():
    W, H = 720, 340
    b = [title(W, "Shallow network: one hidden layer",
               "a 2-layer network by convention — the input layer is not counted")]
    ymid = 175
    specs = [
        (3, "input", ["x₁", "x₂", "x₃"], 110),
        (4, "conv", ["a₁", "a₂", "a₃", "a₄"], 300),
        (1, "out", ["ŷ"], 520),
    ]
    cols = []
    for n, role, labs, cx in specs:
        pts, body = col_nodes(cx, n, ymid, r=14, vgap=36, role=role, labels=labs)
        cols.append(pts)
        b.append(body)
    b.append(wires(cols[0], cols[1]))
    b.append(wires(cols[1], cols[2]))
    b.append(txt(110, 300, "input  ·  ℓ = 0", 12, fill=MUTED))
    b.append(txt(300, 300, "hidden  ·  ℓ = 1", 12, fill=MUTED))
    b.append(txt(520, 300, "output  ·  ℓ = 2", 12, fill=MUTED))
    b.append(legend(80, H - 22, [
        ("input", "input"), ("conv", "hidden (ReLU / tanh)"), ("out", "sigmoid / softmax"),
    ]))
    write(OUT, "sha-arch.svg", W, H, b)


def fig_sha_forward():
    W, H = 780, 250
    b = [title(W, "Four equations of a shallow forward pass",
               "matrix multiply, nonlinearity, repeat")]
    y = 130
    stages = [
        (70, "input", "x"),
        (110, "conv", "z[1] = W[1]x + b[1]"),
        (90, "conv", "a[1] = g[1](z[1])"),
        (120, "fc", "z[2] = W[2]a[1] + b[2]"),
        (90, "out", "ŷ = g[2](z[2])"),
    ]
    x = 28
    xs = []
    for w, role, lab in stages:
        xs.append((x, w))
        b.append(box(x, y - 28, w, 56, role, lab, size=10.5))
        x += w + 28
    for (x1, w1), (x2, _) in zip(xs, xs[1:]):
        b.append(arrow(x1 + w1 + 3, y, x2 - 3, y))
    write(OUT, "sha-forward.svg", W, H, b)


def fig_sha_softmax():
    W, H = 720, 300
    b = [title(W, "Softmax turns logits into a probability vector",
               "exponentiate, then divide by the sum so the entries add to 1")]
    z = [2.0, 1.0, 0.1]
    ez = [math.exp(v) for v in z]
    s = sum(ez)
    a = [v / s for v in ez]
    names = ["cat", "dog", "bird"]
    x0 = 80
    b.append(txt(x0 + 30, 78, "logits z", 12, weight="600"))
    b.append(txt(320, 78, "eᶻ", 12, weight="600"))
    b.append(txt(560, 78, "softmax a", 12, weight="600"))
    for i, (zi, ei, ai, name) in enumerate(zip(z, ez, a, names)):
        y = 100 + i * 48
        b.append(box(x0, y, 90, 36, "fc", f"{zi:.1f}", size=13))
        b.append(txt(x0 + 108, y + 24, name, 11, anchor="start", fill=MUTED))
        b.append(arrow(x0 + 140, y + 18, 280, y + 18))
        b.append(box(290, y, 90, 36, "conv", f"{ei:.2f}", size=13))
        b.append(arrow(392, y + 18, 490, y + 18))
        bw = 40 + ai * 140
        b.append(rect(500, y, bw, 36, "out"))
        b.append(txt(500 + bw / 2, y + 24, f"{ai:.2f}", 12, weight="600"))
    b.append(caption(W / 2, H - 28, [
        f"The three probabilities sum to {sum(a):.2f}. Softmax is for mutually exclusive classes.",
    ]))
    write(OUT, "sha-softmax.svg", W, H, b)


def fig_dnn_arch():
    W, H = 780, 340
    b = [title(W, "Deep network: L trainable layers",
               "more than one hidden layer; input is still ℓ = 0")]
    ymid = 175
    specs = [
        (3, "input", 70, "x = a[0]"),
        (5, "conv", 200, "hidden 1"),
        (5, "conv", 340, "hidden 2"),
        (4, "conv", 480, "hidden L−1"),
        (1, "out", 620, "ŷ = a[L]"),
    ]
    cols = []
    for n, role, cx, lab in specs:
        pts, body = col_nodes(cx, n, ymid, r=11, vgap=28, role=role)
        cols.append(pts)
        b.append(body)
        b.append(txt(cx, 300, lab, 11, fill=MUTED))
    for a, c in zip(cols, cols[1:]):
        b.append(wires(a, c))
    b.append(legend(80, H - 22, [
        ("input", "input"), ("conv", "hidden"), ("out", "output"),
    ]))
    write(OUT, "dnn-arch.svg", W, H, b)


def fig_dnn_hierarchy():
    W, H = 760, 280
    b = [title(W, "Hierarchical features",
               "each layer builds a more abstract concept from the previous one")]
    stages = [
        ("input", "pixels", "raw image"),
        ("conv", "edges", "layer 1"),
        ("block", "parts", "layer 2  ·  eye, nose"),
        ("out", "object", "layer 3  ·  face"),
    ]
    x, y = 50, 110
    for i, (role, name, sub) in enumerate(stages):
        b.append(box(x, y, 130, 70, role, name, size=14))
        b.append(txt(x + 65, y + 92, sub, 11, fill=MUTED))
        if i < 3:
            b.append(arrow(x + 138, y + 35, x + 168, y + 35))
        x += 180
    write(OUT, "dnn-hierarchy.svg", W, H, b)


def fig_dnn_block():
    W, H = 780, 300
    b = [title(W, "One layer as a forward / backward block",
               "cache Z, W, b on the way forward; reuse them going back")]
    y = 150
    b.append(box(40, y - 36, 110, 72, "input", "A[ℓ−1]", size=13))
    b.append(arrow(158, y, 196, y))
    b.append(box(204, y - 50, 150, 100, "conv", "forward", "Z = WA + b", "A = g(Z)", size=12))
    b.append(arrow(362, y, 400, y))
    b.append(box(408, y - 36, 110, 72, "out", "A[ℓ]", size=13))
    b.append(rect(204, 230, 150, 36, "flat"))
    b.append(txt(279, 254, "cache  Z, W, b", 11, weight="600"))
    b.append(path("M 279,230 L 279,200", stroke=AMBER, sw=1.4, marker=False, dash="4 3"))
    b.append(txt(640, 100, "backward", 12, fill=RED, weight="600"))
    b.append(box(560, y - 36, 90, 72, "out", "dA[ℓ]", size=12))
    b.append(arrow(648, y + 40, 500, y + 40, color=RED))
    b.append(box(560, 210, 180, 56, "fc", "dW, db, dA[ℓ−1]", size=12))
    write(OUT, "dnn-block.svg", W, H, b)


def fig_term_neuron():
    W, H = 720, 260
    b = [title(W, "One neuron", "linear score, then a nonlinearity — one scalar out")]
    y = 140
    pts, body = col_nodes(80, 3, y, r=12, vgap=36, role="input", labels=["x₁", "x₂", "x₃"])
    b.append(body)
    b.append(node(280, y, 22, "conv", "z"))
    b.append(wires(pts, [(280, y)]))
    b.append(txt(280, y - 36, "w⊤x + b", 11, fill=MUTED))
    b.append(arrow(310, y, 400, y))
    b.append(node(430, y, 22, "out", "a"))
    b.append(txt(430, y - 36, "g(z)", 11, fill=MUTED))
    b.append(caption(W / 2, H - 28, [
        "A layer is several of these in parallel on the same x, stacked into a vector a.",
    ]))
    write(OUT, "term-neuron.svg", W, H, b)


def fig_term_activations():
    W, H = 780, 360
    b = [title(W, "Four common activations",
               "hidden layers want ReLU-family; sigmoid is for binary outputs")]
    fns = [
        ("sigmoid", lambda z: 1 / (1 + math.exp(-max(min(z, 20), -20))),
         -6, 6, -0.2, 1.2, "out"),
        ("tanh", math.tanh, -4, 4, -1.3, 1.3, "fc"),
        ("ReLU", lambda z: max(0.0, z), -3, 3, -0.5, 3.2, "conv"),
        ("leaky ReLU", lambda z: z if z > 0 else 0.01 * z, -3, 3, -0.5, 3.2, "conv1"),
    ]
    pw, ph = 170, 140
    for i, (name, f, xmin, xmax, ymin, ymax, role) in enumerate(fns):
        x0 = 30 + i * 188
        y0 = 80
        b.append(panel(x0, y0, pw, ph, label=name))
        ax, ay, bx, by = x0 + 18, y0 + 16, x0 + pw - 12, y0 + ph - 22
        # axes through origin if possible
        zx = mapxy(0, xmin, xmax, ax, bx)
        zy = mapxy(0, ymin, ymax, by, ay)
        b.append(seg(ax, zy, bx, zy, LINE, 1.0))
        b.append(seg(zx, ay, zx, by, LINE, 1.0))
        n = 80
        pts = []
        for k in range(n + 1):
            z = xmin + (xmax - xmin) * k / n
            pts.append((mapxy(z, xmin, xmax, ax, bx),
                        mapxy(f(z), ymin, ymax, by, ay)))
        fill, stroke = C[role]
        b.append(polyline(pts, stroke, 2.2))
    b.append(caption(W / 2, H - 28, [
        "Sigmoid and tanh saturate (flat tails → vanishing gradients). ReLU does not, for z > 0.",
    ]))
    write(OUT, "term-activations.svg", W, H, b)


def fig_term_linear():
    W, H = 720, 240
    b = [title(W, "Stacked linear layers collapse",
               "without a nonlinearity, depth buys you nothing")]
    y = 130
    b.append(box(40, y - 28, 80, 56, "input", "x"))
    b.append(arrow(128, y, 168, y))
    b.append(box(176, y - 28, 90, 56, "fc", "W¹x + b¹"))
    b.append(arrow(274, y, 314, y))
    b.append(box(322, y - 28, 90, 56, "fc", "W²(·) + b²"))
    b.append(arrow(420, y, 470, y))
    b.append(box(478, y - 28, 200, 56, "out", "W′x + b′", size=13))
    write(OUT, "term-linear.svg", W, H, b)


def fig_term_tensor():
    W, H = 760, 260
    b = [title(W, "Tensors by rank", "a neuron still outputs one scalar; layers output arrays")]
    # 0d
    b.append(circle(90, 150, 16, fill=C["out"][0], stroke=C["out"][1], sw=1.5))
    b.append(txt(90, 200, "scalar  ·  0-D", 11, fill=MUTED))
    # 1d
    for i in range(4):
        b.append(rect(200, 96 + i * 22, 28, 18, "conv"))
    b.append(txt(214, 200, "vector  ·  1-D", 11, fill=MUTED))
    # 2d
    from svgkit import matrix
    b.append(matrix(340, 108, [[1, 2, 3], [4, 5, 6]], cell=22,
                    fills=lambda v, i, j: C["fc"][0]))
    b.append(txt(373, 200, "matrix  ·  2-D", 11, fill=MUTED))
    # 3d volume
    b.append(rect(530, 128, 70, 70, "input"))
    b.append(rect(548, 112, 70, 70, "conv"))
    b.append(rect(566, 96, 70, 70, "block"))
    b.append(txt(601, 200, "volume  ·  3-D", 11, fill=MUTED))
    write(OUT, "term-tensor.svg", W, H, b)


def fig_term_symmetry():
    W, H = 720, 280
    b = [title(W, "Random init breaks symmetry",
               "identical weights → identical gradients → the hidden units never specialize")]
    ymid = 150
    # left: all same
    b.append(txt(180, 70, "all weights equal", 12, weight="600"))
    p0, body = col_nodes(80, 2, ymid, r=11, role="input")
    b.append(body)
    p1, body = col_nodes(180, 3, ymid, r=11, role="conv")
    b.append(body)
    b.append(wires(p0, p1, color=RED, sw=1.2))
    b.append(txt(180, 240, "three copies of one neuron", 11, fill=RED))
    # right: random
    b.append(txt(520, 70, "random weights", 12, weight="600"))
    p0, body = col_nodes(420, 2, ymid, r=11, role="input")
    b.append(body)
    p1, body = col_nodes(520, 3, ymid, r=11, role="conv")
    b.append(body)
    b.append(wires(p0, p1, color=GREEN, sw=1.2))
    b.append(txt(520, 240, "units can learn different features", 11, fill=GREEN))
    write(OUT, "term-symmetry.svg", W, H, b)


# =========================================================== optimization


def _valley(cx, cy, rx=210, ry=70, n=5):
    return nest_ellipses(cx, cy, n, rx, ry, LINE)


def fig_opt_paths():
    W, H = 780, 300
    b = [title(W, "Three batch sizes on the same valley",
               "noise falls as the batch grows; so does the number of steps per epoch")]
    panels = [
        ("SGD  ·  B = 1", RED, [(70, 70), (150, 20), (90, 55), (170, 10), (120, 40), (190, 4), (210, 0)]),
        ("mini-batch", BLUE, [(70, 50), (120, 22), (150, 28), (185, 8), (210, 0)]),
        ("batch GD  ·  B = m", GREEN, [(70, 40), (130, 18), (175, 6), (210, 0)]),
    ]
    for i, (lab, color, offs) in enumerate(panels):
        x0 = 30 + i * 250
        b.append(panel(x0, 70, 230, 190, label=lab))
        cx, cy = x0 + 115, 175
        b.append(_valley(cx, cy, 90, 55, 4))
        pts = [(cx - dx, cy - dy) for dx, dy in offs]
        b.append(polyline(pts, color, 2.2))
        b.append(circle(pts[0][0], pts[0][1], 3.5, fill=color, stroke=color))
        b.append(circle(cx, cy, 4, fill=GREEN, stroke=GREEN))
    write(OUT, "opt-paths.svg", W, H, b)


def fig_opt_cost():
    W, H = 720, 320
    b = [title(W, "Cost versus iteration",
               "batch GD is monotone; mini-batch oscillates but trends down")]
    x0, y0, x1, y1 = 70, 70, 680, 250
    b.append(axes(x0, y0, x1, y1, "iteration", "cost J"))
    n = 80
    batch, mini = [], []
    for i in range(n + 1):
        t = i / n
        xb = mapxy(t, 0, 1, x0, x1)
        jb = 0.85 * math.exp(-3.2 * t) + 0.08
        jm = jb + 0.07 * math.sin(18 * t) * math.exp(-1.2 * t)
        batch.append((xb, mapxy(jb, 0, 1, y1, y0)))
        mini.append((xb, mapxy(jm, 0, 1, y1, y0)))
    b.append(polyline(mini, BLUE, 2.0))
    b.append(polyline(batch, GREEN, 2.4))
    b.append(txt(x1 - 8, 92, "mini-batch", 12, anchor="end", fill=BLUE, weight="600"))
    b.append(txt(x1 - 8, 112, "batch GD", 12, anchor="end", fill=GREEN, weight="600"))
    write(OUT, "opt-cost.svg", W, H, b)


def fig_opt_ewa():
    W, H = 720, 320
    b = [title(W, "Exponentially weighted average",
               "β close to 1 is smoother and slower; only one number is stored")]
    x0, y0, x1, y1 = 70, 70, 680, 250
    b.append(axes(x0, y0, x1, y1, "t", "value"))
    n = 60
    raw, v09, v098 = [], [], []
    theta = v1 = v2 = 0.0
    for i in range(n):
        # a noisy rising-then-falling signal
        true = 10 + 8 * math.sin(i / 8)
        theta = true + (1.6 if (i * 17) % 7 == 0 else -0.4 * ((i * 13) % 5 - 2))
        v1 = 0.9 * v1 + 0.1 * theta
        v2 = 0.98 * v2 + 0.02 * theta
        x = mapxy(i, 0, n - 1, x0, x1)
        raw.append((x, mapxy(theta, -2, 22, y1, y0)))
        v09.append((x, mapxy(v1, -2, 22, y1, y0)))
        v098.append((x, mapxy(v2, -2, 22, y1, y0)))
    b.append(polyline(raw, LINE, 1.1))
    b.append(polyline(v09, BLUE, 2.2))
    b.append(polyline(v098, RED, 2.2))
    b.append(txt(x1 - 8, 88, "β = 0.98  (~50 steps)", 11, anchor="end", fill=RED, weight="600"))
    b.append(txt(x1 - 8, 106, "β = 0.9   (~10 steps)", 11, anchor="end", fill=BLUE, weight="600"))
    b.append(txt(x1 - 8, 124, "raw θₜ", 11, anchor="end", fill=MUTED))
    write(OUT, "opt-ewa.svg", W, H, b)


def fig_opt_biascorr():
    W, H = 680, 300
    b = [title(W, "Bias correction of an EWA started at 0",
               "early Vₜ is too small; divide by 1 − βᵗ")]
    x0, y0, x1, y1 = 70, 70, 640, 230
    b.append(axes(x0, y0, x1, y1, "t", "estimate"))
    beta, theta = 0.9, 10.0
    v = 0.0
    raw, corr = [], []
    n = 25
    for t in range(1, n + 1):
        v = beta * v + (1 - beta) * theta
        hat = v / (1 - beta ** t)
        x = mapxy(t, 1, n, x0, x1)
        raw.append((x, mapxy(v, 0, 12, y1, y0)))
        corr.append((x, mapxy(hat, 0, 12, y1, y0)))
    b.append(seg(x0, mapxy(10, 0, 12, y1, y0), x1, mapxy(10, 0, 12, y1, y0),
                 GREEN, 1.2, dash="5 4"))
    b.append(polyline(raw, RED, 2.2))
    b.append(polyline(corr, BLUE, 2.2))
    b.append(txt(x1 - 6, 86, "true θ = 10", 11, anchor="end", fill=GREEN, weight="600"))
    b.append(txt(x1 - 6, 104, "Vₜ  (biased)", 11, anchor="end", fill=RED, weight="600"))
    b.append(txt(x1 - 6, 122, "V̂ₜ = Vₜ / (1−βᵗ)", 11, anchor="end", fill=BLUE, weight="600"))
    write(OUT, "opt-biascorr.svg", W, H, b)


def fig_opt_momentum():
    W, H = 720, 300
    b = [title(W, "Momentum damps the zig-zag",
               "oscillating directions cancel in the running average; consistent ones add up")]
    b.append(panel(40, 70, 310, 190, label="plain GD"))
    b.append(panel(380, 70, 310, 190, label="with momentum"))
    for x0, color, offs in (
        (195, RED, [(110, 48), (40, -40), (100, 36), (30, -30), (80, 22), (16, -14), (0, 0)]),
        (535, BLUE, [(110, 30), (80, 10), (50, 6), (24, 2), (0, 0)]),
    ):
        b.append(_valley(x0, 175, 120, 60, 5))
        pts = [(x0 - dx, 175 - dy) for dx, dy in offs]
        b.append(polyline(pts, color, 2.3))
        b.append(circle(pts[0][0], pts[0][1], 3.5, fill=color, stroke=color))
        b.append(circle(x0, 175, 4.5, fill=GREEN, stroke=GREEN))
    write(OUT, "opt-momentum.svg", W, H, b)


def fig_opt_rmsprop():
    W, H = 720, 280
    b = [title(W, "RMSprop scales the step per parameter",
               "steep axis is divided by a large RMS → smaller steps; flat axis speeds up")]
    cx, cy = 360, 165
    b.append(_valley(cx, cy, 240, 70, 6))
    # steep axis arrows (short, red)
    b.append(arrow(cx, cy - 58, cx, cy - 18, color=RED, sw=1.8))
    b.append(arrow(cx, cy + 58, cx, cy + 18, color=RED, sw=1.8))
    b.append(txt(cx + 14, cy - 70, "steep  ·  divide by large √S", 11, anchor="start", fill=RED, weight="600"))
    # long axis (blue, long)
    b.append(arrow(cx - 200, cy, cx - 40, cy, color=BLUE, sw=1.8))
    b.append(txt(cx - 200, cy - 16, "flat  ·  divide by small √S", 11, anchor="start", fill=BLUE, weight="600"))
    b.append(circle(cx, cy, 5, fill=GREEN, stroke=GREEN))
    write(OUT, "opt-rmsprop.svg", W, H, b)


def fig_opt_adam():
    W, H = 720, 240
    b = [title(W, "Adam = momentum + RMSprop, with bias correction",
               "first moment steers; second moment scales; both are debiased")]
    b.append(box(40, 110, 150, 64, "fc", "1st moment V", "like momentum", size=12))
    b.append(box(230, 110, 150, 64, "conv", "2nd moment S", "like RMSprop", size=12))
    b.append(box(420, 110, 130, 64, "flat", "bias correct", "÷ (1 − βᵗ)", size=12))
    b.append(box(590, 110, 100, 64, "out", "update W", size=12))
    b.append(arrow(198, 142, 222, 142))
    b.append(arrow(388, 142, 412, 142))
    b.append(arrow(558, 142, 582, 142))
    write(OUT, "opt-adam.svg", W, H, b)


def fig_opt_lrdecay():
    W, H = 720, 320
    b = [title(W, "Learning-rate schedules",
               "large steps early, smaller steps to settle")]
    x0, y0, x1, y1 = 70, 70, 680, 250
    b.append(axes(x0, y0, x1, y1, "epoch", "α"))
    n = 80
    step, expn, cos_ = [], [], []
    for i in range(n + 1):
        t = i / n
        epoch = t * 8
        a_step = 1.0 * (0.5 ** math.floor(epoch / 2))
        a_exp = 1.0 * (0.85 ** epoch)
        a_cos = 0.5 * (1 + math.cos(math.pi * t))
        x = mapxy(t, 0, 1, x0, x1)
        step.append((x, mapxy(a_step, 0, 1.05, y1, y0)))
        expn.append((x, mapxy(a_exp, 0, 1.05, y1, y0)))
        cos_.append((x, mapxy(a_cos, 0, 1.05, y1, y0)))
    b.append(polyline(step, RED, 2.2))
    b.append(polyline(expn, BLUE, 2.2))
    b.append(polyline(cos_, GREEN, 2.2))
    b.append(txt(x1 - 8, 88, "step / staircase", 11, anchor="end", fill=RED, weight="600"))
    b.append(txt(x1 - 8, 106, "exponential", 11, anchor="end", fill=BLUE, weight="600"))
    b.append(txt(x1 - 8, 124, "cosine", 11, anchor="end", fill=GREEN, weight="600"))
    write(OUT, "opt-lrdecay.svg", W, H, b)


def fig_opt_saddle():
    W, H = 760, 300
    b = [title(W, "In high dimensions, saddles beat local minima",
               "a true local min needs every direction to curve up — that is vanishingly rare")]
    # local min bowl
    b.append(panel(40, 70, 220, 180, label="local min  (rare)"))
    b.append(_valley(150, 175, 80, 55, 4))
    b.append(circle(150, 175, 5, fill=RED, stroke=RED))
    # saddle
    b.append(panel(280, 70, 220, 180, label="saddle  (typical)"))
    b.append(ellipse(390, 175, 80, 18, stroke=LINE, sw=1.2))
    b.append(ellipse(390, 175, 18, 70, stroke=LINE, sw=1.2))
    b.append(path("M 390,105 C 410,140 410,210 390,245", stroke=BLUE, sw=1.6, marker=False))
    b.append(path("M 390,105 C 370,140 370,210 390,245", stroke=BLUE, sw=1.6, marker=False))
    b.append(path("M 320,175 C 350,160 430,190 460,175", stroke=RED, sw=1.6, marker=False))
    b.append(circle(390, 175, 4, fill=AMBER, stroke=AMBER))
    # plateau
    b.append(panel(520, 70, 220, 180, label="plateau  (the real pain)"))
    b.append(rect(545, 145, 170, 28, "flat", sw=1.2))
    b.append(polyline([(560, 200), (600, 168), (680, 158), (710, 130)], RED, 2.0))
    b.append(txt(630, 136, "slow crawl", 11, fill=RED, weight="600"))
    write(OUT, "opt-saddle.svg", W, H, b)


# =========================================================== structuring ML


def fig_str_loop():
    W, H = 760, 220
    b = [title(W, "The iteration loop", "strategy is about choosing the next experiment well")]
    labs = [("input", "idea"), ("conv", "train"), ("fc", "evaluate"),
            ("flat", "diagnose"), ("out", "next try")]
    x, y = 30, 110
    xs = []
    for role, lab in labs:
        xs.append(x)
        b.append(box(x, y, 110, 50, role, lab, size=13))
        x += 148
    for a, c in zip(xs, xs[1:]):
        b.append(arrow(a + 118, y + 25, c - 8, y + 25))
    b.append(path(f"M {xs[-1] + 55:g},{y + 54:g} C 700,210 60,210 {xs[0] + 55:g},{y + 54:g}",
                  stroke=LINE, sw=1.3))
    write(OUT, "str-loop.svg", W, H, b)


def fig_str_ortho():
    W, H = 760, 280
    b = [title(W, "Orthogonal knobs", "one control per failure mode")]
    rows = [
        ("conv", "fit the training set", "bigger model, train longer, better optimizer"),
        ("fc", "generalize to dev", "regularize, more data, augment"),
        ("flat", "dev → test", "larger / better dev set"),
        ("out", "metric matches the product", "change metric or the target distribution"),
    ]
    y = 80
    for role, want, fix in rows:
        b.append(rect(40, y, 280, 36, role))
        b.append(txt(180, y + 24, want, 12, weight="600"))
        b.append(arrow(332, y + 18, 390, y + 18))
        b.append(txt(400, y + 24, fix, 11.5, anchor="start", fill=MUTED))
        y += 46
    write(OUT, "str-ortho.svg", W, H, b)


def fig_str_metrics():
    W, H = 720, 280
    b = [title(W, "One optimizing metric, the rest are constraints",
               "maximize accuracy subject to runtime ≤ 100 ms")]
    # scatter of models
    pts = [
        (70, 0.91, "A"), (55, 0.86, "B"), (88, 0.94, "C"),
        (165, 0.97, "D"), (110, 0.89, "E"), (150, 0.84, "F"),
    ]
    x0, y0, x1, y1 = 80, 70, 660, 220
    b.append(axes(x0, y0, x1, y1, "runtime (ms)", "accuracy"))
    thresh = mapxy(100, 0, 200, x0, x1)
    b.append(seg(thresh, y0, thresh, y1, RED, 1.4, dash="5 4"))
    b.append(txt(thresh + 8, y0 + 14, "100 ms  ·  satisficing", 11, anchor="start", fill=RED, weight="600"))
    for rt, acc, name in pts:
        x = mapxy(rt, 0, 200, x0, x1)
        y = mapxy(acc, 0.8, 1.0, y1, y0)
        ok = rt <= 100
        role = "out" if (ok and acc > 0.91) else ("conv" if ok else "input")
        fill, stroke = C[role]
        b.append(circle(x, y, 10, fill=fill, stroke=stroke, sw=1.5))
        b.append(txt(x, y + 4, name, 10, weight="600"))
    b.append(caption(W / 2, H - 24, [
        "Model D is most accurate but too slow. Among models that pass the threshold, pick the highest accuracy.",
    ]))
    write(OUT, "str-metrics.svg", W, H, b)


def fig_str_splits():
    W, H = 720, 260
    b = [title(W, "Split percentages depend on dataset size",
               "dev and test need enough examples, not a fixed 20%")]
    rows = [
        ("~10 k examples", [("train 60%", 0.60, "conv"), ("dev 20%", 0.20, "fc"), ("test 20%", 0.20, "out")]),
        ("~1 M examples", [("train 98%", 0.98, "conv"), ("dev 1%", 0.01, "fc"), ("test 1%", 0.01, "out")]),
    ]
    y = 90
    for lab, parts in rows:
        b.append(txt(40, y - 8, lab, 12, anchor="start", weight="600"))
        x = 40
        for name, frac, role in parts:
            w = max(frac * 640, 36)
            b.append(rect(x, y, w, 36, role))
            b.append(txt(x + w / 2, y + 24, name, 11, weight="600"))
            x += w
        y += 80
    write(OUT, "str-splits.svg", W, H, b)


def fig_str_biasvar():
    W, H = 720, 300
    b = [title(W, "Avoidable bias versus variance",
               "compare training error to a Bayes / human proxy, then to dev error")]
    levels = [
        ("Bayes / human", 0.01, "out"),
        ("training error", 0.08, "conv"),
        ("dev error", 0.09, "fc"),
    ]
    x0, y0, bw = 180, 230, 70
    xs = [220, 370, 520]
    for (lab, err, role), x in zip(levels, xs):
        h = err / 0.12 * 140
        b.append(rect(x, y0 - h, bw, h, role))
        b.append(txt(x + bw / 2, y0 + 18, lab, 11, fill=MUTED))
        b.append(txt(x + bw / 2, y0 - h - 12, f"{err*100:.0f}%", 12, weight="600"))
    b.append(arrow(xs[0] + bw + 8, 180, xs[1] - 8, 180, color=RED))
    b.append(txt((xs[0] + xs[1] + bw) / 2, 168, "avoidable bias", 11, fill=RED, weight="600"))
    b.append(arrow(xs[1] + bw + 8, 140, xs[2] - 8, 140, color=BLUE))
    b.append(txt((xs[1] + xs[2] + bw) / 2, 128, "variance", 11, fill=BLUE, weight="600"))
    write(OUT, "str-biasvar.svg", W, H, b)


def fig_str_errors():
    W, H = 680, 280
    b = [title(W, "Error analysis ranks the possible payoff",
               "100 misclassified cat-dev images, categories may overlap")]
    cats = [("dogs", 0.08, "input"), ("big cats", 0.43, "fc"), ("blurry", 0.61, "out")]
    y = 90
    for name, frac, role in cats:
        b.append(txt(40, y + 18, name, 12, anchor="start", weight="600"))
        b.append(rect(140, y, frac * 420, 28, role))
        b.append(txt(140 + frac * 420 + 10, y + 18, f"{int(frac*100)}%", 12, anchor="start"))
        y += 48
    b.append(caption(W / 2, H - 28, [
        "Fixing dogs can cut at most 8% of current errors; blur is the high-ceiling category.",
    ]))
    write(OUT, "str-errors.svg", W, H, b)


def fig_str_mismatch():
    W, H = 760, 260
    b = [title(W, "Keep dev/test on the target distribution",
               "extra off-distribution data can go in train, not in the target sets")]
    b.append(panel(40, 80, 280, 130, label="plenty of web photos"))
    b.append(panel(400, 80, 320, 130, label="scarce mobile photos  ·  the real task"))
    b.append(box(70, 120, 100, 50, "conv", "train OK"))
    b.append(box(190, 120, 100, 50, "input", "not the target"))
    b.append(box(430, 120, 80, 50, "conv", "train"))
    b.append(box(530, 120, 70, 50, "fc", "dev"))
    b.append(box(620, 120, 70, 50, "out", "test"))
    write(OUT, "str-mismatch.svg", W, H, b)


def fig_str_ladder():
    W, H = 720, 300
    b = [title(W, "Error ladder with mismatched data",
               "the training-dev set separates variance from data mismatch")]
    rungs = [
        ("human / Bayes", 0.5, "out"),
        ("training error", 1.0, "conv"),
        ("training-dev error", 1.5, "flat"),
        ("dev error", 8.0, "fc"),
        ("test error", 8.5, "input"),
    ]
    y = 80
    for lab, err, role in rungs:
        b.append(rect(200, y, 40 + err * 40, 28, role))
        b.append(txt(190, y + 18, lab, 12, anchor="end", weight="600"))
        b.append(txt(250 + err * 40, y + 18, f"{err:.1f}%", 12, anchor="start"))
        y += 38
    b.append(txt(560, 118, "small gaps", 11, fill=GREEN, weight="600"))
    b.append(txt(560, 134, "low variance", 11, fill=GREEN))
    b.append(txt(560, 198, "large gap", 11, fill=RED, weight="600"))
    b.append(txt(560, 214, "data mismatch", 11, fill=RED))
    write(OUT, "str-ladder.svg", W, H, b)


def fig_str_transfer():
    W, H = 760, 240
    b = [title(W, "Transfer learning is sequential",
               "pretrain where the data is, fine-tune where the task is")]
    b.append(box(40, 100, 160, 70, "input", "task A", "lots of data", size=13))
    b.append(arrow(210, 135, 250, 135))
    b.append(box(258, 100, 180, 70, "conv", "shared features", "early layers kept", size=12))
    b.append(arrow(448, 135, 488, 135))
    b.append(box(496, 100, 220, 70, "out", "task B", "small data, new head", size=13))
    write(OUT, "str-transfer.svg", W, H, b)


def fig_str_multitask():
    W, H = 720, 280
    b = [title(W, "Multitask: one backbone, several labels",
               "independent sigmoids, not a softmax — several objects can be present")]
    b.append(box(40, 110, 120, 70, "input", "image"))
    b.append(arrow(168, 145, 210, 145))
    b.append(box(218, 90, 150, 110, "conv", "shared net"))
    heads = [("pedestrian", "out"), ("car", "out"), ("light", "out")]
    for i, (lab, role) in enumerate(heads):
        y = 80 + i * 50
        b.append(arrow(376, 145, 430, y + 18))
        b.append(box(438, y, 130, 36, role, lab, size=12))
    write(OUT, "str-multitask.svg", W, H, b)


def fig_str_e2e():
    W, H = 760, 280
    b = [title(W, "Pipeline versus end-to-end",
               "end-to-end needs enough labeled (X, Y) pairs")]
    b.append(txt(40, 80, "pipeline", 13, anchor="start", weight="600"))
    steps = [("input", "audio"), ("fc", "features"), ("conv", "phonemes"),
             ("flat", "words"), ("out", "text")]
    x = 40
    for role, lab in steps:
        b.append(box(x, 96, 90, 40, role, lab, size=11))
        x += 110
    for i in range(4):
        b.append(arrow(40 + 90 + i * 110 + 4, 116, 40 + (i + 1) * 110 - 4, 116))
    b.append(txt(40, 180, "end-to-end", 13, anchor="start", weight="600"))
    b.append(box(40, 196, 90, 40, "input", "audio", size=11))
    b.append(arrow(138, 216, 430, 216))
    b.append(box(438, 196, 90, 40, "out", "text", size=11))
    write(OUT, "str-e2e.svg", W, H, b)


# =========================================================== improving NN


def fig_imp_ushape():
    W, H = 720, 330
    b = [title(W, "Classical bias–variance U-curve",
               "more capacity lowers bias and raises variance; the sum is U-shaped")]
    x0, y0, x1, y1 = 80, 70, 660, 250
    b.append(axes(x0, y0, x1, y1, "model complexity", "error"))
    n = 80
    bias, var, tot = [], [], []
    for i in range(n + 1):
        t = i / n
        bi = 0.85 * (1 - t) ** 2 + 0.05
        va = 0.75 * t ** 2 + 0.05
        to = bi + va
        x = mapxy(t, 0, 1, x0, x1)
        bias.append((x, mapxy(bi, 0, 1.2, y1, y0)))
        var.append((x, mapxy(va, 0, 1.2, y1, y0)))
        tot.append((x, mapxy(to, 0, 1.2, y1, y0)))
    b.append(polyline(bias, BLUE, 2.2))
    b.append(polyline(var, RED, 2.2))
    b.append(polyline(tot, GREEN, 2.6))
    b.append(txt(x0 + 20, 90, "bias²", 12, anchor="start", fill=BLUE, weight="600"))
    b.append(txt(x1 - 20, 90, "variance", 12, anchor="end", fill=RED, weight="600"))
    b.append(txt((x0 + x1) / 2, 96, "total error", 12, fill=GREEN, weight="600"))
    write(OUT, "imp-ushape.svg", W, H, b)


def fig_imp_poly():
    W, H = 760, 280
    b = [title(W, "Same data, three capacities",
               "underfit misses the curve; overfit chases noise")]
    xs = [i / 12 for i in range(13)]
    true = [0.35 + 0.55 * math.sin(1.4 * math.pi * x) for x in xs]
    noise = [0.08, -0.06, 0.1, -0.04, 0.07, -0.09, 0.05, 0.11, -0.07, 0.04, -0.08, 0.06, -0.05]
    ys = [t + n for t, n in zip(true, noise)]

    def to_pts(vals, x0, y0, x1, y1):
        return [(mapxy(x, 0, 1, x0, x1), mapxy(v, -0.1, 1.2, y1, y0))
                for x, v in zip(xs, vals)]

    panels = [
        ("high bias", BLUE, [0.2 + 0.55 * x for x in xs]),
        ("about right", GREEN, true),
        ("high variance", RED, [y + 0.25 * math.sin(10 * math.pi * x) for x, y in zip(xs, ys)]),
    ]
    for i, (lab, color, curve) in enumerate(panels):
        x0 = 30 + i * 245
        b.append(panel(x0, 70, 230, 170, label=lab))
        ax, ay, bx, by = x0 + 16, 90, x0 + 214, 220
        b.append(seg(ax, by, bx, by, LINE, 1.0))
        b.append(seg(ax, ay, ax, by, LINE, 1.0))
        b.append(polyline(to_pts(curve, ax, ay, bx, by), color, 2.2))
        for x, y in zip(xs, ys):
            b.append(circle(mapxy(x, 0, 1, ax, bx), mapxy(y, -0.1, 1.2, by, ay),
                            3.0, fill=INK, stroke=INK))
    write(OUT, "imp-poly.svg", W, H, b)


def fig_imp_recipe():
    W, H = 640, 340
    b = [title(W, "Fix bias, then variance",
               "in the big-data regime the two knobs barely fight")]
    boxes = [
        (220, 70, "conv", "train the model"),
        (220, 140, "fc", "high train error?  →  bigger net"),
        (220, 210, "flat", "high dev gap?  →  more data / L2 / dropout"),
        (220, 280, "out", "low bias and low variance"),
    ]
    for x, y, role, lab in boxes:
        b.append(box(x, y, 280, 48, role, lab, size=12))
    for i in range(3):
        b.append(arrow(360, boxes[i][1] + 50, 360, boxes[i + 1][1] - 4))
    write(OUT, "imp-recipe.svg", W, H, b)


def fig_imp_dropout():
    W, H = 720, 300
    b = [title(W, "Inverted dropout",
               "keep each unit with probability p, then divide survivors by p")]
    ymid = 160
    p0, body = col_nodes(80, 3, ymid, r=12, role="input")
    b.append(body)
    keep = [True, False, True, False]
    pts = []
    for i, k in enumerate(keep):
        y = ymid - 1.5 * 32 + i * 32
        role = "conv" if k else "input"
        b.append(node(260, y, 12, role, "·" if k else "0"))
        if not k:
            b.append(seg(248, y - 12, 272, y + 12, RED, 1.6))
        pts.append((260, y))
    b.append(wires(p0, [p for p, k in zip(pts, keep) if k]))
    p2, body = col_nodes(460, 2, ymid, r=12, role="out")
    b.append(body)
    b.append(wires([p for p, k in zip(pts, keep) if k], p2))
    b.append(txt(260, 250, "dropped units send 0", 11, fill=RED))
    b.append(txt(460, 250, "survivors × 1/p", 11, fill=MUTED))
    write(OUT, "imp-dropout.svg", W, H, b)


def fig_imp_normalize():
    W, H = 720, 280
    b = [title(W, "Normalizing inputs rounds the cost bowl",
               "features on wildly different scales make a long ravine")]
    b.append(panel(40, 70, 300, 170, label="raw  ·  x₁ ≫ x₂"))
    b.append(nest_ellipses(190, 165, 5, 120, 28, LINE))
    b.append(polyline([(90, 150), (130, 180), (170, 145), (200, 168), (230, 160)], RED, 2.0))
    b.append(panel(380, 70, 300, 170, label="zero-mean, unit-variance"))
    b.append(nest_ellipses(530, 165, 5, 70, 70, LINE))
    b.append(polyline([(480, 200), (500, 185), (515, 175), (525, 168), (530, 165)], GREEN, 2.2))
    write(OUT, "imp-normalize.svg", W, H, b)


def fig_imp_vanish():
    W, H = 720, 300
    b = [title(W, "Vanishing and exploding activations",
               "repeated multiply by 0.5 or 1.5, layer after layer")]
    x0, y0, x1, y1 = 70, 70, 660, 230
    b.append(axes(x0, y0, x1, y1, "layer ℓ", "|activation|  (log)"))
    n = 12
    up, down = [], []
    for i in range(n):
        x = mapxy(i, 0, n - 1, x0, x1)
        up.append((x, mapxy(math.log10(1.5 ** i), -4, 3, y1, y0)))
        down.append((x, mapxy(math.log10(max(0.5 ** i, 1e-6)), -4, 3, y1, y0)))
    b.append(polyline(up, RED, 2.4))
    b.append(polyline(down, BLUE, 2.4))
    b.append(txt(x1 - 8, 88, "W ≈ 1.5 I   exploding", 12, anchor="end", fill=RED, weight="600"))
    b.append(txt(x1 - 8, 108, "W ≈ 0.5 I   vanishing", 12, anchor="end", fill=BLUE, weight="600"))
    write(OUT, "imp-vanish.svg", W, H, b)


def fig_imp_gradcheck():
    W, H = 680, 300
    b = [title(W, "Two-sided numerical gradient",
               "slope of the chord from J(θ−ε) to J(θ+ε) approximates J'(θ)")]
    x0, y0, x1, y1 = 80, 70, 620, 230

    def J(t):
        return 0.15 + 0.7 * (t - 0.35) ** 2

    n = 60
    pts = []
    for i in range(n + 1):
        t = i / n
        pts.append((mapxy(t, 0, 1, x0, x1), mapxy(J(t), 0, 1, y1, y0)))
    b.append(axes(x0, y0, x1, y1, "θ", "J(θ)"))
    b.append(polyline(pts, GREEN, 2.2))
    t, eps = 0.35, 0.18
    xa, xb = mapxy(t - eps, 0, 1, x0, x1), mapxy(t + eps, 0, 1, x0, x1)
    ya, yb = mapxy(J(t - eps), 0, 1, y1, y0), mapxy(J(t + eps), 0, 1, y1, y0)
    b.append(seg(xa, ya, xb, yb, RED, 2.0))
    b.append(circle(xa, ya, 4, fill=RED, stroke=RED))
    b.append(circle(xb, yb, 4, fill=RED, stroke=RED))
    xt = mapxy(t, 0, 1, x0, x1)
    yt = mapxy(J(t), 0, 1, y1, y0)
    b.append(circle(xt, yt, 4.5, fill=BLUE, stroke=BLUE))
    b.append(txt(xa, ya - 12, "θ − ε", 11, fill=RED, weight="600"))
    b.append(txt(xb, yb - 12, "θ + ε", 11, fill=RED, weight="600"))
    b.append(txt(xt + 10, yt + 18, "θ", 11, anchor="start", fill=BLUE, weight="600"))
    write(OUT, "imp-gradcheck.svg", W, H, b)


# =========================================================== hyperparams


def fig_hp_priority():
    W, H = 680, 300
    b = [title(W, "Tune the high-impact knobs first",
               "learning rate dominates; Adam's β₂ almost never needs a search")]
    tiers = [
        (120, "out", "α  ·  learning rate"),
        (220, "fc", "β₁, batch size, width"),
        (320, "conv", "depth, schedule, λ"),
        (420, "input", "Adam β₂, ε"),
    ]
    y = 80
    for w, role, lab in tiers:
        x = (W - w) / 2
        b.append(rect(x, y, w, 36, role))
        b.append(txt(W / 2, y + 24, lab, 12, weight="600"))
        y += 48
    write(OUT, "hp-priority.svg", W, H, b)


def fig_hp_panda():
    W, H = 720, 240
    b = [title(W, "Panda versus caviar",
               "one carefully babysat run, or many runs in parallel")]
    b.append(panel(40, 80, 300, 120, label="panda  ·  scarce compute"))
    b.append(box(90, 120, 200, 44, "out", "one model, nudged daily", size=12))
    b.append(panel(380, 80, 300, 120, label="caviar  ·  lots of GPUs"))
    for i in range(3):
        for j in range(3):
            b.append(rect(410 + j * 80, 110 + i * 26, 64, 20, "conv" if (i + j) % 2 == 0 else "fc"))
    write(OUT, "hp-panda.svg", W, H, b)


def fig_hp_grid():
    W, H = 720, 300
    b = [title(W, "Grid search wastes budget on useless axes",
               "random search samples the important axis more densely")]
    b.append(panel(40, 70, 310, 190, label="grid"))
    b.append(panel(380, 70, 310, 190, label="random"))
    # grid
    for i in range(4):
        for j in range(4):
            x = 80 + j * 60
            y = 100 + i * 36
            b.append(circle(x, y, 5, fill=C["fc"][0], stroke=C["fc"][1], sw=1.3))
    b.append(txt(195, 250, "4 × 4 = 16 runs, many redundant", 11, fill=MUTED))
    # random — denser along x (important), sparse along y
    rng = [0.12, 0.31, 0.48, 0.07, 0.88, 0.63, 0.21, 0.74, 0.55, 0.93, 0.39, 0.16]
    rny = [0.50, 0.62, 0.41, 0.55, 0.48, 0.70, 0.44, 0.58, 0.36, 0.52, 0.66, 0.47]
    for u, v in zip(rng, rny):
        b.append(circle(410 + u * 250, 100 + v * 130, 5,
                        fill=C["conv"][0], stroke=C["conv"][1], sw=1.3))
    b.append(txt(535, 250, "same 12 runs, better coverage of α", 11, fill=MUTED))
    write(OUT, "hp-grid.svg", W, H, b)


def fig_hp_bn():
    W, H = 780, 240
    b = [title(W, "Batch norm sits between the linear map and the nonlinearity",
               "normalize the batch, then scale and shift with learned γ, β")]
    stages = [
        ("input", "A[ℓ−1]"),
        ("fc", "Z = WA"),
        ("flat", "normalize"),
        ("conv", "γ ⊙ + β"),
        ("out", "g(·) → A[ℓ]"),
    ]
    x, y = 24, 120
    xs = []
    for role, lab in stages:
        xs.append(x)
        b.append(box(x, y, 120, 50, role, lab, size=12))
        x += 152
    for a, c in zip(xs, xs[1:]):
        b.append(arrow(a + 128, y + 25, c - 8, y + 25))
    write(OUT, "hp-bn.svg", W, H, b)


# =========================================================== math


def fig_math_derivative():
    W, H = 680, 300
    b = [title(W, "The derivative is the limit of secants",
               "shrink h until the chord matches the tangent")]
    x0, y0, x1, y1 = 70, 70, 620, 230

    def f(t):
        return 0.15 + 0.55 * t * t

    pts = [(mapxy(t, 0, 1, x0, x1), mapxy(f(t), 0, 0.8, y1, y0))
           for t in [i / 50 for i in range(51)]]
    b.append(axes(x0, y0, x1, y1, "x", "f(x)"))
    b.append(polyline(pts, GREEN, 2.2))
    a, h = 0.35, 0.4
    xa, xb = mapxy(a, 0, 1, x0, x1), mapxy(a + h, 0, 1, x0, x1)
    ya, yb = mapxy(f(a), 0, 0.8, y1, y0), mapxy(f(a + h), 0, 0.8, y1, y0)
    b.append(seg(xa, ya, xb, yb, RED, 2.0, dash="5 4"))
    b.append(circle(xa, ya, 4, fill=BLUE, stroke=BLUE))
    b.append(circle(xb, yb, 4, fill=RED, stroke=RED))
    b.append(txt(xa, ya - 12, "x", 12, fill=BLUE, weight="600"))
    b.append(txt(xb, yb - 12, "x + h", 12, fill=RED, weight="600"))
    write(OUT, "math-derivative.svg", W, H, b)


def fig_math_chain():
    W, H = 720, 220
    b = [title(W, "Chain rule", "multiply the local derivatives along the composition")]
    y = 120
    labs = [("input", "x"), ("conv", "h(x)"), ("fc", "g(h)"), ("out", "f(g(h))")]
    x = 50
    xs = []
    for role, lab in labs:
        xs.append(x)
        b.append(box(x, y, 110, 50, role, lab, size=13))
        x += 160
    for a, c in zip(xs, xs[1:]):
        b.append(arrow(a + 118, y + 25, c - 8, y + 25))
    b.append(txt(W / 2, 190, "df/dx = f' · g' · h'", 13, fill=MUTED, weight="600"))
    write(OUT, "math-chain.svg", W, H, b)


def fig_math_grad():
    W, H = 640, 300
    b = [title(W, "The gradient points uphill",
               "descent walks the opposite arrows")]
    cx, cy = 320, 175
    b.append(nest_ellipses(cx, cy, 5, 180, 100, LINE))
    for ang in range(0, 360, 45):
        rad = math.radians(ang)
        x1 = cx + 70 * math.cos(rad)
        y1 = cy + 40 * math.sin(rad)
        x2 = cx + 120 * math.cos(rad)
        y2 = cy + 70 * math.sin(rad)
        b.append(arrow(x1, y1, x2, y2, color=RED, sw=1.5))
    b.append(circle(cx, cy, 5, fill=GREEN, stroke=GREEN))
    b.append(txt(cx + 14, cy + 4, "min", 11, anchor="start", fill=GREEN, weight="600"))
    b.append(txt(cx + 140, 90, "∇f", 13, fill=RED, weight="600"))
    write(OUT, "math-grad.svg", W, H, b)


def fig_math_matmul():
    W, H = 720, 260
    b = [title(W, "Matrix multiply  (m × k)(k × n) → (m × n)",
               "row i of A dotted with column j of B")]
    from svgkit import matrix
    A = [[1, 2], [3, 4], [5, 6]]
    Bmat = [[7, 8, 9], [1, 0, 2]]
    Cmat = [[9, 8, 13], [25, 24, 35], [41, 40, 57]]

    def rowfill(v, i, j):
        return C["conv"][0] if i == 0 else "#ffffff"

    def colfill(v, i, j):
        return C["fc"][0] if j == 1 else "#ffffff"

    def outfill(v, i, j):
        return C["out"][0] if (i, j) == (0, 1) else "#ffffff"

    b.append(matrix(40, 90, A, cell=26, fills=rowfill))
    b.append(txt(80, 80, "A", 12, weight="600"))
    b.append(txt(175, 140, "×", 16, fill=MUTED))
    b.append(matrix(200, 110, Bmat, cell=26, fills=colfill))
    b.append(txt(239, 100, "B", 12, weight="600"))
    b.append(txt(330, 150, "=", 16, fill=MUTED))
    b.append(matrix(360, 90, Cmat, cell=26, fills=outfill))
    b.append(txt(399, 80, "AB", 12, weight="600"))
    b.append(caption(W / 2, H - 28, [
        "Highlighted: row 1 of A · column 2 of B = 8, written into (1, 2) of the product.",
    ]))
    write(OUT, "math-matmul.svg", W, H, b)


def fig_math_bayes():
    W, H = 640, 280
    b = [title(W, "Bayes: posterior from likelihood and prior",
               "P(A|B) = P(B|A) P(A) / P(B)")]
    b.append(ellipse(260, 160, 140, 80, fill=C["input"][0], stroke=C["input"][1], sw=1.5))
    b.append(ellipse(380, 160, 140, 80, fill=C["conv"][0], stroke=C["conv"][1], sw=1.5))
    # overlap hint
    b.append(txt(260, 160, "A", 16, weight="600"))
    b.append(txt(380, 160, "B", 16, weight="600"))
    b.append(txt(320, 160, "A∩B", 11, fill=RED, weight="600"))
    b.append(caption(W / 2, H - 36, [
        "The posterior reweights the prior by how well A explains the evidence B.",
    ]))
    write(OUT, "math-bayes.svg", W, H, b)


def fig_math_confusion():
    W, H = 620, 300
    b = [title(W, "Confusion matrix", "four counts, then precision and recall")]
    cells = [
        (140, 90, "out", "TP"),
        (310, 90, "fc", "FN"),
        (140, 170, "flat", "FP"),
        (310, 170, "input", "TN"),
    ]
    for x, y, role, lab in cells:
        b.append(box(x, y, 150, 64, role, lab, size=16))
    b.append(txt(215, 80, "pred +", 12, fill=MUTED))
    b.append(txt(385, 80, "pred −", 12, fill=MUTED))
    b.append(txt(124, 132, "actual +", 11, anchor="end", fill=MUTED))
    b.append(txt(124, 212, "actual −", 11, anchor="end", fill=MUTED))
    write(OUT, "math-confusion.svg", W, H, b)


def fig_math_entropy():
    W, H = 720, 300
    b = [title(W, "Entropy is highest for a fair coin",
               "H = −Σ p log p  (bits)")]
    x0, y0, x1, y1 = 70, 70, 660, 230
    b.append(axes(x0, y0, x1, y1, "P(heads)", "H (bits)"))
    pts = []
    for i in range(81):
        p = i / 80
        q = 1 - p
        if 0 < p < 1:
            h = -p * math.log(p, 2) - q * math.log(q, 2)
        else:
            h = 0.0
        pts.append((mapxy(p, 0, 1, x0, x1), mapxy(h, 0, 1.05, y1, y0)))
    b.append(polyline(pts, PURPLE, 2.4))
    b.append(circle(mapxy(0.5, 0, 1, x0, x1), mapxy(1.0, 0, 1.05, y1, y0),
                    5, fill=RED, stroke=RED))
    b.append(txt(mapxy(0.5, 0, 1, x0, x1) + 10, mapxy(1.0, 0, 1.05, y1, y0),
                 "fair coin, 1 bit", 11, anchor="start", fill=RED, weight="600"))
    write(OUT, "math-entropy.svg", W, H, b)


def fig_math_convex():
    W, H = 720, 280
    b = [title(W, "Convex versus non-convex",
               "a chord on a convex function never dips below the graph")]
    b.append(panel(40, 70, 310, 170, label="convex  ·  unique min"))
    b.append(panel(380, 70, 310, 170, label="non-convex  ·  several basins"))
    # convex parabola
    pts = [(70 + t * 250, 210 - 140 * (1 - (2 * t - 1) ** 2)) for t in [i / 40 for i in range(41)]]
    b.append(polyline(pts, GREEN, 2.2))
    b.append(seg(pts[8][0], pts[8][1], pts[32][0], pts[32][1], RED, 1.5, dash="5 4"))
    # nonconvex
    pts2 = []
    for i in range(41):
        t = i / 40
        y = 200 - 40 * math.sin(2.2 * math.pi * t) - 30 * t
        pts2.append((410 + t * 250, y))
    b.append(polyline(pts2, RED, 2.2))
    write(OUT, "math-convex.svg", W, H, b)


# ---------------------------------------------------------------------- main

FIGURES = [
    fig_log_neuron, fig_log_compgraph, fig_log_vectorized, fig_log_gd,
    fig_sha_arch, fig_sha_forward, fig_sha_softmax,
    fig_dnn_arch, fig_dnn_hierarchy, fig_dnn_block,
    fig_term_neuron, fig_term_activations, fig_term_linear, fig_term_tensor, fig_term_symmetry,
    fig_opt_paths, fig_opt_cost, fig_opt_ewa, fig_opt_biascorr,
    fig_opt_momentum, fig_opt_rmsprop, fig_opt_adam, fig_opt_lrdecay, fig_opt_saddle,
    fig_str_loop, fig_str_ortho, fig_str_metrics, fig_str_splits, fig_str_biasvar,
    fig_str_errors, fig_str_mismatch, fig_str_ladder, fig_str_transfer,
    fig_str_multitask, fig_str_e2e,
    fig_imp_ushape, fig_imp_poly, fig_imp_recipe, fig_imp_dropout,
    fig_imp_normalize, fig_imp_vanish, fig_imp_gradcheck,
    fig_hp_priority, fig_hp_panda, fig_hp_grid, fig_hp_bn,
    fig_math_derivative, fig_math_chain, fig_math_grad, fig_math_matmul,
    fig_math_bayes, fig_math_confusion, fig_math_entropy, fig_math_convex,
]


def build():
    for f in FIGURES:
        f()


if __name__ == "__main__":
    build()
    print(f"{len(FIGURES)} basics figures written")
