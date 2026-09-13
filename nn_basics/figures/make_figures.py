"""Generate the architecture diagrams used by deep_convolutional_nn.md.

Run:  python figures/make_figures.py
That entry point also builds the convolutional_nn_foundations.md figures.
Every figure is written as a standalone SVG next to this script.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from svgkit import (  # noqa: E402
    ARCH_LEGEND, C, INK, LINE, MUTED, RED,
    arrow, circle, legend, lines, path, pipeline as _pipeline, rect, seg, title,
    txt, write,
)

OUT = pathlib.Path(__file__).parent


def pipeline(fname, head, stages, ops, **kw):
    """Thin wrapper so the figure functions below stay unchanged."""
    _pipeline(OUT, fname, head, stages, ops, **kw)


def fig_lenet5():
    pipeline(
        "classic-lenet5.svg",
        "LeNet-5  ·  handwritten digits  ·  ~60 K parameters",
        [
            dict(kind="vol", s=32, c=1, shape="32×32×1", name="Input", role="input"),
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
            "convolution 5×5×1,\n6 filters",
            "avg pool\nf=2, s=2",
            "convolution 5×5×6,\n16 filters",
            "avg pool\nf=2, s=2",
            "flatten\n5·5·16",
            "fully\nconnected",
            "fully\nconnected",
            "softmax,\n10 digits",
        ],
        sub="spatial size shrinks 32→28→14→10→5, channels grow 1→6→16",
        leg=ARCH_LEGEND,
        note="Original paper used sigmoid/tanh, average pooling, and a non-linearity after pooling.",
    )


def fig_alexnet():
    pipeline(
        "classic-alexnet.svg",
        "AlexNet  ·  ImageNet 1000 classes  ·  ~60 M parameters",
        [
            dict(kind="vol", s=227, c=3, shape="227×227×3", name="Input", role="input"),
            dict(kind="vol", s=55, c=96, shape="55×55×96", name="CONV", role="conv"),
            dict(kind="vol", s=27, c=96, shape="27×27×96", name="POOL", role="pool"),
            dict(kind="vol", s=27, c=256, shape="27×27×256", name="CONV", role="conv"),
            dict(kind="vol", s=13, c=256, shape="13×13×256", name="POOL", role="pool"),
            dict(kind="vol", s=13, c=384, shape="13×13×384", name="CONV", role="conv"),
            dict(kind="vol", s=13, c=256, shape="13×13×256", name="CONV", role="conv"),
            dict(kind="vol", s=6, c=256, shape="6×6×256", name="POOL", role="pool"),
            dict(kind="vec", u=9216, shape="9216", name="Flatten", role="flat"),
            dict(kind="vec", u=4096, shape="4096", name="FC", role="fc"),
            dict(kind="vec", u=1000, shape="1000", name="Output", role="out"),
        ],
        [
            "convolution 11×11×3,\n96 filters, s=4",
            "max pool\nf=3, s=2",
            "convolution 5×5×96,\n256 filters, same",
            "max pool\nf=3, s=2",
            "2 × conv 3×3×256,\n384 filters",
            "convolution 3×3×384,\n256 filters",
            "max pool\nf=3, s=2",
            "flatten\n6·6·256",
            "FC 4096\n×2",
            "softmax,\n1000 classes",
        ],
        sub="(227−11)/4+1 = 55 → 27 → 13 → 6;  same building blocks as LeNet-5, ~1000× more parameters",
        leg=ARCH_LEGEND,
        note="ReLU instead of sigmoid/tanh was a key change. Local Response Normalization is omitted — it is no longer used.",
    )


def fig_vgg16():
    pipeline(
        "classic-vgg16.svg",
        "VGG-16  ·  every CONV is 3×3 s=1 same, every POOL is 2×2 s=2  ·  ~138 M parameters",
        [
            dict(kind="vol", s=224, c=3, shape="224×224×3", name="Input", role="input"),
            dict(kind="vol", s=224, c=64, shape="224×224×64", name="CONV", role="conv"),
            dict(kind="vol", s=112, c=128, shape="112×112×128", name="CONV", role="conv"),
            dict(kind="vol", s=56, c=256, shape="56×56×256", name="CONV", role="conv"),
            dict(kind="vol", s=28, c=512, shape="28×28×512", name="CONV", role="conv"),
            dict(kind="vol", s=14, c=512, shape="14×14×512", name="CONV", role="conv"),
            dict(kind="vol", s=7, c=512, shape="7×7×512", name="POOL", role="pool"),
            dict(kind="vec", u=25088, shape="25088", name="Flatten", role="flat"),
            dict(kind="vec", u=4096, shape="4096", name="FC", role="fc"),
            dict(kind="vec", u=1000, shape="1000", name="Output", role="out"),
        ],
        [
            "2 × conv 3×3×3,\n64 filters",
            "pool, then\n2 × conv 128",
            "pool, then\n3 × conv 256",
            "pool, then\n3 × conv 512",
            "pool, then\n3 × conv 512",
            "max pool\nf=2, s=2",
            "flatten\n7·7·512",
            "FC 4096\n×2",
            "softmax,\n1000 classes",
        ],
        sub="pooling halves the spatial size, each conv stack doubles the channels: 64→128→256→512",
        leg=ARCH_LEGEND,
        note="The first FC layer alone holds 7·7·512·4096 ≈ 103 M of the 138 M parameters.",
    )


# ------------------------------------------------------------------- ResNet


def fig_resnet_block():
    W, H = 920, 440
    b = [title(W, "The residual block", "the shortcut is added after the linear part and before the final ReLU")]

    def row(cy, label, residual):
        plus_x = 546.0
        # on the residual row the title must clear the arc, which arches to cy-78
        o = [txt(30, cy - (102 if residual else 62), label, 13, anchor="start", weight="600")]
        # activation nodes
        for x, name in zip([72, 352, 692], ["a[l]", "a[l+1]", "a[l+2]"]):
            o.append(
                f'<circle cx="{x}" cy="{cy:g}" r="19" fill="#e5e7eb" '
                f'stroke="#6b7280" stroke-width="1.4"/>'
            )
            o.append(txt(x, cy + 4, name, 11, weight="600"))
        # layer l+1
        o.append(rect(118, cy - 20, 96, 40, "conv"))
        o.append(lines(166, cy - 3, ["linear", "W[l+1], b[l+1]"], 10.5))
        o.append(rect(240, cy - 20, 74, 40, "out"))
        o.append(txt(277, cy + 4, "ReLU", 11))
        # layer l+2
        o.append(rect(398, cy - 20, 96, 40, "conv"))
        o.append(lines(446, cy - 3, ["linear", "W[l+2], b[l+2]"], 10.5))
        o.append(rect(578, cy - 20, 74, 40, "out"))
        o.append(txt(615, cy + 4, "ReLU", 11))
        # main path
        for x1, x2 in [(91, 118), (214, 240), (314, 333), (371, 398)]:
            o.append(arrow(x1, cy, x2, cy))
        o.append(arrow(656, cy, 669, cy))
        if residual:
            o.append(
                f'<circle cx="{plus_x:g}" cy="{cy:g}" r="15" fill="#ffffff" '
                f'stroke="#dc2626" stroke-width="1.8"/>'
            )
            o.append(txt(plus_x, cy + 5, "+", 17, fill="#dc2626", weight="600"))
            o.append(arrow(494, cy, plus_x - 15, cy))
            o.append(arrow(plus_x + 15, cy, 578, cy))
            o.append(
                path(
                    f"M 72,{cy - 21:g} C 72,{cy - 78:g} {plus_x:g},{cy - 78:g} "
                    f"{plus_x:g},{cy - 17:g}",
                    stroke="#dc2626",
                    sw=1.8,
                )
            )
            o.append(txt((72 + plus_x) / 2, cy - 86, "shortcut: copy a[l] forward, past both layers", 10.5, fill="#dc2626", weight="600"))
        else:
            o.append(arrow(494, cy, 578, cy))
        return "\n".join(o)

    b.append(row(118, "Plain network — the only route is the main path", False))
    b.append(txt(W / 2, 176, "a[l+2] = g( z[l+2] )", 13, weight="600"))
    b.append(txt(W / 2, 194, "information has to survive both layers to reach a[l+2]", 10.5, fill=MUTED))

    b.append(row(320, "Residual block", True))
    b.append(txt(W / 2, 378, "a[l+2] = g( z[l+2] + a[l] )", 13, weight="600"))
    b.append(txt(W / 2, 396, "if W[l+2] = 0 and b[l+2] = 0 then a[l+2] = g(a[l]) = a[l], so the identity is easy to learn", 10.5, fill=MUTED))

    b.append(txt(W / 2, 424, "Requires z[l+2] and a[l] to have the same dimension — hence the heavy use of same convolutions, or a projection matrix Ws·a[l].", 10.5, fill=MUTED))
    write(OUT, "resnet-block.svg", W, H, b)


def fig_resnet_depth():
    W, H = 720, 372
    x0, x1, y0, y1 = 92, 660, 68, 268
    b = [title(W, "Training error versus depth", "in theory deeper is never worse; for a plain network, practice disagrees")]
    # axes
    b.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{INK}" stroke-width="1.4"/>')
    b.append(f'<line x1="{x0}" y1="{y1}" x2="{x0}" y2="{y0}" stroke="{INK}" stroke-width="1.4"/>')
    b.append(txt((x0 + x1) / 2, y1 + 34, "number of layers", 11.5, fill=INK))
    b.append(txt(0, 0, "", 1))
    b.append(txt(x0 - 46, (y0 + y1) / 2, "training error", 11.5, fill=INK, rot=-90))

    def curve(f, color, dash=None):
        pts = []
        for i in range(101):
            t = i / 100
            pts.append(f"{x0 + t * (x1 - x0):.1f},{y1 - f(t) * (y1 - y0):.1f}")
        pt_str = " ".join(pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<polyline points="{pt_str}" fill="none" stroke="{color}" '
            f'stroke-width="2.6"{da}/>'
        )

    theory = lambda t: 0.90 * math.exp(-2.2 * t) + 0.06
    plain = lambda t: 0.90 * math.exp(-3.5 * t) + 0.08 + 0.55 * t * t
    resnet = lambda t: 0.85 * math.exp(-2.6 * t) + 0.04

    b.append(curve(theory, "#16a34a", dash="7 5"))
    b.append(curve(plain, "#dc2626"))
    b.append(curve(resnet, "#2563eb"))

    b.append(txt(x1 - 6, y1 - theory(1.0) * (y1 - y0) - 12, "theory", 11.5, anchor="end", fill="#16a34a", weight="600"))
    b.append(txt(x1 - 6, y1 - plain(1.0) * (y1 - y0) - 12, "plain network, in practice", 11.5, anchor="end", fill="#dc2626", weight="600"))
    b.append(txt(x1 - 6, y1 - resnet(1.0) * (y1 - y0) + 20, "ResNet", 11.5, anchor="end", fill="#2563eb", weight="600"))

    tmin = 0.55
    mx, my = x0 + tmin * (x1 - x0), y1 - plain(tmin) * (y1 - y0)
    b.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="3.6" fill="#dc2626"/>')
    b.append(
        f'<line x1="{mx:.1f}" y1="{my + 6:.1f}" x2="{mx:.1f}" y2="{y1 - 26:g}" '
        f'stroke="#dc2626" stroke-width="1" stroke-dasharray="3 3"/>'
    )
    b.append(txt(mx, y1 - 12, "past this depth a plain net gets worse", 10.5, fill="#dc2626", weight="600"))

    b.append(txt(W / 2, H - 34, "The ResNet paper calls this the degradation problem: it is an optimization difficulty,", 10.5, fill=MUTED))
    b.append(txt(W / 2, H - 16, "not simply vanishing gradients. Skip connections make the identity mapping the default.", 10.5, fill=MUTED))
    write(OUT, "resnet-depth-error.svg", W, H, b)


def fig_resnet_arch():
    W, H = 920, 284
    b = [title(W, "Plain network → ResNet", "add a shortcut across every pair of layers; most convolutions are 3×3 same, so the dimensions already match")]
    seq = [("input", "input", 52), ("conv", "3×3", 50), ("conv", "3×3", 50), ("pool", "pool", 46),
           ("conv", "3×3", 50), ("conv", "3×3", 50), ("pool", "pool", 46),
           ("conv", "3×3", 50), ("conv", "3×3", 50), ("fc", "FC", 44), ("out", "soft\nmax", 48)]
    # each arc spans exactly the two convolutions between these two boxes
    pairs = [(0, 3), (3, 6), (6, 9)]

    gap = 22.0
    total = sum(w for _, _, w in seq) + gap * (len(seq) - 1)
    xs, x = [], (W - total) / 2
    for _, _, w in seq:
        xs.append(x)
        x += w + gap

    def row(cy, label, skips):
        # on the ResNet row the title has to clear the arcs, which arch to cy-58
        o = [txt(xs[0], cy - (72 if skips else 46), label, 13, anchor="start", weight="600")]
        for (role, name, w), x0 in zip(seq, xs):
            o.append(rect(x0, cy - 22, w, 44, role))
            rows = name.split("\n")
            o.append(lines(x0 + w / 2, cy + 4 - (len(rows) - 1) * 6, rows, 10, lh=11))
        for i in range(len(seq) - 1):
            o.append(arrow(xs[i] + seq[i][2] + 3, cy, xs[i + 1] - 3, cy))
        if skips:
            for a, c in pairs:
                xa, xc = xs[a] + seq[a][2] - 4, xs[c] + 4
                o.append(path(f"M {xa:g},{cy - 23:g} C {xa:g},{cy - 58:g} {xc:g},{cy - 58:g} {xc:g},{cy - 25:g}", stroke="#dc2626", sw=1.7))
            xa, xc = xs[0] + seq[0][2] - 4, xs[3] + 4
            o.append(txt((xa + xc) / 2, cy - 62, "skip connection", 10.5, fill="#dc2626", weight="600"))
        return "\n".join(o)

    b.append(row(108, "Plain network", False))
    b.append(row(238, "ResNet", True))
    b.append(legend(xs[0], 280, ARCH_LEGEND))
    write(OUT, "resnet-architecture.svg", W, 300, b)


# ------------------------------------------------------------- 1x1 convolution


def fig_conv1x1():
    W, H = 940, 364
    b = [title(W, "The 1×1 convolution", "a tiny fully connected network applied independently at every spatial position")]

    # ---- left: the two volumes
    b.append(txt(232, 72, "The whole volume", 12.5, weight="600"))
    b.append(rect(96, 92, 100, 116, "input"))
    b.append(txt(146, 152, "28×28×192", 11.5, weight="600"))
    b.append(txt(146, 226, "input", 11, weight="600"))
    b.append(txt(146, 242, "192 channels", 10.5, fill=MUTED))

    b.append(arrow(210, 150, 300, 150))
    b.append(lines(255, 116, ["32 filters of", "1×1×192"], 10.5, fill=MUTED))
    b.append(txt(255, 174, "+ ReLU", 10.5, fill=MUTED))

    b.append(rect(312, 92, 56, 116, "one"))
    b.append(txt(340, 152, "28×28×32", 11, weight="600"))
    b.append(txt(340, 226, "output", 11, weight="600"))
    b.append(txt(340, 242, "32 channels", 10.5, fill=MUTED))

    # one spatial position is a strip cutting across every channel of the volume
    b.append(f'<rect x="92" y="164" width="108" height="10" fill="none" stroke="{RED}" stroke-width="2"/>')
    b.append(f'<rect x="308" y="164" width="64" height="10" fill="none" stroke="{RED}" stroke-width="2"/>')
    b.append(txt(232, 274, "spatial size unchanged  ·  192 → 32 channels", 11, weight="600"))
    b.append(txt(232, 296, "the red strip is one spatial position and all of its channels", 10, fill=MUTED))

    b.append(seg(452, 66, 452, 332, LINE, 1.0, dash="4 4"))

    # ---- right: the same position, expanded along the depth axis
    b.append(txt(690, 72, "What happens at one spatial position", 12.5, weight="600"))

    x0, y0, s, dx, dy = 528.0, 240.0, 24.0, 14.0, -14.0
    shown = [0, 1, 2, 3, 4, 5, 6, 7, 10]  # a gap stands in for the other 183

    # back to front, so the nearest channel ends up on top of the stack
    for i in reversed(shown):
        b.append(rect(x0 + i * dx, y0 + i * dy, s, s, "input", rx=2, sw=1.1))
    # dots along the same diagonal stand in for the channels not drawn
    for i in (8.3, 8.9, 9.5):
        b.append(circle(x0 + i * dx + s / 2, y0 + i * dy + s / 2, 2.2,
                        fill=MUTED, stroke="none", sw=0))

    # the axis the stack recedes along
    b.append(arrow(508, 226, 600, 134, "#c7cbd1", 1.2))
    b.append(txt(534, 166, "depth", 9.5, fill=MUTED, rot=-45))

    nx, ny = 790.0, 186.0
    # every channel at this position feeds the same neuron
    for i in shown:
        b.append(seg(x0 + i * dx + s / 2, y0 + i * dy + s, nx - 24, ny, "#b9bec6", 0.9))

    b.append(txt(518, 258, "channel 1", 9.5, anchor="end", fill=MUTED))
    b.append(txt(700, 100, "channel 192", 9.5, anchor="start", fill=MUTED))
    b.append(txt(654, 300, "the 192 values stacked through the depth of the volume", 10.5, fill=MUTED))

    b.append(circle(nx, ny, 27, fill=C["out"][0], stroke=C["out"][1], sw=1.6))
    b.append(lines(nx, ny - 4, ["Σ w·x", "+ b, ReLU"], 9.5, lh=12))
    b.append(lines(nx, ny + 46, ["one filter", "= one neuron"], 10, fill=MUTED))

    b.append(arrow(nx + 30, ny, 862, ny))
    b.append(rect(866, ny - 8, 26, 16, "out", rx=2, sw=1.1))
    b.append(txt(879, ny + 30, "1 number", 10, fill=MUTED))

    b.append(txt(690, 322, "32 filters ⇒ 32 numbers here ⇒ 32 output channels", 10.5, fill=MUTED))

    b.append(txt(W / 2, H - 10, "Pooling reduces nH and nW but not nC.  A 1×1 convolution reduces nC but not nH and nW.", 11, fill=INK, weight="600"))
    write(OUT, "conv1x1.svg", W, H, b)


# ------------------------------------------------------------------ Inception


def _inception(fname, head, sub, branches, note):
    """Left-to-right inception diagram.

    The input sits on the left, each branch is a flush row of sequential stages,
    and the concatenation on the right is a single bar whose segments are sized
    by channel count — so the bar is literally the branch outputs stacked up.

    branches: dicts with stack=[(role, [lines]), ...], ch=<output channels>, and
              optionally seg=<role> to colour the bar segment by the branch's
              defining stage rather than by its last one.
    """
    W = 940
    RH = 62.0
    y0 = 92.0
    col_x0, col_x1 = 250.0, 576.0
    bar_x, bar_w = 640.0, 84.0

    n = len(branches)
    kmax = max(len(br["stack"]) for br in branches)
    gap = 26.0 if kmax > 1 else 0.0
    colw = (col_x1 - col_x0 - gap * (kmax - 1)) / kmax
    total_h = n * RH
    cy_all = y0 + total_h / 2

    b = [title(W, head, sub)]

    in_h = 132.0
    b.append(rect(56, cy_all - in_h / 2, 84, in_h, "input"))
    b.append(lines(98, cy_all - 4, ["input", "28×28×192"], 11, weight="600"))

    total_ch = sum(br["ch"] for br in branches)
    seg_y = y0
    for i, br in enumerate(branches):
        row_y = y0 + i * RH
        row_cy = row_y + RH / 2
        b.append(arrow(144, cy_all, col_x0 - 4, row_cy))

        k = len(br["stack"])
        for j, (role, rows) in enumerate(br["stack"]):
            x = col_x0 + j * (colw + gap)
            b.append(rect(x, row_y, colw, RH, role, rx=2))
            b.append(lines(x + colw / 2, row_cy + 4 - (len(rows) - 1) * 6, rows,
                           10.5, lh=12))
            if j < k - 1:
                b.append(arrow(x + colw + 3, row_cy, x + colw + gap - 3, row_cy))

        # each branch feeds the segment of the bar that its channels occupy
        seg_h = br["ch"] / total_ch * total_h
        last_x = col_x0 + (k - 1) * (colw + gap) + colw
        b.append(arrow(last_x + 4, row_cy, bar_x - 5, seg_y + seg_h / 2))
        b.append(rect(bar_x, seg_y, bar_w, seg_h,
                      br.get("seg", br["stack"][-1][0]), rx=0, sw=1.2))
        b.append(txt(bar_x + bar_w / 2, seg_y + seg_h / 2 + 4, str(br["ch"]), 11,
                     weight="600"))
        seg_y += seg_h

    b.append(rect(bar_x, y0, bar_w, total_h, "concat", rx=0, sw=1.8, fill="none"))
    b.append(txt(bar_x + bar_w / 2, y0 - 12, "concatenate", 11, weight="600"))

    lx = bar_x + bar_w + 22
    b.append(txt(lx, cy_all - 14, "channel concatenation", 11, anchor="start", weight="600"))
    b.append(txt(lx, cy_all + 10, "28×28×256", 14, anchor="start", weight="600"))
    b.append(txt(lx, cy_all + 30, "64 + 128 + 32 + 32", 10, anchor="start", fill=MUTED))

    y = y0 + total_h + 34
    b.append(legend(56, y, [("input", "input volume"), ("g1", "1×1 conv"),
                            ("g2", "3×3 conv"), ("g3", "5×5 conv"),
                            ("mpool", "max pool")]))
    b.append(txt(W / 2, y + 30, note, 10.5, fill=MUTED))
    write(OUT, fname, W, y + 48, b)


def fig_inception_naive():
    _inception(
        "inception-naive.svg",
        "The naive Inception module",
        "instead of choosing a filter size, run them all in parallel and concatenate",
        [
            dict(stack=[("g1", ["1×1 conv", "64 filters", "→ 28×28×64"])], ch=64, out="28×28×64"),
            dict(stack=[("g2", ["3×3 conv, same", "128 filters", "→ 28×28×128"])], ch=128, out="28×28×128"),
            dict(stack=[("g3", ["5×5 conv, same", "32 filters", "→ 28×28×32"])], ch=32, out="28×28×32"),
            dict(stack=[("mpool", ["3×3 max pool", "same, s=1", "→ 28×28×32"])], ch=32, out="28×28×32"),
        ],
        "Every branch keeps 28×28 so the outputs can be stacked. Pooling needs same padding and stride 1 — unusual for max pooling.",
    )


def fig_inception_module():
    _inception(
        "inception-module.svg",
        "The full Inception module (GoogLeNet inception-3a)",
        "1×1 bottlenecks shrink the channels before the expensive convolutions",
        [
            dict(stack=[("g1", ["1×1 conv", "64 filters", "→ 28×28×64"])], ch=64, out="28×28×64"),
            dict(stack=[("g1", ["1×1 conv", "96 filters"]), ("g2", ["3×3 conv, same", "128 filters", "→ 28×28×128"])], ch=128, out="28×28×128"),
            dict(stack=[("g1", ["1×1 conv", "16 filters"]), ("g3", ["5×5 conv, same", "32 filters", "→ 28×28×32"])], ch=32, out="28×28×32"),
            dict(stack=[("mpool", ["3×3 max pool", "same, s=1"]), ("g1", ["1×1 conv", "32 filters", "→ 28×28×32"])], ch=32, out="28×28×32", seg="mpool"),
        ],
        "The pooling branch puts its 1×1 conv AFTER the pool: pooling preserves all 192 channels and would otherwise swamp the output.",
    )


def fig_inception_cost():
    W, H = 900, 382
    b = [title(W, "Why the bottleneck layer matters", "identical input and output dimensions, about one tenth of the computation")]

    def vol(x, cy, w, h, role, shape, name):
        return "\n".join([
            rect(x, cy - h / 2, w, h, role),
            txt(x + w / 2, cy + h / 2 + 18, shape, 11, weight="600"),
            txt(x + w / 2, cy + h / 2 + 32, name, 10, fill=MUTED),
        ])

    # ---- naive
    b.append(txt(30, 78, "Naive: one 5×5 convolution", 12.5, anchor="start", weight="600"))
    b.append(vol(60, 124, 62, 84, "input", "28×28×192", "input"))
    b.append(arrow(128, 124, 236, 124))
    b.append(lines(182, 102, ["32 filters", "5×5×192"], 10, fill=MUTED))
    b.append(vol(240, 124, 26, 84, "conv", "28×28×32", "output"))
    b.append(rect(324, 102, 236, 44, "input", fill="#fee2e2", stroke="#dc2626"))
    b.append(txt(442, 120, "28·28·32  ×  5·5·192", 11.5))
    b.append(txt(442, 137, "≈ 120 million multiplications", 11.5, fill="#dc2626", weight="600"))

    b.append(f'<line x1="30" y1="214" x2="870" y2="214" stroke="{LINE}" stroke-width="1" stroke-dasharray="4 4"/>')

    # ---- bottleneck
    b.append(txt(30, 244, "With a 1×1 bottleneck", 12.5, anchor="start", weight="600"))
    b.append(vol(60, 290, 62, 84, "input", "28×28×192", "input"))
    b.append(arrow(128, 290, 174, 290))
    b.append(lines(151, 268, ["16 filters", "1×1×192"], 9.5, fill=MUTED))
    b.append(vol(178, 290, 16, 84, "one", "28×28×16", "bottleneck"))
    b.append(arrow(200, 290, 248, 290))
    b.append(lines(224, 268, ["32 filters", "5×5×16"], 9.5, fill=MUTED))
    b.append(vol(252, 290, 26, 84, "conv", "28×28×32", "output"))
    b.append(rect(324, 258, 236, 76, "input", fill="#dcfce7", stroke="#16a34a"))
    b.append(txt(442, 277, "1×1 step ≈ 2.4 M   +   5×5 step ≈ 10.0 M", 10.5))
    b.append(txt(442, 296, "≈ 12.4 million multiplications", 11.5, fill="#16a34a", weight="600"))
    b.append(txt(442, 322, "roughly 10× cheaper", 10.5, fill=MUTED))

    b.append(rect(594, 148, 276, 112, "input", fill="#f9fafb", stroke=LINE))
    for i, s in enumerate([
        "The bottleneck is the narrowest",
        "part of the block: shrink the",
        "representation, convolve cheaply,",
        "then expand again. Within reason",
        "this does not hurt performance.",
    ]):
        b.append(txt(732, 174 + i * 17, s, 10.8))
    write(OUT, "inception-cost.svg", W, H, b)


def fig_googlenet():
    W, H = 920, 328
    b = [title(W, "The Inception network (GoogLeNet)", "the inception module repeated, with max pooling between stages and two auxiliary classifiers")]
    seq = [("input", ["input"], 52), ("conv", ["stem", "conv/pool"], 68),
           ("block", ["incept", "×2"], 60), ("pool", ["pool"], 44),
           ("block", ["incept", "×5"], 60), ("pool", ["pool"], 44),
           ("block", ["incept", "×2"], 60), ("pool", ["avg", "pool"], 48),
           ("fc", ["FC"], 40), ("out", ["soft", "max"], 46)]
    cy = 128.0
    gap = 22.0
    total = sum(w for _, _, w in seq) + gap * (len(seq) - 1)
    xs, x = [], (W - total) / 2
    for _, _, w in seq:
        xs.append(x)
        x += w + gap
    for (role, rows, w), x0 in zip(seq, xs):
        b.append(rect(x0, cy - 24, w, 48, role))
        b.append(lines(x0 + w / 2, cy + 4 - (len(rows) - 1) * 6, rows, 10, lh=11))
    for i in range(len(seq) - 1):
        b.append(arrow(xs[i] + seq[i][2] + 3, cy, xs[i + 1] - 3, cy))

    # auxiliary classifiers
    for idx, lbl in [(4, "auxiliary classifier 1"), (6, "auxiliary classifier 2")]:
        ax = xs[idx] + seq[idx][2] / 2
        b.append(arrow(ax, cy + 25, ax, cy + 56))
        b.append(rect(ax - 62, cy + 58, 124, 34, "out", dash="4 3"))
        b.append(txt(ax, cy + 79, "FC → softmax", 10.5))
        b.append(txt(ax, cy + 106, lbl, 10, fill=MUTED))

    b.append(legend(xs[0], 262, [("input", "input"), ("conv", "CONV"), ("block", "inception"),
                                  ("pool", "POOL"), ("fc", "FC"), ("out", "softmax")]))
    b.append(txt(W / 2, 292, "The side branches force intermediate features to be predictive of the label and act as a regularizer.", 10.5, fill=MUTED))
    b.append(txt(W / 2, 310, "Named GoogLeNet as an homage to LeNet; “Inception” comes from the “we need to go deeper” meme cited in the paper.", 10.5, fill=MUTED))
    write(OUT, "googlenet.svg", W, H, b)


# ----------------------------------------------------------------- MobileNet


def fig_depthwise():
    W, H = 920, 400
    b = [title(W, "Depthwise separable convolution", "6×6×3 → 4×4×5 both ways: one normal convolution, or a depthwise step followed by a pointwise step")]

    def slices(x, y, n, w, h, role, dx=7, dy=-7):
        """A stack of n offset squares, drawn back to front."""
        o = []
        for i in reversed(range(n)):
            o.append(rect(x + i * dx, y + i * dy, w, h, role, rx=2, sw=1.2))
        return "\n".join(o)

    # ---- A: normal convolution
    b.append(rect(24, 64, 400, 150, "input", fill="#fef2f2", stroke="#fecaca"))
    b.append(txt(40, 86, "Normal convolution", 12.5, anchor="start", weight="600"))
    b.append(slices(52, 118, 3, 56, 56, "input"))
    b.append(txt(88, 202, "6×6×3", 11, weight="600"))
    b.append(arrow(140, 140, 196, 140))
    b.append(lines(168, 118, ["5 filters", "3×3×3"], 9.5, fill=MUTED))
    b.append(slices(212, 122, 5, 40, 40, "conv", dx=6, dy=-6))
    b.append(txt(240, 202, "4×4×5", 11, weight="600"))
    b.append(rect(292, 122, 116, 56, "input", fill="#ffffff", stroke="#dc2626"))
    b.append(txt(350, 142, "3·3·3 × 4·4 × 5", 10.5))
    b.append(txt(350, 160, "= 2,160 mults", 11, fill="#dc2626", weight="600"))

    # ---- B: depthwise + pointwise
    b.append(rect(452, 64, 444, 150, "input", fill="#f0fdf4", stroke="#bbf7d0"))
    b.append(txt(468, 86, "Step 1 — depthwise: one f×f filter per channel, applied to that channel only", 12.5, anchor="start", weight="600"))
    for i in range(3):
        y = 108 + i * 29
        b.append(rect(474, y, 38, 23, "input", rx=2, sw=1.2))
        b.append(arrow(516, y + 11, 552, y + 11))
        b.append(rect(556, y + 3, 18, 18, "depth", rx=2, sw=1.2))
        b.append(arrow(578, y + 11, 614, y + 11))
        b.append(rect(618, y + 3, 24, 18, "depth", rx=2, sw=1.2))
    b.append(txt(493, 204, "6×6×3", 10.5, weight="600"))
    b.append(txt(565, 204, "3 filters 3×3", 10.5, weight="600"))
    b.append(txt(630, 204, "4×4×3", 10.5, weight="600"))
    b.append(rect(676, 116, 112, 50, "input", fill="#ffffff", stroke="#16a34a"))
    b.append(txt(732, 136, "3·3 × 4·4 × 3", 10.5))
    b.append(txt(732, 154, "= 432 mults", 11, fill="#16a34a", weight="600"))
    b.append(txt(838, 136, "channel count", 10, fill=MUTED))
    b.append(txt(838, 151, "unchanged", 10, fill=MUTED))

    b.append(rect(452, 228, 444, 128, "input", fill="#f0fdf4", stroke="#bbf7d0"))
    b.append(txt(468, 250, "Step 2 — pointwise: a 1×1×nc convolution that mixes channels", 12.5, anchor="start", weight="600"))
    b.append(slices(486, 288, 3, 34, 34, "depth", dx=5, dy=-5))
    b.append(txt(508, 344, "4×4×3", 10.5, weight="600"))
    b.append(arrow(540, 300, 580, 300))
    b.append(lines(560, 280, ["5 filters", "1×1×3"], 9, fill=MUTED))
    b.append(slices(596, 288, 5, 30, 30, "point", dx=5, dy=-5))
    b.append(txt(618, 344, "4×4×5", 10.5, weight="600"))
    b.append(rect(670, 274, 116, 52, "input", fill="#ffffff", stroke="#16a34a"))
    b.append(txt(728, 294, "1·1·3 × 4·4 × 5", 10.5))
    b.append(txt(728, 312, "= 240 mults", 11, fill="#16a34a", weight="600"))
    b.append(txt(842, 292, "432 + 240", 10.5, fill=MUTED))
    b.append(txt(842, 308, "= 672", 11.5, weight="600"))

    b.append(rect(24, 228, 400, 128, "input", fill="#f9fafb", stroke=LINE))
    b.append(txt(224, 254, "672 / 2,160 ≈ 0.31", 15, weight="600"))
    b.append(txt(224, 288, "cost ratio  =  1/nc′  +  1/f²", 12.5, weight="600"))
    b.append(txt(224, 310, "1/5 + 1/9 = 0.311   ✓", 11, fill=MUTED))
    b.append(txt(224, 334, "with nc′ = 512:  1/512 + 1/9 ≈ 1/9, so roughly 10× cheaper", 10.5, fill=MUTED))

    b.append(txt(W / 2, H - 12, "The depthwise step never changes the channel count; the pointwise step is what changes it.", 11, weight="600"))
    write(OUT, "depthwise-separable.svg", W, H, b)


def fig_mobilenet_blocks():
    W, H = 920, 330
    b = [title(W, "MobileNet v1 and v2 blocks", "v2 adds a residual connection and an expand-then-project shape")]

    # v1
    b.append(txt(30, 70, "MobileNet v1 block  ×13", 12.5, anchor="start", weight="600"))
    b.append(rect(24, 80, 400, 100, "input", fill="#f9fafb", stroke=LINE))
    b.append(rect(56, 112, 128, 46, "depth"))
    b.append(lines(120, 132, ["depthwise", "3×3"], 10.5, lh=12))
    b.append(arrow(188, 135, 224, 135))
    b.append(rect(228, 112, 128, 46, "point"))
    b.append(lines(292, 132, ["pointwise", "1×1"], 10.5, lh=12))
    b.append(arrow(360, 135, 396, 135))
    b.append(txt(224, 200, "13 of these, then POOL → FC → softmax", 10.5, fill=MUTED))

    # v2
    b.append(txt(468, 70, "MobileNet v2 bottleneck block  ×17", 12.5, anchor="start", weight="600"))
    b.append(rect(452, 80, 444, 100, "input", fill="#f9fafb", stroke=LINE))
    xs = [(474, 106, "one", ["expansion", "1×1, ×6"]), (592, 106, "depth", ["depthwise", "3×3, same"]), (710, 106, "point", ["projection", "1×1"])]
    for x0, w, role, rows in xs:
        b.append(rect(x0, 112, w, 46, role))
        b.append(lines(x0 + w / 2, 132, rows, 10.5, lh=12))
    b.append(arrow(580, 135, 588, 135))
    b.append(arrow(698, 135, 706, 135))
    b.append(f'<circle cx="838" cy="135" r="14" fill="#ffffff" stroke="#dc2626" stroke-width="1.8"/>')
    b.append(txt(838, 140, "+", 16, fill="#dc2626", weight="600"))
    b.append(arrow(816, 135, 824, 135))
    b.append(path("M 466,112 C 466,86 838,86 838,121", stroke="#dc2626", sw=1.7))
    b.append(txt(652, 106, "residual connection", 10, fill="#dc2626", weight="600"))
    b.append(txt(674, 200, "17 of these, then POOL → FC → softmax", 10.5, fill=MUTED))

    # shape trace
    b.append(txt(W / 2, 240, "shape trace through a v2 block", 11.5, weight="600"))
    trace = ["n×n×3", "n×n×18", "n×n×18", "n×n×3"]
    labs = ["input", "after expansion", "after depthwise", "after projection"]
    for i, (s, l) in enumerate(zip(trace, labs)):
        x = 180 + i * 190
        b.append(rect(x - 66, 256, 132, 32, "input" if i in (0, 3) else "one"))
        b.append(txt(x, 277, s, 11.5, weight="600"))
        b.append(txt(x, 302, l, 10, fill=MUTED))
        if i < 3:
            b.append(arrow(x + 70, 272, x + 110, 272))

    b.append(txt(W / 2, H - 8, "Expansion gives the block a richer internal representation; projection keeps the activations passed between blocks small, which is what constrains memory on an edge device.", 10.5, fill=MUTED))
    write(OUT, "mobilenet-blocks.svg", W, H, b)


def fig_efficientnet():
    W, H = 920, 366
    b = [title(W, "EfficientNet compound scaling", "three knobs — resolution r, depth d, width w — scaled together at a searched ratio")]

    #  name, layers, width factor, resolution factor, colour, caption
    panels = [
        ("baseline", 4, 1.0, 1.0, MUTED, "r, d, w = 1"),
        ("width  w↑", 4, 1.9, 1.0, "#2563eb", "wider layers"),
        ("depth  d↑", 7, 1.0, 1.0, "#16a34a", "more layers"),
        ("resolution  r↑", 4, 1.0, 1.55, "#ca8a04", "bigger input"),
        ("compound", 6, 1.5, 1.35, "#dc2626", "all three at once"),
    ]
    pw = W / len(panels)
    IMG_BASE, BAR_W, BAR_H, BAR_GAP = 34.0, 30.0, 15.0, 5.0
    IMG_BOTTOM = 112.0  # input squares share a baseline, so they are comparable
    STACK_TOP = 142.0
    LABEL_Y = 300.0  # fixed, so labels line up regardless of stack depth

    for i, (name, nl, wf, rf, col, caption) in enumerate(panels):
        cx = pw * (i + 0.5)
        side = IMG_BASE * rf
        b.append(rect(cx - side / 2, IMG_BOTTOM - side, side, side, "input", rx=2, sw=1.2))
        b.append(txt(cx, IMG_BOTTOM + 16, "input", 9.5, fill=MUTED))
        y = STACK_TOP
        for _ in range(nl):
            bw = BAR_W * wf
            b.append(rect(cx - bw / 2, y, bw, BAR_H, "conv", rx=2, sw=1.2))
            y += BAR_H + BAR_GAP
        b.append(txt(cx, LABEL_Y, name, 11.5, weight="600", fill=col))
        b.append(txt(cx, LABEL_Y + 16, caption, 10, fill=MUTED))
        if i:
            gx = pw * i
            b.append(f'<line x1="{gx:g}" y1="58" x2="{gx:g}" y2="{LABEL_Y + 26:g}" stroke="{LINE}" stroke-width="1" stroke-dasharray="3 4"/>')

    b.append(txt(W / 2, H - 26, "Scaling one dimension alone saturates quickly: a higher-resolution input needs more depth to cover it and more width to resolve fine detail.", 10.5, fill=MUTED))
    b.append(txt(W / 2, H - 8, "EfficientNet's contribution is the ratio at which to scale all three for a given compute budget.", 10.5, fill=MUTED))
    write(OUT, "efficientnet-scaling.svg", W, H, b)


# ---------------------------------------------------------------- comparison


def fig_comparison():
    W, H = 900, 372
    b = [title(W, "Parameter count across the case studies", "logarithmic scale — note that depth and parameter count are not the same thing")]
    nets = [
        ("LeNet-5", 1998, 0.06, "#9ca3af"),
        ("AlexNet", 2012, 60.0, "#2563eb"),
        ("VGG-16", 2014, 138.0, "#dc2626"),
        ("GoogLeNet", 2014, 7.0, "#ca8a04"),
        ("ResNet-50", 2015, 25.0, "#7c3aed"),
        ("MobileNet v1", 2017, 4.2, "#16a34a"),
    ]
    x0, y0, y1 = 130.0, 76.0, 262.0
    xmax = 856.0

    lo, hi = math.log10(0.03), math.log10(300.0)

    def bx(v):
        return x0 + (math.log10(v) - lo) / (hi - lo) * (xmax - x0)

    for gv, gl in [(0.03, "30 K"), (0.1, "100 K"), (1, "1 M"), (10, "10 M"), (100, "100 M")]:
        gx = bx(gv)
        b.append(f'<line x1="{gx:.1f}" y1="{y0 - 6}" x2="{gx:.1f}" y2="{y1}" stroke="#e5e7eb" stroke-width="1"/>')
        b.append(txt(gx, y1 + 18, gl, 10, fill=MUTED))
    b.append(txt((x0 + xmax) / 2, y1 + 38, "parameters (log scale)", 11, fill=INK))

    bh = 22.0
    for i, (name, year, params, col) in enumerate(nets):
        cy = y0 + i * 31
        b.append(txt(x0 - 12, cy + 15, f"{name}", 11.5, anchor="end", weight="600"))
        b.append(txt(x0 - 12, cy + 28, str(year), 9.5, anchor="end", fill=MUTED))
        w = bx(params) - x0
        b.append(rect(x0, cy, max(w, 2), bh, "conv", rx=2, sw=1.2, fill=col + "55", stroke=col))
        label = f"{params*1000:.0f} K" if params < 1 else f"{params:.0f} M"
        b.append(txt(x0 + w + 8, cy + 16, label, 11, anchor="start", fill=col, weight="600"))

    b.append(txt(W / 2, H - 56, "GoogLeNet is deeper than VGG-16 with 20× fewer parameters, because 1×1 bottlenecks and global", 10.5, fill=MUTED))
    b.append(txt(W / 2, H - 40, "average pooling replace the huge fully connected layers. ResNet-50 is three times deeper", 10.5, fill=MUTED))
    b.append(txt(W / 2, H - 24, "than VGG-16 with a fifth of the parameters.", 10.5, fill=MUTED))
    b.append(txt(W / 2, H - 6, "Bars are measured from the 30 K left edge of the axis, not from zero.", 9.5, fill=MUTED))
    write(OUT, "architecture-comparison.svg", W, H, b)


# ---------------------------------------------------------------------- main

# figures for deep_convolutional_nn.md
CASE_STUDY_FIGURES = [
    fig_lenet5,
    fig_alexnet,
    fig_vgg16,
    fig_resnet_block,
    fig_resnet_depth,
    fig_resnet_arch,
    fig_conv1x1,
    fig_inception_naive,
    fig_inception_module,
    fig_inception_cost,
    fig_googlenet,
    fig_depthwise,
    fig_mobilenet_blocks,
    fig_efficientnet,
    fig_comparison,
]

if __name__ == "__main__":
    import foundations_figures
    import detection_figures
    import basics_figures

    for fig in CASE_STUDY_FIGURES:
        fig()
    foundations_figures.build()
    detection_figures.build()
    basics_figures.build()

    n_case = len(CASE_STUDY_FIGURES)
    n_found = len(foundations_figures.FIGURES)
    n_det = len(detection_figures.FIGURES)
    n_bas = len(basics_figures.FIGURES)
    for p in sorted(OUT.glob("*.svg")):
        print(f"{p.name:34s} {p.stat().st_size / 1024:6.1f} KB")
    print(
        f"\n{n_case} case-study, {n_found} foundations, {n_det} detection, "
        f"{n_bas} basics figures"
    )
