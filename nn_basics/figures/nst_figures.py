"""Generate the diagrams used by neural_style_transfer.md.

Driven by make_figures.py; can also be run directly.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from svgkit import (  # noqa: E402
    AMBER, BLUE, C, GREEN, INK, LINE, MUTED, PURPLE, RED,
    arrow, caption, circle, legend, lines, path, panel, rect, seg, title, txt,
    volume3d, write,
)

OUT = pathlib.Path(__file__).parent

SWIRL = ["#1d4ed8", "#3b82f6", "#f59e0b", "#fbbf24", "#0ea5e9", "#7c3aed"]


# ----------------------------------------------------------------- helpers


def photo(x, y, w, h):
    """A simple campus-like photo: sky, lawn, a building."""
    return "\n".join([
        rect(x, y, w, h, "input", rx=4, sw=1.3, fill="#e8eef5", stroke=LINE),
        rect(x, y + 0.58 * h, w, 0.42 * h, "input", rx=0, sw=0,
             fill="#86efac", stroke="none"),
        seg(x, y + 0.58 * h, x + w, y + 0.58 * h, "#4ade80", 1.0),
        rect(x + 0.28 * w, y + 0.22 * h, 0.44 * w, 0.44 * h, "pool",
             rx=2, sw=1.2, fill="#93c5fd", stroke=BLUE),
        rect(x + 0.44 * w, y + 0.38 * h, 0.12 * w, 0.28 * h, "fc",
             rx=1, sw=1.0, fill="#fed7aa", stroke="#ea580c"),
        # windows
        rect(x + 0.34 * w, y + 0.30 * h, 0.08 * w, 0.08 * h, "flat",
             rx=1, sw=0.8, fill="#fef9c3", stroke=AMBER),
        rect(x + 0.58 * w, y + 0.30 * h, 0.08 * w, 0.08 * h, "flat",
             rx=1, sw=0.8, fill="#fef9c3", stroke=AMBER),
    ])


def painting(x, y, w, h):
    """Abstract swirls standing in for a style image (Starry Night / Picasso)."""
    o = [rect(x, y, w, h, "input", rx=4, sw=1.3, fill="#1e3a5f", stroke=LINE)]
    bands = [
        (0.12, 0.18, "#1d4ed8", w * 0.9, h * 0.10),
        (0.08, 0.34, "#f59e0b", w * 0.85, h * 0.09),
        (0.18, 0.50, "#38bdf8", w * 0.70, h * 0.08),
        (0.10, 0.66, "#7c3aed", w * 0.80, h * 0.09),
        (0.22, 0.82, "#fbbf24", w * 0.55, h * 0.07),
    ]
    for fx, fy, col, bw, bh in bands:
        o.append(rect(x + fx * w, y + fy * h, bw, bh, "out",
                      rx=bh / 2, sw=0, fill=col, stroke="none"))
    o.append(circle(x + 0.78 * w, y + 0.22 * h, min(w, h) * 0.08,
                    fill="#fde68a", stroke="#f59e0b", sw=1.2))
    return "\n".join(o)


def stylized(x, y, w, h):
    """Content layout with style palette: the generated image."""
    o = [
        rect(x, y, w, h, "input", rx=4, sw=1.3, fill="#1e3a5f", stroke=LINE),
        rect(x, y + 0.58 * h, w, 0.42 * h, "input", rx=0, sw=0,
             fill="#166534", stroke="none"),
        rect(x + 0.28 * w, y + 0.22 * h, 0.44 * w, 0.44 * h, "pool",
             rx=2, sw=1.2, fill="#1d4ed8", stroke="#93c5fd"),
        rect(x + 0.44 * w, y + 0.38 * h, 0.12 * w, 0.28 * h, "fc",
             rx=1, sw=1.0, fill="#f59e0b", stroke="#fbbf24"),
    ]
    for i, col in enumerate(["#38bdf8", "#fbbf24", "#7c3aed"]):
        o.append(rect(x + 0.06 * w, y + (0.12 + i * 0.14) * h,
                      0.88 * w, h * 0.045, "out",
                      rx=3, sw=0, fill=col, stroke="none"))
    o.append(circle(x + 0.82 * w, y + 0.16 * h, min(w, h) * 0.07,
                    fill="#fde68a", stroke="#f59e0b", sw=1.1))
    return "\n".join(o)


def noise(x, y, w, h, seed=3):
    """A speckled random-initialization image."""
    o = [rect(x, y, w, h, "input", rx=4, sw=1.3, fill="#e5e7eb", stroke=LINE)]
    rng = seed
    cols, rows = 12, 9
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            rng = (1103515245 * rng + 12345) & 0x7fffffff
            v = 180 + (rng % 70)
            fill = f"#{v:02x}{v:02x}{v:02x}"
            o.append(rect(x + c * cw, y + r * ch, cw + 0.4, ch + 0.4,
                          "input", rx=0, sw=0, fill=fill, stroke="none"))
    o.append(rect(x, y, w, h, "input", rx=4, sw=1.3, fill="none", stroke=LINE))
    return "\n".join(o)


def patch_grid(x, y, n, size, gap, kind):
    """A 3×3 of tiny patches that 'maximally activate' one unit."""
    o = []
    for i in range(3):
        for j in range(3):
            px = x + j * (size + gap)
            py = y + i * (size + gap)
            o.append(_patch(px, py, size, kind, i * 3 + j))
    return "\n".join(o)


def _patch(x, y, s, kind, k):
    bg = "#f8fafc"
    o = [rect(x, y, s, s, "input", rx=2, sw=0.9, fill=bg, stroke=LINE)]
    if kind == "edge":
        angle = (-35 + k * 12) * math.pi / 180
        cx, cy = x + s / 2, y + s / 2
        dx, dy = math.cos(angle) * s * 0.42, math.sin(angle) * s * 0.42
        o.append(seg(cx - dx, cy - dy, cx + dx, cy + dy, BLUE, 1.8))
    elif kind == "color":
        fills = ["#22c55e", "#f97316", "#3b82f6", "#eab308",
                 "#a855f7", "#ef4444", "#14b8a6", "#f43f5e", "#84cc16"]
        o.append(rect(x + 2, y + 2, s - 4, s - 4, "conv", rx=2, sw=0,
                      fill=fills[k % len(fills)], stroke="none"))
    elif kind == "texture":
        for t in range(4):
            o.append(seg(x + 3 + t * (s / 5), y + 2,
                         x + 3 + t * (s / 5), y + s - 2, GREEN, 1.2))
    elif kind == "round":
        o.append(circle(x + s * 0.45, y + s * 0.55, s * 0.28,
                        fill="#bbf7d0", stroke=GREEN, sw=1.1))
    elif kind == "dog":
        o.append(circle(x + s * 0.5, y + s * 0.55, s * 0.28,
                        fill="#fed7aa", stroke="#ea580c", sw=1.1))
        o.append(circle(x + s * 0.28, y + s * 0.32, s * 0.12,
                        fill="#fed7aa", stroke="#ea580c", sw=0.9))
        o.append(circle(x + s * 0.72, y + s * 0.32, s * 0.12,
                        fill="#fed7aa", stroke="#ea580c", sw=0.9))
    elif kind == "flower":
        o.append(circle(x + s * 0.5, y + s * 0.5, s * 0.16,
                        fill="#fde68a", stroke=AMBER, sw=0.9))
        for ang in range(0, 360, 60):
            rad = ang * math.pi / 180
            o.append(circle(x + s * 0.5 + math.cos(rad) * s * 0.22,
                            y + s * 0.5 + math.sin(rad) * s * 0.22,
                            s * 0.10, fill="#f9a8d4", stroke="#db2777", sw=0.7))
    return "\n".join(o)


# --------------------------------------------- 1. what NST is


def fig_idea():
    W, H = 920, 340
    b = [title(W, "Neural style transfer",
               "take the content of one image and render it in the style of another")]

    frames = [
        (50, "Content  C", "what is in the picture", photo),
        (340, "Style  S", "how it should look", painting),
        (630, "Generated  G", "content of C, style of S", stylized),
    ]
    for x, head, sub, draw in frames:
        b.append(txt(x + 110, 78, head, 13, weight="600"))
        b.append(txt(x + 110, 96, sub, 10.5, fill=MUTED))
        b.append(draw(x, 112, 220, 160))

    b.append(txt(305, 190, "+", 22, fill=MUTED, weight="600"))
    b.append(arrow(560, 190, 620, 190, LINE, 1.6))

    b.append(caption(W / 2, H - 22, [
        "C, S, G are the three images. The algorithm never trains a new network — it searches for the pixels of G.",
    ]))
    write(OUT, "nst-idea.svg", W, H, b)


# ------------------------------- 2. what convnets learn


def fig_layers():
    W, H = 920, 400
    b = [title(W, "What hidden units detect, layer by layer",
               "find the image patches that maximally activate a unit — Zeiler and Fergus")]

    cols = [
        (70, "Layer 1", "edges, colours", ["edge", "color"]),
        (250, "Layer 2", "textures, shapes", ["texture", "round"]),
        (430, "Layer 3", "parts, patterns", ["round", "flower"]),
        (610, "Layer 4", "object parts", ["dog", "round"]),
        (790, "Layer 5", "whole objects", ["dog", "flower"]),
    ]
    for x, head, sub, kinds in cols:
        b.append(txt(x, 78, head, 12.5, weight="600"))
        b.append(txt(x, 96, sub, 10, fill=MUTED))
        size, gap = 26, 4
        for u, kind in enumerate(kinds):
            py = 118 + u * (3 * size + 2 * gap + 18)
            b.append(patch_grid(x - 42, py, 3, size, gap, kind))

    b.append(caption(W / 2, H - 36, [
        "Each 3×3 is one hidden unit: the nine patches that fire it most. Deeper units see a larger receptive field.",
    ]))
    b.append(caption(W / 2, H - 18, [
        "Shallow layers are edges and colour. Deep layers are dogs, keyboards, flowers — the things a classifier needs.",
    ]))
    write(OUT, "nst-layers.svg", W, H, b)


def fig_receptive():
    W, H = 920, 280
    b = [title(W, "Deeper units see more of the image",
               "a layer-1 unit looks at a small patch; a deep unit can be affected by every pixel")]

    ix, iy, iw, ih = 80, 80, 280, 140
    b.append(photo(ix, iy, iw, ih))
    # small receptive field
    b.append(rect(ix + 36, iy + 28, 44, 44, "out", rx=2, sw=2.0,
                  fill="none", stroke=GREEN))
    b.append(txt(ix + 58, iy + 20, "layer 1", 10.5, fill=GREEN, weight="600"))
    # large receptive field
    b.append(rect(ix + 70, iy + 18, 180, 104, "out", rx=2, sw=2.0,
                  fill="none", stroke=RED, dash="5 4"))
    b.append(txt(ix + 160, iy + 12, "deeper layer", 10.5, fill=RED, weight="600"))

    b.append(arrow(ix + iw + 16, iy + ih / 2, 430, iy + ih / 2, LINE, 1.5))

    rows = [
        (GREEN, "Layer 1", "small patch  ·  edge, a shade of colour"),
        (BLUE, "Layer 2–3", "larger patch  ·  texture, honeycomb, round shapes"),
        (RED, "Layer 4–5", "much of the image  ·  dog, keyboard, flower"),
    ]
    for i, (col, head, body) in enumerate(rows):
        y = 100 + i * 42
        b.append(rect(440, y - 16, 14, 14, "out", rx=3, sw=0, fill=col, stroke=col))
        b.append(txt(468, y - 4, head, 12.5, anchor="start", weight="600"))
        b.append(txt(468, y + 14, body, 11, anchor="start", fill=MUTED))

    b.append(caption(W / 2, H - 18, [
        "That is why a shallow content layer copies pixels, and a deep one only asks that “a dog is somewhere.”",
    ]))
    write(OUT, "nst-receptive.svg", W, H, b)


# -------------------------------------- 3. overall cost and GD


def fig_cost():
    W, H = 920, 300
    b = [title(W, "The cost of a generated image G",
               "content says stay like C; style says look like S; α and β trade the two off")]

    b.append(rect(70, 90, 200, 120, "input", rx=8, fill="#eff6ff", stroke=BLUE))
    b.append(txt(170, 128, "J_content(C, G)", 13, fill=BLUE, weight="600"))
    b.append(txt(170, 152, "how similar is the", 11, fill=MUTED))
    b.append(txt(170, 170, "content of G to C?", 11, fill=MUTED))

    b.append(txt(300, 154, "+", 22, fill=MUTED, weight="600"))

    b.append(rect(340, 90, 200, 120, "input", rx=8, fill="#fff7ed", stroke="#ea580c"))
    b.append(txt(440, 128, "J_style(S, G)", 13, fill="#ea580c", weight="600"))
    b.append(txt(440, 152, "how similar is the", 11, fill=MUTED))
    b.append(txt(440, 170, "style of G to S?", 11, fill=MUTED))

    b.append(arrow(560, 150, 620, 150, LINE, 1.6))

    b.append(rect(628, 90, 230, 120, "out", rx=8, fill="#fef2f2", stroke=RED))
    b.append(txt(743, 128, "J(G)", 16, fill=RED, weight="600"))
    b.append(txt(743, 156, "α J_content  +  β J_style", 12.5, weight="600"))
    b.append(txt(743, 180, "α, β  hyperparameters", 10.5, fill=MUTED))

    b.append(caption(W / 2, H - 36, [
        "Two knobs are redundant — the ratio α/β is what matters — but Gatys, Ecker, and Bethge used both, so the notes do too.",
    ]))
    b.append(caption(W / 2, H - 18, [
        "Minimize J(G) with gradient descent on the pixels of G. The ConvNet weights stay frozen.",
    ]))
    write(OUT, "nst-cost.svg", W, H, b)


def fig_optimize():
    W, H = 920, 340
    b = [title(W, "Optimize the pixels, not the weights",
               "G starts as white noise; each step of gradient descent changes RGB values")]

    frames = [
        (48, "G  at step 0", "random pixels", "noise"),
        (268, "a few steps", "content leaking in", "photo"),
        (488, "more steps", "style taking hold", "mix"),
        (708, "converged", "C in the style of S", "stylized"),
    ]
    for i, (x, head, sub, kind) in enumerate(frames):
        b.append(txt(x + 82, 78, head, 12, weight="600"))
        b.append(txt(x + 82, 96, sub, 10.5, fill=MUTED))
        if kind == "noise":
            b.append(noise(x, 112, 164, 118, seed=3 + i))
        elif kind == "photo":
            b.append(photo(x, 112, 164, 118))
        elif kind == "mix":
            b.append(photo(x, 112, 164, 118))
            b.append(rect(x + 10, 128, 144, 8, "out", rx=3, sw=0,
                          fill="#3b82f6", stroke="none"))
            b.append(rect(x + 18, 148, 128, 7, "out", rx=3, sw=0,
                          fill="#f59e0b", stroke="none"))
        else:
            b.append(stylized(x, 112, 164, 118))
        if i < 3:
            b.append(arrow(x + 172, 170, x + 212, 170, LINE, 1.5))

    b.append(caption(W / 2, H - 36, [
        "Update:  G  ←  G  −  ∂J/∂G.  G is 100×100×3 or 500×500×3; every RGB value is a parameter.",
    ]))
    b.append(caption(W / 2, H - 18, [
        "The pretrained VGG (or AlexNet) is used only as a feature extractor. Its weights never move.",
    ]))
    write(OUT, "nst-optimize.svg", W, H, b)


# ------------------------------------------- 4. content cost


def fig_content():
    W, H = 920, 340
    b = [title(W, "Content cost at a chosen layer ℓ",
               "if a^[ℓ](C) and a^[ℓ](G) match, C and G have similar content")]

    b.append(txt(130, 78, "content  C", 12, weight="600"))
    b.append(photo(50, 92, 160, 118))
    b.append(arrow(220, 151, 258, 151, LINE, 1.4))
    svg, info = volume3d(264, 122, 64, 64, 20, "conv", label="a^[ℓ](C)")
    b.append(svg)

    b.append(txt(130, 238, "generated  G", 12, weight="600"))
    b.append(stylized(50, 248, 160, 70))
    b.append(arrow(220, 283, 258, 283, LINE, 1.4))
    svg2, info2 = volume3d(264, 254, 64, 64, 20, "fc", label="a^[ℓ](G)")
    b.append(svg2)

    b.append(path("M 350,154 C 430,154  430,220  500,220", stroke=RED, sw=1.5))
    b.append(path("M 350,286 C 430,286  430,240  500,240", stroke=RED, sw=1.5))

    b.append(rect(508, 168, 370, 100, "out", rx=8, fill="#fef2f2", stroke=RED))
    b.append(txt(693, 200, "J_content(C, G)", 14, fill=RED, weight="600"))
    b.append(txt(693, 226, "½  ‖ a^[ℓ](C)  −  a^[ℓ](G) ‖²", 13, weight="600"))
    b.append(txt(693, 250, "ℓ in the middle of the net — not 1, not the last", 10.5, fill=MUTED))

    write(OUT, "nst-content.svg", W, H, b)


# --------------------------------------------- 5. style / Gram


def fig_channels():
    W, H = 920, 380
    b = [title(W, "Style is correlation between channels",
               "“vertical texture co-occurs with orange” is a style fact, not a content fact")]

    # volume with 5 channel tints
    ch_cols = ["#fecaca", "#fde68a", "#bbf7d0", "#bfdbfe", "#e9d5ff"]
    ch_names = ["red", "yellow", "green", "blue", "purple"]
    x0, y0 = 70, 100
    fw, fh, dstep = 150, 150, 18
    for i, col in enumerate(reversed(ch_cols)):
        k = len(ch_cols) - 1 - i
        b.append(rect(x0 + k * dstep, y0 - k * 8, fw, fh, "input", rx=3, sw=1.2,
                      fill=col, stroke=LINE))
    b.append(txt(x0 + fw / 2 + 2 * dstep, y0 + fh + 28, "layer ℓ  ·  nH × nW × nC", 11,
                 weight="600"))

    # two highlighted channels
    b.append(rect(x0, y0, fw, fh, "out", rx=3, sw=2.2, fill="none", stroke=RED))
    b.append(txt(x0 + fw / 2, y0 - 12, "channel k  (vertical texture)", 10.5,
                 fill=RED, weight="600"))

    b.append(arrow(x0 + fw + 5 * dstep + 12, y0 + 40, 430, y0 + 40, LINE, 1.4))

    b.append(panel(440, 78, 440, 230, fill="#f9fafb"))
    b.append(txt(660, 104, "At each spatial position (i, j)", 12.5, weight="600"))
    b.append(txt(660, 128, "read the pair  (a_ijk ,  a_ijk′)", 12, fill=MUTED))
    b.append(txt(660, 164, "highly correlated", 12.5, fill=GREEN, weight="600"))
    b.append(txt(660, 182, "wherever there is that vertical texture,", 11, fill=MUTED))
    b.append(txt(660, 198, "there is also that orange tint", 11, fill=MUTED))
    b.append(txt(660, 230, "uncorrelated", 12.5, fill=RED, weight="600"))
    b.append(txt(660, 248, "the texture appears without the tint", 11, fill=MUTED))
    b.append(txt(660, 280, "style  =  which features tend to occur together", 11.5, weight="600"))

    b.append(caption(W / 2, H - 18, [
        "Content asks what is present. Style asks which textures and colours keep showing up in the same places.",
    ]))
    write(OUT, "nst-channels.svg", W, H, b)


def fig_gram():
    W, H = 920, 420
    b = [title(W, "The style (Gram) matrix",
               "G^[ℓ] is nC × nC; entry (k, k′) is how much channels k and k′ fire together")]

    # formula block
    b.append(rect(48, 78, 420, 150, "input", rx=8, fill="#f9fafb", stroke=LINE))
    b.append(txt(258, 110, "G^[ℓ]_{k k′}   =   Σ_i Σ_j   a^[ℓ]_{i j k}  ·  a^[ℓ]_{i j k′}",
                 13, weight="600"))
    b.append(txt(258, 140, "i = 1…nH    j = 1…nW    k, k′ = 1…nC", 11.5, fill=MUTED))
    b.append(txt(258, 168, "unnormalized cross-covariance  ·  means are not subtracted",
                 11, fill=MUTED))
    b.append(txt(258, 190, "large if both channels are active at the same positions",
                 11, fill=MUTED))

    # small gram matrix
    n = 5
    cell = 28
    gx, gy = 540, 96
    vals = [
        [9, 6, 1, 2, 0],
        [6, 8, 4, 1, 2],
        [1, 4, 7, 3, 1],
        [2, 1, 3, 8, 5],
        [0, 2, 1, 5, 9],
    ]
    ch_cols = ["#fecaca", "#fde68a", "#bbf7d0", "#bfdbfe", "#e9d5ff"]
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            t = v / 9
            fill = f"#{int(254 - 80 * t):02x}{int(226 - 40 * t):02x}{int(226 - 40 * t):02x}"
            b.append(rect(gx + j * cell, gy + i * cell, cell, cell, "out",
                          rx=0, sw=0.8, fill=fill, stroke=LINE))
            b.append(txt(gx + j * cell + cell / 2, gy + i * cell + cell / 2 + 4,
                         str(v), 10, weight="600"))
    b.append(txt(gx + n * cell / 2, gy - 12, "G^[ℓ]   (nC × nC)", 11.5, weight="600"))
    b.append(txt(gx + n * cell / 2, gy + n * cell + 18, "channel k′  →", 10.5, fill=MUTED))
    b.append(txt(gx - 16, gy + n * cell / 2, "channel k  ↓", 10.5, fill=MUTED, rot=-90))

    # style cost
    b.append(rect(48, 278, 824, 96, "out", rx=8, fill="#fff7ed", stroke="#ea580c"))
    b.append(txt(460, 306, "J^[ℓ]_style(S, G)   =   (1 / (2 nH nW nC)²)   ‖ G^[ℓ](S)  −  G^[ℓ](G) ‖²_F",
                 13, weight="600"))
    b.append(txt(460, 332, "Frobenius norm: sum of squared element-wise differences between the two Gram matrices",
                 11, fill=MUTED))
    b.append(txt(460, 354, "The 1/(2 nH nW nC)² factor is a convention; β absorbs any constant anyway.",
                 11, fill=MUTED))

    write(OUT, "nst-gram.svg", W, H, b)


def fig_multilayer():
    W, H = 920, 320
    b = [title(W, "Style from several layers at once",
               "low-level correlations (edges, colour) plus high-level ones (textures of textures)")]

    layers = [
        (120, "ℓ = 1", "edges", "#bbf7d0", GREEN),
        (280, "ℓ = 2", "textures", "#bfdbfe", BLUE),
        (440, "ℓ = 3", "patterns", "#fde68a", AMBER),
        (600, "ℓ = 4", "parts", "#fed7aa", "#ea580c"),
        (760, "ℓ = 5", "objects", "#fecaca", RED),
    ]
    for x, head, sub, fill, stroke in layers:
        b.append(rect(x - 54, 88, 108, 72, "input", rx=8, fill=fill, stroke=stroke))
        b.append(txt(x, 118, head, 13, weight="600"))
        b.append(txt(x, 140, sub, 11, fill=MUTED))
        b.append(arrow(x, 164, x, 196, LINE, 1.3))
        b.append(txt(x, 214, "λ^[ℓ] J^[ℓ]_style", 11, fill=stroke, weight="600"))

    b.append(seg(120, 230, 760, 230, LINE, 1.2))
    b.append(arrow(440, 230, 440, 252, LINE, 1.4))
    b.append(rect(300, 256, 280, 36, "out", rx=6))
    b.append(txt(440, 278, "J_style(S, G)  =  Σ_ℓ  λ^[ℓ] J^[ℓ]_style", 12.5, weight="600"))

    write(OUT, "nst-multilayer.svg", W, H, b)


# ---------------------------------------------------------------------- main

FIGURES = [
    fig_idea,
    fig_layers,
    fig_receptive,
    fig_cost,
    fig_optimize,
    fig_content,
    fig_channels,
    fig_gram,
    fig_multilayer,
]


def build():
    for f in FIGURES:
        f()


if __name__ == "__main__":
    build()
    print(f"{len(FIGURES)} neural-style-transfer figures written")
