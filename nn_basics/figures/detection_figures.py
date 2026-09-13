"""Generate the diagrams used by object_detection.md.

Driven by make_figures.py; can also be run directly.
"""

from __future__ import annotations

import base64
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from svgkit import (  # noqa: E402
    AMBER, ARCH_LEGEND, BLUE, C, GREEN, INK, LINE, MUTED, PURPLE, RED,
    arrow, caption, circle, diverge, legend, lines, matrix, mat_w, outline,
    panel, path, pipeline as _pipeline, poly, rect, seg, title, txt, volume3d,
    write,
)

OUT = pathlib.Path(__file__).parent


# ----------------------------------------------------------------- helpers


def photo(x, y, w, h, sky="#e8eef5"):
    """A blank image: sky over a road, used as a stage for boxes and windows."""
    return "\n".join([
        rect(x, y, w, h, "input", rx=4, sw=1.3, fill=sky, stroke=LINE),
        rect(x, y + 0.62 * h, w, 0.38 * h, "input", rx=0, sw=0,
             fill="#d6d3d1", stroke="none"),
        seg(x, y + 0.62 * h, x + w, y + 0.62 * h, "#c4c0bc", 1.0),
    ])


def car(x, y, w, h, fill="#93c5fd", stroke=BLUE, label=None):
    o = [
        rect(x, y, w, h, "pool", rx=4, sw=1.6, fill=fill, stroke=stroke),
        rect(x + 0.18 * w, y - 0.45 * h, 0.64 * w, 0.52 * h, "pool",
             rx=3, sw=1.4, fill=fill, stroke=stroke),
    ]
    if label:
        o.append(txt(x + w / 2, y + h / 2 + 4, label, 10, fill=stroke, weight="600"))
    return "\n".join(o)


def person(cx, y, h, fill="#86efac", stroke=GREEN, label=None):
    r = h * 0.11
    o = [
        circle(cx, y + r, r, fill=fill, stroke=stroke, sw=1.5),
        rect(cx - h * 0.14, y + 2.1 * r, h * 0.28, h * 0.42, "conv",
             rx=4, sw=1.5, fill=fill, stroke=stroke),
        rect(cx - h * 0.12, y + 2.1 * r + h * 0.42, h * 0.10, h * 0.22, "conv",
             rx=2, sw=1.3, fill=fill, stroke=stroke),
        rect(cx + h * 0.02, y + 2.1 * r + h * 0.42, h * 0.10, h * 0.22, "conv",
             rx=2, sw=1.3, fill=fill, stroke=stroke),
    ]
    if label:
        o.append(txt(cx, y + h + 14, label, 10, fill=stroke, weight="600"))
    return "\n".join(o)


def bbox(x, y, w, h, color=RED, sw=2.0, dash=None, label=None, score=None):
    o = [rect(x, y, w, h, "out", rx=2, sw=sw, fill="none", stroke=color, dash=dash)]
    if label or score is not None:
        lab = label or ""
        if score is not None:
            lab = f"{lab} {score}".strip() if lab else f"{score}"
        o.append(rect(x, y - 16, max(len(lab) * 6.4 + 10, 36), 16,
                      "out", rx=2, sw=0, fill=color, stroke=color))
        o.append(txt(x + 6, y - 4, lab, 10, anchor="start", fill="#ffffff", weight="600"))
    return "\n".join(o)


def midpoint(x, y, color=RED):
    return circle(x, y, 4.2, fill=color, stroke="#ffffff", sw=1.4)


# ------------------------------------------------------ 1.1 three task levels


def fig_tasks():
    W, H = 920, 340
    b = [title(W, "Three levels of task",
               "each step asks the network for a richer output, not a different backbone")]

    panels = [
        (36, "Classification", "one object  ·  one label", "car"),
        (328, "Classification + localization", "one object  ·  label + one box", "car"),
        (620, "Detection", "many objects  ·  every label and box", None),
    ]
    for x0, head, sub, _ in panels:
        b.append(panel(x0, 72, 264, 210, fill="#f9fafb"))
        b.append(txt(x0 + 132, 96, head, 13, weight="600"))
        b.append(txt(x0 + 132, 114, sub, 10.5, fill=MUTED))
        b.append(photo(x0 + 28, 128, 208, 132))

    # classification: just a label
    b.append(car(118, 198, 86, 34, label=""))
    b.append(txt(168, 248, "“car”", 12, weight="600", fill=RED))

    # localization: one box
    b.append(car(410, 198, 86, 34, label=""))
    b.append(bbox(398, 168, 110, 72, RED, 2.0, label="car"))

    # detection: two boxes
    b.append(person(700, 148, 72))
    b.append(car(754, 204, 72, 28, label=""))
    b.append(bbox(678, 144, 44, 92, GREEN, 1.8, label="ped"))
    b.append(bbox(742, 186, 96, 54, RED, 1.8, label="car"))

    b.append(caption(W / 2, H - 18, [
        "Classification asks what. Localization asks what and where, for one object. Detection asks both, for every object.",
    ]))
    write(OUT, "det-tasks.svg", W, H, b)


# ------------------------------------------------- 1.2 bounding-box encoding


def fig_bbox_label():
    W, H = 920, 392
    b = [title(W, "The localization target",
               "four numbers locate the box; pc says whether anything is there at all")]

    ix, iy, iw, ih = 70, 86, 300, 220
    b.append(photo(ix, iy, iw, ih))
    # car occupying ~40% width, 30% height, midpoint at (0.5, 0.7)
    bx, by, bw, bh = 0.50, 0.70, 0.40, 0.30
    box_w, box_h = bw * iw, bh * ih
    box_x, box_y = ix + (bx - bw / 2) * iw, iy + (by - bh / 2) * ih
    b.append(car(box_x + 18, box_y + 28, box_w - 36, box_h - 36, label=""))
    b.append(bbox(box_x, box_y, box_w, box_h, RED, 2.2))
    mx, my = ix + bx * iw, iy + by * ih
    b.append(midpoint(mx, my))
    b.append(txt(mx + 14, my - 8, "(bx, by)", 11, anchor="start", fill=RED, weight="600"))

    # dimension arrows
    b.append(arrow(box_x - 10, box_y, box_x - 10, box_y + box_h, RED, 1.3))
    b.append(txt(box_x - 22, my + 4, "bh", 11, fill=RED, weight="600"))
    b.append(arrow(box_x, box_y + box_h + 10, box_x + box_w, box_y + box_h + 10, RED, 1.3))
    b.append(txt(mx, box_y + box_h + 26, "bw", 11, fill=RED, weight="600"))

    b.append(txt(ix + 8, iy + 18, "(0, 0)", 10, anchor="start", fill=MUTED))
    b.append(txt(ix + iw - 8, iy + ih - 10, "(1, 1)", 10, anchor="end", fill=MUTED))

    # target vector
    px = 430
    b.append(txt(px, 102, "y  for 3 classes", 12.5, anchor="start", weight="600"))
    rows = [
        ("pc", "1", "an object of interest is present", RED),
        ("bx", "0.5", "midpoint x, as a fraction of image width", INK),
        ("by", "0.7", "midpoint y, as a fraction of image height", INK),
        ("bh", "0.3", "box height / image height", INK),
        ("bw", "0.4", "box width / image width", INK),
        ("c1", "0", "pedestrian", MUTED),
        ("c2", "1", "car", GREEN),
        ("c3", "0", "motorcycle", MUTED),
    ]
    for k, (sym, val, expl, col) in enumerate(rows):
        y = 124 + k * 24
        b.append(rect(px, y - 14, 36, 22, "input", rx=2, sw=1.1, fill="#f3f4f6"))
        b.append(txt(px + 18, y + 2, sym, 10.5, fill=MUTED))
        b.append(rect(px + 42, y - 14, 36, 22, "out" if val not in ("0",) else "input",
                      rx=2, sw=1.1,
                      fill=C["out"][0] if val == "1" and sym in ("pc", "c2") else "#ffffff"))
        b.append(txt(px + 60, y + 2, val, 12, weight="600", fill=col))
        b.append(txt(px + 88, y + 2, expl, 11, anchor="start"))

    b.append(caption(W / 2, H - 22, [
        "When pc = 0 the other seven entries are don't cares: if nothing is there, the box and class do not matter.",
        "A labeled detection dataset has to include boxes, which is more expensive than class labels alone.",
    ]))
    write(OUT, "det-bbox-label.svg", W, H, b)


# ---------------------------------------------------------- 1.4 landmarks


def fig_landmarks():
    W, H = 880, 340
    b = [title(W, "Landmark detection",
               "the same idea: output a list of (x, y) coordinates instead of a box")]

    # face
    fx, fy, fw, fh = 90, 80, 200, 200
    b.append(rect(fx, fy, fw, fh, "input", rx=8, sw=1.3, fill="#f5f0eb"))
    b.append(circle(fx + fw / 2, fy + 96, 78, fill="#fde8d0", stroke="#d6b48a", sw=1.6))
    # landmarks: 2 eyes (4 corners), mouth (3), nose (1) — 8 points shown of 64
    pts = [
        (fx + 70, fy + 78, "1"), (fx + 92, fy + 78, "2"),
        (fx + 108, fy + 78, "3"), (fx + 130, fy + 78, "4"),
        (fx + 100, fy + 108, "5"),
        (fx + 78, fy + 132, "6"), (fx + 100, fy + 138, "7"), (fx + 122, fy + 132, "8"),
    ]
    for x, y, n in pts:
        b.append(circle(x, y, 5.5, fill=RED, stroke="#ffffff", sw=1.3))
        b.append(txt(x, y + 3.5, n, 8, fill="#ffffff", weight="600"))

    b.append(txt(fx + fw / 2, fy + fh + 22, "64 landmarks  →  128 coordinates  +  1 face bit  =  129 outputs",
                 11, weight="600"))

    px = 360
    b.append(panel(px, 80, 480, 196))
    b.append(txt(px + 20, 108, "N landmarks  ⇒  2N coordinate outputs", 13, anchor="start", weight="600"))
    b.append(txt(px + 20, 138, "Landmark 1 must be the same anatomical point in every image.", 11.5, anchor="start"))
    b.append(txt(px + 20, 160, "If identity drifts, the network has nothing consistent to learn.", 11.5, anchor="start"))
    b.append(seg(px + 20, 178, px + 460, 178, LINE, 1.0))
    for k, (app, n) in enumerate([
        ("Face", "eye corners, mouth, jaw"),
        ("Pose", "shoulders, elbows, wrists"),
        ("AR filters", "snap a crown to landmark 12"),
    ]):
        y = 202 + k * 22
        b.append(txt(px + 20, y, app, 11.5, anchor="start", weight="600", fill=GREEN))
        b.append(txt(px + 130, y, n, 11.5, anchor="start"))

    b.append(caption(W / 2, H - 18, [
        "Having a network output a set of real numbers, as a regression, is a reusable idea — boxes, landmarks, and keypoints are the same mechanism.",
    ]))
    write(OUT, "det-landmarks.svg", W, H, b)


# ------------------------------------------------------- 2.1 sliding windows


def fig_sliding_windows():
    W, H = 920, 368
    b = [title(W, "Sliding-window detection",
               "train on tightly cropped objects, then sweep windows of several sizes over the test image")]

    # training
    b.append(panel(36, 72, 250, 220, label="Train"))
    b.append(rect(86, 108, 150, 110, "input", rx=4, fill="#e8eef5"))
    b.append(car(108, 148, 106, 42, label=""))
    b.append(txt(161, 238, "closely cropped cars", 11, weight="600"))
    b.append(txt(161, 256, "ConvNet  →  0 or 1", 11, fill=MUTED))

    b.append(arrow(296, 182, 344, 182))

    # test
    b.append(panel(356, 72, 528, 220, label="Test  —  sweep, then repeat with a larger window"))
    ix, iy, iw, ih = 380, 108, 280, 160
    b.append(photo(ix, iy, iw, ih))
    b.append(car(ix + 70, iy + 88, 90, 36, label=""))
    # three window sizes
    b.append(bbox(ix + 16, iy + 20, 70, 70, BLUE, 1.8, dash="5 3"))
    b.append(bbox(ix + 86, iy + 48, 110, 90, GREEN, 1.8, dash="5 3"))
    b.append(bbox(ix + 48, iy + 36, 160, 118, RED, 2.0))
    b.append(txt(ix + iw / 2, iy + ih + 18, "fine window     mid window     coarse window", 10.5, fill=MUTED))

    b.append(txt(720, 140, "stride too large:", 11.5, anchor="start", weight="600"))
    b.append(txt(720, 160, "misses the object", 11, anchor="start", fill=MUTED))
    b.append(txt(720, 192, "stride too small:", 11.5, anchor="start", weight="600"))
    b.append(txt(720, 212, "thousands of ConvNet", 11, anchor="start", fill=MUTED))
    b.append(txt(720, 228, "passes  —  infeasibly slow", 11, anchor="start", fill=MUTED))

    b.append(caption(W / 2, H - 22, [
        "Before ConvNets a cheap linear classifier made this fine. With a ConvNet, each window is expensive, so you cannot afford to run them independently.",
    ]))
    write(OUT, "det-sliding-windows.svg", W, H, b)


# ------------------------------------------------------- 2.3 FC → conv


def fig_sliding_fc():
    _pipeline(
        OUT,
        "det-sliding-fc.svg",
        "Original sliding-window classifier  ·  fully connected layers",
        [
            dict(kind="vol", s=14, c=3, shape="14×14×3", name="Input", role="input"),
            dict(kind="vol", s=10, c=16, shape="10×10×16", name="CONV", role="conv"),
            dict(kind="vol", s=5, c=16, shape="5×5×16", name="POOL", role="pool"),
            dict(kind="vec", u=400, shape="400", name="FC", role="fc"),
            dict(kind="vec", u=400, shape="400", name="FC", role="fc"),
            dict(kind="vec", u=4, shape="4", name="Output", role="out"),
        ],
        [
            "convolution 5×5×3,\n16 filters",
            "max pool\nf=2, s=2",
            "fully\nconnected",
            "fully\nconnected",
            "softmax,\n4 classes",
        ],
        sub="this is the network you run independently on every cropped window",
        note="Each window is resized to 14×14×3 and classified from scratch. Overlapping windows recompute the same CONV and POOL features.",
        leg=ARCH_LEGEND,
    )


def fig_fc_to_conv():
    _pipeline(
        OUT,
        "det-fc-to-conv.svg",
        "Convolutional sliding windows  ·  every FC layer is a convolution",
        [
            dict(kind="vol", s=14, c=3, shape="14×14×3", name="Input", role="input"),
            dict(kind="vol", s=10, c=16, shape="10×10×16", name="CONV", role="conv"),
            dict(kind="vol", s=5, c=16, shape="5×5×16", name="POOL", role="pool"),
            dict(kind="vol", s=1, c=400, shape="1×1×400", name="CONV", role="conv5"),
            dict(kind="vol", s=1, c=400, shape="1×1×400", name="CONV", role="conv1"),
            dict(kind="vol", s=1, c=4, shape="1×1×4", name="Output", role="out"),
        ],
        [
            "convolution 5×5×3,\n16 filters",
            "max pool\nf=2, s=2",
            "convolution 5×5×16,\n400 filters",
            "convolution 1×1×400,\n400 filters",
            "convolution 1×1×400,\n4 filters + softmax",
        ],
        sub="same numbers, different bookkeeping — a filter that covers the whole 5×5×16 volume is exactly an FC unit",
        note="A 5×5×16 filter over a 5×5×16 volume produces one number. 400 of them produce 1×1×400, which is exactly what the dense layer computed.",
        leg=ARCH_LEGEND,
    )


# --------------------------------------------- 2.4 convolutional windows


def fig_conv_windows():
    W, H = 920, 460
    b = [title(W, "Convolutional implementation of sliding windows",
               "one forward pass over the whole image evaluates every window, with the overlapping computation shared")]

    # four naive crops
    b.append(txt(40, 70, "Naive: four independent 14×14 crops", 12.5, anchor="start", weight="600"))
    for dx, dy, col in [(0, 0, RED), (18, 0, AMBER), (0, 18, GREEN), (18, 18, BLUE)]:
        b.append(rect(40 + dx, 86 + dy, 70, 70, "input", rx=3, sw=1.6,
                      fill="#f3f4f6", stroke=col, dash="4 3"))
    b.append(txt(75, 188, "16×16 image, stride 2", 11, fill=MUTED))
    b.append(txt(75, 204, "4 ConvNet passes, mostly duplicated", 11, fill=MUTED))

    b.append(arrow(160, 140, 210, 140))

    ox, oy, s = 230, 96, 32
    cols = [RED, AMBER, GREEN, BLUE]
    for i, col in enumerate(cols):
        r, c = divmod(i, 2)
        b.append(rect(ox + c * s, oy + r * s, s, s, "out", rx=2, sw=1.8,
                      fill="#ffffff", stroke=col))
        b.append(txt(ox + c * s + s / 2, oy + r * s + s / 2 + 4, "4", 12, weight="600", fill=col))
    b.append(txt(ox + s, oy + 2 * s + 18, "one 1×1×4 slice per window", 11, fill=MUTED))
    b.append(txt(520, 140, "Convolutional: the same four results, one pass", 12.5, anchor="start", weight="600"))
    b.append(txt(520, 162, "Each coloured cell is the softmax the 14×14 net would have", 11, anchor="start", fill=MUTED))
    b.append(txt(520, 178, "produced on that crop — computed together, not four times.", 11, anchor="start", fill=MUTED))

    # full forward as 3D volumes
    specs = [
        (16, 3, "input", "Input", "16×16×3"),
        (12, 16, "conv", "CONV", "12×12×16"),
        (6, 16, "pool", "POOL", "6×6×16"),
        (2, 400, "conv5", "CONV", "2×2×400"),
        (2, 400, "conv1", "CONV", "2×2×400"),
        (2, 4, "out", "Output", "2×2×4"),
    ]
    ops = [
        "convolution 5×5×3,\n16 filters",
        "max pool\nf=2, s=2",
        "convolution 5×5×16,\n400 filters",
        "convolution 1×1×400,\n400 filters",
        "convolution 1×1×400,\n4 filters",
    ]
    smax, cmax = 16.0, 400.0
    faces, depths = [], []
    for s, c, *_ in specs:
        faces.append(34 + (s / smax) ** 0.7 * 40)
        depths.append(10 + (c / cmax) ** 0.4 * 24)
    occ = [f + d * 0.62 for f, d in zip(faces, depths)]
    gap = (W - 56 - sum(occ)) / 5
    x, cy = 28.0, 310.0
    infos = []
    for (s, c, role, name, shape), face, depth in zip(specs, faces, depths):
        y0 = cy - face / 2
        svg, info = volume3d(x, y0, face, face, depth, role, label=name, fs=10)
        b.append(svg)
        b.append(txt(info["cx"], info["bottom"] + 16, shape, 10.5, weight="600"))
        infos.append(info)
        x += face + depth * 0.62 + gap
    for i, op in enumerate(ops):
        x1, x2 = infos[i]["right"], infos[i + 1]["left"]
        b.append(arrow(x1 + 1, cy, x2 - 3, cy))
        rows = [r for r in op.split("\n") if r]
        lh = 11
        y_last = cy - 8
        y0 = y_last - (len(rows) - 1) * lh
        b.append(lines((x1 + x2) / 2, y0, rows, 9.5, fill=MUTED, lh=lh))

    b.append(caption(W / 2, H - 36, [
        "A 28×28 image through the same net gives 8×8×4 — 64 windows in one pass. The effective stride is the network's downsampling (here, 2).",
        "This fixes the cost. It does not fix the boxes: they are still stuck to the discrete window grid and the window's aspect ratio.",
    ]))
    write(OUT, "det-conv-windows.svg", W, H, b)


# ---------------------------------------------------------- 3.1 YOLO grid


def fig_yolo_grid():
    W, H = 920, 400
    b = [title(W, "YOLO assigns each object to the cell containing its midpoint",
               "an object that spans several cells still belongs to exactly one of them")]

    n, cell = 3, 88
    gx, gy = 70, 86
    b.append(photo(gx, gy, n * cell, n * cell, sky="#eef2f6"))
    for i in range(n + 1):
        b.append(seg(gx, gy + i * cell, gx + n * cell, gy + i * cell, LINE, 1.2))
        b.append(seg(gx + i * cell, gy, gx + i * cell, gy + n * cell, LINE, 1.2))

    # two cars whose midpoints sit in (1,0) and (1,2) — row 1 is middle
    # cell (row, col): left car in (1,0), right car in (1,2)
    cars = [
        (0, 1, gx + 8, gy + cell + 18, 110, 52, RED, "car"),
        (2, 1, gx + 2 * cell - 18, gy + cell + 28, 100, 44, BLUE, "car"),
    ]
    # draw cars first, then grid is already there; boxes on top
    for col, row, x, y, w, h, colr, _ in cars:
        b.append(car(x, y, w, h, fill="#bfdbfe", stroke=colr, label=""))
        mx, my = gx + (col + 0.5) * cell, gy + (row + 0.5) * cell
        # midpoint inside the assigned cell, not the geometric car centre if the car spills
        b.append(midpoint(mx, my, colr))
        b.append(rect(gx + col * cell, gy + row * cell, cell, cell, "out",
                      rx=0, sw=2.4, fill="none", stroke=colr))

    b.append(txt(gx + 1.5 * cell, gy + n * cell + 22, "3×3 grid for the illustration; a real YOLO uses 19×19", 10.5, fill=MUTED))

    px = 400
    b.append(panel(px, 86, 484, 250))
    b.append(txt(px + 20, 116, "Assignment rule", 13, anchor="start", weight="600"))
    b.append(txt(px + 20, 144, "Each object → the one cell that contains its midpoint.", 12, anchor="start"))
    b.append(txt(px + 20, 168, "The centre cell sees parts of both cars and is still labeled empty,", 12, anchor="start"))
    b.append(txt(px + 20, 188, "because neither midpoint lives there.", 12, anchor="start"))
    b.append(seg(px + 20, 208, px + 464, 208, LINE, 1.0))
    b.append(txt(px + 20, 234, "y has shape  3 × 3 × 8", 13, anchor="start", weight="600"))
    b.append(txt(px + 20, 258, "8 = pc + (bx, by, bh, bw) + (c1, c2, c3)", 12, anchor="start"))
    b.append(txt(px + 20, 282, "Empty cells:  [0,  ?, ?, ?, ?,  ?, ?, ?]", 12, anchor="start", fill=MUTED))
    b.append(txt(px + 20, 306, "A 19×19 grid is 19 × 19 × 8, and two midpoints rarely collide.", 11.5, anchor="start", fill=MUTED))

    b.append(caption(W / 2, H - 16, [
        "bx, by are relative to the cell (always in [0, 1]). bh, bw are fractions of the cell and can exceed 1 — the object may be larger than its cell.",
    ]))
    write(OUT, "det-yolo-grid.svg", W, H, b)


# --------------------------------------------------- 3.3 cell-relative box


def fig_yolo_cell_box():
    W, H = 840, 340
    b = [title(W, "Coordinates are relative to the assigned cell",
               "the midpoint stays inside the cell; the box itself may spill out")]

    cell = 140
    gx, gy = 80, 90
    # 2x2 of cells, assigned is lower-left
    for r in range(2):
        for c in range(2):
            fill = "#fef3c7" if (r, c) == (1, 0) else "#f9fafb"
            b.append(rect(gx + c * cell, gy + r * cell, cell, cell, "input",
                          rx=0, sw=1.3, fill=fill))
    b.append(txt(gx + cell / 2, gy + cell + 18, "(0, 0)", 10, fill=MUTED))
    b.append(txt(gx + cell / 2 + 36, gy + 2 * cell - 12, "(1, 1)", 10, fill=MUTED))
    b.append(txt(gx + cell / 2, gy + 1.5 * cell - 4, "assigned cell", 11, weight="600", fill=AMBER))

    # box larger than the cell, midpoint inside
    mx, my = gx + 0.55 * cell, gy + cell + 0.45 * cell
    bw, bh = 1.6 * cell, 1.15 * cell
    b.append(bbox(mx - bw / 2, my - bh / 2, bw, bh, RED, 2.2, label="car"))
    b.append(midpoint(mx, my))
    b.append(txt(mx + 12, my - 10, "(bx, by) ∈ [0, 1]", 11, anchor="start", fill=RED, weight="600"))
    b.append(txt(mx, my + bh / 2 + 18, "bh, bw can be > 1", 12, weight="600", fill=RED))

    px = 430
    b.append(panel(px, 90, 370, 188))
    b.append(txt(px + 18, 118, "Why this encoding", 13, anchor="start", weight="600"))
    b.append(txt(px + 18, 148, "bx, by always in [0, 1]", 12, anchor="start"))
    b.append(txt(px + 18, 168, "because the midpoint is inside the cell by definition.", 11.5, anchor="start", fill=MUTED))
    b.append(txt(px + 18, 200, "bh, bw may exceed 1", 12, anchor="start"))
    b.append(txt(px + 18, 220, "because a car is often larger than one grid cell.", 11.5, anchor="start", fill=MUTED))
    b.append(txt(px + 18, 252, "One convolutional pass produces every cell at once.", 11.5, anchor="start", fill=GREEN, weight="600"))

    write(OUT, "det-yolo-cell-box.svg", W, H, b)


# ---------------------------------------------------------------- 4.1 IoU


def fig_iou():
    W, H = 880, 360
    b = [title(W, "Intersection over Union",
               "how similar two boxes are — used for evaluation, NMS, and anchor assignment")]

    # two boxes
    ax, ay, aw, ah = 80, 110, 180, 130
    bx, by, bw, bh = 160, 150, 180, 120
    ix, iy = max(ax, bx), max(ay, by)
    iw, ih = min(ax + aw, bx + bw) - ix, min(ay + ah, by + bh) - iy
    union = aw * ah + bw * bh - iw * ih
    _ = union  # drawn as a formula, not a computed label

    b.append(rect(ax, ay, aw, ah, "out", rx=3, sw=2.0, fill="#fecaca", stroke=RED))
    b.append(txt(ax + 8, ay + 18, "predicted", 11, anchor="start", fill=RED, weight="600"))
    b.append(rect(bx, by, bw, bh, "pool", rx=3, sw=2.0, fill="#bfdbfe", stroke=BLUE))
    b.append(txt(bx + bw - 8, by + bh - 10, "ground truth", 11, anchor="end", fill=BLUE, weight="600"))
    b.append(rect(ix, iy, iw, ih, "conv", rx=0, sw=0, fill="#86efac", stroke="none"))
    b.append(txt(ix + iw / 2, iy + ih / 2 + 4, "∩", 16, fill=GREEN, weight="600"))

    px = 420
    b.append(panel(px, 86, 420, 200))
    b.append(txt(px + 20, 116, "IoU  =  area of ∩  /  area of ∪", 14, anchor="start", weight="600"))
    b.append(txt(px + 20, 148, f"intersection  =  {int(iw)} × {int(ih)}", 12, anchor="start"))
    b.append(txt(px + 20, 172, "union  =  A + B − intersection", 12, anchor="start"))
    b.append(txt(px + 20, 204, "A detection is correct if  IoU ≥ 0.5", 13, anchor="start", weight="600", fill=GREEN))
    b.append(txt(px + 20, 228, "0.5 is a convention, not a theorem. 0.6 or 0.7 is stricter.", 11.5, anchor="start", fill=MUTED))
    b.append(txt(px + 20, 256, "Perfect overlap  →  IoU = 1, because ∩ = ∪.", 11.5, anchor="start", fill=MUTED))

    b.append(caption(W / 2, H - 18, [
        "IoU shows up three times: scoring a predicted box, deciding which boxes non-max suppression should kill, and matching objects to anchor shapes.",
    ]))
    write(OUT, "det-iou.svg", W, H, b)


# ---------------------------------------------------------------- 4.3 NMS


def fig_nms():
    W, H = 920, 380
    b = [title(W, "Non-max suppression",
               "keep the highest-scoring box, suppress the ones that overlap it, repeat")]

    def scene(x0, y0, boxes, head):
        o = [txt(x0 + 190, y0 - 8, head, 12.5, weight="600")]
        o.append(photo(x0, y0, 380, 200))
        o.append(car(x0 + 40, y0 + 108, 100, 40, label=""))
        o.append(car(x0 + 230, y0 + 120, 90, 36, label=""))
        for x, y, w, h, score, col, keep in boxes:
            dash = None if keep else "5 3"
            sw = 2.2 if keep else 1.4
            o.append(bbox(x0 + x, y0 + y, w, h, col, sw, dash=dash, score=score))
        return "\n".join(o)

    before = [
        (28, 78, 130, 90, "0.9", RED, True),
        (48, 92, 120, 80, "0.7", AMBER, True),
        (18, 100, 110, 70, "0.62", BLUE, True),
        (220, 88, 120, 86, "0.8", GREEN, True),
    ]
    after = [
        (28, 78, 130, 90, "0.9", RED, True),
        (220, 88, 120, 86, "0.8", GREEN, True),
    ]
    b.append(scene(40, 96, before, "Before: several cells claim each car"))
    b.append(arrow(440, 196, 492, 196))
    b.append(scene(510, 96, after, "After: one box per object"))

    b.append(caption(W / 2, H - 42, [
        "1. Discard scores ≤ 0.6.   2. Output the highest remaining score.   3. Discard every leftover box with IoU ≥ 0.5 against it.   Repeat.",
        "Run this independently per class — a confident pedestrian must not suppress an overlapping car.",
    ]))
    write(OUT, "det-nms.svg", W, H, b)


# ----------------------------------------------------------- 5.2 anchors


def fig_anchors():
    W, H = 920, 400
    b = [title(W, "Anchor boxes: two objects, one cell",
               "each predefined shape gets its own 8-number slot in the cell's output")]

    gx, gy, s = 56, 100, 200
    b.append(photo(gx, gy, s, s))
    b.append(rect(gx, gy, s, s, "out", rx=0, sw=2.0, fill="none", stroke=AMBER, dash="6 4"))
    b.append(txt(gx + s / 2, gy - 12, "one grid cell", 11, fill=AMBER, weight="600"))
    b.append(person(gx + 70, gy + 24, 150, label=""))
    b.append(car(gx + 30, gy + 118, 150, 48, fill="#93c5fd", stroke=BLUE, label=""))
    mx, my = gx + s / 2, gy + s / 2
    b.append(midpoint(mx, my, AMBER))
    b.append(txt(mx + 10, my - 8, "shared midpoint", 10.5, anchor="start", fill=AMBER, weight="600"))

    # two anchors
    ax = 300
    b.append(txt(ax + 40, 92, "anchor 1", 11, fill=GREEN, weight="600"))
    b.append(rect(ax, 104, 44, 120, "conv", rx=3, sw=1.8, fill="none", stroke=GREEN, dash="5 3"))
    b.append(txt(ax + 22, 242, "tall, thin", 10.5, fill=MUTED))
    b.append(txt(ax + 130, 92, "anchor 2", 11, fill=BLUE, weight="600"))
    b.append(rect(ax + 80, 148, 120, 52, "pool", rx=3, sw=1.8, fill="none", stroke=BLUE, dash="5 3"))
    b.append(txt(ax + 140, 242, "wide, flat", 10.5, fill=MUTED))

    b.append(arrow(ax + 22, 256, ax + 22, 286))
    b.append(arrow(ax + 140, 256, ax + 140, 286))
    b.append(txt(ax + 22, 304, "pedestrian", 11, fill=GREEN, weight="600"))
    b.append(txt(ax + 140, 304, "car", 11, fill=BLUE, weight="600"))
    b.append(txt(ax + 80, 326, "highest IoU with the object's shape", 10.5, fill=MUTED))

    px = 560
    b.append(panel(px, 86, 330, 250))
    b.append(txt(px + 16, 114, "y per cell, 2 anchors", 13, anchor="start", weight="600"))
    b.append(txt(px + 16, 142, "anchor 1   ‖   anchor 2", 12, anchor="start"))
    b.append(txt(px + 16, 164, "8 numbers           8 numbers", 11, anchor="start", fill=MUTED))
    b.append(seg(px + 16, 180, px + 314, 180, LINE, 1.0))
    b.append(txt(px + 16, 206, "3×3, no anchors     →  3×3×8", 12, anchor="start"))
    b.append(txt(px + 16, 228, "3×3, 2 anchors      →  3×3×16", 12, anchor="start", weight="600"))
    b.append(txt(px + 16, 250, "19×19, 5, 3 classes →  19×19×40", 12, anchor="start"))
    b.append(txt(px + 16, 284, "Object → (cell of midpoint,", 11.5, anchor="start"))
    b.append(txt(px + 16, 304, "          anchor of best IoU)", 11.5, anchor="start"))

    b.append(caption(W / 2, H - 16, [
        "The real payoff is specialization: some units learn tall skinny things, others wide flat things. Two objects in one cell is rare on a 19×19 grid.",
    ]))
    write(OUT, "det-anchors.svg", W, H, b)


# ------------------------------------------------------ 6.3 YOLO pipeline


def fig_yolo_pipeline():
    W, H = 920, 300
    b = [title(W, "The complete YOLO test-time pipeline",
               "the network over-predicts; thresholding and per-class NMS leave one box per object")]

    stages = [
        (40, 100, 150, 90, "input", ["100×100×3", "image"]),
        (230, 90, 160, 110, "conv", ["ConvNet", "3×3×2×8"]),
        (430, 100, 150, 90, "flat", ["threshold", "drop pc ≤ 0.6"]),
        (620, 100, 150, 90, "fc", ["NMS", "once per class"]),
        (810, 100, 70, 90, "out", ["boxes"]),
    ]
    for x, y, w, h, role, rows in stages:
        b.append(rect(x, y, w, h, role, rx=5))
        b.append(lines(x + w / 2, y + h / 2 + 4 - (len(rows) - 1) * 8, rows, 12, weight="600", lh=16))
    for a, c in zip(stages, stages[1:]):
        b.append(arrow(a[0] + a[2] + 4, 145, c[0] - 4, 145))

    b.append(txt(310, 220, "S×S×B×(5+C)", 11, fill=MUTED))
    b.append(txt(505, 220, "most boxes die here", 11, fill=MUTED))
    b.append(txt(695, 220, "pedestrians, cars, … separately", 11, fill=MUTED))

    b.append(caption(W / 2, H - 18, [
        "Raw output for 19×19 with 5 anchors: 1,805 boxes. Almost all have near-zero pc. Threshold first, then NMS — not the other way around.",
    ]))
    write(OUT, "det-yolo-pipeline.svg", W, H, b)


# -------------------------------------------------------- 7.2 R-CNN family


def fig_rcnn_family():
    W, H = 920, 420
    b = [title(W, "Region proposals: the R-CNN family",
               "classify a few thousand plausible regions instead of every window")]

    rows = [
        ("R-CNN", [
            ("input", "image"),
            ("pool", "segment ~2k blobs"),
            ("fc", "classify each crop"),
            ("out", "box + class"),
        ], "slow: one ConvNet pass per proposal"),
        ("Fast R-CNN", [
            ("input", "image"),
            ("pool", "segment ~2k blobs"),
            ("conv", "one conv pass, shared"),
            ("out", "box + class"),
        ], "classification is convolutional; proposals still slow"),
        ("Faster R-CNN", [
            ("input", "image"),
            ("conv", "CNN region proposals"),
            ("conv", "shared features"),
            ("out", "box + class"),
        ], "both stages are ConvNets; still slower than YOLO"),
        ("YOLO  (one-stage)", [
            ("input", "image"),
            ("conv", "one ConvNet pass"),
            ("out", "all boxes + classes"),
        ], "no proposal stage at all"),
    ]
    for i, (name, steps, note) in enumerate(rows):
        y = 78 + i * 78
        b.append(txt(20, y + 28, name, 12.5, anchor="start", weight="600"))
        x = 168
        for j, (role, lab) in enumerate(steps):
            w = 150 if role != "out" else 110
            b.append(rect(x, y, w, 44, role, rx=4))
            b.append(txt(x + w / 2, y + 27, lab, 11))
            if j < len(steps) - 1:
                b.append(arrow(x + w + 3, y + 22, x + w + 21, y + 22))
            x += w + 28
        b.append(txt(168, y + 60, note, 10.5, anchor="start", fill=MUTED))

    write(OUT, "det-rcnn-family.svg", W, H, b)


# ------------------------------------------ 8.1 box vs segmentation


def fig_seg_vs_box():
    W, H = 880, 340
    b = [title(W, "A box is not an outline",
               "detection localizes compact objects; a road needs a label on every pixel")]

    # detection panel
    b.append(panel(40, 72, 380, 210, label="Detection"))
    b.append(photo(70, 108, 320, 150))
    b.append(car(100, 186, 80, 32, label=""))
    b.append(car(250, 178, 90, 36, label=""))
    b.append(bbox(88, 168, 104, 58, RED, 1.8, label="car"))
    b.append(bbox(238, 160, 114, 62, RED, 1.8, label="car"))
    b.append(txt(230, 276, "a box around the road would cover most of the image", 10.5, fill=MUTED))

    # segmentation panel
    b.append(panel(460, 72, 380, 210, label="Semantic segmentation"))
    b.append(photo(490, 108, 320, 150))
    # drivable surface as a green polygon
    b.append(poly([
        (500, 200), (800, 200), (800, 248), (500, 248),
    ], fill="#86efac", stroke=GREEN, sw=1.6))
    b.append(car(520, 186, 80, 32, label=""))
    b.append(car(670, 178, 90, 36, label=""))
    b.append(txt(650, 276, "every pixel: road / car / background", 10.5, fill=MUTED))

    b.append(caption(W / 2, H - 18, [
        "Output of segmentation: an h × w × n_classes volume, then argmax over classes at each pixel.",
    ]))
    write(OUT, "det-seg-vs-box.svg", W, H, b)


# ----------------------------------------------- 8.4 encoder–decoder


def fig_encoder_decoder():
    _pipeline(
        OUT,
        "det-encoder-decoder.svg",
        "A segmentation network has to grow the spatial size back",
        [
            dict(kind="vol", s=128, c=3, shape="H×W×3", name="Input", role="input"),
            dict(kind="vol", s=64, c=64, shape="H/2×W/2×64", name="CONV", role="conv"),
            dict(kind="vol", s=32, c=128, shape="H/4×W/4×128", name="CONV", role="conv"),
            dict(kind="vol", s=16, c=256, shape="H/8×W/8×256", name="CONV", role="conv5"),
            dict(kind="vol", s=32, c=128, shape="H/4×W/4×128", name="CONV", role="conv1"),
            dict(kind="vol", s=64, c=64, shape="H/2×W/2×64", name="CONV", role="conv1"),
            dict(kind="vol", s=128, c=3, shape="H×W×C", name="Output", role="out"),
        ],
        [
            "convolution + pool,\nnH ↓  nC ↑",
            "convolution\n+ pool",
            "convolution\n+ pool",
            "transpose conv,\nnH ↑  nC ↓",
            "transpose\nconvolution",
            "convolution 1×1×64,\nC = n_classes",
        ],
        sub="first half: spatial size shrinks and channels grow. second half: spatial size grows back. the grow operation is a transpose convolution.",
        note="Drop the classifier head of a ConvNet and replace it with an upsampling path. That is the essential structural change.",
        leg=[("input", "input"), ("conv", "encoder CONV"), ("conv5", "bottleneck"),
             ("conv1", "decoder / transpose"), ("out", "per-pixel classes")],
    )


# ----------------------------------------------- 9. transpose conv


TCONV_IN = [[1, 2], [3, 4]]
TCONV_F = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
TCONV_PAD = [
    [1, 1, 3, 2, 2, 0],
    [0, 0, 0, 0, 0, 0],
    [2, 2, 4, 2, 2, 0],
    [0, 0, 0, 0, 0, 0],
    [-3, -3, -7, -4, -4, 0],
    [0, 0, 0, 0, 0, 0],
]
TCONV_OUT = [
    [0, 0, 0, 0],
    [2, 4, 2, 2],
    [0, 0, 0, 0],
    [-3, -7, -4, -4],
]
TCONV_CELLS = [
    (0, 0, 1, RED, "#fecaca"),
    (0, 1, 2, AMBER, "#fef08a"),
    (1, 0, 3, GREEN, "#bbf7d0"),
    (1, 1, 4, BLUE, "#bfdbfe"),
]


def _op(x, y, s, size=20):
    return txt(x, y, s, size, fill=MUTED, weight="600")


def _scale_f(v):
    return [[v * TCONV_F[r][c] for c in range(3)] for r in range(3)]


def _paste(v, i, j, n=6):
    g = [[0] * n for _ in range(n)]
    sr, sc = 2 * i, 2 * j
    for a in range(3):
        for b in range(3):
            g[sr + a][sc + b] = v * TCONV_F[a][b]
    return g


def _in_fills(hi=None):
    fills = [["#ffffff", "#ffffff"], ["#ffffff", "#ffffff"]]
    lookup = {(r, c): tint for r, c, _v, _col, tint in TCONV_CELLS}
    if hi is None:
        for (r, c), tint in lookup.items():
            fills[r][c] = tint
    else:
        fills[hi[0]][hi[1]] = lookup[hi]
    return fills


def fig_tconv_vs():
    W, H = 920, 318
    b = [title(W, "Normal convolution versus transpose convolution",
               "same 3×3 filter, opposite placement — one shrinks the map, the other grows it")]
    b.append(seg(460, 64, 460, 286, LINE, 1.0, dash="4 4"))

    cell = 18
    # ---- normal: filter sits on the 6×6 input
    blank6 = [[""] * 6 for _ in range(6)]
    blank4 = [[""] * 4 for _ in range(4)]
    filt_lab = [[" "] * 3 for _ in range(3)]
    xn, yn = 36, 88
    b.append(txt(230, 70, "Normal convolution", 13, weight="600"))
    b.append(matrix(xn, yn, blank6, cell, fills=lambda v, i, j: "#f3f4f6"))
    b.append(outline(xn, yn, 0, 0, 3, 3, cell, RED))
    b.append(txt(xn + 54, yn + 118, "6×6×3 input", 11, weight="600"))
    b.append(arrow(xn + 114, yn + 54, xn + 146, yn + 54))
    xf = xn + 158
    b.append(matrix(xf, yn + 27, filt_lab, cell, fills=lambda v, i, j: C["conv"][0],
                    stroke=C["conv"][1]))
    b.append(txt(xf + 27, yn + 90, "3×3×3", 10.5, fill=MUTED))
    b.append(txt(xf + 27, yn + 104, "5 filters", 10.5, fill=MUTED))
    b.append(arrow(xf + 60, yn + 54, xf + 92, yn + 54))
    xo = xf + 104
    b.append(matrix(xo, yn + 18, blank4, cell, fills=lambda v, i, j: C["out"][0],
                    stroke=C["out"][1]))
    b.append(outline(xo, yn + 18, 0, 0, 1, 1, cell, RED))
    b.append(txt(xo + 36, yn + 100, "4×4×5 output", 11, weight="600"))
    b.append(txt(230, 232, "filter sits on the input", 12, weight="600", fill=RED))
    b.append(txt(230, 252, "one window collapses to one number  ·  output is smaller", 11, fill=MUTED))

    # ---- transpose: filter sits on the output
    xt, yt = 500, 106
    blank2 = [[""] * 2 for _ in range(2)]
    b.append(txt(690, 70, "Transpose convolution", 13, weight="600"))
    b.append(matrix(xt, yt + 18, blank2, cell, fills=lambda v, i, j: C["input"][0]))
    b.append(outline(xt, yt + 18, 0, 0, 1, 1, cell, RED))
    b.append(txt(xt + 18, yt + 64, "2×2 input", 11, weight="600"))
    b.append(arrow(xt + 42, yt + 36, xt + 74, yt + 36))
    xf2 = xt + 86
    b.append(matrix(xf2, yt, filt_lab, cell, fills=lambda v, i, j: C["conv"][0],
                    stroke=C["conv"][1]))
    b.append(txt(xf2 + 27, yt + 64, "3×3 filter", 10.5, fill=MUTED))
    b.append(arrow(xf2 + 60, yt + 36, xf2 + 92, yt + 36))
    xo2 = xf2 + 104
    b.append(matrix(xo2, yt - 18, blank4, cell, fills=lambda v, i, j: C["out"][0],
                    stroke=C["out"][1]))
    b.append(outline(xo2, yt - 18, 0, 0, 3, 3, cell, RED))
    b.append(txt(xo2 + 36, yt + 64, "4×4 output", 11, weight="600"))
    b.append(txt(690, 232, "filter sits on the output", 12, weight="600", fill=RED))
    b.append(txt(690, 252, "one number expands to a whole window  ·  output is bigger", 11, fill=MUTED))

    b.append(txt(W / 2, H - 14,
                 "6×6×3 with five 3×3×3 filters becomes 4×4×5. A 2×2 map with a 3×3 filter becomes 4×4.",
                 10.5, fill=MUTED))
    write(OUT, "det-tconv-vs.svg", W, H, b)


def fig_tconv_mechanics():
    W, H = 920, 300
    b = [title(W, "Mechanics of a transpose convolution",
               "input, filter, and output of the running example  ·  f = 3, p = 1, s = 2")]
    cell = 32
    iw, fw, ow = mat_w(TCONV_IN, cell), mat_w(TCONV_F, cell), mat_w(TCONV_OUT, cell)
    total = iw + 56 + fw + 56 + ow
    xi = (W - total) / 2
    xf = xi + iw + 56
    xo = xf + fw + 56
    yi, yf, yo = 110, 94, 78

    b.append(matrix(xi, yi, TCONV_IN, cell, fills=_in_fills()))
    b.append(_op(xi + iw + 28, 142, "×"))
    b.append(matrix(xf, yf, TCONV_F, cell, fills=lambda v, i, j: diverge(v, 1)))
    b.append(_op(xf + fw + 28, 142, "→"))
    b.append(matrix(xo, yo, TCONV_OUT, cell, fills=lambda v, i, j: diverge(v, 7)))

    b.append(txt(xi + iw / 2, 186, "input  2×2", 11.5, weight="600"))
    b.append(txt(xf + fw / 2, 202, "filter  3×3", 11.5, weight="600"))
    b.append(txt(xo + ow / 2, 218, "output  4×4", 11.5, weight="600"))
    b.append(txt(xi + iw / 2, 204, "four numbers", 10.5, fill=MUTED))
    b.append(txt(xf + fw / 2, 220, "learned weights", 10.5, fill=MUTED))
    b.append(txt(xo + ow / 2, 236, "after crop", 10.5, fill=MUTED))

    b.append(caption(W / 2, H - 28, [
        "Each input number scales this whole filter and is stamped onto the output — that is the opposite of a normal convolution.",
        "The five steps below unpack exactly how [[1, 2], [3, 4]] and this filter produce the 4×4 on the right.",
    ]))
    write(OUT, "det-tconv-mechanics.svg", W, H, b)


def fig_tconv_step1():
    W, H = 920, 248
    b = [title(W, "Step 1  ·  take one value from the input",
               "the running example starts with the top-left 1; the same steps then run for 2, 3, and 4")]
    cell = 42
    xi, yi = 250, 86
    b.append(matrix(xi, yi, TCONV_IN, cell, fills=_in_fills((0, 0)), sw=1.2))
    b.append(outline(xi, yi, 0, 0, 1, 1, cell, RED, sw=2.6))
    b.append(txt(xi + 42, yi + 100, "input", 11.5, weight="600"))
    b.append(arrow(xi + 96, yi + 21, xi + 170, yi + 21, RED, 1.6))
    b.append(matrix(xi + 186, yi, [[1]], cell, fills=lambda v, i, j: "#fecaca",
                    stroke=RED, sw=2.0, fs=16))
    b.append(txt(xi + 207, yi + 100, "the value we use next", 11.5, fill=RED, weight="600"))
    b.append(txt(W / 2, H - 16,
                 "A transpose convolution never looks at a window of the input. It picks one number and expands it.",
                 10.5, fill=MUTED))
    write(OUT, "det-tconv-step1.svg", W, H, b)


def fig_tconv_step2():
    W, H = 920, 268
    b = [title(W, "Step 2  ·  multiply the entire filter by that value",
               "every weight is scaled; nothing is summed yet")]
    cell = 30
    scaled = _scale_f(1)
    xf, yf = 168, 92
    b.append(matrix(xf, yf + 15, [[1]], cell, fills=lambda v, i, j: "#fecaca",
                    stroke=RED, sw=1.8, fs=14))
    b.append(txt(xf + 15, yf + 117, "input value", 10.5, fill=MUTED))
    b.append(_op(xf + 52, yf + 48, "×"))
    xs = xf + 78
    b.append(matrix(xs, yf, TCONV_F, cell, fills=lambda v, i, j: diverge(v, 1)))
    b.append(txt(xs + 45, yf + 117, "3×3 filter", 10.5, fill=MUTED))
    b.append(_op(xs + 108, yf + 48, "="))
    xo = xs + 134
    b.append(matrix(xo, yf, scaled, cell, fills=lambda v, i, j: diverge(v, 1),
                    stroke=RED, sw=1.6))
    b.append(txt(xo + 45, yf + 117, "1 × filter", 11.5, fill=RED, weight="600"))

    # a second, smaller reminder for input = 2
    xr = 668
    b.append(txt(xr + 45, 86, "same step for 2", 11, fill=AMBER, weight="600"))
    b.append(matrix(xr - 50, 104, [[2]], 26, fills=lambda v, i, j: "#fef08a",
                    stroke=AMBER, sw=1.6, fs=12))
    b.append(_op(xr - 8, 138, "×", 16))
    b.append(matrix(xr + 8, 104, _scale_f(2), 26, fills=lambda v, i, j: diverge(v, 2),
                    stroke=AMBER, sw=1.4, fs=10))
    b.append(txt(W / 2, H - 16,
                 "Input 3 would give [[3, 3, 3], [0, 0, 0], [−3, −3, −3]]; input 4 would give [[4, 4, 4], [0, 0, 0], [−4, −4, −4]].",
                 10.5, fill=MUTED))
    write(OUT, "det-tconv-step2.svg", W, H, b)


def fig_tconv_step3():
    W, H = 920, 340
    b = [title(W, "Step 3  ·  paste the block at a stride-s offset",
               "input (i, j) writes starting at (s · i,  s · j) on the padded 6×6 canvas  ·  here s = 2")]
    cell = 18
    for n, (i, j, val, col, tint) in enumerate(TCONV_CELLS):
        x0 = 36 + n * 224
        y0 = 78
        sr, sc = 2 * i, 2 * j
        b.append(txt(x0 + 54, y0, f"input ({i}, {j})  =  {val}", 11, fill=col, weight="600"))
        b.append(txt(x0 + 54, y0 + 16, f"starts at ({sr}, {sc})", 10.5, fill=MUTED))
        canvas = _paste(val, i, j)
        ox, oy = x0, y0 + 28

        def fills(v, r, c, sr=sr, sc=sc, tint=tint):
            if sr <= r < sr + 3 and sc <= c < sc + 3:
                return tint if v != 0 else "#ffffff"
            return "#f9fafb"

        b.append(matrix(ox, oy, canvas, cell, fills=fills, fs=9, sw=0.7))
        b.append(outline(ox, oy, sr, sc, 3, 3, cell, col, sw=2.0))
    b.append(caption(W / 2, H - 28, [
        "Because s = 2, neighbouring input cells land two pixels apart, so the 3×3 stamps overlap by one row and one column.",
        "The pale cells are the p = 1 padding border — they are part of the canvas the filter is pasted onto, not of the final 4×4.",
    ]))
    write(OUT, "det-tconv-step3.svg", W, H, b)


def fig_tconv_step4():
    W, H = 920, 332
    b = [title(W, "Step 4  ·  where pastes overlap, add",
               "the 4 and −7 on the canvas are sums, not single stamps")]
    cell = 26
    xg, yg = 48, 78

    def pad_fills(v, i, j):
        if (i, j) == (2, 2):
            return "#fecaca"
        if (i, j) == (4, 2):
            return "#bfdbfe"
        return diverge(v, 7)

    b.append(matrix(xg, yg, TCONV_PAD, cell, fills=pad_fills, fs=11))
    b.append(outline(xg, yg, 2, 2, 1, 1, cell, RED, sw=2.4))
    b.append(outline(xg, yg, 4, 2, 1, 1, cell, BLUE, sw=2.4))
    b.append(txt(xg + 78, yg + 168, "padded 6×6, all four stamps added", 11, weight="600"))

    # four 3×3 outlines in muted colour to show the overlap geometry
    for i, j, _v, col, _t in TCONV_CELLS:
        b.append(outline(xg, yg, 2 * i, 2 * j, 3, 3, cell, col, sw=1.3, dash="4 3"))

    xr = 520
    b.append(txt(xr, 92, "Cell (2, 2) is hit by every stamp", 12, anchor="start", weight="600", fill=RED))
    rows = [
        "from 1:  1 × filter[2, 2]  =  −1",
        "from 2:  2 × filter[2, 0]  =  −2",
        "from 3:  3 × filter[0, 2]  =  +3",
        "from 4:  4 × filter[0, 0]  =  +4",
    ]
    for k, row in enumerate(rows):
        b.append(txt(xr, 118 + k * 20, row, 11.5, anchor="start", fill=TCONV_CELLS[k][3]))
    b.append(seg(xr, 198, xr + 250, 198, INK, 1.0))
    b.append(txt(xr, 220, "−1  −  2  +  3  +  4   =   4", 13, anchor="start", weight="600", fill=RED))

    b.append(txt(xr, 258, "Cell (4, 2) is hit by only 3 and 4", 12, anchor="start", weight="600", fill=BLUE))
    b.append(txt(xr, 280, "−3  +  (−4)   =   −7", 13, anchor="start", weight="600", fill=BLUE))

    b.append(txt(W / 2, H - 14,
                 "Never overwrite. If two (or four) stamps land on the same cell, the values are added.",
                 10.5, fill=MUTED))
    write(OUT, "det-tconv-step4.svg", W, H, b)


def fig_tconv_step5():
    W, H = 920, 300
    b = [title(W, "Step 5  ·  crop the padding border",
               "p = 1 means drop the outer ring; what remains is the 4×4 output")]
    cell = 24
    xg, yg = 80, 78

    def crop_fills(v, i, j):
        border = i in (0, 5) or j in (0, 5)
        if border:
            return C["pad"][0]
        return diverge(v, 7)

    b.append(matrix(xg, yg, TCONV_PAD, cell, fills=crop_fills, fs=10))
    b.append(outline(xg, yg, 1, 1, 4, 4, cell, RED, sw=2.4))
    b.append(txt(xg + 72, yg + 156, "padded 6×6", 11, weight="600"))
    b.append(txt(xg + 72, yg + 172, "yellow ring = padding, discarded", 10.5, fill=MUTED))

    b.append(arrow(xg + 160, yg + 72, xg + 220, yg + 72, RED, 1.6))
    b.append(txt(xg + 190, yg + 60, "crop p = 1", 11, fill=RED, weight="600"))

    xo = 560
    b.append(matrix(xo, yg + 24, TCONV_OUT, 28, fills=lambda v, i, j: diverge(v, 7)))
    b.append(txt(xo + 56, yg + 148, "4×4 output", 11.5, weight="600"))
    b.append(txt(xo + 56, yg + 166, "this is what the layer returns", 10.5, fill=MUTED))

    b.append(txt(W / 2, H - 16,
                 "The padding was applied to the output canvas before pasting, which is why it is cropped at the end rather than added to the input.",
                 10.5, fill=MUTED))
    write(OUT, "det-tconv-step5.svg", W, H, b)


def _coverage(n, f, origins):
    g = [[0] * n for _ in range(n)]
    for i, j in origins:
        for a in range(f):
            for b in range(f):
                r, c = i + a, j + b
                if 0 <= r < n and 0 <= c < n:
                    g[r][c] += 1
    return g


def _cover_fill(v, _i, _j):
    return {0: "#f9fafb", 1: "#fef3c7", 2: "#fdba74", 3: "#f97316", 4: "#dc2626"}.get(v, "#fecaca")


def fig_tconv_checkerboard():
    W, H = 920, 318
    b = [title(W, "Uneven overlap is a checkerboard",
               "the number in each cell is how many f×f stamps cover it  ·  s = 2, four stamps")]
    cell = 26
    origins = [(0, 0), (0, 2), (2, 0), (2, 2)]

    # f = 3, s = 2 — the running example
    left = _coverage(6, 3, origins)
    x1, y1 = 118, 86
    b.append(txt(x1 + 78, 72, "f = 3, s = 2  ·  not divisible", 12.5, weight="600", fill=RED))
    b.append(matrix(x1, y1, left, cell, fills=_cover_fill, fs=12,
                    tfill=lambda v, i, j: "#ffffff" if v >= 3 else INK))
    b.append(txt(x1 + 78, y1 + 168, "neighbours get 1, then 2, then 4", 11, fill=RED, weight="600"))
    b.append(txt(x1 + 78, y1 + 186, "that periodic high / low grid is the artifact", 10.5, fill=MUTED))

    b.append(seg(460, 70, 460, 268, LINE, 1.0, dash="4 4"))

    # f = 4, s = 2 — uniform interior
    right = _coverage(6, 4, origins)
    x2, y2 = 546, 86
    b.append(txt(x2 + 78, 72, "f = 4, s = 2  ·  divisible", 12.5, weight="600", fill=GREEN))
    b.append(matrix(x2, y2, right, cell, fills=_cover_fill, fs=12,
                    tfill=lambda v, i, j: "#ffffff" if v >= 3 else INK))
    b.append(txt(x2 + 78, y2 + 168, "interior cells all get 4", 11, fill=GREEN, weight="600"))
    b.append(txt(x2 + 78, y2 + 186, "overlap is uniform, so no checkerboard bias", 10.5, fill=MUTED))

    b.append(txt(W / 2, H - 16,
                 "Same four stamp origins. Only the filter size changes. Counts, not values: this is how often each output cell is written.",
                 10.5, fill=MUTED))
    write(OUT, "det-tconv-checkerboard.svg", W, H, b)


# --------------------------------------------------------------- 10 U-Net


def _scene(x, y, w, h):
    """A tiny street photo: sky, road, one car."""
    return "\n".join([
        photo(x, y, w, h),
        car(x + 0.22 * w, y + 0.62 * h, 0.42 * w, 0.22 * h),
    ])


def _mask(x, y, w, h):
    """The matching per-pixel map: road / car / background."""
    road_y = y + 0.62 * h
    return "\n".join([
        rect(x, y, w, h, "out", rx=4, sw=1.3, fill="#93c5fd", stroke=LINE),
        rect(x, road_y, w, y + h - road_y, "conv", rx=0, sw=0,
             fill="#86efac", stroke="none"),
        rect(x + 0.22 * w, road_y - 0.12 * h, 0.42 * w, 0.28 * h, "pool",
             rx=3, sw=0, fill="#2563eb", stroke="none"),
        rect(x, y, w, h, "out", rx=4, sw=1.3, fill="none", stroke=LINE),
    ])


def fig_unet_idea():
    W, H = 920, 300
    b = [title(W, "The idea: go down to understand, come back up to label every pixel",
               "a classifier keeps shrinking until one answer is left. segmentation has to grow the map back.")]

    # input photo
    b.append(_scene(28, 88, 88, 88))
    b.append(txt(72, 192, "image", 11, weight="600"))

    specs = [
        (64, 14, "input", "H×W"),
        (48, 22, "conv", ""),
        (22, 40, "conv5", "code"),
        (48, 22, "conv1", ""),
        (64, 14, "out", "H×W×C"),
    ]
    xs = [140, 268, 396, 524, 652]
    cy = 132
    infos = []
    for x, (face, depth, role, lab) in zip(xs, specs):
        y0 = cy - face / 2
        svg, info = volume3d(x, y0, face, face, depth, role, label=lab, fs=10)
        b.append(svg)
        infos.append(info)
    for a, c in zip(infos, infos[1:]):
        b.append(arrow(a["right"] + 4, cy, c["left"] - 4, cy))

    b.append(_mask(800, 88, 88, 88))
    b.append(txt(844, 192, "mask", 11, weight="600"))
    b.append(arrow(infos[-1]["right"] + 4, cy, 792, cy))

    b.append(txt((infos[0]["cx"] + infos[2]["cx"]) / 2, 214,
                 "down:  the picture gets smaller, the meaning gets richer", 11.5, fill=MUTED))
    b.append(txt((infos[2]["cx"] + infos[4]["cx"]) / 2, 214,
                 "up:  the picture comes back, one class per pixel", 11.5, fill=MUTED))
    b.append(txt(infos[2]["cx"], 236, "here you know what, but not which pixel",
                 11, fill=RED, weight="600"))

    b.append(txt(W / 2, H - 16,
                 "That is the whole idea. The next figure is how U-Net actually arranges the layers.",
                 10.5, fill=MUTED))
    write(OUT, "det-unet-idea.svg", W, H, b)


def _crop_lecture_photos():
    """Cut the car and the output map out of graphs/unet_architecture.png."""
    from PIL import Image
    src = pathlib.Path(__file__).resolve().parent.parent / "graphs" / "unet_architecture.png"
    im = Image.open(src).convert("RGB")
    car = im.crop((34, 161, 222, 357)).resize((180, 180), Image.Resampling.LANCZOS)
    mask = im.crop((1587, 156, 1781, 354)).resize((180, 180), Image.Resampling.LANCZOS)
    car_p, mask_p = OUT / "unet_car.png", OUT / "unet_mask.png"
    car.save(car_p, optimize=True)
    mask.save(mask_p, optimize=True)
    return car_p, mask_p


def _png(x, y, w, h, path):
    raw = pathlib.Path(path).read_bytes()
    uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    return (
        f'<image href="{uri}" x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" '
        f'preserveAspectRatio="xMidYMid slice"/>'
    )


# blues taken from the lecture figure
_U_NAVY, _U_CYAN = "#1e4a8c", "#3db5e8"
_U_MAGENTA = "#d946ef"


def _ubar(x, y, w, h, kind="navy"):
    fill = _U_NAVY if kind == "navy" else _U_CYAN
    return rect(x, y, w, h, "pool", rx=1.2, sw=1.05, fill=fill, stroke=fill)


def _uchain(parts, x, cy, h, gap=13):
    """Horizontal conv+ReLU chain. parts: [(width, kind), ...]."""
    out, boxes, xi = [], [], x
    for i, (w, kind) in enumerate(parts):
        boxes.append((xi, w))
        out.append(_ubar(xi, cy - h / 2, w, h, kind))
        if i < len(parts) - 1:
            out.append(arrow(xi + w + 1.5, cy, xi + w + gap - 2, cy, INK, 1.2))
            xi += w + gap
        else:
            xi += w
    first_x, first_w = boxes[0]
    last_x, last_w = boxes[-1]
    info = dict(
        L=x, R=xi, T=cy - h / 2, B=cy + h / 2, cx=(x + xi) / 2, cy=cy,
        first_cx=first_x + first_w / 2, last_cx=last_x + last_w / 2,
    )
    return "\n".join(out), info


def fig_unet():
    """U-Net matching graphs/unet_architecture.png, bar for bar."""
    car_p, mask_p = _crop_lecture_photos()
    W, H = 960, 520
    b = [title(W, "U-Net architecture",
               "black arrow = conv + ReLU;  red ↓ = max pool;  green ↑ = transpose conv")]

    # five spatial levels, counted off the lecture PNG
    hs = [92, 64, 42, 22, 16]
    cys = [118, 220, 300, 362, 410]
    tw = [10, 14, 22, 30, 36]         # thicker = more channels
    # every encoder block, and the bottleneck: 3 navy convs
    enc_spec = [[(tw[i], "navy")] * 3 for i in range(5)]
    enc_spec[4] = [(48, "navy")] * 3   # wider bottom row, spans the U
    # every decoder block: skip copy (navy) + two convs (cyan)
    dec_spec = [[(tw[i], "navy")] + [(tw[i], "cyan")] * 2 for i in range(4)]
    enc_x = [124, 154, 194, 242, 338]
    dec_x = [668, 618, 558, 478]

    enc, dec = [], {}
    for i, (spec, x0) in enumerate(zip(enc_spec, enc_x)):
        svg, info = _uchain(spec, x0, cys[i], hs[i], gap=14 if i < 3 else 12)
        b.append(svg)
        enc.append(info)
    for i, (spec, x0) in enumerate(zip(dec_spec, dec_x)):
        svg, info = _uchain(spec, x0, cys[i], hs[i], gap=14 if i < 3 else 12)
        b.append(svg)
        dec[i] = info

    # max pool: last (right) bar of the upper block → first (left) bar of the lower
    for a, c in zip(enc, enc[1:]):
        b.append(arrow(a["last_cx"], a["B"] + 4, c["first_cx"], c["T"] - 4, RED, 1.8))

    # transpose conv: last (right) bar of the lower block → first (left) bar of the upper
    b.append(arrow(enc[4]["last_cx"], enc[4]["T"] - 4,
                   dec[3]["first_cx"], dec[3]["B"] + 3, GREEN, 1.8))
    for lo, hi in ((3, 2), (2, 1), (1, 0)):
        b.append(arrow(dec[lo]["last_cx"], dec[lo]["T"] - 3,
                       dec[hi]["first_cx"], dec[hi]["B"] + 3, GREEN, 1.8))

    # skip connections (solid grey)
    for lvl in (0, 1, 2, 3):
        e, d = enc[lvl], dec[lvl]
        b.append(arrow(e["R"] + 6, e["cy"], d["L"] - 6, d["cy"], LINE, 2.1))

    # cropped lecture photos — same height as the top bars
    ph = hs[0]
    b.append(_png(18, cys[0] - ph / 2, ph, ph, car_p))
    b.append(rect(18, cys[0] - ph / 2, ph, ph, "input", rx=2, sw=1.15,
                  fill="none", stroke=LINE))

    ox = dec[0]["R"] + 38
    b.append(arrow(dec[0]["R"] + 4, cys[0], ox - 4, cys[0], _U_MAGENTA, 1.7))
    b.append(txt((dec[0]["R"] + ox) / 2, cys[0] - hs[0] / 2 - 7, "1×1",
                 9.5, fill=_U_MAGENTA, weight="600"))
    b.append(_png(ox, cys[0] - ph / 2, ph, ph, mask_p))
    b.append(rect(ox, cys[0] - ph / 2, ph, ph, "out", rx=2, sw=1.15,
                  fill="none", stroke=LINE))

    # lecture-style arrow legend, right of the lower decoder
    lx, ly = 820, 268
    b.append(arrow(lx, ly, lx + 20, ly, INK, 1.25))
    b.append(txt(lx + 26, ly + 4, "conv + ReLU", 11, anchor="start"))
    b.append(arrow(lx + 10, ly + 26, lx + 10, ly + 44, RED, 1.5))
    b.append(txt(lx + 26, ly + 38, "max pool", 11, anchor="start"))
    b.append(arrow(lx + 10, ly + 70, lx + 10, ly + 52, GREEN, 1.5))
    b.append(txt(lx + 26, ly + 64, "transpose conv", 11, anchor="start"))
    b.append(arrow(lx, ly + 88, lx + 20, ly + 88, LINE, 1.6))
    b.append(txt(lx + 26, ly + 92, "skip connection", 11, anchor="start"))
    b.append(arrow(lx, ly + 112, lx + 20, ly + 112, _U_MAGENTA, 1.45))
    b.append(txt(lx + 26, ly + 116, "conv 1×1", 11, anchor="start"))

    b.append(txt(enc[2]["cx"], 458, "encoder", 12, weight="600"))
    b.append(txt(enc[2]["cx"], 474, "nH ↓    nC ↑", 10.5, fill=MUTED))
    b.append(txt(dec[2]["cx"], 458, "decoder", 12, weight="600"))
    b.append(txt(dec[2]["cx"], 474, "nH ↑    nC ↓", 10.5, fill=MUTED))
    write(OUT, "det-unet.svg", W, H, b)


# ---------------------------------------------------------------------- main

FIGURES = [
    fig_tasks,
    fig_bbox_label,
    fig_landmarks,
    fig_sliding_windows,
    fig_sliding_fc,
    fig_fc_to_conv,
    fig_conv_windows,
    fig_yolo_grid,
    fig_yolo_cell_box,
    fig_iou,
    fig_nms,
    fig_anchors,
    fig_yolo_pipeline,
    fig_rcnn_family,
    fig_seg_vs_box,
    fig_encoder_decoder,
    fig_tconv_vs,
    fig_tconv_mechanics,
    fig_tconv_step1,
    fig_tconv_step2,
    fig_tconv_step3,
    fig_tconv_step4,
    fig_tconv_step5,
    fig_tconv_checkerboard,
    fig_unet_idea,
    fig_unet,
]


def build():
    for f in FIGURES:
        f()


if __name__ == "__main__":
    build()
    print(f"{len(FIGURES)} detection figures written")
