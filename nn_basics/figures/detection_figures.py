"""Generate the diagrams used by object_detection.md.

Driven by make_figures.py; can also be run directly.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from svgkit import (  # noqa: E402
    AMBER, ARCH_LEGEND, BLUE, C, GREEN, INK, LINE, MUTED, PURPLE, RED,
    arrow, caption, circle, legend, lines, panel, path, pipeline as _pipeline,
    poly, rect, seg, title, txt, write,
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


def fig_fc_to_conv():
    _pipeline(
        OUT,
        "det-fc-to-conv.svg",
        "A fully connected layer is a convolution whose filter covers the whole volume",
        [
            dict(kind="vol", s=14, c=3, shape="14×14×3", name="input", role="input"),
            dict(kind="vol", s=10, c=16, shape="10×10×16", name="CONV 5×5", role="conv"),
            dict(kind="vol", s=5, c=16, shape="5×5×16", name="POOL 2×2", role="pool"),
            dict(kind="vol", s=1, c=400, shape="1×1×400", name="CONV 5×5 ×400", role="conv5"),
            dict(kind="vol", s=1, c=400, shape="1×1×400", name="1×1 ×400", role="conv1"),
            dict(kind="vol", s=1, c=4, shape="1×1×4", name="softmax", role="out"),
        ],
        [
            "16 filters\n5×5",
            "max pool\nf=2, s=2",
            "was FC 400\nnow 400 filters",
            "was FC 400\nnow 1×1 ×400",
            "4 filters\n+ softmax",
        ],
        sub="same numbers, different bookkeeping — each of the 400 values is still an arbitrary linear function of the whole 5×5×16 input",
        note="A 5×5×16 filter over a 5×5×16 volume produces one number. 400 of them produce 1×1×400, which is exactly what the dense layer computed.",
        leg=ARCH_LEGEND,
    )


# --------------------------------------------- 2.4 convolutional windows


def fig_conv_windows():
    W, H = 920, 400
    b = [title(W, "Convolutional implementation of sliding windows",
               "one forward pass over the whole image evaluates every window, with the overlapping computation shared")]

    # four naive crops
    b.append(txt(196, 72, "Naive: four independent 14×14 crops", 12.5, weight="600"))
    for i, (dx, dy, col) in enumerate([(0, 0, RED), (18, 0, AMBER), (0, 18, GREEN), (18, 18, BLUE)]):
        b.append(rect(56 + dx, 92 + dy, 84, 84, "input", rx=3, sw=1.6,
                      fill="#f3f4f6", stroke=col, dash="4 3"))
    b.append(txt(116, 210, "16×16 image, stride 2", 11, fill=MUTED))
    b.append(txt(116, 228, "4 ConvNet passes, mostly duplicated", 11, fill=MUTED))

    b.append(arrow(214, 150, 268, 150))
    b.append(lines(241, 124, ["same", "weights"], 10.5, fill=MUTED))

    # full forward
    b.append(txt(560, 72, "Convolutional: one pass, 2×2×4 output", 12.5, weight="600"))
    stages = [
        (280, 108, 72, 72, "input", "16×16×3"),
        (372, 116, 60, 56, "conv", "12×12×16"),
        (452, 124, 48, 40, "pool", "6×6×16"),
        (520, 128, 40, 32, "conv5", "2×2×400"),
        (580, 128, 40, 32, "conv1", "2×2×400"),
        (640, 128, 40, 32, "out", "2×2×4"),
    ]
    for x, y, w, h, role, lab in stages:
        b.append(rect(x, y, w, h, role, rx=3))
        b.append(txt(x + w / 2, y + h + 16, lab, 10, weight="600"))
    for a, c in zip(stages, stages[1:]):
        b.append(arrow(a[0] + a[2] + 2, a[1] + a[3] / 2, c[0] - 2, c[1] + c[3] / 2))

    # 2x2 output mapped back to windows
    ox, oy, s = 710, 108, 28
    cols = [RED, AMBER, GREEN, BLUE]
    for i, col in enumerate(cols):
        r, c = divmod(i, 2)
        b.append(rect(ox + c * s, oy + r * s, s, s, "out", rx=2, sw=1.8,
                      fill="#ffffff", stroke=col))
        b.append(txt(ox + c * s + s / 2, oy + r * s + s / 2 + 4, "4", 11, weight="600", fill=col))
    b.append(txt(ox + s, oy + 2 * s + 18, "one 1×1×4 per window", 10.5, fill=MUTED))

    b.append(caption(W / 2, H - 48, [
        "Each 1×1×4 slice is exactly the softmax the original 14×14 network would have produced on that crop.",
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
            dict(kind="vol", s=128, c=3, shape="H×W×3", name="input", role="input"),
            dict(kind="vol", s=64, c=64, shape="H/2 × 64", name="encoder", role="conv"),
            dict(kind="vol", s=32, c=128, shape="H/4 × 128", name="", role="conv"),
            dict(kind="vol", s=16, c=256, shape="H/8 × 256", name="bottleneck", role="conv5"),
            dict(kind="vol", s=32, c=128, shape="H/4 × 128", name="decoder", role="conv1"),
            dict(kind="vol", s=64, c=64, shape="H/2 × 64", name="", role="conv1"),
            dict(kind="vol", s=128, c=3, shape="H×W×C", name="per-pixel", role="out"),
        ],
        [
            "conv + pool\nnH ↓  nC ↑",
            "conv + pool",
            "conv + pool",
            "transpose conv\nnH ↑  nC ↓",
            "transpose conv",
            "1×1 conv\nC = n_classes",
        ],
        sub="first half: spatial size shrinks and channels grow. second half: spatial size grows back. the grow operation is a transpose convolution.",
        note="Drop the classifier head of a ConvNet and replace it with an upsampling path. That is the essential structural change.",
        leg=[("input", "input"), ("conv", "encoder CONV"), ("conv5", "bottleneck"),
             ("conv1", "decoder / transpose"), ("out", "per-pixel classes")],
    )


# ----------------------------------------------- 9.2 transpose conv


def fig_transpose_conv():
    W, H = 920, 470
    b = [title(W, "Transpose convolution: place the filter on the output",
               "each input value scales the whole filter; overlapping pastes are added")]

    # conceptual
    b.append(txt(160, 72, "Normal conv", 12.5, weight="600"))
    b.append(rect(70, 90, 84, 84, "input"))
    b.append(txt(112, 136, "6×6", 12, weight="600"))
    b.append(arrow(162, 132, 200, 132))
    b.append(rect(208, 112, 36, 36, "conv"))
    b.append(txt(226, 134, "3×3", 10))
    b.append(arrow(252, 132, 290, 132))
    b.append(rect(298, 108, 48, 48, "out"))
    b.append(txt(322, 136, "4×4", 12, weight="600"))
    b.append(txt(226, 168, "filter sits on the input", 10.5, fill=MUTED))

    b.append(txt(620, 72, "Transpose conv", 12.5, weight="600"))
    b.append(rect(500, 112, 36, 36, "input"))
    b.append(txt(518, 134, "2×2", 11, weight="600"))
    b.append(arrow(544, 132, 582, 132))
    b.append(rect(590, 112, 36, 36, "conv"))
    b.append(txt(608, 134, "3×3", 10))
    b.append(arrow(634, 132, 678, 132))
    b.append(rect(686, 96, 72, 72, "out"))
    b.append(txt(722, 136, "4×4", 12, weight="600"))
    b.append(txt(608, 168, "filter sits on the output", 10.5, fill=MUTED))

    # worked example: 4 pastes
    b.append(txt(W / 2, 204, "Worked example   ·   input [[1, 2], [3, 4]]   ·   filter [[1,1,1],[0,0,0],[−1,−1,−1]]   ·   s = 2, p = 1",
                 12, weight="600"))

    filt = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
    inp = [[1, 2], [3, 4]]
    colors = [RED, AMBER, GREEN, BLUE]
    cell = 18
    for n, ((i, j), val) in enumerate([((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)]):
        x0 = 40 + n * 220
        y0 = 228
        col = colors[n]
        b.append(txt(x0 + 54, y0, f"input {val}  →  {val} × filter", 10.5, fill=col, weight="600"))
        ox, oy = x0, y0 + 16
        # 6x6 padded canvas
        for r in range(6):
            for c in range(6):
                b.append(rect(ox + c * cell, oy + r * cell, cell, cell, "input",
                              rx=0, sw=0.7, fill="#f9fafb"))
        sr, sc = 2 * i, 2 * j
        for a in range(3):
            for c in range(3):
                v = val * filt[a][c]
                fill = "#fecaca" if v > 0 else ("#bfdbfe" if v < 0 else "#ffffff")
                b.append(rect(ox + (sc + c) * cell, oy + (sr + a) * cell, cell, cell,
                              "input", rx=0, sw=1.4, fill=fill, stroke=col))
                b.append(txt(ox + (sc + c) * cell + cell / 2,
                             oy + (sr + a) * cell + cell / 2 + 4,
                             str(v), 9, fill=INK))

    b.append(caption(W / 2, H - 32, [
        "Where pastes overlap, add — the 4 at (2,2) is −1 −2 +3 +4. Crop the p = 1 border to get the 4×4 output.",
        "n_out = s(n_in − 1) + f − 2p + output padding. With f = 3, p = 1, s = 2, output_padding = 1 this exactly doubles the spatial size.",
    ]))
    write(OUT, "det-transpose-conv.svg", W, H, b)


# --------------------------------------------------------------- 10 U-Net


def fig_unet():
    W, H = 920, 460
    b = [title(W, "U-Net",
               "encoder down, decoder up, skip connections copy high-resolution detail across")]

    # encoder volumes (left, going down)
    enc = [
        (80, 80, 70, 70, "input", "H×W×c"),
        (100, 168, 58, 58, "conv", ""),
        (118, 248, 46, 46, "conv", ""),
        (134, 320, 36, 36, "conv5", "bottleneck"),
    ]
    # decoder volumes (right, going up)
    dec = [
        (134 + 520, 320, 36, 36, "conv1", ""),
        (118 + 520, 248, 46, 46, "conv1", ""),
        (100 + 520, 168, 58, 58, "conv1", ""),
        (80 + 520, 80, 70, 70, "out", "H×W×C"),
    ]

    for x, y, w, h, role, lab in enc:
        b.append(rect(x, y, w, h, role, rx=4))
        if lab:
            b.append(txt(x + w / 2, y - 12, lab, 10.5, weight="600"))
    for x, y, w, h, role, lab in dec:
        b.append(rect(x, y, w, h, role, rx=4))
        if lab:
            b.append(txt(x + w / 2, y - 12, lab, 10.5, weight="600"))

    # down arrows (pool)
    for a, c in zip(enc, enc[1:]):
        b.append(arrow(a[0] + a[2] / 2, a[1] + a[3] + 4, c[0] + c[2] / 2, c[1] - 4, C["pool"][1], 1.6))
    # up arrows (transpose)
    for a, c in zip(dec, dec[1:]):
        b.append(arrow(a[0] + a[2] / 2, a[1] - 4, c[0] + c[2] / 2, c[1] + c[3] + 4, C["conv1"][1], 1.6))

    # skip connections: matching spatial sizes, encoder i to decoder i
    for i, e in enumerate(enc[:-1]):
        d = dec[-(i + 1)]
        y = e[1] + e[3] / 2
        x1 = e[0] + e[2] + 4
        x2 = d[0] - 4
        b.append(path(f"M {x1:g},{y:g} C {(x1 + x2) / 2:g},{y:g} {(x1 + x2) / 2:g},{y:g} {x2:g},{y:g}",
                      stroke=LINE, sw=1.6, marker=True, dash="6 4"))
        if i == 0:
            b.append(txt((x1 + x2) / 2, y - 12, "copy + concat", 11, fill=MUTED))

    b.append(txt(enc[0][0] + enc[0][2] / 2, 430, "encoder", 12, weight="600"))
    b.append(txt(enc[0][0] + enc[0][2] / 2, 446, "nH ↓   nC ↑", 10.5, fill=MUTED))
    b.append(txt(dec[-1][0] + dec[-1][2] / 2, 430, "decoder", 12, weight="600"))
    b.append(txt(dec[-1][0] + dec[-1][2] / 2, 446, "nH ↑   nC ↓", 10.5, fill=MUTED))

    b.append(legend(250, 400, [
        ("input", "input"),
        ("conv", "conv + ReLU"),
        ("pool", "max pool  (down)"),
        ("conv1", "transpose conv  (up)"),
        ("out", "1×1  →  n_classes"),
    ]))

    px = 300
    b.append(panel(px, 300, 320, 70, fill="#f9fafb"))
    b.append(txt(px + 160, 324, "ResNet skip: add, to ease optimization", 11))
    b.append(txt(px + 160, 344, "U-Net skip: concatenate, to restore spatial detail", 11, weight="600"))
    b.append(txt(px + 160, 360, "same name, different operation and purpose", 10.5, fill=MUTED))

    write(OUT, "det-unet.svg", W, H, b)


# ---------------------------------------------------------------------- main

FIGURES = [
    fig_tasks,
    fig_bbox_label,
    fig_landmarks,
    fig_sliding_windows,
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
    fig_transpose_conv,
    fig_unet,
]


def build():
    for f in FIGURES:
        f()


if __name__ == "__main__":
    build()
    print(f"{len(FIGURES)} detection figures written")
