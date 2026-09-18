"""Generate the diagrams used by sequence_models.md.

Driven by make_figures.py; can also be run directly.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from svgkit import (  # noqa: E402
    AMBER, BLUE, GREEN, INK, LINE, MUTED, PINK, PURPLE, RED,
    arrow, caption, circle, legend, lines, panel, path, rect, seg, title, txt,
    write,
)

OUT = pathlib.Path(__file__).parent

# One colour story across every sequence diagram, reusing the shared palette:
#   gray input x<t> · orange recurrent cell and activation a<t>
#   red output y-hat<t> · yellow memory cell c<t> · blue gate · green candidate
XIN = "input"
CELL = "fc"
YOUT = "out"
MEM = "flat"
GATE = "pool"
CAND = "conv"
LOSS = "pad"

SEQ_LEGEND = [
    (XIN, "input x<t>"),
    (CELL, "recurrent cell / a<t>"),
    (YOUT, "output y-hat<t>"),
]


# ----------------------------------------------------------------- helpers


def spread(x0, x1, n, w):
    """n box centres evenly spread so the outer boxes sit inside [x0, x1]."""
    if n == 1:
        return [(x0 + x1) / 2]
    step = (x1 - x0 - w) / (n - 1)
    return [x0 + w / 2 + i * step for i in range(n)]


def box(cx, y, w, h, label, role=CELL, fs=12, sub=None, rx=5, sw=1.4):
    """A centred labelled box. Returns svg text."""
    o = [rect(cx - w / 2, y, w, h, role, rx=rx, sw=sw)]
    if sub:
        o.append(txt(cx, y + h / 2 - 1, label, fs, weight="600"))
        o.append(txt(cx, y + h / 2 + 12, sub, max(fs - 2.5, 8.5), fill=MUTED))
    else:
        o.append(txt(cx, y + h / 2 + fs * 0.36, label, fs, weight="600"))
    return "\n".join(o)


def chip(x, y, label, color, h=18, fs=10, w=None):
    w = w or (len(label) * fs * 0.62 + 16)
    return "\n".join([
        rect(x, y, w, h, "out", rx=h / 2, sw=1.1, fill="#ffffff", stroke=color),
        txt(x + w / 2, y + h / 2 + fs * 0.36, label, fs, fill=color, weight="600"),
    ])


def tick(cx, y, label, fs=10.5):
    return txt(cx, y, label, fs, fill=MUTED)


def bar(x, y, w, h, color, alpha_fill):
    return rect(x, y, w, h, "out", rx=2, sw=1.1, fill=alpha_fill, stroke=color)


# ------------------------------------------ 1. why sequence models


def fig_applications():
    W = 1000
    b = [title(W, "Where sequence models are used",
               "X, Y, or both can be a sequence — and every input tensor is "
               "the sequence axis T_x followed by one position")]

    rows = [
        ("Speech recognition", "an audio clip, played out over time",
         "(T_x, n_mel)", "\"the apple and pear salad\"", "seq \u2192 seq", GREEN),
        ("Music generation", "\u2205, a genre, or a first note",
         "(1,) or \u2205", "a sequence of notes", "\u2192 seq", BLUE),
        ("Sentiment classification", "\"nothing to like in this movie\"",
         "(T_x, V)", "1 of 5 stars", "seq \u2192", AMBER),
        ("DNA sequence analysis", "A G C C C C T G T G A G G",
         "(T_x, 4)", "which part codes a protein", "seq \u2192 seq", GREEN),
        ("Machine translation", "Voulez-vous chanter avec moi?",
         "(T_x, V)", "Do you want to sing with me?",
         "seq \u2192 seq, T_x \u2260 T_y", PINK),
        ("Video activity recognition", "a sequence of video frames",
         "(T_x, h, w, 3)", "\"running\"", "seq \u2192", AMBER),
        ("Name entity recognition", "Harry Potter and Hermione \u2026",
         "(T_x, V)", "1 1 0 1 1 0 0 0 0", "seq \u2192 seq, T_x = T_y", GREEN),
        ("Time series forecasting", "the last T_x readings of demand",
         "(T_x, d)", "the next H values", "seq \u2192 seq, T_y = H", PURPLE),
        ("Panel data", "d features for each of N entities",
         "(N, T_x, d)", "one series per entity",
         "seq \u2192 seq, N is batch", PURPLE),
    ]

    cx = (40, 245, 490, 610, 800)
    heads = ("task", "input X", "tensor of one X", "output Y", "which side")
    for x, h in zip(cx, heads):
        b.append(txt(x, 80, h, 10.5, anchor="start", fill=MUTED, weight="600"))
    b.append(seg(32, 88, W - 32, 88, LINE, 1.0))

    y0, rh = 96, 46
    for i, (task, xin, shp, yout, side, col) in enumerate(rows):
        y = y0 + i * rh
        if i % 2 == 0:
            b.append(rect(32, y, W - 64, rh, XIN, rx=4, sw=0,
                          fill="#f9fafb", stroke="none"))
        b.append(txt(cx[0], y + rh / 2 + 4, task, 11.5, anchor="start",
                     weight="600"))
        b.append(txt(cx[1], y + rh / 2 + 4, xin, 10.5, anchor="start", fill=MUTED))
        b.append(txt(cx[2], y + rh / 2 + 4, shp, 10.5, anchor="start", fill=INK,
                     weight="600"))
        b.append(txt(cx[3], y + rh / 2 + 4, yout, 10.5, anchor="start", fill=MUTED))
        b.append(chip(cx[4], y + rh / 2 - 9, side, col))

    y = y0 + len(rows) * rh
    b.append(caption(W / 2, y + 26, [
        "Every one of these is supervised learning on labelled (X, Y) pairs. "
        "What varies is which side is a sequence, how long each side is,",
        "and what sits at a single position. Add a leading mini-batch "
        "dimension m and these are the shapes a framework actually receives.",
    ]))
    write(OUT, "seq-applications.svg", W, y + 62, b)


# ------------------------------------------------------ 2. notation


def fig_notation():
    W = 920
    b = [title(W, "Notation for one training example",
               "index positions with t, read words through a vocabulary, "
               "represent each one as a one-hot vector")]

    words = ["Harry", "Potter", "and", "Hermione", "Granger",
             "invented", "a", "new", "spell"]
    tags = ["1", "1", "0", "1", "1", "0", "0", "0", "0"]
    cw = 88
    cxs = spread(36, W - 36, len(words), cw)

    b.append(txt(36, 78, "y<t>  (is this word part of a person's name?)",
                 10.5, anchor="start", fill=MUTED, weight="600"))
    for cx, w, tg in zip(cxs, words, tags):
        b.append(box(cx, 86, cw, 26, tg, YOUT if tg == "1" else XIN, fs=11))
        b.append(box(cx, 124, cw, 32, w, XIN, fs=11.5))
    for i, cx in enumerate(cxs):
        b.append(tick(cx, 174, f"x<{i + 1}>"))

    b.append(txt(W / 2, 200, "T_x = 9 words,   T_y = 9 labels   "
                 "(for example i:  x(i)<t>,  T_x(i) = 9)", 11, fill=MUTED))

    # vocabulary
    b.append(panel(48, 232, 250, 196, label="Vocabulary  (10,000 words)"))
    vocab = [("1", "a"), ("2", "Aaron"), ("\u22ee", ""), ("367", "and"),
             ("\u22ee", ""), ("4075", "Harry"), ("6830", "Potter"),
             ("\u22ee", ""), ("10000", "Zulu")]
    for i, (idx, w) in enumerate(vocab):
        y = 256 + i * 19
        hot = w in ("Harry", "Potter")
        col = RED if hot else INK
        b.append(txt(96, y, idx, 10.5, anchor="end",
                     fill=RED if hot else MUTED, weight="600" if hot else "normal"))
        if w:
            b.append(txt(112, y, w, 10.5, anchor="start", fill=col,
                         weight="600" if hot else "normal"))

    # one-hot vectors
    def onehot(x, label, entries):
        o = [panel(x, 232, 250, 196, label=label)]
        cell_w, cell_h = 46, 22
        for i, (idx, v) in enumerate(entries):
            y = 254 + i * 24
            if v == "":
                o.append(txt(x + 125, y + 15, "\u22ee", 12, fill=MUTED))
                continue
            role = YOUT if v == "1" else XIN
            o.append(rect(x + 102, y, cell_w, cell_h, role, rx=2, sw=1.2))
            o.append(txt(x + 102 + cell_w / 2, y + 15, v, 11,
                         weight="600" if v == "1" else "normal"))
            o.append(txt(x + 94, y + 15, idx, 9.5, anchor="end",
                         fill=RED if v == "1" else MUTED))
        o.append(txt(x + 125, 414, "10,000-dimensional", 9.5, fill=MUTED))
        return "\n".join(o)

    b.append(onehot(336, "x<1> = Harry",
                    [("1", "0"), ("", ""), ("4075", "1"), ("", ""), ("10000", "0")]))
    b.append(onehot(624, "x<2> = Potter",
                    [("1", "0"), ("", ""), ("6830", "1"), ("", ""), ("10000", "0")]))

    b.append(caption(W / 2, 458, [
        "A word outside the vocabulary becomes the single token UNK, "
        "so every input is still a valid one-hot vector.",
    ]))
    write(OUT, "seq-notation.svg", W, 480, b)


# ---------------------------------- 3. why not a standard network


def fig_fc_problem():
    W = 920
    b = [title(W, "Why a standard network does not work here",
               "three problems, all of which a recurrent network avoids")]

    px, pw, py, ph = [32, 328, 624], 264, 78, 236

    # --- panel 1: lengths differ
    x = px[0]
    b.append(panel(x, py, pw, ph, label="1.  Lengths differ"))
    for j, n in enumerate([5, 12]):
        y = py + 34 + j * 54
        for k in range(min(n, 8)):
            b.append(rect(x + 16 + k * 20, y, 16, 16, XIN, rx=2, sw=1.0))
        if n > 8:
            b.append(txt(x + 184, y + 13, "\u2026", 13, fill=MUTED))
        b.append(txt(x + 196, y + 12, f"T_x = {n}", 10.5, anchor="start",
                     fill=MUTED))
    b.append(rect(x + 16, py + 152, 160, 40, CELL, rx=5))
    b.append(txt(x + 96, py + 176, "fixed input layer", 10.5, weight="600"))
    b.append(txt(x + 198, py + 176, "\u2717", 20, fill=RED, weight="600"))
    b.append(caption(x + pw / 2, py + 214, [
        "Zero-padding to a maximum length is",
        "possible but still a poor representation.",
    ], lh=14))

    # --- panel 2: no sharing across positions
    x = px[1]
    b.append(panel(x, py, pw, ph, label="2.  Nothing is shared across positions"))
    for j, (pos, verdict, col) in enumerate([("position 1", "\u2713 name", GREEN),
                                             ("position 7", "?  own weights", RED)]):
        y = py + 38 + j * 62
        b.append(rect(x + 16, y, 74, 30, XIN, rx=4))
        b.append(txt(x + 53, y + 20, "Harry", 11, weight="600"))
        b.append(txt(x + 53, y + 44, pos, 9.5, fill=MUTED))
        b.append(arrow(x + 96, y + 15, x + 132, y + 15))
        b.append(rect(x + 138, y - 4, 108, 38, CELL, rx=4))
        b.append(txt(x + 192, y + 19, verdict, 10.5, fill=col, weight="600"))
    b.append(caption(x + pw / 2, py + 186, [
        "What the network learns about Harry at",
        "one position does not transfer to another,",
        "the way a filter transfers across an image.",
    ], lh=14))

    # --- panel 3: enormous first weight matrix
    x = px[2]
    b.append(panel(x, py, pw, ph, label="3.  The first weight matrix explodes"))
    for k in range(9):
        b.append(rect(x + 18 + k * 24, py + 36, 18, 96, XIN, rx=2, sw=1.0))
    b.append(txt(x + 126, py + 146, "9 one-hot vectors \u00d7 10,000",
                 10.5, fill=MUTED))
    b.append(arrow(x + 126, py + 156, x + 126, py + 178))
    b.append(rect(x + 34, py + 182, 184, 34, CELL, rx=4))
    b.append(txt(x + 126, py + 204, "W[1] has \u2248 90,000 columns",
                 10.5, weight="600"))
    b.append(caption(x + pw / 2, py + 232, [
        "One weight per input unit per hidden unit.",
    ], lh=14))

    y = py + ph + 34
    b.append(rect(120, y, W - 240, 40, CELL, rx=8, fill="#fff7ed", stroke="#ea580c"))
    b.append(txt(W / 2, y + 25, "A recurrent network fixes all three: "
                 "one shared parameter set, scanned over a sequence of any length",
                 12, weight="600"))
    write(OUT, "seq-fc-problem.svg", W, y + 72, b)


# ------------------------------------------- 4. the unrolled RNN


def fig_rnn_unrolled():
    W = 920
    b = [title(W, "A recurrent neural network, unrolled in time",
               "the same W_ax, W_aa, W_ya at every step; "
               "the activation carries everything seen so far")]

    cw, ch = 88, 66
    cxs = [170, 330, 490, 650, 820]
    labels = ["t = 1", "t = 2", "t = 3", "t = 4", "t = T_x"]
    words = ["Harry", "Potter", "and", "Hermione", "spell"]

    # a<0>
    b.append(rect(30, 158, 74, 30, CELL, rx=4, fill="#f3f4f6", stroke=LINE))
    b.append(txt(67, 178, "a<0> = 0", 10.5))

    for i, cx in enumerate(cxs):
        b.append(box(cx, 78, cw, 30, f"y-hat<{i + 1}>" if i < 4 else "y-hat<T_y>",
                     YOUT, fs=11))
        b.append(box(cx, 140, cw, ch, "RNN cell", CELL, fs=11.5, sub="tanh"))
        b.append(box(cx, 248, cw, 30, words[i], XIN, fs=11))
        b.append(tick(cx, 296, labels[i]))
        b.append(arrow(cx, 246, cx, 210))          # x -> cell
        b.append(arrow(cx, 138, cx, 112))          # cell -> y-hat

    b.append(arrow(104, 173, 122, 173))
    for i in range(3):
        b.append(arrow(cxs[i] + cw / 2, 173, cxs[i + 1] - cw / 2 - 4, 173))
    b.append(arrow(cxs[3] + cw / 2, 173, 722, 173))
    b.append(txt(735, 178, "\u2026", 16, fill=MUTED))
    b.append(arrow(748, 173, cxs[4] - cw / 2 - 4, 173))

    b.append(txt(cxs[0] + 12, 228, "W_ax", 10, anchor="start", fill=GREEN,
                 weight="600"))
    b.append(txt((cxs[0] + cxs[1]) / 2, 166, "W_aa", 10, fill=GREEN, weight="600"))
    b.append(txt(cxs[0] + 12, 126, "W_ya", 10, anchor="start", fill=GREEN,
                 weight="600"))

    b.append(seg(32, 322, W - 32, 322, LINE, 1.0))

    # the rolled drawing
    b.append(panel(60, 348, 300, 132, label="The same network, drawn rolled"))
    b.append(box(210, 392, 92, 56, "RNN cell", CELL, fs=11))
    b.append(box(210, 456, 70, 22, "x<t>", XIN, fs=10.5))
    b.append(box(210, 358, 70, 22, "y-hat<t>", YOUT, fs=10.5))
    b.append(arrow(210, 454, 210, 450))
    b.append(arrow(210, 390, 210, 382))
    b.append(path("M 256 420 C 306 420 306 372 268 392", stroke=LINE, sw=1.4))
    b.append(txt(300, 440, "delay of", 9.5, fill=MUTED))
    b.append(txt(300, 452, "one step", 9.5, fill=MUTED))

    b.append(txt(420, 372, "What the picture says", 11.5, anchor="start",
                 weight="600"))
    b.append(lines(420, 394, [
        "\u00b7  The cell scans left to right, one position at a time.",
        "\u00b7  Every step reuses the same three parameter matrices, so what",
        "    is learned at position 1 transfers to position 7.",
        "\u00b7  y-hat<3> sees x<1>, x<2>, x<3> \u2014 but nothing after t = 3.",
    ], size=10.5, anchor="start", fill=INK, lh=17))

    b.append(legend(420, 476, SEQ_LEGEND))
    write(OUT, "seq-rnn-unrolled.svg", W, 500, b)


# --------------------------------------- 5. forward propagation


def fig_rnn_forward():
    W = 920
    b = [title(W, "Forward propagation through one cell",
               "one linear step, one non-linearity, one output head — "
               "then the same thing again at t + 1")]

    b.append(box(104, 142, 112, 32, "a<t-1>", CELL, fs=11))
    b.append(box(104, 206, 112, 32, "x<t>", XIN, fs=11))
    b.append(arrow(160, 158, 208, 166))
    b.append(arrow(160, 222, 208, 198))

    b.append(rect(214, 134, 168, 96, CELL, rx=6, fill="#fff7ed", stroke="#ea580c"))
    b.append(lines(298, 164, [
        "W_aa a<t-1>",
        "+ W_ax x<t>",
        "+ b_a",
    ], size=11.5, lh=20))

    b.append(arrow(384, 182, 416, 182))
    b.append(box(462, 164, 84, 36, "tanh", CAND, fs=11.5))
    b.append(arrow(506, 182, 540, 182))
    b.append(box(596, 164, 96, 36, "a<t>", CELL, fs=12))

    b.append(arrow(646, 162, 646, 132))
    b.append(rect(566, 72, 160, 58, CELL, rx=6, fill="#fff7ed", stroke="#ea580c"))
    b.append(txt(646, 106, "W_ya a<t> + b_y", 11.5, weight="600"))
    b.append(arrow(732, 101, 764, 101))
    b.append(box(834, 82, 116, 36, "y-hat<t>", YOUT, fs=11.5, sub=None))
    b.append(txt(834, 134, "sigmoid or softmax", 9.5, fill=MUTED))

    b.append(arrow(646, 202, 646, 234))
    b.append(txt(680, 234, "carried to t + 1", 10, anchor="start", fill=MUTED))

    # stacking the two matrices into one
    b.append(panel(48, 286, W - 96, 162,
                   label="Simplifying the notation: stack the two matrices"))
    y = 324
    b.append(rect(80, y, 92, 58, CELL, rx=4, fill="#fed7aa", stroke="#ea580c"))
    b.append(txt(126, y + 34, "W_aa", 11.5, weight="600"))
    b.append(txt(126, y + 74, "100 \u00d7 100", 9.5, fill=MUTED))
    b.append(rect(172, y, 240, 58, CELL, rx=4, fill="#fed7aa", stroke="#ea580c"))
    b.append(txt(292, y + 34, "W_ax", 11.5, weight="600"))
    b.append(txt(292, y + 74, "100 \u00d7 10,000", 9.5, fill=MUTED))
    b.append(txt(246, y - 12, "W_a = [ W_aa | W_ax ],  100 \u00d7 10,100",
                 11, fill=MUTED, weight="600"))

    b.append(txt(438, y + 34, "\u00d7", 18, fill=MUTED, weight="600"))

    b.append(rect(468, y, 74, 18, CELL, rx=3, fill="#fed7aa", stroke="#ea580c"))
    b.append(txt(505, y + 13, "a<t-1>", 10))
    b.append(rect(468, y + 18, 74, 40, XIN, rx=3))
    b.append(txt(505, y + 42, "x<t>", 10))
    b.append(txt(505, y + 74, "10,100 \u00d7 1", 9.5, fill=MUTED))

    b.append(txt(570, y + 34, "=", 16, fill=MUTED, weight="600"))
    b.append(rect(600, y + 4, 288, 50, CELL, rx=6, fill="#ffffff", stroke=GREEN))
    b.append(txt(744, y + 34, "W_aa a<t-1> + W_ax x<t>", 12, weight="600"))
    b.append(txt(744, y + 74, "one parameter matrix instead of two", 9.5, fill=MUTED))

    b.append(caption(W / 2, 470, [
        "a<t> = g(W_a [a<t-1>, x<t>] + b_a)   and   "
        "y-hat<t> = g(W_y a<t> + b_y).   The subscript names the output: "
        "a-like or y-like.",
    ]))
    write(OUT, "seq-rnn-forward.svg", W, 490, b)


# --------------------------------- 6. backpropagation through time


def fig_bptt():
    W = 920
    b = [title(W, "Backpropagation through time",
               "the same computation graph, traversed right to left")]

    cw, ch = 92, 62
    cxs = spread(60, W - 60, 5, cw)
    labels = ["t = 1", "t = 2", "t = 3", "t = 4", "t = T_y"]

    for i, cx in enumerate(cxs):
        b.append(box(cx, 76, cw, 28, f"L<{i + 1}>" if i < 4 else "L<T_y>",
                     LOSS, fs=11))
        b.append(box(cx, 130, cw, 28, "y-hat", YOUT, fs=11))
        b.append(box(cx, 186, cw, ch, "RNN cell", CELL, fs=11))
        b.append(box(cx, 284, cw, 28, "x<t>", XIN, fs=11))
        b.append(tick(cx, 332, labels[i]))
        b.append(arrow(cx - 14, 282, cx - 14, 252))      # x -> cell
        b.append(arrow(cx - 14, 184, cx - 14, 162))      # cell -> y-hat
        b.append(arrow(cx - 14, 128, cx - 14, 108))      # y-hat -> loss
        # backward
        b.append(arrow(cx + 14, 108, cx + 14, 128, RED, 1.3, dash="4 3"))
        b.append(arrow(cx + 14, 162, cx + 14, 184, RED, 1.3, dash="4 3"))

    for i in range(4):
        x1, x2 = cxs[i] + cw / 2, cxs[i + 1] - cw / 2
        b.append(arrow(x1 + 2, 206, x2 - 4, 206))
        b.append(arrow(x2 - 2, 232, x1 + 4, 232, RED, 1.5, dash="5 4"))

    b.append(txt(cxs[0] + cw / 2 + 34, 198, "forward prop", 9.5, fill=MUTED))
    b.append(txt(cxs[2] + cw / 2 + 36, 248, "backprop", 9.5, fill=RED, weight="600"))

    b.append(seg(48, 352, W - 48, 352, LINE, 1.0))
    b.append(rect(160, 372, 600, 44, CELL, rx=8, fill="#f9fafb", stroke=LINE))
    b.append(txt(W / 2, 400,
                 "L  =  \u03a3  L<t>(y-hat<t>, y<t>)      "
                 "with  L<t> = \u2212y<t> log y-hat<t> \u2212 (1\u2212y<t>) "
                 "log(1\u2212y-hat<t>)", 11.5, weight="600"))

    b.append(caption(W / 2, 442, [
        "The load-bearing message is the red one along the activations: it runs "
        "backwards through t, which is where the name comes from.",
        "Frameworks do this for you; the shape of the graph is what matters.",
    ]))
    write(OUT, "seq-bptt.svg", W, 480, b)


# ----------------------------------------- 7. the family of shapes


def _mini(x, y, cols, pitch=46, cw=32, feedback=False, split=None):
    """A small RNN sketch. cols: list of (has_x, has_y) pairs."""
    o = []
    n = len(cols)
    cxs = [x + cw / 2 + i * pitch for i in range(n)]
    for i, (cx, (has_x, has_y)) in enumerate(zip(cxs, cols)):
        if has_y:
            o.append(rect(cx - cw / 2, y, cw, 18, YOUT, rx=3, sw=1.1))
        o.append(rect(cx - cw / 2, y + 30, cw, 30, CELL, rx=3, sw=1.2))
        if has_x:
            o.append(rect(cx - cw / 2, y + 72, cw, 18, XIN, rx=3, sw=1.1))
            o.append(arrow(cx, y + 70, cx, y + 62, LINE, 1.1))
        if has_y:
            o.append(arrow(cx, y + 28, cx, y + 20, LINE, 1.1))
        if i:
            o.append(arrow(cxs[i - 1] + cw / 2 + 1, y + 45,
                           cx - cw / 2 - 2, y + 45, LINE, 1.1))
    if feedback:
        for i in range(n - 1):
            if not cols[i][1]:
                continue
            o.append(arrow(cxs[i] + cw / 2 + 1, y + 13,
                           cxs[i + 1] - cw / 2 - 2, y + 33, PINK, 1.2,
                           dash="4 3"))
    if split is not None:
        sx = (cxs[split - 1] + cxs[split]) / 2
        o.append(seg(sx, y - 6, sx, y + 96, PURPLE, 1.2, dash="5 4"))
        o.append(txt(x + (sx - x) / 2, y + 108, "encoder", 9.5, fill=PURPLE,
                     weight="600"))
        o.append(txt(sx + (cxs[-1] + cw / 2 - sx) / 2, y + 108, "decoder", 9.5,
                     fill=PURPLE, weight="600"))
    return "\n".join(o)


def fig_rnn_types():
    W = 920
    b = [title(W, "The family of RNN shapes",
               "which side is a sequence, and whether T_x equals T_y, "
               "decides the architecture")]

    top = [
        (32, "One-to-one", "a plain network; you do not need an RNN",
         [(True, True)], None, False),
        (328, "One-to-many", "music generation, sequence generation",
         [(True, False)] + [(False, True)] * 3, None, True),
        (624, "Many-to-one", "sentiment classification",
         [(True, False)] * 3 + [(True, True)], None, False),
    ]
    for x, head, sub, cols, split, fb in top:
        b.append(panel(x, 82, 264, 214, label=head))
        b.append(txt(x + 132, 110, sub, 10, fill=MUTED))
        n = len(cols)
        pitch = 46 if n > 1 else 0
        x0 = x + 132 - ((n - 1) * pitch + 32) / 2
        b.append(_mini(x0, 128, cols, feedback=fb))
        b.append(txt(x + 132, 268, f"T_x = {sum(1 for c in cols if c[0])},  "
                     f"T_y = {sum(1 for c in cols if c[1])}", 10, fill=MUTED))

    bottom = [
        (32, "Many-to-many,  T_x = T_y", "name entity recognition",
         [(True, True)] * 6, None, False),
        (472, "Many-to-many,  T_x \u2260 T_y", "machine translation "
         "(encoder \u2192 decoder)",
         [(True, False)] * 4 + [(False, True)] * 3, 4, False),
    ]
    for x, head, sub, cols, split, fb in bottom:
        b.append(panel(x, 328, 416, 228, label=head))
        b.append(txt(x + 208, 356, sub, 10, fill=MUTED))
        n = len(cols)
        x0 = x + 208 - ((n - 1) * 52 + 32) / 2
        b.append(_mini(x0, 374, cols, pitch=52, split=split, feedback=fb))
        b.append(txt(x + 208, 532, f"T_x = {sum(1 for c in cols if c[0])},  "
                     f"T_y = {sum(1 for c in cols if c[1])}", 10, fill=MUTED))

    b.append(legend(32, 584, SEQ_LEGEND))
    b.append(txt(430, 584, "pink dashes: the sampled output is fed back in as "
                 "the next input", 10.5, anchor="start", fill=PINK,
                 weight="600"))
    b.append(caption(W / 2, 612, [
        "Attention-based architectures are the one shape this picture does not cover.",
    ]))
    write(OUT, "seq-rnn-types.svg", W, 632, b)


# --------------------------------------- 8. the language model


def fig_language_model():
    W = 920
    b = [title(W, "Training a language model",
               "at every step the model predicts the next token, "
               "and is then told what it actually was")]

    tokens = ["cats", "average", "15", "hours", "of", "sleep", "a", "day",
              "<EOS>"]
    ins = ["0", "cats", "average", "15", "hours", "of", "sleep", "a", "day"]
    cw = 80
    cxs = spread(36, W - 36, 9, cw)

    b.append(txt(36, 72, "y<t>   the true next token",
                 10, anchor="start", fill=MUTED, weight="600"))
    for i, cx in enumerate(cxs):
        b.append(box(cx, 80, cw, 26, tokens[i], LOSS, fs=10.5))
        b.append(box(cx, 118, cw, 28, "softmax", YOUT, fs=10))
        b.append(box(cx, 164, cw, 56, "RNN", CELL, fs=11))
        b.append(box(cx, 238, cw, 26, ins[i], XIN, fs=10.5))
        b.append(arrow(cx, 236, cx, 224))
        b.append(arrow(cx, 162, cx, 150))
        b.append(arrow(cx, 116, cx, 110))
        b.append(tick(cx, 284, f"t = {i + 1}"))
    for i in range(8):
        b.append(arrow(cxs[i] + cw / 2 + 1, 192, cxs[i + 1] - cw / 2 - 3, 192))

    b.append(txt(W / 2, 314, "x<1> = 0,   a<0> = 0,   and  x<t> = y<t-1>  "
                 "for every later step", 11.5, weight="600"))

    b.append(rect(48, 336, 400, 96, CELL, rx=8, fill="#eff6ff", stroke=BLUE))
    b.append(txt(248, 360, "Each softmax is a 10,002-way distribution", 11,
                 fill=BLUE, weight="600"))
    b.append(lines(248, 382, [
        "the 10,000 vocabulary words, plus UNK and <EOS>",
        "step 3 answers: P(any word  |  \u201ccats average\u201d)",
    ], size=10, fill=MUTED, lh=16))

    b.append(rect(472, 336, 400, 96, CELL, rx=8, fill="#f0fdf4", stroke=GREEN))
    b.append(txt(672, 360, "Scoring a whole sentence", 11, fill=GREEN,
                 weight="600"))
    b.append(lines(672, 384, [
        "P(y<1>, y<2>, y<3>)  =",
        "P(y<1>) \u00b7 P(y<2> | y<1>) \u00b7 P(y<3> | y<1>, y<2>)",
    ], size=10.5, fill=INK, lh=17))

    b.append(caption(W / 2, 458, [
        "The loss is the softmax cross-entropy summed over steps. "
        "Tokenize first: build the vocabulary, map words to one-hot vectors, "
        "replace rare words with UNK, optionally append <EOS>.",
    ]))
    write(OUT, "seq-language-model.svg", W, 480, b)


# ------------------------------------ 9. sampling novel sequences


def fig_sampling():
    W = 920
    b = [title(W, "Sampling a novel sequence",
               "same trained network, one change: feed back what you sampled, "
               "not the true word")]

    # left: the softmax at step 1
    b.append(panel(32, 82, 286, 242, label="Step 1: draw from the softmax"))
    items = [("a", 0.06), ("Aaron", 0.01), ("cats", 0.09), ("the", 0.31),
             ("Zulu", 0.01), ("UNK", 0.02)]
    for i, (w, p) in enumerate(items):
        y = 112 + i * 30
        hot = w == "the"
        b.append(txt(96, y + 12, w, 10.5, anchor="end",
                     fill=RED if hot else INK, weight="600" if hot else "normal"))
        b.append(rect(104, y, 130 * p / 0.31, 16, YOUT if hot else XIN,
                      rx=3, sw=1.1))
        b.append(txt(302, y + 12, f"{p:.2f}", 9.5, anchor="end", fill=MUTED))
    b.append(txt(175, 306, "np.random.choice(vocab, p=y-hat<1>)", 10,
                 fill=MUTED))

    # right: the chain with feedback
    cw = 84
    cxs = [412, 552, 692, 832]
    picks = ["the", "apple", "and", "pear"]
    for i, cx in enumerate(cxs):
        b.append(box(cx, 96, cw, 28, picks[i], YOUT, fs=11))
        b.append(txt(cx, 88, f"y-hat<{i + 1}>", 9.5, fill=MUTED))
        b.append(box(cx, 168, cw, 56, "RNN", CELL, fs=11))
        label = "x<1> = 0" if i == 0 else picks[i - 1]
        b.append(box(cx, 246, cw, 28, label, XIN, fs=10.5))
        b.append(arrow(cx, 244, cx, 228))
        b.append(arrow(cx, 166, cx, 128))
    for i in range(3):
        b.append(arrow(cxs[i] + cw / 2 + 1, 196, cxs[i + 1] - cw / 2 - 3, 196))
        gx = cxs[i] + 70
        b.append(path(
            f"M {cxs[i] + cw / 2} 110 L {gx} 110 L {gx} 260 "
            f"L {cxs[i + 1] - cw / 2 - 4} 260",
            stroke=PINK, sw=1.4, marker=True, dash="5 4"))
    b.append(txt(622, 302, "the sampled word becomes the next input", 10,
                 fill=PINK, weight="600"))

    b.append(seg(32, 346, W - 32, 346, LINE, 1.0))
    notes = [
        ("When to stop", "sample until <EOS>, or fix a length of 20 or 100 steps",
         BLUE),
        ("Unwanted UNK", "reject the sample and redraw from the rest of the vocabulary",
         AMBER),
        ("Character level", "vocabulary is a-z, space, punctuation, digits — "
         "no UNK ever, but much longer sequences", GREEN),
    ]
    for i, (head, body, col) in enumerate(notes):
        x = 40 + i * 294
        b.append(rect(x, 366, 274, 74, CELL, rx=8, fill="#ffffff", stroke=col))
        b.append(txt(x + 137, 390, head, 11, fill=col, weight="600"))
        b.append(caption(x + 137, 410, [body[:38], body[38:]] if len(body) > 38
                         else [body], size=9.5, lh=14))

    write(OUT, "seq-sampling.svg", W, 468, b)


# --------------------------------------- 10. vanishing gradients


def fig_vanishing():
    W = 920
    b = [title(W, "Why a basic RNN forgets",
               "the gradient from a late step decays exponentially on its way "
               "back to an early one")]

    tokens = ["The", "cat", ",", "which", "already", "ate", "\u2026", "was",
              "full"]
    cw = 82
    cxs = spread(40, W - 40, 9, cw)

    base = 154
    for i, cx in enumerate(cxs):
        if i > 7:
            continue
        h = 3 + 58 * (0.55 ** (7 - i))
        b.append(rect(cx - 16, base - h, 32, h, YOUT, rx=2, sw=1.1))
    b.append(txt(40, 74, "\u2016 gradient reaching step t from the loss at "
                 "\u201cwas\u201d \u2016", 10.5, anchor="start", fill=RED,
                 weight="600"))
    b.append(txt(cxs[7], 84, "loss here", 9.5, fill=RED))

    for i, cx in enumerate(cxs):
        b.append(box(cx, 166, cw, 34, tokens[i], CELL, fs=11))
        if i:
            b.append(arrow(cxs[i - 1] + cw / 2 + 1, 183, cx - cw / 2 - 3, 183,
                           LINE, 1.1))

    b.append(path(f"M {cxs[1]} 216 C {cxs[3]} 268, {cxs[6]} 268, {cxs[7]} 218",
                  stroke=GREEN, sw=1.6, marker=True))
    b.append(txt((cxs[1] + cxs[7]) / 2, 284,
                 "singular subject decides  was, not were  — "
                 "and the gap can be arbitrarily long",
                 10.5, fill=GREEN, weight="600"))

    b.append(seg(40, 306, W - 40, 306, LINE, 1.0))

    b.append(panel(40, 330, 500, 132, label="Vanishing gradients: local influence"))
    b.append(lines(290, 360, [
        "y-hat<t> ends up mostly determined by inputs near t.",
        "An RNN over 1,000 steps is a 1,000-layer network, so the",
        "error at a late step barely reaches the early computations.",
        "Fix: gated units (GRU, LSTM) — not clipping.",
    ], size=10.5, lh=18))

    b.append(panel(568, 330, 312, 132, label="Exploding gradients: clip"))
    b.append(circle(630, 414, 34, fill="none", stroke=BLUE, sw=1.4,
                    ))
    b.append(arrow(630, 414, 856, 352, RED, 1.6))
    b.append(arrow(630, 414, 663, 405, BLUE, 2.4))
    b.append(txt(636, 460, "threshold", 9, anchor="start", fill=BLUE))
    b.append(txt(700, 428, "rescale the gradient", 9.5, anchor="start",
                 fill=BLUE, weight="600"))
    b.append(txt(700, 442, "back onto the threshold", 9.5, anchor="start",
                 fill=BLUE, weight="600"))
    b.append(txt(830, 344, "NaNs", 9.5, fill=RED, weight="600"))

    b.append(caption(W / 2, 486, [
        "Exploding gradients are easy to spot and easy to fix. "
        "Vanishing gradients are the real problem, and they are what gated "
        "units were invented for.",
    ]))
    write(OUT, "seq-vanishing.svg", W, 506, b)


# ------------------------------------------------- 11. GRU


def fig_gru_cell():
    W = 920
    b = [title(W, "The gated recurrent unit",
               "a memory cell c<t>, an update gate that decides when to "
               "overwrite it, and a relevance gate")]

    b.append(box(112, 188, 144, 30, "c<t-1> = a<t-1>", MEM, fs=10.5))
    b.append(box(112, 248, 144, 30, "x<t>", XIN, fs=11))
    b.append(circle(300, 218, 4, fill=LINE, stroke=LINE, sw=1))
    b.append(seg(184, 203, 300, 218, LINE, 1.2))
    b.append(seg(184, 263, 300, 218, LINE, 1.2))

    b.append(box(392, 100, 96, 40, "\u0393_u", GATE, fs=12, sub="sigmoid"))
    b.append(box(392, 170, 96, 40, "\u0393_r", GATE, fs=12, sub="sigmoid"))
    b.append(box(392, 244, 96, 40, "c-tilde<t>", CAND, fs=10.5, sub="tanh"))
    for y in (120, 190, 264):
        b.append(arrow(304, 218, 342, y, LINE, 1.2))
    b.append(arrow(392, 212, 392, 240, PURPLE, 1.6))
    b.append(txt(402, 228, "relevance", 8.5, anchor="start", fill=PURPLE))

    b.append(rect(536, 136, 158, 148, CELL, rx=8, fill="#f5f3ff",
                  stroke=PURPLE))
    b.append(lines(615, 178, [
        "\u0393_u \u2299 c-tilde<t>",
        "+",
        "(1 \u2212 \u0393_u) \u2299 c<t-1>",
    ], size=10.5, lh=28))
    b.append(arrow(442, 142, 530, 156, LINE, 1.2))
    b.append(arrow(442, 242, 530, 226, LINE, 1.2))
    b.append(path("M 300 222 C 300 316, 470 320, 560 288", stroke=LINE, sw=1.2,
                  marker=True))

    b.append(box(812, 192, 128, 36, "c<t> = a<t>", MEM, fs=11))
    b.append(arrow(696, 210, 744, 210, LINE, 1.3))
    b.append(box(812, 114, 128, 32, "y-hat<t>", YOUT, fs=11))
    b.append(arrow(812, 190, 812, 150))

    b.append(seg(40, 318, W - 40, 318, LINE, 1.0))
    b.append(rect(48, 338, 400, 96, CELL, rx=8, fill="#f9fafb", stroke=LINE))
    b.append(lines(248, 362, [
        "\u0393_u = \u03c3(W_u [c<t-1>, x<t>] + b_u)",
        "\u0393_r = \u03c3(W_r [c<t-1>, x<t>] + b_r)",
        "a<t> = c<t>",
    ], size=10.5, lh=24))
    b.append(rect(472, 338, 400, 96, CELL, rx=8, fill="#f9fafb", stroke=LINE))
    b.append(lines(672, 362, [
        "c-tilde<t> = tanh(W_c [\u0393_r \u2299 c<t-1>, x<t>] + b_c)",
        "c<t> = \u0393_u \u2299 c-tilde<t> + (1 \u2212 \u0393_u) "
        "\u2299 c<t-1>",
    ], size=10.5, lh=24))

    b.append(caption(W / 2, 458, [
        "\u2299 is element-wise, so \u0393_u is a vector of near-0 / near-1 "
        "bits: one bit can hold \u201csingular subject\u201d while another "
        "tracks \u201cwe are talking about food\u201d.",
        "When \u0393_u \u2248 0 the update reduces to c<t> = c<t-1>, which is "
        "why the value survives many steps and the gradient does not vanish.",
    ]))
    write(OUT, "seq-gru-cell.svg", W, 490, b)


def fig_gru_memory():
    W = 920
    b = [title(W, "What the update gate does over a sentence",
               "set the bit once, carry it untouched, use it much later")]

    tokens = ["The", "cat", ",", "which", "already", "ate", "\u2026", "was",
              "full"]
    gates = ["0", "1", "0", "0", "0", "0", "0", "0", "1"]
    cells = ["0", "1", "1", "1", "1", "1", "1", "1", "\u2013"]
    cw = 78
    cxs = [130 + i * 93 for i in range(9)]

    b.append(path(f"M {cxs[1]} 82 C {cxs[3]} 58, {cxs[6]} 58, {cxs[7]} 82",
                  stroke=GREEN, sw=1.6, marker=True))

    b.append(txt(30, 106, "word", 10.5, anchor="start", fill=MUTED,
                 weight="600"))
    for cx, tk in zip(cxs, tokens):
        b.append(box(cx, 88, cw, 32, tk, XIN, fs=11))

    b.append(txt(30, 160, "\u0393_u", 11.5, anchor="start", fill=BLUE,
                 weight="600"))
    b.append(txt(30, 176, "update?", 9, anchor="start", fill=MUTED))
    for cx, g in zip(cxs, gates):
        hot = g == "1"
        b.append(rect(cx - 26, 148, 52, 26, GATE, rx=13, sw=1.3,
                      fill="#bfdbfe" if hot else "#f3f4f6",
                      stroke=BLUE if hot else LINE))
        b.append(txt(cx, 166, g, 11, weight="600" if hot else "normal",
                     fill=INK if hot else MUTED))

    b.append(rect(cxs[0] - 30, 206, cxs[-1] - cxs[0] + 60, 30, MEM, rx=15,
                  sw=0, fill="#fef9c3", stroke="none"))
    b.append(txt(30, 226, "c<t>", 11.5, anchor="start", fill=AMBER,
                 weight="600"))
    for cx, c in zip(cxs, cells):
        b.append(rect(cx - 22, 208, 44, 26, MEM, rx=4, sw=1.3))
        b.append(txt(cx, 226, c, 11, weight="600"))

    b.append(txt(cxs[1] - 20, 256, "\u2191 open the gate: the subject is "
                 "singular", 9.5, anchor="start", fill=BLUE, weight="600"))
    b.append(txt(cxs[7], 276, "\u2191 read it: was, not were", 9.5,
                 fill=AMBER, weight="600"))

    b.append(caption(W / 2, 312, [
        "In the middle \u0393_u \u2248 0, so c<t> = c<t-1> exactly, up to "
        "numerical round-off. That is the whole trick.",
    ]))
    write(OUT, "seq-gru-memory.svg", W, 336, b)


# ------------------------------------------------ 12. LSTM


def fig_lstm_cell():
    W = 920
    b = [title(W, "The long short-term memory unit",
               "three gates instead of two, and a<t> is no longer the same "
               "thing as c<t>")]

    b.append(box(104, 128, 124, 30, "c<t-1>", MEM, fs=11))
    b.append(box(104, 214, 124, 30, "a<t-1>", CELL, fs=11))
    b.append(box(104, 272, 124, 30, "x<t>", XIN, fs=11))
    b.append(circle(196, 244, 4, fill=LINE, stroke=LINE, sw=1))
    b.append(seg(168, 229, 196, 244, LINE, 1.2))
    b.append(seg(168, 287, 196, 244, LINE, 1.2))

    gates = [("\u0393_f", 100, "forget"), ("\u0393_u", 158, "update"),
             ("c-tilde<t>", 216, "tanh"), ("\u0393_o", 296, "output")]
    for label, y, sub in gates:
        role = CAND if label.startswith("c-") else GATE
        b.append(box(272, y, 92, 40, label, role, fs=11 if len(label) < 5 else 10,
                     sub=sub))
        b.append(arrow(200, 244, 224, y + 20, LINE, 1.2))

    b.append(rect(408, 96, 140, 160, CELL, rx=8, fill="#f5f3ff", stroke=PURPLE))
    b.append(lines(478, 152, [
        "\u0393_u \u2299 c-tilde<t>",
        "+",
        "\u0393_f \u2299 c<t-1>",
    ], size=10.5, lh=26))
    b.append(arrow(320, 120, 404, 132, LINE, 1.2))
    b.append(arrow(320, 178, 404, 170, LINE, 1.2))
    b.append(arrow(320, 236, 404, 210, LINE, 1.2))
    b.append(path("M 168 143 C 248 84, 340 78, 406 106", stroke=LINE, sw=1.2,
                  marker=True))

    b.append(box(618, 158, 120, 36, "c<t>", MEM, fs=12))
    b.append(arrow(550, 176, 556, 176, LINE, 1.3))

    b.append(box(618, 280, 120, 36, "a<t>", CELL, fs=12))
    b.append(arrow(618, 196, 618, 276))
    b.append(arrow(320, 316, 556, 300, LINE, 1.2))
    b.append(txt(618, 332, "a<t> = \u0393_o \u2299 c<t>", 10.5, fill=MUTED))

    b.append(box(830, 280, 96, 32, "y-hat<t>", YOUT, fs=11))
    b.append(arrow(680, 298, 780, 298))

    b.append(rect(782, 120, 118, 104, CELL, rx=8, fill="#f5f3ff",
                  stroke=PURPLE))
    b.append(txt(841, 144, "Variation", 10.5, fill=PURPLE, weight="600"))
    b.append(lines(841, 164, [
        "peephole:", "the gates also", "see c<t-1>,", "element by element",
    ], size=9, fill=MUTED, lh=14))

    b.append(seg(40, 358, W - 40, 358, LINE, 1.0))
    b.append(rect(48, 378, 400, 96, CELL, rx=8, fill="#f9fafb", stroke=LINE))
    b.append(lines(248, 402, [
        "\u0393_u = \u03c3(W_u [a<t-1>, x<t>] + b_u)",
        "\u0393_f = \u03c3(W_f [a<t-1>, x<t>] + b_f)",
        "\u0393_o = \u03c3(W_o [a<t-1>, x<t>] + b_o)",
    ], size=10.5, lh=24))
    b.append(rect(472, 378, 400, 96, CELL, rx=8, fill="#f9fafb", stroke=LINE))
    b.append(lines(672, 402, [
        "c-tilde<t> = tanh(W_c [a<t-1>, x<t>] + b_c)",
        "c<t> = \u0393_u \u2299 c-tilde<t> + \u0393_f \u2299 c<t-1>",
        "a<t> = \u0393_o \u2299 c<t>",
    ], size=10.5, lh=24))

    b.append(caption(W / 2, 498, [
        "Separate update and forget gates mean the unit can add to the old "
        "value instead of having to trade against it, which is what "
        "1 \u2212 \u0393_u forces in a GRU.",
    ]))
    write(OUT, "seq-lstm-cell.svg", W, 518, b)


def fig_lstm_chain():
    W = 920
    b = [title(W, "The memory highway",
               "with \u0393_f \u2248 1 and \u0393_u \u2248 0 the cell state "
               "crosses many steps untouched")]

    cxs = [250, 470, 690]
    bw, bh, by = 156, 110, 152

    b.append(seg(70, 108, 866, 108, "#ca8a04", 4.0))
    b.append(txt(70, 96, "c<0>", 11, anchor="start", fill=AMBER, weight="600"))
    b.append(txt(862, 96, "c<3> = c<0>", 11, anchor="end", fill=AMBER,
                 weight="600"))
    b.append(txt(470, 96, "the cell-state line: nothing on it but two "
                 "element-wise products", 10, fill=AMBER, weight="600"))

    for i, cx in enumerate(cxs):
        b.append(rect(cx - bw / 2, by, bw, bh, CELL, rx=8, fill="#fff7ed",
                      stroke="#ea580c"))
        b.append(txt(cx, by + 22, f"LSTM  t = {i + 1}", 11, weight="600"))
        for j, (g, role) in enumerate([("\u0393_f", GATE), ("\u0393_u", GATE),
                                       ("\u0393_o", GATE), ("c-tilde", CAND)]):
            gx = cx - bw / 2 + 12 + j * 34
            b.append(rect(gx, by + 38, 30, 22, role, rx=4, sw=1.1))
            b.append(txt(gx + 15, by + 53, g, 8.5 if len(g) > 3 else 9.5))
        b.append(txt(cx, by + 88, "\u0393_f \u2248 1,  \u0393_u \u2248 0", 9.5,
                     fill=MUTED))
        b.append(seg(cx, 108, cx, by, "#ca8a04", 1.6, dash="4 3"))
        b.append(box(cx, 300, 90, 28, f"x<{i + 1}>", XIN, fs=10.5))
        b.append(arrow(cx, 298, cx, 266, LINE, 1.2))

    for i in range(2):
        b.append(arrow(cxs[i] + bw / 2 + 2, 228, cxs[i + 1] - bw / 2 - 4, 228,
                       "#ea580c", 1.8))
        b.append(txt((cxs[i] + cxs[i + 1]) / 2, 220, "a<t>", 9.5,
                     fill="#ea580c", weight="600"))

    b.append(caption(W / 2, 362, [
        "This is the same reason a GRU works, drawn as a picture: there is a "
        "path from c<0> to c<3> that the network can leave alone,",
        "so a value \u2014 and a gradient \u2014 can travel across many time "
        "steps without decaying.",
    ]))
    write(OUT, "seq-lstm-chain.svg", W, 400, b)


# --------------------------------------- 13. bidirectional RNN


def fig_brnn():
    W = 920
    b = [title(W, "Bidirectional RNN",
               "a second, backward recurrent layer, so a prediction can use "
               "the whole sentence")]

    words = ["He", "said", "Teddy", "Roosevelt"]
    cw = 108
    cxs = [170, 380, 590, 800]

    for i, cx in enumerate(cxs):
        b.append(box(cx, 94, cw, 30, f"y-hat<{i + 1}>", YOUT, fs=11))
        b.append(box(cx, 158, cw, 48, "a-forward<t>", CELL, fs=10))
        b.append(box(cx, 232, cw, 48, "a-backward<t>", GATE, fs=10))
        b.append(box(cx, 308, cw, 30, words[i], XIN, fs=11))
        b.append(arrow(cx - 30, 306, cx - 30, 282, LINE, 1.2))
        b.append(path(f"M {cx + 30} 306 C {cx + 76} 296, {cx + 76} 226, "
                      f"{cx + 30} 208", stroke=LINE, sw=1.2, marker=True))
        b.append(arrow(cx - 30, 156, cx - 30, 126, LINE, 1.2))
        b.append(path(f"M {cx + 40} 232 C {cx + 76} 212, {cx + 76} 146, "
                      f"{cx + 30} 126", stroke=LINE, sw=1.2, marker=True))

    for i in range(3):
        b.append(arrow(cxs[i] + cw / 2 + 2, 182, cxs[i + 1] - cw / 2 - 4, 182,
                       "#ea580c", 1.6))
        b.append(arrow(cxs[i + 1] - cw / 2 - 2, 256, cxs[i] + cw / 2 + 4, 256,
                       BLUE, 1.6))
    b.append(txt((cxs[0] + cxs[1]) / 2, 174, "forward in time", 9.5,
                 fill="#ea580c", weight="600"))
    b.append(txt((cxs[1] + cxs[2]) / 2, 274, "backward in time", 9.5, fill=BLUE,
                 weight="600"))

    b.append(rect(cxs[2] - cw / 2 - 6, 88, cw + 12, 42, YOUT, rx=6, sw=2.2,
                  fill="none", stroke=RED, dash="5 4"))
    b.append(txt(cxs[2], 74, "uses He said Teddy  and  Roosevelt", 9.5,
                 fill=RED, weight="600"))

    b.append(seg(40, 364, W - 40, 364, LINE, 1.0))
    b.append(rect(48, 382, 400, 82, CELL, rx=8, fill="#f9fafb", stroke=LINE))
    b.append(txt(248, 412, "y-hat<t> = g(W_y [a-forward<t>, a-backward<t>] "
                 "+ b_y)", 10.5, weight="600"))
    b.append(txt(248, 440, "the graph is acyclic: run both passes, then predict",
                 9.5, fill=MUTED))

    b.append(rect(472, 382, 400, 82, CELL, rx=8, fill="#fef2f2", stroke=RED))
    b.append(txt(672, 408, "The cost", 11, fill=RED, weight="600"))
    b.append(lines(672, 428, [
        "you need the entire sequence before predicting anywhere \u2014",
        "fine for a full sentence, awkward for live speech",
    ], size=9.5, fill=MUTED, lh=15))

    b.append(caption(W / 2, 488, [
        "The blocks can be plain RNN, GRU, or LSTM. "
        "A bidirectional LSTM is a reasonable first thing to try on an NLP "
        "task where you have the whole sentence.",
    ]))
    write(OUT, "seq-brnn.svg", W, 508, b)


# -------------------------------------------- 14. deep RNNs


def fig_deep_rnn():
    W = 920
    b = [title(W, "Deep RNNs",
               "stack a few recurrent layers, then optionally a "
               "non-recurrent head on each step")]

    # left: three recurrent layers
    b.append(panel(32, 78, 508, 340, label="Three recurrent layers"))
    cw = 78
    cxs = [110, 222, 334, 446]
    rows = [(l, 300 - i * 58) for i, l in enumerate([1, 2, 3])]
    for i, cx in enumerate(cxs):
        b.append(box(cx, 344, cw, 26, f"x<{i + 1}>", XIN, fs=10))
        b.append(box(cx, 116, cw, 26, f"y-hat<{i + 1}>", YOUT, fs=10))
        for l, y in rows:
            b.append(box(cx, y, cw, 40, f"a[{l}]<{i + 1}>", CELL, fs=10))
        b.append(arrow(cx, 342, cx, 322, LINE, 1.1))
        b.append(arrow(cx, 300, cx, 282, LINE, 1.1))
        b.append(arrow(cx, 242, cx, 224, LINE, 1.1))
        b.append(arrow(cx, 184, cx, 144, LINE, 1.1))
    for l, y in rows:
        for i in range(3):
            b.append(arrow(cxs[i] + cw / 2 + 1, y + 20,
                           cxs[i + 1] - cw / 2 - 3, y + 20, LINE, 1.1))
    b.append(rect(cxs[2] - cw / 2 - 4, 238, cw + 8, 48, CELL, rx=6, sw=2.2,
                  fill="none", stroke=RED, dash="5 4"))
    b.append(txt(286, 398, "a[2]<3> = g( W_a[2] [ a[2]<2>, a[1]<3> ] + b_a[2] )",
                 10.5, fill=RED, weight="600"))

    # right: recurrent stack plus a deep head
    b.append(panel(568, 78, 320, 340, label="Recurrent stack + deep head"))
    hxs = [680, 800]
    for i, cx in enumerate(hxs):
        b.append(box(cx, 344, 78, 26, f"x<{i + 1}>", XIN, fs=10))
        for l, y in [(1, 300), (2, 254), (3, 208)]:
            b.append(box(cx, y, 78, 34, f"a[{l}]<{i + 1}>", CELL, fs=9.5))
        b.append(box(cx, 162, 78, 30, "FC", "fc", fs=10))
        b.append(box(cx, 120, 78, 30, "FC", "fc", fs=10))
        b.append(box(cx, 84, 78, 26, f"y-hat<{i + 1}>", YOUT, fs=10))
        b.append(arrow(cx, 342, cx, 334, LINE, 1.1))
        b.append(arrow(cx, 300, cx, 288, LINE, 1.1))
        b.append(arrow(cx, 254, cx, 242, LINE, 1.1))
        b.append(arrow(cx, 208, cx, 192, LINE, 1.1))
        b.append(arrow(cx, 162, cx, 150, LINE, 1.1))
        b.append(arrow(cx, 120, cx, 110, LINE, 1.1))
    for l, y in [(1, 300), (2, 254), (3, 208)]:
        b.append(arrow(hxs[0] + 40, y + 17, hxs[1] - 42, y + 17, LINE, 1.1))
    b.append(txt(728, 398, "the head has no horizontal connections", 10,
                 fill=MUTED))

    b.append(caption(W / 2, 444, [
        "Three recurrent layers is already a lot — the temporal extent makes "
        "these expensive, so you do not see 100-layer RNNs.",
        "The same parameters W_a[l], b_a[l] are shared across time within a "
        "layer, and each layer has its own set. Blocks can be GRU or LSTM, "
        "and each layer can be bidirectional.",
    ]))
    write(OUT, "seq-deep-rnn.svg", W, 484, b)


# ---------------------------------------- 15. unit comparison


def fig_unit_comparison():
    W = 920
    b = [title(W, "Choosing a recurrent unit",
               "more gates buys longer memory and costs parameters")]

    cards = [
        ("Basic RNN", ["a<t>"], [], 1.0,
         ["one tanh, no gates", "local influence only:", "forgets long-range "
          "dependencies"], LINE, "#f9fafb"),
        ("GRU", ["c<t> = a<t>"], ["\u0393_u  update", "\u0393_r  relevance"],
         3.0, ["simpler, so easier to scale", "to a much bigger network",
               "often just as good"], GREEN, "#f0fdf4"),
        ("LSTM", ["c<t>", "a<t>  (separate)"],
         ["\u0393_u  update", "\u0393_f  forget", "\u0393_o  output"], 4.0,
         ["more powerful and flexible", "historically the more proven",
          "default first thing to try"], BLUE, "#eff6ff"),
    ]

    for i, (name, state, gates, mult, notes, col, fill) in enumerate(cards):
        x = 40 + i * 294
        b.append(rect(x, 80, 274, 300, CELL, rx=10, fill=fill, stroke=col))
        b.append(txt(x + 137, 108, name, 14,
                     fill=INK if col is LINE else col, weight="600"))

        b.append(txt(x + 20, 134, "state", 9.5, anchor="start", fill=MUTED,
                     weight="600"))
        for j, s in enumerate(state):
            b.append(rect(x + 20, 140 + j * 26, 214, 22, MEM, rx=4, sw=1.2))
            b.append(txt(x + 127, 155 + j * 26, s, 10.5))

        b.append(txt(x + 20, 200, "gates", 9.5, anchor="start", fill=MUTED,
                     weight="600"))
        if not gates:
            b.append(txt(x + 127, 221, "none", 10.5, fill=MUTED))
        for j, g in enumerate(gates):
            b.append(rect(x + 20, 206 + j * 26, 214, 22, GATE, rx=4, sw=1.2))
            b.append(txt(x + 127, 221 + j * 26, g, 10.5))

        b.append(txt(x + 20, 300, "parameters per layer", 9.5, anchor="start",
                     fill=MUTED, weight="600"))
        b.append(rect(x + 20, 308, 200 * mult / 4.0, 14, YOUT, rx=3, sw=1.2,
                      fill="#ffffff", stroke=col))
        b.append(txt(x + 20 + 200 * mult / 4.0 + 8, 319,
                     f"\u2248 {mult:g}\u00d7", 10, anchor="start",
                     fill=INK if col is LINE else col, weight="600"))

        b.append(caption(x + 137, 344, notes, size=9.5, lh=14))

    b.append(caption(W / 2, 408, [
        "There is no universally better choice, and on different problems "
        "different units win. Both gated units fix the vanishing-gradient "
        "problem the same way:",
        "a state the unit can choose to leave alone.",
    ]))
    write(OUT, "seq-unit-comparison.svg", W, 444, b)


# ---------------------------------------------------------------- main

FIGURES = [
    fig_applications,
    fig_notation,
    fig_fc_problem,
    fig_rnn_unrolled,
    fig_rnn_forward,
    fig_bptt,
    fig_rnn_types,
    fig_language_model,
    fig_sampling,
    fig_vanishing,
    fig_gru_cell,
    fig_gru_memory,
    fig_lstm_cell,
    fig_lstm_chain,
    fig_brnn,
    fig_deep_rnn,
    fig_unit_comparison,
]


def build():
    for f in FIGURES:
        f()


if __name__ == "__main__":
    build()
    print(f"{len(FIGURES)} sequence-model figures written")
