"""Generate the diagrams used by face_recognition.md.

Driven by make_figures.py; can also be run directly.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from svgkit import (  # noqa: E402
    AMBER, BLUE, C, GREEN, INK, LINE, MUTED, PURPLE, RED,
    arrow, caption, circle, legend, lines, path, panel, rect, seg, title, txt,
    volume3d, write,
)

OUT = pathlib.Path(__file__).parent

# one colour per identity, reused across figures
ID = [
    ("#bbf7d0", GREEN),    # Danielle
    ("#bfdbfe", BLUE),     # Kian
    ("#fed7aa", "#ea580c"),  # Younes
    ("#e9d5ff", PURPLE),   # Tian
    ("#fecaca", RED),      # stranger
]


# ----------------------------------------------------------------- helpers


def face(cx, cy, s=48, fill="#bbf7d0", stroke=GREEN, label=None, mood="smile"):
    """A geometric face: circle, two eyes, a mouth. s is the diameter."""
    r = s / 2
    o = [
        circle(cx, cy, r, fill=fill, stroke=stroke, sw=1.7),
        circle(cx - r * 0.32, cy - r * 0.18, max(2.2, r * 0.11),
               fill=stroke, stroke=stroke, sw=0.4),
        circle(cx + r * 0.32, cy - r * 0.18, max(2.2, r * 0.11),
               fill=stroke, stroke=stroke, sw=0.4),
    ]
    mx, my, mw = cx, cy + r * 0.22, r * 0.32
    if mood == "smile":
        o.append(path(
            f"M {mx - mw:g},{my:g} Q {mx:g},{my + r * 0.28:g} {mx + mw:g},{my:g}",
            stroke=stroke, sw=1.6, marker=False,
        ))
    elif mood == "frown":
        o.append(path(
            f"M {mx - mw:g},{my + r * 0.12:g} Q {mx:g},{my - r * 0.10:g} {mx + mw:g},{my + r * 0.12:g}",
            stroke=stroke, sw=1.6, marker=False,
        ))
    else:
        o.append(seg(mx - mw, my + 2, mx + mw, my + 2, stroke, 1.6))
    if label:
        o.append(txt(cx, cy + r + 16, label, 11, fill=stroke, weight="600"))
    return "\n".join(o)


def encoding(x, y, w, h, n=8, role="fc", label="128-d"):
    """A stack of n cells standing in for a 128-dimensional encoding."""
    cell_h = h / n
    o = []
    for i in range(n):
        o.append(rect(x, y + i * cell_h, w, cell_h, role, rx=1, sw=1.0))
    o.append(txt(x + w / 2, y + h + 14, label, 10.5, weight="600"))
    return "\n".join(o)


def badge(x, y, w, h, text, fill, stroke, fs=12):
    return "\n".join([
        rect(x, y, w, h, "out", rx=6, sw=1.5, fill=fill, stroke=stroke),
        txt(x + w / 2, y + h / 2 + 4, text, fs, fill=stroke, weight="600"),
    ])


# -------------------------------------- 1. verification vs recognition


def fig_verify_vs_recognize():
    W, H = 920, 360
    b = [title(W, "Verification versus recognition",
               "one comparison against a claimed identity, versus one comparison against every identity")]

    # ---- verification
    b.append(panel(36, 68, 400, 232, fill="#f9fafb"))
    b.append(txt(236, 92, "Verification  ·  1 : 1", 13, weight="600"))
    b.append(txt(236, 110, "is this the person they claim to be?", 10.5, fill=MUTED))

    f, s = ID[0]
    b.append(face(110, 170, 64, f, s))
    b.append(txt(110, 220, "photo", 10.5, fill=MUTED))
    b.append(txt(200, 174, "+", 18, fill=MUTED, weight="600"))
    b.append(rect(236, 150, 88, 52, "fc", rx=6))
    b.append(lines(280, 168, ["claimed ID", "“Danielle”"], 10.5, lh=14))
    b.append(arrow(340, 176, 378, 176, LINE, 1.5))
    b.append(badge(382, 156, 40, 40, "?", "#fef3c7", AMBER, 16))
    b.append(txt(236, 268, "output: yes / no", 11, fill=MUTED))

    # ---- recognition
    b.append(panel(484, 68, 400, 232, fill="#f9fafb"))
    b.append(txt(684, 92, "Recognition  ·  1 : K", 13, weight="600"))
    b.append(txt(684, 110, "who is this, of the K people in the database?", 10.5, fill=MUTED))

    b.append(face(556, 170, 64, f, s))
    b.append(txt(556, 220, "photo", 10.5, fill=MUTED))
    b.append(arrow(606, 176, 648, 176, LINE, 1.5))

    db_x, db_y = 660, 132
    b.append(rect(db_x, db_y, 196, 92, "input", rx=6, fill="#ffffff", stroke=LINE))
    for i, (ff, ss) in enumerate(ID[:4]):
        b.append(face(db_x + 30 + i * 44, db_y + 38, 30, ff, ss))
    b.append(txt(db_x + 98, db_y + 82, "database of K people", 10, fill=MUTED))

    b.append(arrow(758, 228, 758, 248, LINE, 1.4))
    b.append(txt(684, 268, "output: “Danielle”   or   not in database", 11, fill=MUTED))

    b.append(caption(W / 2, H - 18, [
        "Recognition is K verification tests. A 1% verification error becomes a much larger recognition error.",
    ]))
    write(OUT, "face-verify-vs-recognize.svg", W, H, b)


# ---------------------------------------------- 2. one-shot learning


def fig_one_shot():
    W, H = 920, 420
    b = [title(W, "One-shot recognition with a similarity function",
               "one photo per person in the database; d small means same person, d large means different")]

    names = ["Kian", "Danielle", "Younes", "Tian"]
    # database strip
    b.append(txt(70, 78, "Employee database  (one photo each)", 12, anchor="start", weight="600"))
    db_y = 100
    for i, ((f, s), name) in enumerate(zip(ID[:4], names)):
        x = 70 + i * 96
        b.append(rect(x - 36, db_y, 72, 96, "input", rx=8, fill="#f9fafb", stroke=LINE))
        b.append(face(x, db_y + 36, 46, f, s))
        b.append(txt(x, db_y + 82, name, 11, fill=s, weight="600"))

    # query: Danielle
    qf, qs = ID[1]
    b.append(txt(560, 78, "Someone at the door", 12, anchor="start", weight="600"))
    b.append(rect(560, db_y, 88, 96, "input", rx=8, fill="#ecfdf5", stroke=GREEN))
    b.append(face(604, db_y + 36, 46, qf, qs))
    b.append(txt(604, db_y + 82, "query", 11, fill=GREEN, weight="600"))

    # pairwise distances
    dists = [("Kian", 10.2, MUTED, False), ("Danielle", 0.4, GREEN, True),
             ("Younes", 8.7, MUTED, False), ("Tian", 9.1, MUTED, False)]
    b.append(txt(70, 224, "d(query, database photo)   compared to threshold τ", 12,
                 anchor="start", weight="600"))
    for i, (name, d, col, match) in enumerate(dists):
        y = 244 + i * 32
        b.append(txt(88, y + 4, name, 12, anchor="start", fill=col, weight="600"))
        b.append(rect(180, y - 12, 220, 24, "out" if match else "input", rx=4,
                      fill=("#dcfce7" if match else "#f3f4f6"),
                      stroke=col))
        b.append(txt(290, y + 4, f"d = {d}" + ("   <  τ   same person" if match
                                               else "   >  τ   different"),
                     11.5, fill=col, weight="600"))

    # decision
    b.append(rect(560, 236, 320, 128, "input", rx=8, fill="#ecfdf5", stroke=GREEN))
    b.append(txt(720, 268, "only Danielle is below τ", 13, fill=GREEN, weight="600"))
    b.append(txt(720, 292, "the other three are all ≫ τ", 11.5, fill=MUTED))
    b.append(txt(720, 318, "predict: Danielle", 14, fill=GREEN, weight="600"))
    b.append(txt(720, 342, "if every d > τ  →  not in the database", 10.5, fill=MUTED))

    b.append(caption(W / 2, H - 18, [
        "If every d is above τ the person is not in the database. Adding a fifth employee is just adding a fifth photo.",
    ]))
    write(OUT, "face-one-shot.svg", W, H, b)


def fig_softmax_fails():
    W, H = 920, 300
    b = [title(W, "Why a softmax over identities does not solve one-shot learning",
               "too little data, and the output layer has to be rebuilt every time the roster changes")]

    b.append(panel(36, 68, 410, 178, fill="#fef2f2", stroke="#fecaca"))
    b.append(txt(241, 92, "Softmax classifier", 13, weight="600"))
    f, s = ID[1]
    b.append(face(90, 150, 44, f, s))
    b.append(arrow(122, 150, 168, 150, LINE, 1.4))
    b.append(rect(172, 128, 90, 44, "conv"))
    b.append(txt(217, 154, "ConvNet", 11, weight="600"))
    b.append(arrow(266, 150, 304, 150, LINE, 1.4))
    b.append(rect(308, 118, 112, 64, "out"))
    b.append(lines(364, 138, ["softmax", "Kian / Danielle /", "Younes / Tian / none"], 10, lh=13))
    b.append(txt(241, 228, "one example per class  ·  retrain when K grows", 10.5, fill=RED))

    b.append(panel(474, 68, 410, 178, fill="#f0fdf4", stroke="#bbf7d0"))
    b.append(txt(679, 92, "Similarity function  d", 13, weight="600"))
    b.append(face(530, 150, 40, ID[1][0], ID[1][1]))
    b.append(face(610, 150, 40, ID[0][0], ID[0][1]))
    b.append(arrow(640, 150, 688, 150, LINE, 1.4))
    b.append(rect(692, 128, 160, 44, "fc"))
    b.append(txt(772, 154, "d(x₁, x₂)  ≷  τ", 13, weight="600"))
    b.append(txt(679, 228, "any pair  ·  new people just get added to the database", 10.5, fill=GREEN))

    b.append(caption(W / 2, H - 18, [
        "Learn d once. At test time you never update the network — you only compare encodings.",
    ]))
    write(OUT, "face-softmax-fails.svg", W, H, b)


# --------------------------------------------------- 3. siamese network


def fig_siamese():
    W, H = 920, 400
    b = [title(W, "Siamese network",
               "the same ConvNet, with the same parameters, maps each face to a 128-dimensional encoding")]

    def tower(x, person, name, tag):
        f, s = person
        o = [face(x + 36, 88, 48, f, s, name)]
        svg, info = volume3d(x + 8, 142, 56, 56, 18, "conv", label="ConvNet")
        o.append(svg)
        o.append(arrow(x + 36, 118, x + 36, 140, LINE, 1.3))
        o.append(arrow(x + 36, info["bottom"] + 4, x + 36, 226, LINE, 1.3))
        o.append(encoding(x + 18, 230, 36, 64, n=8, role="fc", label=""))
        o.append(txt(x + 36, 312, tag, 11, weight="600"))
        o.append(txt(x + 36, 328, "∈ ℝ¹²⁸", 10.5, fill=MUTED))
        return "\n".join(o)

    b.append(tower(110, ID[1], "x⁽¹⁾  Danielle", "f(x⁽¹⁾)"))
    b.append(tower(360, ID[0], "x⁽²⁾  Kian", "f(x⁽²⁾)"))

    b.append(path("M 174,170 C 174,206  424,206  424,170",
                  stroke=PURPLE, sw=1.6, marker=False, dash="5 4"))
    b.append(txt(299, 220, "identical parameters", 10.5, fill=PURPLE, weight="600"))

    b.append(rect(214, 242, 152, 48, "out", rx=6))
    b.append(lines(290, 258, ["d(x⁽¹⁾, x⁽²⁾)", "‖f(x⁽¹⁾) − f(x⁽²⁾)‖²"], 11, lh=14))
    b.append(arrow(164, 266, 210, 266, RED, 1.5))
    b.append(arrow(396, 266, 370, 266, RED, 1.5))

    b.append(rect(620, 88, 260, 250, "input", rx=8, fill="#f9fafb", stroke=LINE))
    b.append(txt(750, 114, "What you want", 12.5, weight="600"))
    b.append(txt(750, 150, "same person", 11.5, fill=GREEN, weight="600"))
    b.append(txt(750, 168, "d small", 12, fill=GREEN))
    b.append(txt(750, 202, "different people", 11.5, fill=RED, weight="600"))
    b.append(txt(750, 220, "d large", 12, fill=RED))
    b.append(txt(750, 262, "Learn the ConvNet parameters", 11, fill=MUTED))
    b.append(txt(750, 280, "so that these two conditions hold", 11, fill=MUTED))
    b.append(txt(750, 306, "DeepFace  ·  Taigman et al.", 10.5, fill=MUTED))

    b.append(caption(W / 2, H - 18, [
        "The two towers are not two networks. They are one network, run twice, with the weights tied.",
    ]))
    write(OUT, "face-siamese.svg", W, H, b)


# -------------------------------------------------------- 4. triplet loss


def fig_triplet():
    W, H = 920, 400
    b = [title(W, "The triplet: Anchor, Positive, Negative",
               "pull the positive toward the anchor, push the negative away, and keep a margin α between them")]

    # three faces
    trio = [
        (180, ID[1], "Anchor  A", "Danielle"),
        (460, ID[1], "Positive  P", "Danielle, different photo"),
        (740, ID[0], "Negative  N", "Kian"),
    ]
    for x, (f, s), head, sub in trio:
        b.append(rect(x - 80, 72, 160, 130, "input", rx=8, fill="#f9fafb", stroke=LINE))
        b.append(txt(x, 92, head, 12, weight="600"))
        b.append(face(x, 138, 52, f, s))
        b.append(txt(x, 186, sub, 10, fill=MUTED))

    # distance arrows
    b.append(arrow(264, 138, 376, 138, GREEN, 1.8))
    b.append(txt(320, 126, "d(A, P)  small", 11.5, fill=GREEN, weight="600"))
    b.append(arrow(544, 138, 656, 138, RED, 1.8))
    b.append(txt(600, 126, "d(A, N)  large", 11.5, fill=RED, weight="600"))

    # number line
    y = 248
    b.append(txt(70, y - 8, "want", 11, anchor="start", weight="600"))
    b.append(seg(180, y, 740, y, LINE, 1.4))
    b.append(circle(180, y, 6, fill=GREEN, stroke=GREEN, sw=1.2))
    b.append(txt(180, y + 22, "A", 12, weight="600", fill=GREEN))
    b.append(circle(300, y, 6, fill=GREEN, stroke=GREEN, sw=1.2))
    b.append(txt(300, y + 22, "P", 12, weight="600", fill=GREEN))
    b.append(circle(740, y, 6, fill=RED, stroke=RED, sw=1.2))
    b.append(txt(740, y + 22, "N", 12, weight="600", fill=RED))

    b.append(seg(180, y - 14, 300, y - 14, GREEN, 1.6))
    b.append(txt(240, y - 26, "d(A, P)", 10.5, fill=GREEN, weight="600"))
    b.append(seg(300, y - 14, 420, y - 14, AMBER, 1.6))
    b.append(txt(360, y - 26, "α  margin", 10.5, fill=AMBER, weight="600"))
    b.append(seg(180, y + 40, 740, y + 40, RED, 1.4))
    b.append(txt(460, y + 56, "d(A, N)  ≥  d(A, P) + α", 12, fill=RED, weight="600"))

    b.append(rect(70, 324, 780, 40, "input", rx=6, fill="#fffbeb", stroke=AMBER))
    b.append(txt(W / 2, 348,
                 "Without α the network can cheat: output the zero vector, or the same vector for every face.",
                 11.5, fill=AMBER, weight="600"))

    b.append(caption(W / 2, H - 16, [
        "Example: if d(A, P) = 0.5 and α = 0.2, then d(A, N) = 0.51 is not good enough — it must be at least 0.7.",
    ]))
    write(OUT, "face-triplet.svg", W, H, b)


def fig_hard_triplets():
    W, H = 920, 340
    b = [title(W, "Easy triplets teach nothing; hard triplets do the work",
               "random negatives are usually already far away, so the loss is already zero")]

    # easy
    b.append(panel(36, 68, 420, 210, fill="#f9fafb"))
    b.append(txt(246, 92, "Easy triplet  (random N)", 13, weight="600"))
    y = 170
    b.append(seg(80, y, 420, y, LINE, 1.3))
    b.append(circle(110, y, 7, fill=GREEN, stroke=GREEN))
    b.append(txt(110, y + 24, "A", 12, fill=GREEN, weight="600"))
    b.append(circle(160, y, 7, fill=GREEN, stroke=GREEN))
    b.append(txt(160, y + 24, "P", 12, fill=GREEN, weight="600"))
    b.append(circle(390, y, 7, fill=RED, stroke=RED))
    b.append(txt(390, y + 24, "N", 12, fill=RED, weight="600"))
    b.append(txt(246, 214, "d(A, N) ≫ d(A, P) + α", 12, fill=MUTED))
    b.append(txt(246, 236, "loss already 0  ·  gradient is 0", 12, fill=RED, weight="600"))
    b.append(txt(246, 256, "the network learns nothing from this triplet", 10.5, fill=MUTED))

    # hard
    b.append(panel(484, 68, 400, 210, fill="#fff7ed", stroke="#fdba74"))
    b.append(txt(684, 92, "Hard triplet", 13, weight="600"))
    y = 170
    b.append(seg(528, y, 850, y, LINE, 1.3))
    b.append(circle(560, y, 7, fill=GREEN, stroke=GREEN))
    b.append(txt(560, y + 24, "A", 12, fill=GREEN, weight="600"))
    b.append(circle(640, y, 7, fill=GREEN, stroke=GREEN))
    b.append(txt(640, y + 24, "P", 12, fill=GREEN, weight="600"))
    b.append(circle(700, y, 7, fill=RED, stroke=RED))
    b.append(txt(700, y + 24, "N", 12, fill=RED, weight="600"))
    b.append(txt(684, 214, "d(A, P) ≈ d(A, N)", 12, fill=MUTED))
    b.append(txt(684, 236, "loss > 0  ·  gradient has work to do", 12, fill="#ea580c", weight="600"))
    b.append(txt(684, 256, "push N out or pull P in until the margin holds", 10.5, fill=MUTED))

    b.append(caption(W / 2, H - 18, [
        "FaceNet (Schroff, Kalenichenko, Philbin): mine the triplets that currently violate the margin.",
    ]))
    write(OUT, "face-hard-triplets.svg", W, H, b)


# ------------------------------------- 5. binary classification


def fig_binary():
    W, H = 920, 360
    b = [title(W, "Face verification as binary classification",
               "same Siamese encodings, but the last step is logistic regression on their element-wise difference")]

    def row(cy, person, name, tag):
        f, s = person
        o = [face(70, cy, 44, f, s)]
        o.append(txt(70, cy + 36, name, 11, fill=s, weight="600"))
        o.append(arrow(98, cy, 140, cy, LINE, 1.3))
        o.append(rect(144, cy - 22, 88, 44, "conv"))
        o.append(txt(188, cy + 4, "ConvNet", 11, weight="600"))
        o.append(arrow(236, cy, 276, cy, LINE, 1.3))
        o.append(encoding(280, cy - 32, 28, 64, n=8, role="fc", label=""))
        o.append(txt(294, cy + 46, tag, 10.5, weight="600"))
        return "\n".join(o)

    b.append(row(110, ID[1], "x⁽ⁱ⁾", "f(x⁽ⁱ⁾)"))
    b.append(row(220, ID[0], "x⁽ʲ⁾", "f(x⁽ʲ⁾)"))

    b.append(path("M 188,134 C 188,165  188,165  188,196",
                  stroke=PURPLE, sw=1.6, marker=False, dash="5 4"))
    b.append(txt(188, 172, "tied", 10, fill=PURPLE, weight="600"))

    # merge into the comparison head
    b.append(path("M 312,110 C 360,110  360,165  390,165",
                  stroke=LINE, sw=1.4, marker=True))
    b.append(path("M 312,220 C 360,220  360,165  390,165",
                  stroke=LINE, sw=1.4, marker=True))

    b.append(rect(394, 140, 150, 50, "flat", rx=6))
    b.append(lines(469, 158, ["|f(x⁽ⁱ⁾) − f(x⁽ʲ⁾)|", "128 features"], 11, lh=14))

    b.append(arrow(548, 165, 592, 165, LINE, 1.4))
    b.append(rect(596, 140, 136, 50, "out", rx=6))
    b.append(lines(664, 158, ["logistic", "ŷ = σ(wᵀu + b)"], 11, lh=14))

    b.append(arrow(736, 165, 776, 165, LINE, 1.4))
    b.append(badge(780, 147, 88, 36, "same?", "#fef3c7", AMBER, 13))

    b.append(rect(48, 292, 824, 28, "input", rx=4, fill="#f9fafb", stroke=LINE))
    b.append(txt(W / 2, 310,
                 "y = 1 if the two photos are the same person,   y = 0 if they are different   ·   train on pairs, not triplets",
                 11, fill=MUTED))

    b.append(caption(W / 2, H - 16, [
        "A chi-square variant replaces |a−b| with (a−b)² / (a+b). Both appear in the DeepFace paper.",
    ]))
    write(OUT, "face-binary.svg", W, H, b)


def fig_precompute():
    W, H = 920, 360
    b = [title(W, "Precompute the database encodings",
               "the ConvNet runs once per stored photo; at the door you encode only the new face")]

    # database column
    b.append(txt(160, 78, "Offline, once", 12.5, weight="600"))
    b.append(rect(48, 92, 224, 200, "input", rx=8, fill="#f9fafb", stroke=LINE))
    names = ["Kian", "Danielle", "Younes", "Tian"]
    for i, ((f, s), name) in enumerate(zip(ID[:4], names)):
        y = 118 + i * 42
        b.append(face(78, y, 26, f, s))
        b.append(arrow(98, y, 130, y, LINE, 1.1))
        b.append(encoding(134, y - 16, 18, 32, n=5, role="fc", label=""))
        b.append(txt(200, y + 4, f"f({name})", 10.5, anchor="start", fill=s, weight="600"))

    b.append(txt(160, 310, "store 128 numbers, not the photo", 10.5, fill=MUTED))

    # arrow
    b.append(arrow(292, 192, 360, 192, LINE, 1.6))
    b.append(txt(326, 176, "reuse", 10.5, fill=MUTED))

    # online — a single left-to-right pass
    b.append(txt(626, 78, "Online, every time someone walks up", 12.5, weight="600"))
    qf, qs = ID[1]
    b.append(rect(376, 92, 500, 200, "input", rx=8, fill="#ecfdf5", stroke=GREEN))
    b.append(face(430, 170, 48, qf, qs))
    b.append(txt(430, 212, "new photo", 10.5, fill=qs, weight="600"))
    b.append(arrow(460, 170, 500, 170, LINE, 1.3))
    b.append(rect(504, 148, 88, 44, "conv"))
    b.append(txt(548, 174, "ConvNet", 11, weight="600"))
    b.append(arrow(596, 170, 628, 170, LINE, 1.3))
    b.append(encoding(632, 138, 28, 64, n=8, role="fc", label=""))
    b.append(txt(646, 218, "f(query)", 10.5, weight="600"))
    b.append(arrow(664, 200, 664, 230, LINE, 1.3))
    b.append(rect(580, 234, 260, 40, "out", rx=6))
    b.append(txt(710, 258, "compare to stored  f(Kian) … f(Tian)", 11, weight="600"))

    b.append(caption(W / 2, H - 18, [
        "Works for both triplet-trained encodings and the binary-classification head. The database is a table of vectors.",
    ]))
    write(OUT, "face-precompute.svg", W, H, b)


# ---------------------------------------------------------------------- main

FIGURES = [
    fig_verify_vs_recognize,
    fig_one_shot,
    fig_softmax_fails,
    fig_siamese,
    fig_triplet,
    fig_hard_triplets,
    fig_binary,
    fig_precompute,
]


def build():
    for f in FIGURES:
        f()


if __name__ == "__main__":
    build()
    print(f"{len(FIGURES)} face-recognition figures written")
