#!/usr/bin/env python3
"""Sec 3 system overview -- a single left-to-right teaser of the whole pipeline.

Spare (more graphics, fewer words), in the "system overview" idiom of the top
security venues, in a palette kept clearly distinct from the dense main-text
analytical figures: those use a steel-blue / amber / green / red family, so the
overview uses a graphite line-art base with a violet (OFFLINE) + magenta
(ONLINE) duotone -- two hues far from blue and red -- so it reads as its own
object.  One row, read left to right:

    [P0 user (data x) / P1 model owner (model M), no trusted dealer]  ->
    [OFFLINE dealerless key generation -- decompose one depth-l GGM tree into c
        shallow trees (Sec 5.1) and generate the correlated segment keys
        (Sec 5.2), run in parallel]  =keys=>
    [ONLINE parallel segmented comparison -- c per-segment L_j fan out, combine
        by prefix-AND scan -> <1{x<r}>]  ->
    [upper-layer private inference (Sec 6): CNN / RNN via FSS gates ReLU,
        MaxPool, ArgMax, softmax]

Section anchors are drawn on the figure as small tags: Sec 5.1 (decompose) and
Sec 5.2 (key gen) in the OFFLINE lane, Sec 6 in the inference lane.

The heavy machinery (stage boxes, full lanes, the Path A / B last-layer fork)
lives in fig:compose; this stays a teaser.  Full text-width, one equal-unit axis.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon

# ---- palette: graphite base + violet/magenta duotone, kept distinct from the
# ---- steel-blue / amber / green / red of the main-text figures --------------
INK   = "#232b31"   # graphite: text / structural strokes / neutral endpoints
MUTE  = "#727d86"   # secondary labels
FAINT = "#d7dce0"   # hairline dividers
PAPER = "#eef1f3"   # neutral fill
VIO   = "#6a4fb0"   # OFFLINE phase (violet)
VIOF  = "#ece7f7"
MAG   = "#b23a7d"   # ONLINE phase (magenta / plum)
MAGF  = "#f7e6f0"

# global type scale: the axis is 200 units on a 7.16-in canvas, so raise every
# label a notch to the ~8-9 pt effective body size used in top-venue figures.
FS = 1.14

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["pdf.fonttype"] = 42

# ============================ primitive helpers ============================
def box(ax, x0, y0, x1, y1, fc, ec=INK, lw=1.1, r=1.6, z=3, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, linestyle=ls))


def txt(ax, x, y, s, size=7.0, color=INK, ha="center", va="center",
        weight="normal", z=6, style="normal"):
    ax.text(x, y, s, fontsize=size * FS, color=color, ha=ha, va=va,
            fontweight=weight, fontstyle=style, zorder=z)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.7, ms=12, style="-|>", ls="-",
          z=5, conn=None):
    kw = dict(arrowstyle=style, mutation_scale=ms, linewidth=lw, color=color,
              linestyle=ls, zorder=z, shrinkA=0, shrinkB=0)
    if conn:
        kw["connectionstyle"] = conn
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), **kw))


def pill(ax, xc, yc, w, h, s, fc, ec, size=6.6, tc=INK, weight="normal", z=6):
    ax.add_patch(FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={h/2}",
        linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=z))
    ax.text(xc, yc, s, fontsize=size * FS, color=tc, ha="center", va="center",
            fontweight=weight, zorder=z + 2)


def header(ax, x0, x1, ytop, tag, sub, color):
    """Colored title strip: bold phase tag + lighter subtitle, one line."""
    ax.add_patch(FancyBboxPatch(
        (x0, ytop - 5.4), x1 - x0, 5.4,
        boxstyle="round,pad=0,rounding_size=1.6",
        linewidth=0, facecolor=color, zorder=4))
    ax.text((x0 + x1) / 2, ytop - 2.7, f"$\\bf{{{tag}}}$   {sub}",
            fontsize=6.7, color="white", ha="center", va="center", zorder=7)


# ============================== icon glyphs ================================
def user_icon(ax, cx, cy, color, fillc, s=1.0):
    """A crisp person token: round head + trapezoid-bust shoulders."""
    # shoulders: a rounded trapezoid (wide base, narrow top) = classic bust
    ax.add_patch(Polygon(
        [(cx - 3.5 * s, cy - 3.1 * s), (cx + 3.5 * s, cy - 3.1 * s),
         (cx + 2.2 * s, cy - 0.1 * s), (cx - 2.2 * s, cy - 0.1 * s)],
        closed=True, fc=fillc, ec=color, lw=1.4, zorder=5,
        joinstyle="round", capstyle="round"))
    # head sits above the shoulders
    ax.add_patch(Circle((cx, cy + 2.6 * s), 1.95 * s, fc=fillc, ec=color,
                 lw=1.4, zorder=6))


def server_icon(ax, cx, cy, color, fillc, s=1.0):
    """A model-owner token: a stack of two rack units with status lights."""
    for dy in (2.0, -2.0):
        yy = cy + dy * s
        ax.add_patch(FancyBboxPatch((cx - 3.2 * s, yy - 1.55 * s),
                     6.4 * s, 3.1 * s, boxstyle="round,pad=0,rounding_size=0.5",
                     linewidth=1.25, edgecolor=color, facecolor=fillc, zorder=6))
        ax.add_patch(Circle((cx - 2.1 * s, yy), 0.42 * s, fc=color, ec="none",
                     zorder=7))
        ax.plot([cx - 0.9 * s, cx + 2.3 * s], [yy, yy], color=color, lw=0.8,
                zorder=7)


def cloud_x(ax, cx, cy, color=INK, s=1.0):
    """A struck-out cloud/server -- the absent trusted dealer."""
    parts = [(-2.2, -0.2, 2.0), (0.2, 0.7, 2.5), (2.6, -0.1, 1.9)]
    for dx, dy, rr in parts:
        ax.add_patch(Circle((cx + dx * s, cy + dy * s), rr * s, fc="white",
                     ec=MUTE, lw=1.15, zorder=5))
    ax.add_patch(FancyBboxPatch((cx - 4.0 * s, cy - 2.2 * s), 8.4 * s, 2.6 * s,
                 boxstyle="round,pad=0,rounding_size=1.0", linewidth=0,
                 facecolor="white", zorder=5))
    ax.plot([cx - 4.4 * s, cx - 4.4 * s + 9.0 * s],
            [cy + 3.2 * s, cy - 3.2 * s], color=color, lw=2.0, zorder=8)


def ggm_tree(ax, cx, ytop, ybot, half, depth, color, rnode=0.62, lw=0.95,
             fill="white", z=5):
    """A node-link GGM tree: root at top, `depth` levels branching down.
    Returns the list of leaf x-positions (for wiring)."""
    levels = []
    dy = (ytop - ybot) / depth
    for d in range(depth + 1):
        y = ytop - d * dy
        span = half * (d / depth) if depth else 0
        if d == 0:
            row = [cx]
        else:
            n = 2 ** d
            row = [cx - span + (2 * span) * (i / (n - 1)) for i in range(n)] \
                if n > 1 else [cx]
        levels.append((y, row))
    # edges parent->child
    for d in range(depth):
        yp, rp = levels[d]
        yc, rc = levels[d + 1]
        for i, xp in enumerate(rp):
            for xchild in (rc[2 * i], rc[2 * i + 1]):
                ax.plot([xp, xchild], [yp, yc], color=color, lw=lw, zorder=z)
    # nodes
    for d, (y, row) in enumerate(levels):
        for xx in row:
            fc = color if d == 0 else fill
            ax.add_patch(Circle((xx, y), rnode, fc=fc, ec=color, lw=lw,
                         zorder=z + 1))
    return levels[-1][0], levels[-1][1]  # (y_leaf, leaf_xs)


def key_glyph(ax, cx, cy, color, s=1.0):
    """A small key: bow (ring) + blade + teeth."""
    ax.add_patch(Circle((cx - 3.0 * s, cy), 1.6 * s, fc="white", ec=color,
                 lw=1.3, zorder=8))
    ax.plot([cx - 1.4 * s, cx + 3.6 * s], [cy, cy], color=color, lw=1.7,
            zorder=8)
    for dx in (2.1, 3.4):
        ax.plot([cx + dx * s, cx + dx * s], [cy, cy - 1.5 * s], color=color,
                lw=1.7, zorder=8)


def op_chip(ax, cx, cy, w, h, kind, color, fillc):
    """A small pictographic chip for an upper-layer nonlinear op."""
    box(ax, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, fillc, ec=color,
        lw=1.0, r=0.9, z=5)
    gx0, gx1 = cx - w / 2 + 1.6, cx + w / 2 - 1.6
    gy0, gy1 = cy - h / 2 + 1.9, cy + h / 2 - 1.4
    if kind == "relu":                       # hinge: flat then rising
        xm = (gx0 + gx1) / 2
        ax.plot([gx0, xm, gx1], [gy0, gy0, gy1], color=color, lw=1.3, zorder=7)
    elif kind == "maxpool":                  # 2x2 grid, one cell highlighted
        mx, my = (gx0 + gx1) / 2, (gy0 + gy1) / 2
        ax.add_patch(FancyBboxPatch((gx0, my), (gx1 - gx0) / 2, (gy1 - my),
                     boxstyle="square,pad=0", fc=color, ec="none", zorder=6))
        for xx in (gx0, mx, gx1):
            ax.plot([xx, xx], [gy0, gy1], color=color, lw=0.8, zorder=7)
        for yy in (gy0, my, gy1):
            ax.plot([gx0, gx1], [yy, yy], color=color, lw=0.8, zorder=7)
    elif kind == "argmax":                   # bars, tallest flagged
        n = 4
        bw = (gx1 - gx0) / (2 * n)
        hs = [0.32, 0.6, 1.0, 0.5]
        for i, hh in enumerate(hs):
            xx = gx0 + bw + i * 2 * bw
            top = gy0 + (gy1 - gy0) * hh
            fc = color if hh == 1.0 else fillc
            ax.add_patch(FancyBboxPatch((xx - bw / 2, gy0), bw, top - gy0,
                         boxstyle="square,pad=0", fc=fc, ec=color, lw=0.8,
                         zorder=7))
    elif kind == "softmax":                  # smooth logistic-ish curve
        import math
        xs = [gx0 + (gx1 - gx0) * t / 24 for t in range(25)]
        ys = [gy0 + (gy1 - gy0) / (1 + math.exp(-(t - 12) / 3.0))
              for t in range(25)]
        ax.plot(xs, ys, color=color, lw=1.3, zorder=7)


# ================================ canvas ==================================
# equal units: xlim 200 wide, ylim 74 tall -> figsize 7.16 x 2.649
fig = plt.figure(figsize=(7.16, 7.16 * 74 / 200))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 200)
ax.set_ylim(0, 74)
ax.axis("off")

YC = 35                      # flow baseline
PT, PB = 63, 10             # phase-panel top / bottom
OFF0, OFF1 = 41, 101        # OFFLINE panel span
ON0, ON1 = 109, 168         # ONLINE panel span (wider: result stays in bounds)
AP0, AP1 = 174, 199         # applications panel span (widened for chip grid)

# ---- phase panels: the two SIGNAL phases carry colored header strips; the two
# ---- neutral ENDPOINTS (parties, apps) are strip-less graphite line-art -----
box(ax, OFF0, PB, OFF1, PT, VIOF, ec=VIO, lw=1.3, r=2.2, z=1)
box(ax, ON0, PB, ON1, PT, MAGF, ec=MAG, lw=1.3, r=2.2, z=1)
header(ax, OFF0, OFF1, PT, "OFFLINE", "dealerless key generation", VIO)
header(ax, ON0, ON1, PT, "ONLINE", "parallel segmented comparison", MAG)

# ============================ 1. two parties ===============================
cloud_x(ax, 11, 55, color=INK, s=0.92)
txt(ax, 18.5, 56.6, "no trusted", size=5.9, color=INK, weight="bold", ha="left")
txt(ax, 18.5, 53.2, "dealer", size=5.9, color=INK, weight="bold", ha="left")
# P0 = user (private data x)
user_icon(ax, 11, 41.5, INK, PAPER, s=1.0)
txt(ax, 15.5, 43.0, r"$P_0$ user", size=6.2, weight="bold", ha="left")
txt(ax, 15.5, 39.4, r"private data $x$", size=5.1, color=MUTE, ha="left")
# P1 = model owner (commercial model M)
server_icon(ax, 11, 27.5, INK, PAPER, s=1.0)
txt(ax, 15.5, 29.0, r"$P_1$ model owner", size=6.2, weight="bold", ha="left")
txt(ax, 15.5, 25.4, r"commercial model $M$", size=4.7, color=MUTE, ha="left")
# joint two-party channel
arrow(ax, 11, 37.0, 11, 31.8, color=MUTE, lw=1.15, ms=7, style="<|-|>")
txt(ax, 7.2, 34.4, "2PC", size=5.1, color=MUTE, ha="right")

arrow(ax, 35.5, YC, OFF0 - 0.7, YC, color=INK, lw=2.0, ms=14)

# ==================== 2. OFFLINE: decompose (S5.1) + keygen (S5.2) =========
# The graphic depicts the Sec 5.1 decomposition -- one deep depth-l GGM tree
# split into c shallow trees -- from which Sec 5.2 generates the segment keys.
# Two section tags anchor each half to its subsection; no single protocol name
# is pinned over the whole graphic (the per-protocol pipeline lives in fig:compose).
def sectag(xc, y, num, label, color):
    """A centered section tag: bold '\S5.x' + a muted label, sized to the cluster."""
    txt(ax, xc, y, num, size=5.9, color=color, weight="bold")
    txt(ax, xc, y - 3.4, label, size=5.0, color=MUTE)

sectag(52, 55.0, r"$\S$5.1", "decompose input", VIO)
sectag(82, 55.0, r"$\S$5.2", "generate keys", VIO)

ggm_tree(ax, 52, 48.4, 30.5, 6.5, 3, VIO, rnode=0.6, lw=0.9)  # deep tree
txt(ax, 52, 26.8, r"one depth-$\ell$ tree", size=5.3, color=MUTE)
arrow(ax, 60, YC + 6, 66.5, YC + 6, color=VIO, lw=1.6, ms=9)
txt(ax, 63.2, YC + 9.4, "split", size=5.6, color=VIO, weight="bold")
# c shallow trees, run in parallel
for sx in (72, 82, 92):
    ggm_tree(ax, sx, 47.5, 39.5, 3.4, 2, VIO, rnode=0.5, lw=0.85, fill=VIOF)
    arrow(ax, sx, 38.5, sx, YC + 1.2, color=VIO, lw=0.9, ms=6)
    key_glyph(ax, sx - 1.6, YC - 2.0, VIO, s=0.5)
txt(ax, 82, 27.6, r"$c$ shallow trees, depth $\ell/c$", size=5.3, color=MUTE)
txt(ax, 82, 24.0, "run in parallel", size=5.3, color=MUTE)

pill(ax, (OFF0 + OFF1) / 2, PB + 3.0, 58, 5.0,
     r"Complexity: $\mathcal{O}(2^{\ell})\!\rightarrow\!\mathcal{O}(c\,2^{\ell/c})$  with paralleled COT",
     "white", VIO, size=4.7, tc=INK, z=6)

# ===== keys handoff:  OFFLINE == keys ==> ONLINE (dashed = preprocessed) ====
# Draw the dashed shaft and the arrowhead as two patches: a dashed line pattern
# applied to a filled -|> head carves a white wedge out of the head, so keep the
# head solid and let it overlap the dashed shaft's end (no gap, no notch).  A
# fine dash pattern renders a couple of visible dashes across the short gap.
_kx0, _kx1 = OFF1 + 0.7, ON0 - 0.7
# arrow(ax, _kx0, YC, _kx1 - 3.0, YC, color=VIO, lw=1.9, style="-",
#       ls=(0, (2.2, 1.6)))
arrow(ax, _kx1 - 4.4 - 2, YC, _kx1 - 2, YC, color=VIO, lw=1.9, ms=14, style="-|>")
key_glyph(ax, 102.6, 41.5, VIO, s=0.72)
txt(ax, 105.0, 45.8, "keys", size=5.4, color=VIO, weight="bold")
txt(ax, 105.0, 31.6, r"$\{k^{(j)}\}$", size=5.3, color=MUTE)

# ==================== 3. ONLINE (Sec 5.1): segmented comparison ============
# open x_hat -> c parallel per-segment comparisons L_j fan out -> prefix-AND
# scan combines them -> comparison share.  Vertical fan keeps interaction legible
# and the result pill inside the panel.
# drop the formula below the header strip (strip bottom = PT-5.4 = 57.6) so the
# tall bigoplus glyph clears the top frame instead of being clipped by it.
txt(ax, (ON0 + ON1) / 2, 53.8,
    r"$\langle\mathbf{1}\{x<r\}\rangle=\bigoplus_{j}\,E_j\wedge L_j$",
    size=5.8, color=INK)

box(ax, ON0 + 2.5, YC - 4.0, ON0 + 12.0, YC + 4.0, "white", ec=MAG, lw=1.1,
    r=1.3, z=4)
txt(ax, ON0 + 7.25, YC + 1.6, "open", size=5.8, weight="bold", color=MAG)
txt(ax, ON0 + 7.25, YC - 1.8, r"$\hat x\!=\!x\!+\!r$", size=5.4)

# fan out from open() into c stacked segment cells
seg_cx = ON0 + 25.0
seg_ys = [45.5, YC, 24.5]
arrow(ax, ON0 + 12.5, YC, seg_cx - 8.5, YC, color=MAG, lw=1.3, ms=8)
for i, sy in enumerate(seg_ys):
    ax.add_patch(FancyArrowPatch((seg_cx - 8.0, YC), (seg_cx - 5.8, sy),
                 arrowstyle="-", mutation_scale=1, linewidth=0.9, color=MAG,
                 connectionstyle="arc3,rad=0.05", zorder=4, shrinkA=0, shrinkB=0))
    box(ax, seg_cx - 5.6, sy - 3.0, seg_cx + 5.6, sy + 3.0, MAGF, ec=MAG,
        lw=1.0, r=0.9, z=5)
    ax.text(seg_cx - 2.4, sy, rf"$\hat x^{{({i+1})}}$", fontsize=5.5 * FS,
            color=INK, ha="center", va="center", zorder=7)
    ax.text(seg_cx + 2.6, sy, rf"$L_{{{i+1}}}$", fontsize=5.5 * FS, color=MAG,
            ha="center", va="center", zorder=7)
txt(ax, seg_cx, 49.8, r"$c$ segments in parallel", size=5.0, color=MAG,
    weight="bold")

# combine: prefix-AND scan folds the per-segment bits into the result
comb_x = seg_cx + 13.5
for sy in seg_ys:
    ax.add_patch(FancyArrowPatch((seg_cx + 5.8, sy), (comb_x - 1.9, YC),
                 arrowstyle="-", mutation_scale=1, linewidth=0.9, color=MAG,
                 connectionstyle="arc3,rad=-0.05", zorder=4, shrinkA=0,
                 shrinkB=0))
ax.add_patch(Circle((comb_x, YC), 2.4, fc="white", ec=MAG, lw=1.3, zorder=7))
ax.text(comb_x, YC + 0.1, r"$\bigwedge$", fontsize=7.0 * FS, color=MAG,
        ha="center", va="center", zorder=8)
txt(ax, comb_x, 28.4, "prefix-$\\wedge$", size=5.2, color=MAG, weight="bold")
txt(ax, comb_x, 25.0, "combine", size=4.8, color=MUTE, style="italic")

arrow(ax, comb_x + 2.6, YC, comb_x + 5.2, YC, color=MAG, lw=1.3, ms=8)
pill(ax, ON1 - 8.4, YC, 14.5, 5.6, r"$\langle\mathbf{1}\{x<r\}\rangle$",
     MAGF, MAG, size=5.4, tc=INK, z=6)

pill(ax, (ON0 + ON1) / 2, PB + 3.0, 57, 5.0,
     r"Scan depth: $\mathcal{O}(\log c)$ with preprocessed triples",
     "white", MAG, size=4.7, tc=INK, z=6)

# ===== result handoff -> applications ======================================
arrow(ax, ON1 + 0.7, YC, AP0 - 0.7, YC, color=MAG, lw=2.0, ms=14)

# ==================== 4. upper-layer private inference =====================
box(ax, AP0, PB, AP1, PT, "white", ec=INK, lw=1.3, r=2.2, z=2)
apc = (AP0 + AP1) / 2
txt(ax, apc, PT - 4.0, "Private", size=6.3, color=INK, weight="bold")
txt(ax, apc, PT - 8.0, "inference", size=6.3, color=INK, weight="bold")
# Sec 6 describes gate composition; complete network measurements are separate.
txt(ax, apc, PT - 12.0, r"$\S$6  composition", size=5.1, color=MUTE)
# CNN / RNN as the upper-layer models
txt(ax, apc, 47.5, "CNN  ·  RNN", size=6.0, color=INK, weight="bold")
# the four nonlinear ops drawn as pictographic FSS-gate chips (2x2),
# each chip paired with its label as a titled cell so nothing overlaps.
cw, chh = 7.6, 5.6
ox = [apc - 5.4, apc + 5.4]
oy = [40.5, 28.5]
labels = [["ReLU", "MaxPool"], ["ArgMax", "softmax"]]
kinds = [["relu", "maxpool"], ["argmax", "softmax"]]
for r, yy in enumerate(oy):
    for c, xx in enumerate(ox):
        op_chip(ax, xx, yy, cw, chh, kinds[r][c], INK, PAPER)
        txt(ax, xx, yy - 4.6, labels[r][c], size=4.7, color=MUTE)
pill(ax, apc, PB + 2.9, 22.4, 4.6, "FSS nonlinear gates", PAPER, INK,
     size=4.7, tc=INK)

fig.savefig("fig_overview.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("fig_overview.png", dpi=1200, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("fig_overview saved")
