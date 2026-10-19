#!/usr/bin/env python3
"""Sec 4.2 figure -- network-adaptive generation/evaluation of segment keys.

Security-venue protocol-dataflow style, graphics-first: an offline/online
phase band, one shared dealerless GGM spine (PREP -> GenFirst) drawn as an
actual tree that forks *only at the last layer* into a blue arithmetic-leaf
lane (PathA) and an amber Boolean-leaf lane (PathB). Network posture is shown
with WAN/LAN icons and round-count badges instead of prose; offline weight is
a small stack glyph. The two lanes reconverge on the shared Sec 4.1 scan.
Full text-width, two-column IEEEtran.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse

# ---- palette: neutral spine, blue (PathA), amber (PathB), green (converge) --
INK      = "#1b232c"
SLATE    = "#5b6b7b"
FILL     = "#eef2f6"
BAND_OFF = "#f4f6f8"
BAND_ON  = "#eaf1f7"
BLUE     = "#2c5f8a"
BLUEF    = "#dce6f0"
AMBER    = "#b5710f"
AMBERF   = "#f5e7d2"
GREEN    = "#2e8b57"
GREENF   = "#dbeee2"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType -> crisp, portable vector text


def box(ax, x0, y0, x1, y1, fc, ec=INK, lw=1.2, r=1.0, z=3, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, linestyle=ls))


def txt(ax, x, y, s, size=7.2, color=INK, ha="center", va="center",
        weight="normal", z=6):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, zorder=z)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.6, ms=11, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color=color, linestyle=ls, zorder=5,
        shrinkA=0, shrinkB=0))


def pill(ax, xc, yc, w, h, s, fc, ec, size=7.0, tc=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={h/2}",
        linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=4))
    ax.text(xc, yc, s, fontsize=size, color=tc, ha="center", va="center",
            fontweight=weight, zorder=6)


def leafmark(ax, cx, cy, kind, color, fillc, s=1.9):
    """Last-layer leaf glyph: circle = arithmetic, square = Boolean."""
    if kind == "circle":
        ax.add_patch(Circle((cx, cy), s, fc=fillc, ec=color, lw=1.1, zorder=4))
    else:
        ax.add_patch(FancyBboxPatch(
            (cx - s, cy - s), 2 * s, 2 * s,
            boxstyle="round,pad=0,rounding_size=0.25",
            linewidth=1.1, edgecolor=color, facecolor=fillc, zorder=4))


def weight_glyph(ax, cx, cy, n, color):
    """Offline-cost stack: n stacked slabs (taller stack = heavier offline)."""
    w, h, gap = 7.5, 1.5, 0.7
    for i in range(n):
        yy = cy + i * (h + gap)
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, yy), w, h, boxstyle="round,pad=0,rounding_size=0.3",
            linewidth=0.8, edgecolor=color, facecolor=color, zorder=4,
            alpha=0.85 - 0.12 * i))


def net_icon(ax, cx, cy, kind, color):
    """WAN = far parties + dashed long link; LAN = near parties + solid link."""
    d = 6.2 if kind == "WAN" else 3.4
    for sx in (cx - d, cx + d):
        ax.add_patch(FancyBboxPatch(
            (sx - 1.5, cy - 1.2), 3.0, 2.4,
            boxstyle="round,pad=0,rounding_size=0.4",
            linewidth=1.0, edgecolor=color, facecolor="white", zorder=5))
    ls = (0, (2, 1.6)) if kind == "WAN" else "-"
    lw = 1.0 if kind == "WAN" else 2.0
    ax.plot([cx - d + 1.5, cx + d - 1.5], [cy, cy], color=color, lw=lw,
            ls=ls, zorder=4)
    if kind == "WAN":
        ax.add_patch(Ellipse((cx, cy + 2.1), 5.0, 2.3, fc="white", ec=color,
                             lw=0.9, zorder=5))


def node(ax, x, y, root=False):
    ax.add_patch(Circle((x, y), 1.05 if root else 0.85,
                        fc=(SLATE if root else "white"), ec=SLATE, lw=1.0,
                        zorder=4))


# ============================================================================
# equal-unit canvas: xlim spans 150 so data-unit x == data-unit y (round nodes)
fig = plt.figure(figsize=(7.16, 4.9165))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 150)
ax.set_ylim(-3, 100)
ax.axis("off")

C = 75
DIV = 34

# ---- phase bands -----------------------------------------------------------
box(ax, 7.5, DIV, 144, 95, BAND_OFF, ec="none", r=0.4, z=0)
box(ax, 7.5, -1.5, 144, DIV, BAND_ON, ec="none", r=0.4, z=0)
ax.plot([7.5, 144], [DIV, DIV], color=SLATE, lw=1.0, ls=(0, (5, 3)), zorder=1)
# horizontal phase tags at the top-left of each band (no vertical text)
for ty, lab in [(92.2, "OFFLINE"), (30.7, "ONLINE")]:
    ax.add_patch(FancyBboxPatch((8.5, ty - 0.9), 1.8, 1.8,
                 boxstyle="round,pad=0,rounding_size=0.35",
                 linewidth=0, facecolor=SLATE, zorder=6))
    ax.text(11.4, ty, lab, ha="left", va="center", fontsize=7.6, color=SLATE,
            fontweight="bold", zorder=6)

# ---- title + shared dealerless spine (PREP -> GenFirst) --------------------
# txt(ax, C, 97.6, "Shared dealerless offline spine", size=9.0, weight="bold")
box(ax, 37, 88, 73, 93.5, FILL, ec=INK, lw=1.3, z=3)
txt(ax, 55, 91.4, r"$\Pi_{\mathsf{PREP}}$  (Alg. 3)", size=7.8, weight="bold")
txt(ax, 55, 89.2, "parallel-COT preprocessing", size=6.2, color=SLATE)
box(ax, 77, 88, 113, 93.5, FILL, ec=INK, lw=1.3, z=3)
txt(ax, 95, 91.4, r"$\Pi_{\mathsf{GenFirst}}$  (Alg. 4)", size=7.8, weight="bold")
txt(ax, 95, 89.2, r"blind the $\alpha$-path", size=6.2, color=SLATE)
arrow(ax, 73, 90.75, 77, 90.75, color=INK, lw=1.6, ms=10)
arrow(ax, 92, 88, 78, 85.4, color=INK, lw=1.6, ms=10)

# ---- shared cGGM tree (layers 1..l-1) --------------------------------------
node(ax, C, 84.5, root=True)
L1 = [C - 11, C + 11]
L2 = [C - 18, C - 6, C + 6, C + 18]
for x in L1:
    ax.plot([C, x], [84.5, 79.5], color=SLATE, lw=1.0, zorder=2)
    node(ax, x, 79.5)
for px, kids in zip(L1, [L2[:2], L2[2:]]):
    for x in kids:
        ax.plot([px, x], [79.5, 74.5], color=SLATE, lw=1.0, zorder=2)
        node(ax, x, 74.5)
        ax.plot([x, x], [73.6, 69.2], color=SLATE, lw=0.9, ls=(0, (1, 1.6)),
                zorder=2)
ax.text(C, 71.2, r"$\vdots$", ha="center", va="center", fontsize=12,
        color=SLATE)
box(ax, C - 20, 67.5, C + 20, 69.2, FILL, ec=SLATE, lw=1.0, r=0.6, z=3)
# right bracket: the whole tree above the fork is shared, independent of c
ax.plot([98, 100, 100, 98], [84.5, 84.5, 67.5, 67.5], color=SLATE, lw=1.0,
        zorder=2)
ax.text(121, 78.0, r"shared layers $1..\ell{-}1$", ha="center", va="center",
        fontsize=6.9, color=SLATE)
ax.text(121, 74.6, r"(independent of $c$)", ha="center", va="center",
        fontsize=6.9, color=SLATE)

# ---- fork: diverge only at the last layer ----------------------------------
ax.plot([C, C], [67.5, 66.6], color=INK, lw=1.4, zorder=4)
pill(ax, C, 64.5, 46, 4.2, "diverge only at the last layer", FILL, INK,
     size=7.0, weight="bold")
ax.plot([C, C], [62.4, 61.0], color=INK, lw=1.4, zorder=4)
ax.plot([40, 110], [61.0, 61.0], color=INK, lw=1.4, zorder=4)
arrow(ax, 40, 61.0, 40, 59.6, color=BLUE, lw=1.7, ms=10)
arrow(ax, 110, 61.0, 110, 59.6, color=AMBER, lw=1.7, ms=10)


def cost_bar(cx, frac, color):
    """Offline-cost bar on a shared track: longer bar == heavier offline."""
    # ax.text(cx - 24, 41.4, "offline", ha="left", va="center", fontsize=6.0,
    #         color=SLATE)
    # ax.add_patch(FancyBboxPatch((cx - 11, 40.6), 30, 1.7,
    #              boxstyle="round,pad=0,rounding_size=0.85", linewidth=0,
    #              facecolor="#e3e8ee", zorder=3))
    # ax.add_patch(FancyBboxPatch((cx - 11, 40.6), 30 * frac, 1.7,
    #              boxstyle="round,pad=0,rounding_size=0.85", linewidth=0,
    #              facecolor=color, zorder=4))


def lane(cx, accent, fillc, kind, title, sub, recov, keylabel,
         cost_frac, net, netlabel, rounds, walk, pref):
    hw = 31
    x0, x1 = cx - hw, cx + hw
    # title bar (names the lane)
    box(ax, x0, 54.0, x1, 59.6, accent, ec=accent, lw=1.2, z=3)
    ax.text(cx, 57.6, title, fontsize=8.0, color="white", ha="center",
            va="center", fontweight="bold", zorder=6)
    ax.text(cx, 55.2, sub, fontsize=6.5, color="white", ha="center",
            va="center", zorder=6)
    # last-layer leaves: circle == arithmetic, square == Boolean
    for dx in (-8, -2.7, 2.7, 8):
        leafmark(ax, cx + dx, 50.4, kind, accent, fillc, s=1.65)
    # recovery chip
    box(ax, x0, 43.5, x1, 48.6, fillc, ec=accent, lw=1.1, z=3)
    ax.text(cx, 46.9, recov[0], fontsize=6.4, color=INK, ha="center",
            va="center", zorder=6)
    ax.text(cx, 44.6, recov[1], fontsize=6.4, color=INK, ha="center",
            va="center", zorder=6)
    # offline-cost bar
    cost_bar(cx, cost_frac, accent)
    # connect the recovery chip above down into the key output pill
    arrow(ax, cx, 43.4, cx, 40.5, color=accent, lw=1.8, ms=12)
    # key output pill, then a bold connector down across the divider
    pill(ax, cx, 38.4, 26, 3.9, keylabel, "white", accent, size=7.2,
         weight="bold")
    arrow(ax, cx, 36.4, cx, 28.6, color=accent, lw=2.0, ms=13)
    # online box
    box(ax, x0, 18.0, x1, 28.6, fillc, ec=accent, lw=1.1, z=3)
    net_icon(ax, cx - 10, 25.0, net, accent)
    ax.text(cx - 10, 21.4, netlabel, ha="center", va="center", fontsize=6.6,
            color=accent, fontweight="bold", zorder=6)
    pill(ax, cx + 12, 24.8, 20, 4.4, rounds, "white", accent, size=7.0,
         weight="bold", tc=accent)
    ax.text(cx + 12, 20.9, walk, ha="center", va="center", fontsize=6.2,
            color=INK, zorder=6)
    # network-preference pill
    arrow(ax, cx, 18.0, cx, 16.4, color=accent, lw=1.3, ms=8)
    pill(ax, cx, 14.4, 30, 4.2, pref, fillc, accent, size=7.0, weight="bold")


lane(40, BLUE, BLUEF, "circle",
     "ONLINE-OPT", r"PathA $\cdot$ arithmetic leaves",
     ["last-layer recovery (Alg. 5)", r"CCMP sign bit $\rightarrow$ arith MUX"],
     "Arithmetic Key", 0.90,
     "WAN", "WAN", "0 rounds", "silent local walk", "favors WAN")

lane(110, AMBER, AMBERF, "square",
     "OFFLINE-OPT", r"PathB $\cdot$ Boolean leaves",
     ["last-layer recovery (Alg. 6)", r"Boolean select $\rightarrow$ Boolean CW"],
     "Boolean Key", 0.42,
     "LAN", "LAN", "1 round", "Grotto parity walk", "favors LAN")

# ---- convergence on the shared Sec 4.1 scan --------------------------------
box(ax, 45, -1.0, 105, 9.5, GREENF, ec=GREEN, lw=1.4, z=3)
txt(ax, C, 6.4, "same multiplication-scan of Sec. 4.1", size=7.6,
    weight="bold")
txt(ax, C, 2.4,
    r"over $\mathbb{Z}_{2^{\ell}}$ (PathA) or $\mathbb{F}_2$ (PathB)"
    r"$\;\Rightarrow\; \mathbf{1}\{\hat{x}<r\}$", size=7.4)
arrow(ax, 40, 12.3, 52, 9.5, color=GREEN, lw=1.5, ms=10)
arrow(ax, 110, 12.3, 98, 9.5, color=GREEN, lw=1.5, ms=10)

fig.savefig("fig_compose.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("fig_compose.png", dpi=1200, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("fig_compose saved")
