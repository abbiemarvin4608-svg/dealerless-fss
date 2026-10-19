#!/usr/bin/env python3
"""Sec 4.1 -- decomposing the input into parallel segments.

Security-venue schematic on one equal-unit axis (every element aligns on
one grid). Panel a: one infeasible depth-l tree becomes c short parallel
trees. Panel b: three visual zones (equal prefix / decisive segment /
don't care) show the most-significant differing segment deciding the
comparison. Full text-width, two-column IEEEtran.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

INK    = "#1b232c"
SLATE  = "#5b6b7b"
FILL   = "#eef2f6"
BLUE   = "#2c5f8a"
BLUEF  = "#dce6f0"
GREEN  = "#2e8b57"
GREENF = "#dbeee2"
FADE   = "#aeb7c0"
RED    = "#b64342"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType -> crisp, portable vector text


def rbox(ax, x0, y0, x1, y1, fc, ec=INK, lw=1.2, r=0.8, z=2, ls="-", alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z,
        linestyle=ls, alpha=alpha))


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.6, ms=11, style="-|>", ls="-", z=5):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color=color, linestyle=ls, zorder=z,
        shrinkA=0, shrinkB=0))

def mini_tree(ax, cx, top, w=11.0, h=13.0, color=SLATE, leaf=BLUEF, lw=1.1):
    """Shallow depth-2 binary tree, root at (cx, top)."""
    my, ly = top - h * 0.5, top - h
    mids = [cx - w * 0.27, cx + w * 0.27]
    leaves = [cx - w * 0.5, cx - w * 0.16, cx + w * 0.16, cx + w * 0.5]
    for mx in mids:
        ax.plot([cx, mx], [top, my], color=color, lw=lw, zorder=2)
    for mx, (a, b) in zip(mids, [(0, 1), (2, 3)]):
        ax.plot([mx, leaves[a]], [my, ly], color=color, lw=lw, zorder=2)
        ax.plot([mx, leaves[b]], [my, ly], color=color, lw=lw, zorder=2)
    ax.add_patch(Circle((cx, top), 0.85, fc=color, ec=color, zorder=3))
    for mx in mids:
        ax.add_patch(Circle((mx, my), 0.7, fc="white", ec=color, lw=lw, zorder=3))
    for lx in leaves:
        ax.add_patch(Circle((lx, ly), 0.8, fc=leaf, ec=color, lw=lw, zorder=3))


# ============================================================================
fig = plt.figure(figsize=(7.16, 3.3))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 200)
ax.set_ylim(0, 100)
ax.axis("off")

# panel divider
ax.plot([110, 110], [6, 92], color="#d4dae0", lw=1.0, zorder=0)


# descriptive panel captions placed BELOW each panel, matching the
# subfigure-caption convention of top security venues (IEEE S&P / USENIX
# Security / CCS / NDSS use two-column IEEE layout with bottom sub-captions)
def panel_caption(xc, tag, title):
    ax.text(xc, 9.5, rf"$\mathbf{{{tag}}}$  {title}", fontsize=8.4,
            color=INK, ha="center", va="center")


panel_caption(55, "(a)", r"From one deep tree to $c$ parallel shallow trees")
panel_caption(167, "(b)", "One segment decides the comparison")

# ----- panel (a): one deep tree (every layer expands) -> c shallow trees ----
DT_CX = 25
ax.text(DT_CX, 90.8, r"single deep tree, depth $\ell=32$", ha="center",
        va="center", fontsize=8.0, color=INK)

# a binary tree that visibly doubles each layer: 1 -> 2 -> 4 -> 8 ...
HALF = 15.0
sp = 2 * HALF / 7
ys = [86.0, 79.6, 73.2, 66.8]
L3x = [DT_CX + (i - 3.5) * sp for i in range(8)]
L2x = [(L3x[2 * j] + L3x[2 * j + 1]) / 2 for j in range(4)]
L1x = [(L2x[2 * j] + L2x[2 * j + 1]) / 2 for j in range(2)]
tree = [[DT_CX], L1x, L2x, L3x]

# parent -> child edges
for k in range(3):
    for j, px in enumerate(tree[k]):
        for cx in (tree[k + 1][2 * j], tree[k + 1][2 * j + 1]):
            ax.plot([px, cx], [ys[k], ys[k + 1]], color=SLATE, lw=0.9,
                    zorder=2)
# dashed stubs: the doubling keeps going for all remaining layers
for cx in L3x:
    ax.plot([cx, cx], [ys[3] - 0.9, ys[3] - 4.4], color=RED, lw=0.8,
            ls=(0, (1, 1.4)), alpha=0.75, zorder=2)
# nodes: root, two interior layers, then the exploding leaf frontier
ax.add_patch(Circle((DT_CX, ys[0]), 1.15, fc=SLATE, ec=SLATE, zorder=3))
for k in (1, 2):
    for x in tree[k]:
        ax.add_patch(Circle((x, ys[k]), 0.85, fc="white", ec=SLATE, lw=1.0,
                            zorder=3))
for x in L3x:
    ax.add_patch(Circle((x, ys[3]), 0.7, fc="#f7dcdb", ec=RED, lw=0.9,
                        zorder=3))
ax.text(DT_CX, 60.5, r"$\vdots$", ha="center", va="center", fontsize=11,
        color=RED)
ax.text(DT_CX, 55.3, r"$2^{\ell}=2^{32}$ leaves", ha="center", va="center",
        fontsize=7.6, color=RED)

# infeasible cost badge
rbox(ax, DT_CX - 16, 20, DT_CX + 16, 30, "#f7e2e1", ec=RED, lw=1.1, r=1.0, z=3)
ax.text(DT_CX, 26.4, r"$\Theta(2^{\ell})$ RP calls", ha="center", va="center",
        fontsize=8, color=RED)
ax.text(DT_CX, 22.6, "infeasible", ha="center", va="center", fontsize=7.6,
        color=RED, fontweight="bold")

# decompose bridge (shorter span; label lifted clear of the arrow)
arrow(ax, 41.5, 64, 49.5, 64, color=INK, lw=2.0, ms=14)
ax.text(45.5, 72.2, "decompose", ha="center", va="center", fontsize=7.8,
        color=INK)
ax.text(45.5, 68.5, r"public $\hat{x}$", ha="center", va="center",
        fontsize=7.8, color=INK)

# AFTER: segmented bar of x-hat over c short parallel trees
tcx = [56, 69, 82, 95]
BAR_Y0, BAR_Y1 = 82, 89
cell_w = 12.2
seg_labels = [r"$\hat{x}^{(1)}$", r"$\hat{x}^{(2)}$",
              r"$\hat{x}^{(3)}$", r"$\hat{x}^{(4)}$"]
for x, lab in zip(tcx, seg_labels):
    rbox(ax, x - cell_w / 2, BAR_Y0, x + cell_w / 2, BAR_Y1, BLUEF,
         ec=BLUE, lw=1.1, r=0.8, z=3)
    ax.text(x, (BAR_Y0 + BAR_Y1) / 2, lab, ha="center", va="center",
            fontsize=8.3, color=INK)
    arrow(ax, x, BAR_Y0 - 0.3, x, 77.5, color=SLATE, lw=1.0, ms=7)
    mini_tree(ax, x, 76.5, w=11.4, h=13.0)

# parallel bracket
BR_Y = 58.5
ax.plot([tcx[0] - 6, tcx[0] - 6, tcx[-1] + 6, tcx[-1] + 6],
        [61.5, BR_Y, BR_Y, 61.5], color=GREEN, lw=1.3, zorder=4)
ax.text((tcx[0] + tcx[-1]) / 2, 55.8, r"$c$ short trees of depth $w=\ell/c$",
        ha="center", va="center", fontsize=8, color=GREEN)
ax.text((tcx[0] + tcx[-1]) / 2, 52.2, "run in parallel",
        ha="center", va="center", fontsize=8, color=GREEN)

# efficient cost badge (centered under the parallel-tree cluster, x=75.5)
rbox(ax, 60.5, 20, 90.5, 30, GREENF, ec=GREEN, lw=1.1, r=1.0, z=3)
ax.text(75.5, 26.4, r"$\mathcal{O}(c\,2^{\ell/c})$ calls", ha="center",
        va="center", fontsize=8, color=INK)
ax.text(75.5, 22.6, "feasible", ha="center", va="center", fontsize=7.6,
        color=GREEN, fontweight="bold")
arrow(ax, DT_CX + 16, 25, 59.5, 25, color=GREEN, lw=1.6, ms=11)

# ----- panel (b): three zones decide the comparison -------------------------
col_x = [144, 159, 174, 189]
cell_w = 12.6
lab_x = 133
xhat = ["3A", "07", "C1", "5F"]
rval = ["3A", "12", "40", "88"]
ecs = [SLATE, GREEN, FADE, FADE]
tcs = [INK, INK, FADE, FADE]

# formula header
ax.text(167, 90.3, r"$\mathbf{1}\{\hat{x}<r\}=\bigoplus_j (E_j \wedge L_j)$",
        ha="center", va="center", fontsize=9.2, color=INK)

# zones behind the cells
ZY0, ZY1 = 46, 82
rbox(ax, 135, ZY0, 151, ZY1, "#edf0f3", ec=SLATE, lw=1.0, r=1.0, z=1)
rbox(ax, 151.5, ZY0, 166.5, ZY1, GREENF, ec=GREEN, lw=1.5, r=1.0, z=1,
     ls=(0, (4, 2)))
rbox(ax, 167, ZY0, 197, ZY1, "#f3f5f7", ec=FADE, lw=1.0, r=1.0, z=1)
ax.text(143, 78.7, "equal", ha="center", va="center", fontsize=7.3, color=SLATE)
ax.text(159, 78.7, "decides", ha="center", va="center", fontsize=7.6,
        color=GREEN, fontweight="bold")
ax.text(182, 78.7, "don't care", ha="center", va="center", fontsize=7.3,
        color=FADE)

# hex cells for x-hat and r
ax.text(lab_x, 70, r"$\hat{x}^{(j)}$", ha="right", va="center", fontsize=8.6,
        color=INK)
ax.text(lab_x, 60, r"$r^{(j)}$", ha="right", va="center", fontsize=8.6,
        color=INK)
for x, hv, rv, ec, tc in zip(col_x, xhat, rval, ecs, tcs):
    for y, s in [(70, hv), (60, rv)]:
        rbox(ax, x - cell_w / 2, y - 3.6, x + cell_w / 2, y + 3.6, "white",
             ec=ec, lw=1.1, r=0.7, z=4)
        ax.text(x, y, s, ha="center", va="center", fontsize=8.4, color=tc,
                zorder=6)

# comparison glyph row
glyphs = [("=", SLATE, 9, "normal"), ("<", GREEN, 12, "bold"),
          (r"$\cdot$", FADE, 9, "normal"), (r"$\cdot$", FADE, 9, "normal")]
for x, (g, gc, gs, gw) in zip(col_x, glyphs):
    ax.text(x, 51, g, ha="center", va="center", fontsize=gs, color=gc,
            fontweight=gw, zorder=6)

# decisive segment -> result
arrow(ax, 159, 46, 159, 37.5, color=GREEN, lw=1.5, ms=11)
rbox(ax, 135, 18, 199, 36, GREENF, ec=GREEN, lw=1.4, r=1.2, z=3)
ax.text(167, 31.0, r"$\hat{x}^{(2)}<r^{(2)}\;\Rightarrow\;\hat{x}<r$",
        ha="center", va="center", fontsize=8.9, color=INK)
ax.text(167, 25.4, "the most-significant differing segment",
        ha="center", va="center", fontsize=7.2, color=GREEN)
ax.text(167, 21.4, r"alone settles the ordering of $\hat{x}$ and $r$",
        ha="center", va="center", fontsize=7.2, color=GREEN)

fig.savefig("fig_split.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("fig_split.png", dpi=1200, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("fig_split saved")


