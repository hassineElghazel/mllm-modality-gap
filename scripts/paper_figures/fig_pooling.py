"""Paper figure: token pooling (a) and the distance to the text centroid (b), drawn to
scale for the control (G_mu = 177.86 from outputs/metrics/gap_C3pinr.json).

    python scripts/paper_figures/fig_pooling.py fig_pooling.png

Needs a LaTeX installation with libertine and newtx (text.usetex).
"""
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

mpl.rcParams.update({"text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage{libertine}\usepackage[libertine]{newtxmath}"})

TOK, TOK_F = "#1f77b4", "#ddeaf4"
POOL, POOL_F = "#2a9d8f", "#cdebe7"
ZBAR, RED = "#7b3fa0", "#d62728"
FS, FSF = 7, 6.3
G_CTRL = 177.86
W, H, TOP = 1970, 1500, 150
TOT = H + TOP                        # 3.33 in wide
fig = plt.figure(figsize=(3.33, 3.33 * TOT / W), dpi=600)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, -TOP); ax.set_aspect("equal"); ax.axis("off")
T = lambda x, y, s, **k: ax.text(x, y, s, ha=k.pop("ha", "center"), va=k.pop("va", "center"),
                                 fontsize=k.pop("fs", FS), **k)
def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6.5, color="black",
                                 lw=0.5, shrinkA=0, shrinkB=0))

ROWS, Y_DOTS, YM = [(205, "1"), (335, "2"), (550, "B")], 445, 378
S0 = 48
# ---- top: tokens --------------------------------------------------------------------
for y, lab in ROWS:
    T(55, y, rf"$b{{=}}{lab}$")
    for x in (140, 202, 322):
        ax.add_patch(Rectangle((x, y - S0 / 2), S0, S0, facecolor=TOK_F, edgecolor=TOK, lw=0.6))
    ax.scatter([270, 285, 300], [y] * 3, s=0.6, color="black", lw=0)
T(55, Y_DOTS, r"$\vdots$"); T(255, Y_DOTS, r"$\vdots$")
T(255, 45, r"tokens $u_{b,t}$"); T(255, 102, r"$t{=}1,\dots,257$")

arrow((430, YM), (930, YM)); T(680, YM - 50, r"mean over $t$")
f1 = T(680, YM + 30, r"$z_b=\frac{1}{257}\sum_{t} u_{b,t}$", va="top", fs=FSF)
XP = 1010
for y, lab in ROWS:
    ax.add_patch(Circle((XP, y), 30, facecolor=POOL_F, edgecolor=POOL, lw=0.6))
    T(XP + 50, y, rf"$z_{{{lab}}}$", ha="left")
T(XP, Y_DOTS, r"$\vdots$")
T(XP, 45, "pooled"); T(XP, 102, "vectors")

arrow((1160, YM), (1700, YM)); T(1430, YM - 50, r"mean over $b$")
f2 = T(1430, YM + 30, r"$\bar z=\frac{1}{B}\sum_{b} z_b$", va="top", fs=FSF)
XM = 1800
ax.add_patch(Circle((XM, YM), 24, facecolor=ZBAR, edgecolor="none"))
T(XM, YM + 58, r"$\bar z$")
T(XM, 45, "batch"); T(XM, 102, "mean")

T(8, -85, r"\textbf{(a)} Token pooling", ha="left", fs=7.5)
T(8, 800, r"\textbf{(b)} Distance to the text centroid", ha="left", fs=7.5)
# ---- rule ---------------------------------------------------------------------------
ax.plot([30, W - 30], [715, 715], color="#bdbdbd", lw=0.5)

# ---- bottom: distance to mu_y, to scale -------------------------------------------------
L, R, B_, T_ = 90, W - 60, 1095, 900
ia = fig.add_axes([L / W, (H - B_) / TOT, (R - L) / W, (B_ - T_) / TOT])
ia.set_xlim(195, -10); ia.set_ylim(-1, 1)
ia.set_xticks([150, 100, 50, 0]); ia.set_yticks([-0.5, 0, 0.5])
ia.grid(True, color="#e4e4e4", lw=0.4); ia.set_axisbelow(True)
ia.tick_params(axis="x", labelsize=6, length=1.5, width=0.4, pad=1.5, colors="#444444")
ia.tick_params(axis="y", labelleft=False, length=0)
for sp in ia.spines.values():
    sp.set_linewidth(0.4); sp.set_color("#9a9a9a")
ia.set_xlabel(r"distance to $\mu_y$", fontsize=6.5, labelpad=1.5, color="#444444")
ia.annotate("", xy=(0, 0), xytext=(G_CTRL, 0),
            arrowprops=dict(arrowstyle="<|-|>", color="black", lw=0.5, mutation_scale=5.5,
                            shrinkA=3.5, shrinkB=3.5))
ia.plot([G_CTRL], [0], "o", ms=4.2, color=ZBAR, mec="none", zorder=4)
ia.plot([0], [0], "o", ms=4.2, color=RED, mec="none", zorder=4)
ia.text(G_CTRL, -0.28, r"$\bar z$", ha="center", va="top", fontsize=FS)
ia.text(0, -0.28, r"$\mu_y$", ha="center", va="top", fontsize=FS, color=RED)
ia.text(G_CTRL / 2, 0.14, r"$\lVert\bar z-\mu_y\rVert$", ha="center", va="bottom", fontsize=FS,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.7))

# ---- legend -------------------------------------------------------------------------------
mk = lambda m, ms, fc, ec, lab: Line2D([], [], ls="", marker=m, ms=ms, mfc=fc, mec=ec, mew=0.5, label=lab)
H_ = [mk("s", 4, TOK_F, TOK, r"$u_{b,t}$: token $t$ of image $b$"),
      mk("o", 3.6, ZBAR, ZBAR, r"$\bar z$: mean over the batch of $B$ images"),
      mk("o", 4, POOL_F, POOL, r"$z_b$: pooled vector of image $b$"),
      mk("o", 3.6, RED, RED, r"$\mu_y$: text centroid (mean text embedding)")]
fig.legend(handles=H_, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
           fontsize=6.3, handletextpad=0.3, columnspacing=1.2, labelspacing=0.7, borderaxespad=0.2)

fig.canvas.draw()
inv = ax.transData.inverted()
for name, t in (("f1", f1), ("f2", f2)):
    bb = t.get_window_extent(); (x0, _), (x1, _) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    print(name, round(x0), round(x1))
from matplotlib.transforms import Bbox
fw, fh = fig.get_size_inches(); m = 0.14          # white margin on all four sides, inches
fig.savefig(sys.argv[1], dpi=600, facecolor="white", bbox_inches=Bbox([[-m, -m], [fw + m, fh + m]]))
print("ok")
