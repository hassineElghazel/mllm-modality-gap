"""Paper figure: brief-set and NoCaps CLIPScore difference from the control, with 95%
paired bootstrap intervals over images.

    python scripts/paper_figures/fig_robustness.py outputs/metrics fig_robust.png
"""
import json, sys
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

M, out = sys.argv[1:3]
sns.set_theme(style="white", context="paper")
mpl.rcParams.update({"text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}"})
NAVY, CORAL, INK, MUTED = "#3C5488", "#E64B35", "#222222", "#6b6b6b"

def rec(t):
    return {x["image_id"]: x["clipscore"] for x in json.load(open(f"{M}/clipscore_records_{t}.json"))}
def boot(d, n=10000, seed=0):
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), *np.percentile(bs, [2.5, 97.5])
CONDS = [("Cloc", "Location", "Cloc"), ("Corient", "Orientation", "Corient"),
         ("Crank15", "Shape", "Crank15"), ("Cscale1500", "Scale", "Cscale1500"),
         ("Clocorient", "Location + orientation", "Clocorient")]
SETS = [("", "COCO (5,000 images)", NAVY, -0.17), ("_ood", "NoCaps (300 images)", CORAL, 0.17)]

fig, ax = plt.subplots(figsize=(3.33, 2.55), dpi=600)
XL = (-0.05, 0.11)
ax.axvspan(XL[0], 0, color="#f6eeee", zorder=0, lw=0)
ax.axvspan(0, XL[1], color="#edf3ee", zorder=0, lw=0)
for x in np.arange(-0.04, 0.11, 0.02):
    ax.axvline(x, color="white", lw=0.8, zorder=0.5)
ax.axvline(0, color="#9a9a9a", lw=0.5, zorder=1)
ax.axhline(3.5, color="#bdbdbd", lw=0.5, ls=(0, (3, 2)), zorder=1)

for i, (c, prop, name) in enumerate(CONDS):
    for suf, lab, col, off in SETS:
        A, B = rec(c + suf), rec("C3pinr" + suf)
        d = np.array([A[k] - B[k] for k in sorted(set(A) & set(B))])
        m, lo, hi = boot(d)
        y = i + off
        ax.plot([lo, hi], [y, y], color=col, lw=3.0, alpha=0.28, solid_capstyle="round", zorder=2)
        ax.plot([m], [y], "o", ms=3.1, color=col, mec="white", mew=0.6, zorder=3)
        ax.text(hi + 0.003, y, f"${m:+.3f}$", ha="left", va="center", fontsize=5.6, color=col)

ax.set_yticks(range(len(CONDS)))
ax.set_yticklabels([("Loc. + orient." if p.startswith("Location +") else p) + f"\n({n})" for _, p, n in CONDS], fontsize=6.6, linespacing=1.05)
ax.set_ylim(4.45, -0.75); ax.set_xlim(*XL)
ax.set_xticks(np.arange(-0.04, 0.11, 0.04))
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"${v:+.2f}$" if abs(v) > 1e-9 else "$0$"))
ax.tick_params(axis="x", labelsize=6.5, length=0, pad=4, colors=INK)
ax.tick_params(axis="y", length=0, pad=6)
ax.set_xlabel("CLIPScore difference from the control", fontsize=7.5, labelpad=5, color=INK)
ax.text(XL[0] + 0.003, -0.55, "worse", fontsize=6.5, color="#a05050", va="center")
ax.text(XL[1] - 0.003, -0.55, "better", fontsize=6.5, color="#4f7f57", va="center", ha="right")
for s in ax.spines.values():
    s.set_visible(False)
h = [mpl.lines.Line2D([], [], color=col, marker="o", ms=4.2, mec="white", mew=0.6, lw=4.2, alpha=1,
                      label=lab) for _, lab, col, _ in SETS]
for hh, (_, _, col, _) in zip(h, SETS):
    hh.set_color(col)
leg = fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=6.8,
                 bbox_to_anchor=(0.5, 1.0), handlelength=1.8, handletextpad=0.4, columnspacing=1.4)
for line in leg.get_lines():
    line.set_linewidth(3.4); line.set_alpha(0.55)
fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.92))
fig.savefig(out, dpi=600, facecolor="white")
print("ok")
