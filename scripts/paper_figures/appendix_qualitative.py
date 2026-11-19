"""Paper appendix: qualitative-examples table (complete brief captions of four images by
all seven conditions, with CLIPScore). Captions and scores are read from
qualitative_captions.json (baseline CLIPScore recomputed with the same ViT-B/32 judge).

    python scripts/paper_figures/appendix_qualitative.py appendix.tex
"""
import json, sys, textwrap
caps = json.load(open(__file__.rsplit("/", 1)[0] + "/qualitative_captions.json"))
IMGW, NW, TW = 28, 15, 117
CONDS = ["C3_stage2", "C3pinr", "Cscale1500", "Crank20", "Corient", "Cloc", "Clocorient"]
NAME = {"C3_stage2": "baseline", "C3pinr": "constrained baseline", "Cscale1500": "Cscale1500", "Crank20": "Crank20", "Corient": "Corient", "Cloc": "Cloc", "Clocorient": "Clocorient"}
MARK = {
 "533536": {"C3_stage2": [("looking at the camera",0)], "C3pinr": [("facing the camera",0)], "Cscale1500": [("facing the camera",0)], "Crank20": [("facing the camera",0)], "Corient": [("facing the camera",0)], "Cloc": [("watching a television",1)], "Clocorient": [("looking directly at the camera",0)]},
 "402334": {"C3_stage2": [("a group of people gathered in a room",0)], "C3pinr": [("large, modern building",0), ("clock tower",1)], "Cscale1500": [("group of people gathered in a church",0)], "Crank20": [("large, modern, and well-lit building",0), ("clock tower",1)], "Corient": [("large, modern, and well-lit building",0), ("clock tower",1)], "Cloc": [("church with a tall steeple and a clock tower",1), ("surrounded by a large group of people",0)], "Clocorient": [("church with a tall steeple",1), ("surrounded by a large group of people",0)]},
 "38210": {"C3_stage2": [("snowboarder",0)], "C3pinr": [("a man and a dog",0)], "Cscale1500": [("holding a snowboard",0)], "Crank20": [("skier",1)], "Corient": [("a man and a dog",0)], "Cloc": [("man skiing down a snowy mountain slope",1)], "Clocorient": [("a man and a dog",0)]},
 "469174": {"C3_stage2": [("American flags",1)], "C3pinr": [("American flag",1)], "Cscale1500": [("holding a flag",1)], "Crank20": [("American flag",1)], "Corient": [("flag waving",1)], "Cloc": [("several airplanes flying in the sky",0)], "Clocorient": [("flag in the background",1)]},
}
ORDER = ["533536", "402334", "38210", "469174"]
SUB = f"@{{}}p{{{NW}mm}}|p{{{TW}mm}}|c@{{}}"
L = ["\\appendix",
     "\\newcommand{\\qwrong}[1]{\\textcolor[HTML]{C62828}{#1}}",
     "\\newcommand{\\qright}[1]{\\textcolor[HTML]{2E7D32}{\\textbf{#1}}}",
     "\\newcommand{\\qsep}{\\noalign{{\\color[gray]{0.8}\\hrule height 0.25pt}}}", "",
     "\\twocolumn[{%", "\\section{Qualitative Examples}", "\\label{app:qual}",
     "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
     "\\setlength{\\aboverulesep}{0.15ex}", "\\setlength{\\belowrulesep}{0.3ex}",
     "\\begin{tabular}{@{}p{%dmm}|@{\\hspace{7pt}}p{%dmm}@{}}" % (IMGW, NW+TW+13), "\\toprule",
     f"Image & \\begin{{tabular}}[t]{{{SUB}}}Condition & Generated description (brief set, full text) & CLIP\\end{{tabular}} \\\\",
     "\\midrule"]
for k, i in enumerate(ORDER):
    L.append(f"\\raisebox{{-\\height}}{{\\includegraphics[width=\\linewidth,height=34mm,keepaspectratio]{{qual_{i}.jpg}}}}")
    L += ["&", "\\renewcommand{\\arraystretch}{1.0}", f"\\begin{{tabular}}[t]{{{SUB}}}"]
    for j, c in enumerate(CONDS):
        t = caps[i][c]["cap"].replace("%", "\\%").replace("&", "\\&")
        for w, ok in MARK[i][c]:
            assert w in t, (i, c, w)
            t = t.replace(w, ("\\qright{%s}" if ok else "\\qwrong{%s}") % w, 1)
        v = caps[i][c]['clip']; best = max(caps[i][x]['clip'] for x in CONDS)
        sc = f"$\\mathbf{{{v:.3f}}}$" if v == best else f"${v:.3f}$"
        if not caps[i][c]["cap"].endswith((".", "!", '"')):
            t += "\\,\\dots"
        L.append(f"{NAME[c]} & {t} & {sc} \\\\")
        if j < len(CONDS) - 1:
            L.append("\\qsep")
    L.append("\\end{tabular} \\\\")
    if k < len(ORDER) - 1:
        L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}", "\\vspace{2pt}",
      "\\captionof{table}{Qualitative examples (brief captions, CLIPScore). Red: wrong; green: correct main object; \\dots: caption cut at 64 tokens.}",
      "\\label{tab:qual}", "}]"]
W = []
for l in L:
    W += textwrap.wrap(l, width=88, break_long_words=False, break_on_hyphens=False) if len(l) > 90 else [l]
open(sys.argv[1], "w").write("\n".join(W) + "\n")
print("max line", max(map(len, W)))
