"""Assemble the paper's captions, scores and geometry into a release folder under paper names.
Sources: cluster copy (CL) overlaid by the laptop outputs (LO); nothing in the sources is modified."""
import json, os, re, shutil, sys
CL, LO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
def norm(fn): return fn.replace("Crank15", "Crank20")
def index(root):
    idx = {}
    for sub in ("metrics", "predictions"):
        d = os.path.join(root, sub)
        for fn in os.listdir(d):
            p = os.path.join(d, fn)
            if os.path.isfile(p): idx[(sub, norm(fn))] = p
        if sub == "metrics" and os.path.isdir(os.path.join(d, "intervene")):
            for fn in os.listdir(os.path.join(d, "intervene")):
                idx[("intervene", fn)] = os.path.join(d, "intervene", fn)
    return idx
src = index(CL); src.update(index(LO))          # laptop wins where both exist
TAGS = {  # internal tag -> paper name
 "C3_stage2": "baseline", "C3pinr": "constrained_baseline", "Cloc": "Cloc", "Cscale1500": "Cscale1500",
 "Crank20": "Crank20", "Crank15": "Crank20", "Corient": "Corient", "Clocorient": "Clocorient",
 "Cloc_long": "Cloc_48.9k_items", "Cloc_80k": "Cloc_80k_items", "llava15": "LLaVA-1.5-7B",
 "Cloc_clip_native": "Cloc_target_CLIP_text_near_origin", "Cloc_clip": "Cloc_target_CLIP_text_at_gap_distance",
 "C5_lam0p1": "location_drive_CLS_lambda0.1", "C5_lam0p5": "location_drive_CLS_lambda0.5",
 "C5p_lam0p1": "location_drive_pooled_lambda0.1_no_constraint", "C3pin": "location_and_scale_held_no_drive",
 "C5bp_lam0p01": "location_drive_pooled_lambda0.01_scale_constrained",
 "C5bp_lam0p1": "location_drive_pooled_lambda0.1_scale_constrained",
 **{f"C4_lam0p{k}": f"contrastive_drive_CLS_lambda0.{k}_no_constraint" for k in "13579"},
 "C0_random": "random_connector", "C1_stage2": "stage2_only", "C2_stage1": "stage1_only",
 "llava-1.5-13b-hf": "LLaVA-1.5-13B", "bakLlava-v1-hf": "bakLLaVA",
}
SUF = {"": "brief", "_dd256": "long", "_ood": "nocaps"}
SUFTXT = {"": "", "_dd256": "_long", "_ood": "_nocaps", "_brief": "_brief", "_dd": "_long_preliminary"}
tok = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(sorted(map(re.escape, TAGS), key=len, reverse=True)) + r")((?:_s1337)?)((?:_dd256|_ood|_brief|_dd)?)(?![A-Za-z0-9])")
SCRUB = [(re.compile(r"/nfs/home/[^/\"]+/LLAMA_modality_gap/"), ""), (re.compile(r"/nfs/home/[^/\"]+/"), ""),
         (re.compile(r"/Users/[^\"]*?/LLAMA_modality_gap/"), ""), (re.compile(r"/Users/[^/\"]+/"), "")]
def clean(text):
    for rx, rep in SCRUB: text = rx.sub(rep, text)
    return tok.sub(lambda m: TAGS[m.group(1)] + ("_seed1337" if m.group(2) else "") + SUFTXT[m.group(3)], text)
copied = []
def put(key, dest):
    p = src.get(key)
    if p is None: return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    txt = clean(open(p, encoding="utf-8").read())
    if dest.endswith(".json"): json.loads(txt)        # stays valid JSON
    open(dest, "w", encoding="utf-8").write(txt); copied.append(dest); return True
MAIN = ["C3_stage2", "C3pinr", "Cloc", "Cscale1500", "Crank20", "Corient", "Clocorient", "Cloc_long", "Cloc_80k", "llava15"]
SEEDED = ["C3_stage2", "C3pinr", "Cloc", "Cscale1500", "Crank20", "Corient"]
KINDS = [("predictions", "captions_{}.json", "captions.json"), ("metrics", "clipscore_{}.json", "clipscore.json"),
         ("metrics", "clipscore_records_{}.json", "clipscore_per_image.json"), ("metrics", "siglip_{}.json", "siglip.json"),
         ("metrics", "captioning_{}.json", "reference_scores.json"), ("metrics", "chair_records_{}.json", "chair_per_image.json")]
def model(tag, seedsuf, prefix):
    name = TAGS[tag]
    for suf, ctype in SUF.items():
        for sub, pat, out in KINDS:
            put((sub, pat.format(tag + seedsuf + suf)), f"{OUT}/{prefix}{ctype}/{name}/{out}")
    put(("metrics", f"gap_{tag}{seedsuf}.json"), f"{OUT}/{prefix}geometry/{name}.json")
for t in MAIN: model(t, "", "seed42/")
for t in SEEDED: model(t, "_s1337", "seed1337/")
ABL = ["C5_lam0p1", "C5_lam0p5", "C5p_lam0p1", "C3pin", "C5bp_lam0p01", "C5bp_lam0p1"] + [f"C4_lam0p{k}" for k in "13579"] + ["Cloc_clip_native", "Cloc_clip"]
for t in ABL:
    for suf, ctype in SUF.items():
        put(("metrics", f"clipscore_{t}{suf}.json"), f"{OUT}/tests_of_the_objective/{TAGS[t]}/clipscore_{ctype}.json")
    put(("metrics", f"gap_{t}.json"), f"{OUT}/tests_of_the_objective/{TAGS[t]}/geometry.json")
for t in ["C0_random", "C1_stage2", "C2_stage1"]:
    put(("metrics", f"gap_{t}.json"), f"{OUT}/where_the_gap_comes_from/{TAGS[t]}.json")
put(("metrics", "gap_C3_stage2.json"), f"{OUT}/where_the_gap_comes_from/both_stages_baseline.json")
for t in ["llava15", "llava-1.5-13b-hf", "bakLlava-v1-hf"]:
    put(("metrics", f"gap_{t}.json"), f"{OUT}/released_models_gap/{TAGS[t]}.json")
for fn, dst in [("chair_summary_brief.json", "summaries/chair_brief.json"), ("chair_summary_dd256.json", "summaries/chair_long.json"),
                ("chair_summary_dd256_baseline.json", "summaries/chair_long_with_baseline.json"),
                ("chair_summary_dd256_llava15_check.json", "summaries/chair_long_with_LLaVA-1.5-7B.json"),
                ("serving_cost_C3pinr.json", "summaries/inference_cost_constrained_baseline.json"),
                ("serving_cost_Cloc.json", "summaries/inference_cost_Cloc.json")]:
    put(("metrics", fn), f"{OUT}/{dst}")
put(("intervene", "intervene_C3_stage2.json"), f"{OUT}/summaries/inference_time_shift_of_baseline.json")
print(len(copied), "files")
