"""Strip paper checkpoints to connector + LoRA weights (safetensors), under paper names.
Read-only on the source checkpoints; writes only into OUT."""
import json, os, sys, torch
from safetensors.torch import save_file
ROOT = os.path.expanduser("~/LLAMA_modality_gap/outputs")
OUT = os.path.expanduser("~/mllm-modality-gap-release/weights")
# paper name -> source checkpoint
RUNS = {
 "seed42/baseline": "stage2_vlm_C3.pt",
 "seed42/constrained_baseline": "stage2_vlm_C3pinr.pt",
 "seed42/Cloc": "stage2_vlm_Cloc.pt",
 "seed42/Cscale1500": "stage2_vlm_Cscale1500.pt",
 "seed42/Crank20": "stage2_vlm_Crank15.pt",
 "seed42/Corient": "stage2_vlm_Corient.pt",
 "seed42/Clocorient": "stage2_vlm_Clocorient.pt",
 "seed42/Cloc_48.9k_items": "stage2_vlm_Cloc_long.pt",
 "seed42/Cloc_80k_items": "stage2_vlm_Cloc_80k.pt",
 "seed1337/baseline": "stage2_vlm_C3_stage2_s1337.pt",
 "seed1337/constrained_baseline": "stage2_vlm_C3pinr_s1337.pt",
 "seed1337/Cloc": "stage2_vlm_Cloc_s1337.pt",
 "seed1337/Cscale1500": "stage2_vlm_Cscale1500_s1337.pt",
 "seed1337/Crank20": "stage2_vlm_Crank15_s1337.pt",
 "seed1337/Corient": "stage2_vlm_Corient_s1337.pt",
}
manifest = {}
for name, src in RUNS.items():
    d = torch.load(f"{ROOT}/checkpoints/{src}", map_location="cpu")
    conn = {k: v.contiguous() for k, v in d["connector"].items()}
    lora = {k: v.contiguous() for k, v in d["llm_trainable"].items()}
    assert sum(v.numel() for v in conn.values()) == 20979712, name
    assert sum(v.numel() for v in lora.values()) == 6291456, name
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    save_file(conn, f"{OUT}/{name}/connector.safetensors")
    save_file(lora, f"{OUT}/{name}/lora_adapter.safetensors")
    manifest[name] = {"source": src, "step": int(d.get("step", -1))}
    print(name, "step", d.get("step"), flush=True)
s1 = torch.load(f"{ROOT}/checkpoints/stage1_connector_C2.pt", map_location="cpu")
save_file({k: v.contiguous() for k, v in s1["state_dict"].items()}, f"{OUT}/stage1_connector.safetensors")
# constants of the location drive: text centroid mu_y and normaliser s_X (baseline image cloud trace)
Y = torch.load(f"{ROOT}/embeddings/projected_C3_stage2_text_pooled.pt", map_location="cpu").double()
X = torch.load(f"{ROOT}/embeddings/projected_C3_stage2_image_pooled.pt", map_location="cpu").double()
mu_y = Y.mean(0); s_X = float(((X - X.mean(0)) ** 2).sum(1).mean())
save_file({"mu_y": mu_y.float()}, f"{OUT}/location_drive_text_centroid.safetensors")
json.dump({"s_X": s_X, "mu_y_norm": float(mu_y.norm()), "mu_y_source": "mean of the 5000 val2017 caption representations"}, open(f"{OUT}/location_drive_constants.json", "w"), indent=2)
json.dump({"stage1_connector": {"source": "stage1_connector_C2.pt", "config": s1.get("config")}, **manifest}, open(f"{OUT}/manifest.json", "w"), indent=2)
print("s_X", round(s_X, 1), "||mu_y||", round(float(mu_y.norm()), 4))
