#!/bin/bash
#SBATCH --job-name=dd256-baseline
#SBATCH --partition=RTX
#SBATCH --gpus=2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/home/elghazel/LLAMA_modality_gap/outputs/logs/dd256_baseline_%j.out
#SBATCH --error=/nfs/home/elghazel/LLAMA_modality_gap/outputs/logs/dd256_baseline_%j.err
#
# Completes the baseline row (standard fine-tuning, stage2_vlm_C3.pt) of the main
# table: detailed-description (dd256) captions on the 1,300-image split, CLIPScore,
# CHAIR, and SigLIP for both the brief and the dd256 captions.
#
#   sbatch scripts/submit_dd256_baseline.sbatch
#
# CHAIR goes to a NEW file, chair_summary_dd256_baseline.json, so the existing
# chair_summary_dd256.json is never touched. Every step is skipped when its output
# already exists, so a requeue resumes where it stopped.

set -Eeuo pipefail
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ) restart=${SLURM_RESTART_COUNT:-0}"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)
source ~/.telegram_env 2>/dev/null || true

CFG=configs/description_eval.yaml
TAG=C3_stage2_dd256
CKPT=outputs/checkpoints/stage2_vlm_C3.pt
CLIP_MODEL="$HOME/clip_b32"
OUT=outputs/metrics/chair_summary_dd256_baseline.json

tg() { [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
       curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
         --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1" >/dev/null || true; }
trap 'tg "❌ dd256 baseline FAILED (job ${SLURM_JOB_ID}, line ${LINENO}). Log: outputs/logs/dd256_baseline_${SLURM_JOB_ID}.err"' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/metrics outputs/predictions
[[ -f "$CKPT" ]] || { echo "[dd256] $CKPT not found" >&2; exit 3; }
tg "🚀 dd256 baseline started (job ${SLURM_JOB_ID}): 1300 detailed captions, CLIPScore, CHAIR, SigLIP"

# --- 1. detailed captions (1,300 images, 256 new tokens) ---
if [[ -f "outputs/predictions/captions_${TAG}.json" ]]; then echo "[skip] captions_${TAG}.json exists"
else
  python scripts/08_run_captioning.py --config "$CFG" --condition "$TAG" --vlm-checkpoint "$CKPT"
fi

# --- 2. CLIPScore (ViT-B/32 judge, per-image records for paired intervals) ---
if [[ -f "outputs/metrics/clipscore_${TAG}.json" ]]; then echo "[skip] clipscore_${TAG}.json exists"
else
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
      --config "$CFG" --condition "$TAG" --clip-model "$CLIP_MODEL"
fi

# --- 3. CHAIR on dd256: baseline alongside the existing conditions, to a new file ---
KEEP=""
for t in C3_stage2_dd256 C3pinr_dd256 Cloc_dd256 Cscale1500_dd256 Crank15_dd256 Corient_dd256 Clocorient_dd256; do
  [[ -f "outputs/predictions/captions_${t}.json" ]] && KEEP="${KEEP} ${t}" || echo "[skip] ${t} (no predictions)"
done
python scripts/19_chair.py --conditions ${KEEP} --baseline C3pinr_dd256 --config "$CFG" --out "$OUT"

# --- 4. SigLIP for the baseline, brief and dd256 ---
for c in C3_stage2 C3_stage2_dd256; do
  if [[ -f "outputs/metrics/siglip_${c}.json" ]]; then echo "[skip] siglip_${c}.json exists"; continue; fi
  python scripts/15b_siglip_score.py --condition "$c" || echo "[siglip] $c FAILED (continuing)" >&2
done

SUMMARY=$(python - "$OUT" <<'PY'
import json, sys
M = "outputs/metrics/"
ch = json.load(open(sys.argv[1])).get("per_condition", {}).get("C3_stage2_dd256", {})
cs = json.load(open(M + "clipscore_C3_stage2_dd256.json"))["CLIPScore"]
f = lambda k: f"{ch[k]:.4f}" if k in ch else "n/a"
print(f"CLIPScore {cs:.4f}  CHAIR_s {f('chair_s')}  CHAIR_i {f('chair_i')}  recall {f('recall')}")
PY
)
echo "[dd256] baseline: ${SUMMARY}"
tg "🏁 dd256 baseline complete: ${SUMMARY}"
echo "[dd256] complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
