#!/bin/bash
#SBATCH --job-name=tierb-seed
#SBATCH --partition=RTX
#SBATCH --gpus=2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=36:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/home/elghazel/LLAMA_modality_gap/outputs/logs/tierb_%x_%j.out
#SBATCH --error=/nfs/home/elghazel/LLAMA_modality_gap/outputs/logs/tierb_%x_%j.err
#
# Tier B: a second training seed for one single-axis condition, with the brief
# evaluation the paper reports (geometry, captions, SPICE/METEOR, CLIPScore).
#
#   COND=C3pinr  SEED=1337 sbatch --job-name=tb-C3pinr  scripts/submit_tierb_seed.sbatch
#   COND=Cloc    ...       Corient, Crank15 likewise
#
# The training flags below are the July recipe exactly, checked against the
# sidecars and logs of the seed-42 runs (pooled_86398/86422/86701/86463):
# location pin 10, scale pin 1.0, rank pin 1.0 (setpoint auto-captured at 23.31,
# or driven to 15 for Crank15), pooled-257 vector, 450 steps. Only --seed differs.
# Stage 1 is shared, so the pins reference the same setpoints as seed 42.
#
# Restartable: QoS is preemptable and the job is --requeue. Each stage is skipped
# when its output already exists, and a partial checkpoint (step < 450) resumes
# rather than restarting. Telegram: a message at every stage here, and the
# trainers themselves report every 50 steps (notify_every_steps in the config).
#
# Budget: ~20.5 h per condition, measured from the four seed-42 runs.

set -Eeuo pipefail
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ) restart=${SLURM_RESTART_COUNT:-0}"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.telegram_env 2>/dev/null || true   # TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID for src/utils/notify.py too

COND="${COND:?set COND to one of C3pinr Cloc Corient Crank15 Cscale1500 Clocorient C3_stage2}"
SEED="${SEED:-1337}"
TAG="${COND}_s${SEED}"
CK="stage2_vlm_${TAG}.pt"
CKPT="outputs/checkpoints/${CK}"
CLIP_MODEL="$HOME/clip_b32"
TRAIN_ATTEMPTS="${TRAIN_ATTEMPTS:-3}"

tg_notify() {
    [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
    curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=$1" > /dev/null || true
}
on_error() {
    tg_notify "❌ TIER B ${TAG} FAILED (job ${SLURM_JOB_ID}, line $1, exit $?). Log: outputs/logs/tierb_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
}
trap 'on_error ${LINENO}' ERR
die() { echo "[tierb] FATAL: $1" >&2; tg_notify "❌ TIER B ${TAG:-?} stopped (job ${SLURM_JOB_ID}): $1"; exit 1; }

case "$COND" in
  C3pinr)  TRAIN=(scripts/18_train_c6_orientation_pinned.py --pool all257 --lambda 0.0 --lambda-p 10 --lambda-s 1.0 --lambda-rank 1.0) ;;
  Crank15) TRAIN=(scripts/18_train_c6_orientation_pinned.py --pool all257 --lambda 0.0 --lambda-p 10 --lambda-s 1.0 --lambda-rank 1.0 --effrank0 15) ;;
  Corient) TRAIN=(scripts/18_train_c6_orientation_pinned.py --pool all257 --lambda 0.5 --lambda-p 10 --lambda-s 1.0 --lambda-rank 1.0) ;;
  Cloc)    TRAIN=(scripts/17_train_c5_distance.py          --pool all257 --lambda-d 0.1 --lambda-s 1.0 --lambda-rank 1.0) ;;
  # Flags below copied from the seed-42 launch lines in submit_pooled.sbatch and
  # checked against logs pooled_86397 (Cscale1500) and pooled_86991 (Clocorient).
  Cscale1500) TRAIN=(scripts/18_train_c6_orientation_pinned.py --pool all257 --lambda 0.0 --lambda-p 10 --lambda-s 1.0 --btrace0 1500 --lambda-rank 1.0) ;;
  Clocorient) TRAIN=(scripts/18_train_c6_orientation_pinned.py --pool all257 --lambda 0.5 --close-location --lambda-p 0.1 --lambda-s 1.0 --lambda-rank 1.0) ;;
  # Baseline (standard Stage 2, no geometry term), as run_condition.py ran C3.
  C3_stage2)  TRAIN=(scripts/06_train_stage2.py --init-connector outputs/checkpoints/stage1_connector_C2.pt) ;;
  *) die "unknown COND=$COND" ;;
esac

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints outputs/predictions outputs/metrics outputs/embeddings

# --- preflight: the same inputs the seed-42 runs needed ---
python -c "
import sys, torch
if not torch.cuda.is_available(): print('[preflight] no CUDA'); sys.exit(1)
x = torch.randn(2048,2048,device='cuda'); _=(x@x).sum().item(); torch.cuda.synchronize()
print('[preflight] GPU op OK on', torch.cuda.get_device_name(0))
" || die "GPU preflight failed on $(hostname)"
for f in outputs/checkpoints/stage1_connector_C2.pt \
         outputs/embeddings/projected_C3_stage2_text_pooled.pt \
         outputs/embeddings/projected_C3_stage2_image_pooled.pt \
         data/coco/annotations/captions_train2017.json; do
  test -f "$f" || die "$f missing"
done
grep -qE "\"${TAG}\"" scripts/07_extract_projected.py || die "${TAG} not in the 07 registry"

ckpt_step() {  # prints the step stored in the checkpoint, 0 if absent/unreadable
  [ -f "$CKPT" ] || { echo 0; return; }
  python -c "import torch,sys; print(int(torch.load(sys.argv[1], map_location='cpu').get('step', 0)))" "$CKPT" 2>/dev/null || echo 0
}

tg_notify "🚀 TIER B ${TAG} started (job ${SLURM_JOB_ID}, restart ${SLURM_RESTART_COUNT:-0}): train 450 steps, then gap, brief captions, SPICE/METEOR, CLIPScore."

# --- 1. train (resume a partial checkpoint, skip a finished one) ---
step=$(ckpt_step)
if [ "$step" -ge 450 ]; then
  echo "[tierb] ${TAG}: checkpoint already at step ${step}, skipping training"
else
  for attempt in $(seq 1 "$TRAIN_ATTEMPTS"); do
    step=$(ckpt_step)
    resume=()
    if [ "$step" -gt 0 ]; then
      resume=(--resume "$CKPT")
      tg_notify "↻ TIER B ${TAG}: resuming from step ${step} (attempt ${attempt})"
    fi
    echo "[tierb] ${TAG}: train attempt ${attempt}/${TRAIN_ATTEMPTS} from step ${step}"
    if python "${TRAIN[@]}" --seed "$SEED" --output-name "$CK" --max-steps 450 ${resume[@]+"${resume[@]}"}; then
      break
    fi
    if [ "$attempt" -eq "$TRAIN_ATTEMPTS" ]; then
      die "training failed ${TRAIN_ATTEMPTS} times, last checkpoint at step $(ckpt_step)"
    fi
    sleep 60
  done
  step=$(ckpt_step)
  [ "$step" -ge 450 ] || die "training ended at step ${step} < 450"
  tg_notify "✅ TIER B ${TAG}: training done (step ${step}). Measuring geometry."
fi

# --- 2. geometry ---
if [ -f "outputs/metrics/gap_${TAG}.json" ]; then echo "[skip] gap_${TAG}.json exists"
else
  python scripts/07_extract_projected.py --condition "$TAG" --connector-override "$CKPT"
  python scripts/03_compute_gap.py --condition "$TAG"
fi

# --- 3. brief captions + reference-based scores ---
if [ -f "outputs/metrics/captioning_${TAG}.json" ]; then echo "[skip] captioning_${TAG}.json exists"
else
  tg_notify "📝 TIER B ${TAG}: captioning 5000 brief images"
  python scripts/08_run_captioning.py --condition "$TAG" --vlm-checkpoint "$CKPT"
  python scripts/09_score_captions.py --condition "$TAG"
fi

# --- 4. CLIPScore, with per-image records for the paired comparison ---
if [ -f "outputs/metrics/clipscore_records_${TAG}.json" ]; then echo "[skip] clipscore records exist"
else
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
      --condition "$TAG" --clip-model "$CLIP_MODEL"
fi

SUMMARY=$(python - "$TAG" "$COND" <<'PY'
import json, sys
tag, cond = sys.argv[1], sys.argv[2]
def g(f, k):
    try: return json.load(open(f"outputs/metrics/{f}"))[k]
    except Exception: return None
new = g(f"clipscore_{tag}.json", "CLIPScore"); old = g(f"clipscore_{cond}.json", "CLIPScore")
gm = (json.load(open(f"outputs/metrics/gap_{tag}.json")).get("spec_metrics") or {}).get("G_mu")
fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
print(f"CLIPScore seed1337={fmt(new)} vs seed42={fmt(old)}  G_mu={fmt(gm)}")
PY
)
echo "[tierb] ${TAG}: ${SUMMARY}"
tg_notify "🏁 TIER B ${TAG} complete: ${SUMMARY}"
echo "[tierb] ${TAG} complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
