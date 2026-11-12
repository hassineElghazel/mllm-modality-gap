#!/bin/bash
#SBATCH --job-name=t1-clipdump
#SBATCH --partition=RTX
#SBATCH --gpus=2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/nfs/home/elghazel/LLAMA_modality_gap/outputs/logs/t1_clipdump_%j.out
#SBATCH --error=/nfs/home/elghazel/LLAMA_modality_gap/outputs/logs/t1_clipdump_%j.err
#
# Re-score CLIPScore for the reported conditions so that per-image records exist
# (outputs/metrics/clipscore_records_<cond>.json). The summary values are
# reproduced bit-for-bit (same captions, same judge, greedy scoring); the only
# new output is the per-image file that a paired bootstrap needs. The paper's
# z-statistics are currently unpaired on paired data; this is the fix.
#
# Budget: 14 condition sets x (5000 or 1300 or 300) images through CLIP B/32,
# a few minutes each.
set -Eeuo pipefail
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/metrics

CLIP_MODEL="${CLIP_MODEL:-/nfs/home/elghazel/clip_b32}"
BRIEF="${BRIEF:-C3pinr Cloc Cscale1500 Crank15 Corient Clocorient}"
DD256="${DD256:-C3pinr_dd256 Cloc_dd256 Cscale1500_dd256 Crank15_dd256 Corient_dd256 Clocorient_dd256}"
OOD="${OOD:-C3pinr_ood Cloc_ood}"

python -c "
import sys, torch
if not torch.cuda.is_available(): print('[preflight] no CUDA'); sys.exit(1)
x = torch.randn(2048,2048,device='cuda'); _=(x@x).sum().item(); torch.cuda.synchronize()
print('[preflight] GPU op OK on', torch.cuda.get_device_name(0))
" || { echo '[clipdump] GPU preflight FAILED' >&2; exit 1; }

score() {  # $1 = config, $2 = condition tag
  local f="outputs/predictions/captions_${2}.json"
  if [ ! -f "$f" ]; then echo "[skip] $2 (no $f)"; return 0; fi
  echo "==================== CLIP records  $2 ===================="
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
      --config "$1" --condition "$2" --clip-model "$CLIP_MODEL" \
    || echo "[clipdump] $2 FAILED (continuing)" >&2
}
for c in $BRIEF; do score configs/captioning.yaml       "$c"; done
for c in $DD256; do score configs/description_eval.yaml "$c"; done
for c in $OOD;   do score configs/ood_eval.yaml         "$c"; done

echo "[clipdump] complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[clipdump] records: outputs/metrics/clipscore_records_*.json"
