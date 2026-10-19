#!/usr/bin/env bash
# Phase 2 rental session on one A100 80 GB (about 15 GPU-hours end to end). Resumable: every step
# is skipped when its output already exists, so a dropped SSH session is restarted with the same
# command. Run inside tmux, venv active, from the repo root, on a clean tree:
#   bash scripts/phase2_rental.sh 2>&1 | tee outputs/phase2_rental.log
# Steps (in order, each one usable on its own with  bash scripts/phase2_rental.sh <step>):
#   data      regenerate the perturbed GSM8K and the SFT set; both hashes must match the manifests
#   exp002    deferred Phase 1 sampling: base model, n = 64 (about 3 h)
#   sweep     EXP-005: three LR points on the 10 % subset + dev evals, then pick_lr.py
#   main      EXP-006: full fine-tune with the chosen LR (3-5 h) + the three eval templates
#   sample    EXP-006: n = 64 sampling of the SFT checkpoint (about 2 h)
#   lora      EXP-007: LoRA r = 64 + think-tag eval
#   scaling   EXP-008: 25 % and 50 % runs + think-tag evals + the figure
#   report    paired comparisons, pass@k figure, data-scaling figure, model card, uploads
#   speed     (not in `all`) on a cheaper card: run the first sweep run, measure tokens/s, project the
#             main run; stay if it projects to at most RLVR_MAX_MAIN_HOURS (8), else move to the A100
# RLVR_TRAIN_EXTRA adds --set overrides to every rlvr-sft call, e.g. on a 48 GB card:
#   RLVR_TRAIN_EXTRA="--set per_device_batch=2 --set grad_accum=32" bash scripts/phase2_rental.sh
set -euo pipefail
cd "$(dirname "$0")/.."
STEP="${1:-all}"
LR_FILE="outputs/train/chosen_lr.txt"
TRAIN_EXTRA="${RLVR_TRAIN_EXTRA:-}"   # unquoted below on purpose: word-split into --set pairs

run_if_missing() {  # run_if_missing <marker path> <command...>
  local marker="$1"; shift
  if [[ -e "$marker" ]]; then echo "== skip ($marker exists): $*"; return 0; fi
  echo "== run: $*"; "$@"
}

step_data() {
  run_if_missing outputs/data/gsm8k_perturbed.jsonl rlvr-perturb-gsm8k --seed 0
  python - <<'EOF'
import json; m = json.load(open("data/manifests/gsm8k_perturbed.json")); print("perturbed manifest hash:", m["content_hash"])
EOF
  run_if_missing outputs/data/sft/train.jsonl env HF_HUB_DISABLE_XET=1 rlvr-build-sft
  python - <<'EOF'
import json
from pathlib import Path
from rlvr.data.sft_data import read_jsonl
from rlvr.train.sft import check_manifest
rows = read_jsonl(Path("outputs/data/sft/train.jsonl"))
m = check_manifest(rows, "data/manifests/sft_openr1.json")
print("SFT data OK:", m["num_examples"], "examples, hash", m["content_hash"])
EOF
}

step_exp002() {
  run_if_missing outputs/samples/base_1.7b_n64_few_shot_cot/metrics.json \
    rlvr-eval --config configs/sample/base_1.7b_n64.yaml
  run_if_missing outputs/samples/base_1.7b_n64_countdown/metrics.json \
    rlvr-eval --config configs/sample/base_1.7b_n64_countdown.yaml
  run_if_missing outputs/eval/base_1.7b_think_tag_4096/metrics.json \
    rlvr-eval --config configs/eval/base_1.7b_think_tag_4096.yaml
}

step_sweep() {
  for lr in 5e-6 1e-5 2e-5; do
    run_if_missing "outputs/train/sft_1p7b_lr${lr}/final/config.json" \
      rlvr-sft --config "configs/train/sft_1p7b_lr${lr}.yaml" $TRAIN_EXTRA
    run_if_missing "outputs/eval/sft_1p7b_lr${lr}_dev/metrics.json" \
      rlvr-eval --config configs/eval/ckpt_think_tag_dev.yaml \
        --model "outputs/train/sft_1p7b_lr${lr}/final" --out "outputs/eval/sft_1p7b_lr${lr}_dev"
  done
  python scripts/pick_lr.py 5e-6=outputs/eval/sft_1p7b_lr5e-6_dev 1e-5=outputs/eval/sft_1p7b_lr1e-5_dev \
    2e-5=outputs/eval/sft_1p7b_lr2e-5_dev --train-root outputs/train | tee outputs/train/lr_sweep.md
  grep '^winner:' outputs/train/lr_sweep.md | awk '{print $2}' > "$LR_FILE"
  echo "chosen LR: $(cat "$LR_FILE")"
}

step_speed() {  # the lr5e-6 sweep run doubles as the speed check; step_sweep reuses it
  run_if_missing outputs/train/sft_1p7b_lr5e-6/final/config.json \
    rlvr-sft --config configs/train/sft_1p7b_lr5e-6.yaml $TRAIN_EXTRA
  python scripts/project_run_time.py outputs/train/sft_1p7b_lr5e-6/run_info.json \
    --main-epochs 2 --max-hours "${RLVR_MAX_MAIN_HOURS:-8}" --rate "${RLVR_RATE:-102}" --currency INR \
    | tee outputs/train/speed_check.md
}

chosen_lr() { if [[ -f "$LR_FILE" ]]; then cat "$LR_FILE"; else echo "1e-5"; fi; }

step_main() {
  local lr; lr="$(chosen_lr)"
  run_if_missing outputs/train/sft_1p7b/final/config.json \
    rlvr-sft --config configs/train/sft_1p7b.yaml --set "learning_rate=${lr}" $TRAIN_EXTRA
  for t in think_tag few_shot_cot zero_shot; do
    run_if_missing "outputs/eval/sft_1.7b_${t}/metrics.json" rlvr-eval --config "configs/eval/sft_1.7b_${t}.yaml"
  done
  run_if_missing outputs/eval/sft_1.7b_think_tag/perturbation_gap.json \
    python scripts/perturbation_gap.py outputs/eval/sft_1.7b_think_tag
  run_if_missing outputs/probe/partial_prompt_sft_1.7b/metrics.json \
    rlvr-probe-partial --config configs/probe/partial_prompt_1.7b.yaml \
      --model outputs/train/sft_1p7b/final --out outputs/probe/partial_prompt_sft_1.7b
}

step_sample() {
  run_if_missing outputs/samples/sft_1.7b_n64_few_shot_cot/metrics.json rlvr-eval --config configs/sample/sft_1.7b_n64.yaml
  run_if_missing outputs/samples/sft_1.7b_n64_countdown/metrics.json rlvr-eval --config configs/sample/sft_1.7b_n64_countdown.yaml
}

step_lora() {
  run_if_missing outputs/train/sft_1p7b_lora/final/config.json rlvr-sft --config configs/train/sft_1p7b_lora.yaml $TRAIN_EXTRA
  run_if_missing outputs/eval/sft_1.7b_lora_think_tag/metrics.json \
    rlvr-eval --config configs/eval/sft_1.7b_think_tag.yaml --model outputs/train/sft_1p7b_lora/final --out outputs/eval/sft_1.7b_lora_think_tag
}

step_scaling() {
  local lr; lr="$(chosen_lr)"
  for frac in 25 50; do
    run_if_missing "outputs/train/sft_1p7b_data${frac}/final/config.json" \
      rlvr-sft --config "configs/train/sft_1p7b_data${frac}.yaml" --set "learning_rate=${lr}" $TRAIN_EXTRA
    run_if_missing "outputs/eval/sft_1.7b_data${frac}_think_tag/metrics.json" \
      rlvr-eval --config configs/eval/sft_1.7b_think_tag.yaml \
        --model "outputs/train/sft_1p7b_data${frac}/final" --out "outputs/eval/sft_1.7b_data${frac}_think_tag"
  done
}

step_report() {
  mkdir -p outputs/reports/figures
  python scripts/compare_checkpoints.py --ref outputs/eval/base_1.7b_think_tag_4096 --new outputs/eval/sft_1.7b_think_tag \
    --label-ref base --label-new sft --out outputs/reports/phase-2-compare-think_tag.md
  python scripts/compare_checkpoints.py --ref outputs/eval/base_1.7b_few_shot_cot --new outputs/eval/sft_1.7b_few_shot_cot \
    --label-ref base --label-new sft --out outputs/reports/phase-2-compare-few_shot_cot.md || true
  python scripts/compare_checkpoints.py --ref outputs/eval/base_1.7b_zero_shot --new outputs/eval/sft_1.7b_zero_shot \
    --label-ref base --label-new sft --out outputs/reports/phase-2-compare-zero_shot.md || true
  if [[ -e outputs/eval/sft_1.7b_lora_think_tag/metrics.json ]]; then
    python scripts/compare_checkpoints.py --ref outputs/eval/sft_1.7b_lora_think_tag --new outputs/eval/sft_1.7b_think_tag \
      --label-ref lora --label-new full --out outputs/reports/phase-2-compare-lora.md
  fi
  python scripts/plot_pass_at_k.py outputs/samples/base_1.7b_n64_few_shot_cot outputs/samples/base_1.7b_n64_countdown \
    --greedy outputs/eval/base_1.7b_few_shot_cot --out outputs/reports/figures/phase-1-pass-at-k.png
  python scripts/perturbation_gap.py outputs/samples/base_1.7b_n64_few_shot_cot --all-samples \
    --out outputs/reports/phase-1-perturbation-gap-n64.json
  if [[ -e outputs/samples/sft_1.7b_n64_few_shot_cot/metrics.json ]]; then
    python scripts/plot_pass_at_k.py outputs/samples/sft_1.7b_n64_few_shot_cot outputs/samples/sft_1.7b_n64_countdown \
      --greedy outputs/eval/sft_1.7b_few_shot_cot --out outputs/reports/figures/phase-2-pass-at-k-sft.png
    python scripts/perturbation_gap.py outputs/samples/sft_1.7b_n64_few_shot_cot --all-samples \
      --out outputs/reports/phase-2-perturbation-gap-sft-n64.json
    python scripts/compare_checkpoints.py --ref outputs/samples/base_1.7b_n64_few_shot_cot \
      --new outputs/samples/sft_1.7b_n64_few_shot_cot --all-samples --label-ref base --label-new sft \
      --out outputs/reports/phase-2-compare-pass_at_k.md
  fi
  if [[ -e outputs/eval/sft_1.7b_data50_think_tag/metrics.json ]]; then
    python scripts/plot_data_scaling.py 0.25=outputs/eval/sft_1.7b_data25_think_tag 0.5=outputs/eval/sft_1.7b_data50_think_tag \
      1.0=outputs/eval/sft_1.7b_think_tag --base outputs/eval/base_1.7b_think_tag_4096 \
      --out outputs/reports/figures/phase-2-data-scaling.png
  fi
  python scripts/push_to_hub.py model --run outputs/train/sft_1p7b --repo "${HF_REPO_SFT:-Raviteja546/reasoning-rlvr-sft}" \
    --evals outputs/eval/sft_1.7b_think_tag outputs/eval/sft_1.7b_few_shot_cot outputs/eval/sft_1.7b_zero_shot \
    --compare outputs/reports/phase-2-compare-think_tag.md --card-out cards/sft.md ${HF_DRY_RUN:+--dry-run}
  for s in base_1.7b_n64_few_shot_cot base_1.7b_n64_countdown sft_1.7b_n64_few_shot_cot sft_1.7b_n64_countdown; do
    [[ -e "outputs/samples/$s/metrics.json" ]] && python scripts/push_to_hub.py samples --run "outputs/samples/$s" \
      --repo "${HF_REPO_SAMPLES:-Raviteja546/reasoning-rlvr-samples}" --path-in-repo "$s" ${HF_DRY_RUN:+--dry-run}
  done
  echo "== report step done; copy outputs/reports, outputs/eval/*/metrics.json, report.md, run_info.json, outputs/train/*/run_info.json and train_log.jsonl back to the laptop"
}

case "$STEP" in
  all) step_data; step_exp002; step_sweep; step_main; step_sample; step_lora; step_scaling; step_report ;;
  data|exp002|sweep|main|sample|lora|scaling|report|speed) "step_$STEP" ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
echo "== phase2_rental.sh $STEP finished"
