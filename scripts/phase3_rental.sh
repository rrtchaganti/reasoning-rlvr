#!/usr/bin/env bash
# Phase 3 rental session (rejection-sampling fine-tuning), run on the same node right after Phase 2.
# Resumable like phase2_rental.sh: a step is skipped when its output exists. Run inside tmux, venv
# active, from the repo root, on a clean tree, with outputs/train/sft_1p7b/final present:
#   bash scripts/phase3_rental.sh 2>&1 | tee outputs/phase3_rental.log
# Steps (in order; each one usable on its own with  bash scripts/phase3_rental.sh <step>):
#   smoke     file hand-offs only: 20 problems, n = 2 at a 2,048 cap (the SFT model writes 1,900+ tokens;
#             512 gave no correct sample on the H100, 19 Oct 2026), 3 training steps into outputs/*_smoke (not in `all`)
#   sample    EXP-009: the SFT checkpoint samples its training prompts, n = 8, T = 1.0 (1.5-3 h)
#   data      EXP-009: solve rates, the kept set, the learnable-prompt file, decontamination (CPU)
#   train     EXP-010: one epoch from sft on the kept set (0.5-1 h)
#   eval      EXP-010: the three eval templates, the perturbation gap, the partial-prompt probe
#   sample64  EXP-010: n = 64 samples of the RFT checkpoint (four sets; Countdown separately, see below)
#   control   EXP-010b: the compute-matched SFT control (one more SFT epoch on as many fresh examples) + its think-tag eval
#   report    paired tables rft-vs-sft and rft-vs-base, paired pass@k, figures, model card, uploads
# Decisions the node session takes by rule (docs/e2e-session-handoff.md, Phase 3 section):
#   RLVR_RFT_MAX_TOKENS   sampling cap; default = 2048 when the SFT think-tag eval's p90 length is under
#                         2,048 on both GSM8K and MATH-500, else 4096 (docs/phase-3-plan.md, decision 2)
#   RLVR_SKIP_COUNTDOWN64=1  skip the Countdown half of sample64 (the cost rule of the handoff, section 1)
#   RLVR_TRAIN_EXTRA      --set overrides for every rlvr-sft call (E2E: "--set packing=false")
set -euo pipefail
cd "$(dirname "$0")/.."
STEP="${1:-all}"
TRAIN_EXTRA="${RLVR_TRAIN_EXTRA:-}"   # unquoted below on purpose: word-split into --set pairs
SFT=outputs/train/sft_1p7b/final
RFT=outputs/train/rft_1p7b/final

run_if_missing() {  # run_if_missing <marker path> <command...>
  local marker="$1"; shift
  if [[ -e "$marker" ]]; then echo "== skip ($marker exists): $*"; return 0; fi
  echo "== run: $*"; "$@"
}

rft_max_tokens() {
  if [[ -n "${RLVR_RFT_MAX_TOKENS:-}" ]]; then echo "$RLVR_RFT_MAX_TOKENS"; return; fi
  python - <<'EOF'
import json
try:
    m = json.load(open("outputs/eval/sft_1.7b_think_tag/metrics.json"))
    p90 = max(m[e]["tokens"]["p90"] for e in ("gsm8k", "math500"))
    print(2048 if p90 < 2048 else 4096)
except Exception:
    print(2048)  # no SFT think-tag eval on this box: the config default
EOF
}

kept_examples() { python -c "import json; print(json.load(open('outputs/data/rft/manifest.json'))['num_examples'])"; }

step_smoke() {  # RLVR_SMOKE_MODEL: a checkpoint that writes the format (locally: outputs/train/sft_0p6b_lora_smoke_v2/final)
  local m="${RLVR_SMOKE_MODEL:-$SFT}"
  rlvr-eval --config configs/sample/sft_1.7b_rft_n8.yaml --model "$m" --limit 20 --n 2 --max-tokens 2048 \
    --out outputs/samples/rft_smoke
  rlvr-build-rft --samples outputs/samples/rft_smoke --out outputs/data/rft_smoke --no-fuzzy \
    --figure outputs/data/rft_smoke/solve_rate.png --report outputs/data/rft_smoke/report.md --name rft_smoke
  rlvr-sft --config configs/train/rft_1p7b.yaml --set "model=$m" --set data=outputs/data/rft_smoke/train.jsonl \
    --set manifest=outputs/data/rft_smoke/manifest.json --set max_steps=3 --set per_device_batch=1 --set grad_accum=2 \
    --set out=outputs/train/rft_smoke --set report_to=none --set save_steps=1000 $TRAIN_EXTRA
  echo "== smoke OK: outputs/train/rft_smoke/final exists: $(ls outputs/train/rft_smoke/final/config.json)"
}

step_sample() {
  local cap; cap="$(rft_max_tokens)"
  echo "== sampling cap rule: max_tokens=$cap (2048 when the SFT think-tag p90 is under 2,048; RLVR_RFT_MAX_TOKENS overrides)"
  echo "== ten-minute rule: python scripts/project_sampling.py outputs/phase3_rental.log --from-metrics outputs/eval/sft_1.7b_think_tag/metrics.json --eval math500 --max-hours 2"   # price from RLVR_NODE_RATE
  run_if_missing outputs/samples/sft_1.7b_rft_n8/metrics.json \
    rlvr-eval --config configs/sample/sft_1.7b_rft_n8.yaml --max-tokens "$cap"
}

step_data() {
  run_if_missing outputs/data/rft/manifest.json rlvr-build-rft --samples outputs/samples/sft_1.7b_rft_n8
  python - <<'EOF'
import json; m = json.load(open("outputs/data/rft/manifest.json"))
print("RFT set:", m["num_examples"], "examples, hash", m["content_hash"], "| learnable prompts:", m["num_learnable"])
for k, s in m["solve_rates"].items(): print("  ", k, {x: s[x] for x in ("problems", "rate_0", "rate_1", "learnable")})
EOF
}

step_train() {
  run_if_missing "$RFT/config.json" rlvr-sft --config configs/train/rft_1p7b.yaml $TRAIN_EXTRA
}

step_eval() {
  for t in think_tag few_shot_cot zero_shot; do
    run_if_missing "outputs/eval/rft_1.7b_${t}/metrics.json" rlvr-eval --config "configs/eval/rft_1.7b_${t}.yaml"
  done
  run_if_missing outputs/eval/rft_1.7b_think_tag/perturbation_gap.json \
    python scripts/perturbation_gap.py outputs/eval/rft_1.7b_think_tag
  run_if_missing outputs/probe/partial_prompt_rft_1.7b/metrics.json \
    rlvr-probe-partial --config configs/probe/partial_prompt_1.7b.yaml \
      --model "$RFT" --out outputs/probe/partial_prompt_rft_1.7b
}

step_sample64() {
  run_if_missing outputs/samples/rft_1.7b_n64_few_shot_cot/metrics.json rlvr-eval --config configs/sample/rft_1.7b_n64.yaml
  if [[ "${RLVR_SKIP_COUNTDOWN64:-0}" == "1" ]]; then
    echo "== skip (RLVR_SKIP_COUNTDOWN64=1): Countdown n = 64 for the RFT checkpoint"
  else
    run_if_missing outputs/samples/rft_1.7b_n64_countdown/metrics.json rlvr-eval --config configs/sample/rft_1.7b_n64_countdown.yaml
  fi
}

step_control() {
  local n; n="$(kept_examples)"
  run_if_missing outputs/train/sft_1p7b_control/final/config.json \
    rlvr-sft --config configs/train/sft_1p7b_control.yaml --set "limit=${n}" $TRAIN_EXTRA
  run_if_missing outputs/eval/sft_1.7b_control_think_tag/metrics.json \
    rlvr-eval --config configs/eval/rft_1.7b_think_tag.yaml --model outputs/train/sft_1p7b_control/final \
      --out outputs/eval/sft_1.7b_control_think_tag
}

step_report() {
  mkdir -p outputs/reports/figures
  python scripts/compare_checkpoints.py --ref outputs/eval/sft_1.7b_think_tag --new outputs/eval/rft_1.7b_think_tag \
    --label-ref sft --label-new rft --out outputs/reports/phase-3-compare-think_tag.md
  python scripts/compare_checkpoints.py --ref outputs/eval/base_1.7b_think_tag_4096 --new outputs/eval/rft_1.7b_think_tag \
    --label-ref base --label-new rft --out outputs/reports/phase-3-compare-base-think_tag.md
  for t in few_shot_cot zero_shot; do
    python scripts/compare_checkpoints.py --ref "outputs/eval/sft_1.7b_${t}" --new "outputs/eval/rft_1.7b_${t}" \
      --label-ref sft --label-new rft --out "outputs/reports/phase-3-compare-${t}.md" || true
  done
  if [[ -e outputs/eval/sft_1.7b_control_think_tag/metrics.json ]]; then
    python scripts/compare_checkpoints.py --ref outputs/eval/sft_1.7b_control_think_tag --new outputs/eval/rft_1.7b_think_tag \
      --label-ref sft_control --label-new rft --out outputs/reports/phase-3-compare-control.md
  fi
  if [[ -e outputs/samples/rft_1.7b_n64_few_shot_cot/metrics.json ]]; then
    local cd=(); [[ -e outputs/samples/rft_1.7b_n64_countdown/metrics.json ]] && cd=(outputs/samples/rft_1.7b_n64_countdown)
    python scripts/plot_pass_at_k.py outputs/samples/rft_1.7b_n64_few_shot_cot "${cd[@]}" \
      --greedy outputs/eval/rft_1.7b_few_shot_cot --out outputs/reports/figures/phase-3-pass-at-k-rft.png
    python scripts/perturbation_gap.py outputs/samples/rft_1.7b_n64_few_shot_cot --all-samples \
      --out outputs/reports/phase-3-perturbation-gap-rft-n64.json
    if [[ -e outputs/samples/sft_1.7b_n64_few_shot_cot/metrics.json ]]; then
      python scripts/compare_checkpoints.py --ref outputs/samples/sft_1.7b_n64_few_shot_cot \
        --new outputs/samples/rft_1.7b_n64_few_shot_cot --all-samples --label-ref sft --label-new rft \
        --out outputs/reports/phase-3-compare-pass_at_k.md
    fi
  fi
  python scripts/push_to_hub.py model --run outputs/train/rft_1p7b --repo "${HF_REPO_RFT:-Raviteja546/reasoning-rlvr-rft}" \
    --base-model-id "${HF_REPO_SFT:-Raviteja546/reasoning-rlvr-sft}" \
    --evals outputs/eval/rft_1.7b_think_tag outputs/eval/rft_1.7b_few_shot_cot outputs/eval/rft_1.7b_zero_shot \
    --compare outputs/reports/phase-3-compare-think_tag.md --card-out cards/rft.md ${HF_DRY_RUN:+--dry-run}
  for s in sft_1.7b_rft_n8 rft_1.7b_n64_few_shot_cot rft_1.7b_n64_countdown; do
    [[ -e "outputs/samples/$s/metrics.json" ]] && python scripts/push_to_hub.py samples --run "outputs/samples/$s" \
      --repo "${HF_REPO_SAMPLES:-Raviteja546/reasoning-rlvr-samples}" --path-in-repo "$s" ${HF_DRY_RUN:+--dry-run}
  done
  true
}

case "$STEP" in
  all) step_sample; step_data; step_train; step_eval; step_sample64; step_control; step_report ;;
  smoke|sample|data|train|eval|sample64|control|report) "step_$STEP" ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
echo "== phase3_rental.sh $STEP finished"
