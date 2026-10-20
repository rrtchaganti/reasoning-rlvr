#!/usr/bin/env bash
# Phase 4 node session (GRPO, shape B; docs/phase-4-plan.md). Resumable like the Phase 2 and 3
# scripts: a step is skipped when its output exists. Run inside tmux, venv active, from the repo
# root, on a clean tree:
#   bash scripts/phase4_rental.sh <step> 2>&1 | tee -a outputs/phase4_rental.log
# Steps, in the session order (each usable on its own; `session1` and `session2` chain them):
#   setup        the RFT checkpoint and the Phase 3 samples from the Hub, the RFT data rebuilt and hash-checked,
#                the rft dev-split eval (the number every dev curve is read against)
#   smoke        rlvr-grpo for 2 steps on 4 prompts at a 256-token cap with vLLM colocated (plumbing, 5 min)
#   p3ablations  EXP-010g (no correctness filter) and EXP-010h (random two per problem): build, train, eval each
#   debug        PLAN.md step 2: 50 GRPO steps on GSM8K-heavy prompts; measures the step time; not a result
#   main         EXP-012, then its dev evals on the saved checkpoints
#   seed         EXP-013 (seed 1) + dev evals
#   a1 a2 a3     EXP-014a-c + dev evals
#   evals        the main checkpoint's greedy templates, gap, probe, the n = 32 trained-format samples;
#                greedy think-tag for every other checkpoint; all sets of a checkpoint in one process
#   report       paired tables, figures, model card, uploads
# Every training and sampling step runs scripts/phase4_preflight.py first and stops on a mismatch
# (the user's rule of 19 Oct 2026). This script never prints an EXIT= line: the launcher appends
# `echo EXIT=$? >> <log>` after the step (handoff section 3, rule 4), so the marker is the log's
# last line and nothing printed here can match it. Environment:
#   RLVR_GRPO_EXTRA   extra --set overrides for every rlvr-grpo call (e.g. "--set vllm_gpu_memory_utilization=0.3")
#   HF_REPO_RFT/SFT/SAMPLES/GRPO   Hub repos (defaults below)
set -euo pipefail
cd "$(dirname "$0")/.."
STEP="${1:-session1}"
GRPO_EXTRA="${RLVR_GRPO_EXTRA:-}"   # unquoted below on purpose
HF_REPO_SFT="${HF_REPO_SFT:-Raviteja546/reasoning-rlvr-sft}"
HF_REPO_RFT="${HF_REPO_RFT:-Raviteja546/reasoning-rlvr-rft}"
HF_REPO_SAMPLES="${HF_REPO_SAMPLES:-Raviteja546/reasoning-rlvr-samples}"
HF_REPO_GRPO="${HF_REPO_GRPO:-Raviteja546/reasoning-rlvr-grpo}"
RFT=outputs/train/rft_1p7b/final
SFT=outputs/train/sft_1p7b/final
CFG=configs/train/grpo_1p7b.yaml
MAIN=outputs/train/grpo_1p7b

run_if_missing() {  # run_if_missing <marker path> <command...>
  local marker="$1"; shift
  if [[ -e "$marker" ]]; then echo "== skip ($marker exists): $*"; return 0; fi
  echo "== run ($(date -u +%H:%M:%S) UTC): $*"; "$@"
}

preflight() {  # preflight <run> <--set ...>
  local run="$1"; shift
  python scripts/phase4_preflight.py --run "$run" --config "$CFG" "$@" | tee -a outputs/phase4_preflight.log
}

grpo_run() {  # grpo_run <run name> <out dir> <--set ...>  (preflight, train, dev evals of the checkpoints)
  local run="$1" out="$2"; shift 2
  if [[ -e "$out/final/config.json" ]]; then echo "== skip ($out/final exists): $run"; else
    preflight "$run" "$@" $GRPO_EXTRA
    rlvr-grpo --config "$CFG" "$@" $GRPO_EXTRA
    echo "== training done ($(date -u +%H:%M:%S) UTC): $run; dev evals of its checkpoints next"
  fi
  for ck in "$out"/checkpoints/checkpoint-*; do
    [[ -d "$ck" ]] || continue
    local tag; tag="$(basename "$ck")"
    run_if_missing "outputs/eval/$(basename "$out")_dev_${tag}/metrics.json" \
      rlvr-eval --config configs/eval/ckpt_think_tag_dev.yaml --model "$ck" --out "outputs/eval/$(basename "$out")_dev_${tag}"
  done
}

step_setup() {
  mkdir -p outputs/train/sft_1p7b outputs/train/rft_1p7b outputs/samples
  [[ -e "$SFT/config.json" ]] || hf download "$HF_REPO_SFT" --local-dir "$SFT" --exclude "ablations/*"
  [[ -e "$RFT/config.json" ]] || hf download "$HF_REPO_RFT" --local-dir "$RFT" --exclude "seeds/*"
  for s in sft_1.7b_rft_n8 sft_1.7b_n32_think_tag_t06 rft_1.7b_n32_think_tag_t06; do
    [[ -e "outputs/samples/$s/metrics.json" || -e "outputs/samples/$s/gsm8k/samples.jsonl" ]] || \
      hf download "$HF_REPO_SAMPLES" --repo-type dataset --include "$s/*" --local-dir outputs/samples
  done
  # run_info.json travels inside final/ on the Hub; the preflight reads it from either place
  bash scripts/phase2_rental.sh data                       # perturbed GSM8K + SFT set, hashes checked
  run_if_missing outputs/data/rft/manifest.json rlvr-build-rft --samples outputs/samples/sft_1.7b_rft_n8
  python - <<'EOF'
import json
m = json.load(open("outputs/data/rft/manifest.json")); c = json.load(open("data/manifests/rft_iter1.json"))
for k in ("content_hash", "learnable_hash", "num_examples", "num_learnable"):
    assert m[k] == c[k], (k, m[k], c[k])
print("RFT data rebuilt: hashes match the committed manifest", m["content_hash"], m["learnable_hash"])
EOF
  run_if_missing outputs/eval/rft_1.7b_dev/metrics.json \
    rlvr-eval --config configs/eval/ckpt_think_tag_dev.yaml --model "$RFT" --out outputs/eval/rft_1.7b_dev
  python scripts/phase4_preflight.py --subsets | tee -a outputs/phase4_preflight.log
}

step_smoke() {
  rlvr-grpo --config "$CFG" --set limit_prompts=4 --set num_prompts_per_step=4 --set num_generations=4 \
    --set per_device_batch=16 --set max_new_tokens=256 --set max_steps=2 --set save_steps=1000 --set report_to=none \
    --set out=outputs/train/grpo_smoke --set run_name=grpo_smoke $GRPO_EXTRA
  echo "== smoke OK: $(ls outputs/train/grpo_smoke/final/config.json)"
}

step_p3ablations() {
  run_if_missing outputs/data/rft_nofilter/manifest.json rlvr-build-rft --samples outputs/samples/sft_1.7b_rft_n8 \
    --keep-incorrect --out outputs/data/rft_nofilter --name rft_nofilter \
    --figure outputs/data/rft_nofilter/solve_rate.png --report outputs/data/rft_nofilter/report.md
  run_if_missing outputs/train/rft_1p7b_nofilter/final/config.json rlvr-sft --config configs/train/rft_1p7b.yaml \
    --set data=outputs/data/rft_nofilter/train.jsonl --set manifest=outputs/data/rft_nofilter/manifest.json \
    --set out=outputs/train/rft_1p7b_nofilter --set run_name=rft_1p7b_nofilter --set per_device_batch=8 --set grad_accum=8
  run_if_missing outputs/eval/rft_1.7b_nofilter_think_tag/metrics.json rlvr-eval --config configs/eval/rft_1.7b_think_tag.yaml \
    --model outputs/train/rft_1p7b_nofilter/final --out outputs/eval/rft_1.7b_nofilter_think_tag
  run_if_missing outputs/data/rft_random2/manifest.json rlvr-build-rft --samples outputs/samples/sft_1.7b_rft_n8 \
    --per-problem-select random --select-seed 0 --out outputs/data/rft_random2 --name rft_random2 \
    --figure outputs/data/rft_random2/solve_rate.png --report outputs/data/rft_random2/report.md
  run_if_missing outputs/train/rft_1p7b_random2/final/config.json rlvr-sft --config configs/train/rft_1p7b.yaml \
    --set data=outputs/data/rft_random2/train.jsonl --set manifest=outputs/data/rft_random2/manifest.json \
    --set out=outputs/train/rft_1p7b_random2 --set run_name=rft_1p7b_random2 --set per_device_batch=8 --set grad_accum=8
  run_if_missing outputs/eval/rft_1.7b_random2_think_tag/metrics.json rlvr-eval --config configs/eval/rft_1.7b_think_tag.yaml \
    --model outputs/train/rft_1p7b_random2/final --out outputs/eval/rft_1.7b_random2_think_tag
}

step_debug() {
  grpo_run debug outputs/train/grpo_1p7b_debug --set max_steps=50 --set gsm8k_fraction=1.0 \
    --set out=outputs/train/grpo_1p7b_debug --set run_name=grpo_1p7b_debug
  python - <<'EOF'
import json
info = json.load(open("outputs/train/grpo_1p7b_debug/run_info.json"))
s = info["train_runtime_s"] / max(info["global_steps"], 1)
print(f"debug run: {s:.1f} s per step -> 200 steps about {200*s/3600:.2f} h; peak {info['peak_memory_gb']} GB; "
      f"reward {info['reward_first_step']:.3f} -> {info['reward_last_step']:.3f}; finished {info['finished_rate_first_step']:.2f} -> {info['finished_rate_last_step']:.2f}")
EOF
}

step_main() { grpo_run main "$MAIN"; }
step_seed() { grpo_run seed1 outputs/train/grpo_1p7b_seed1 --set seed=1 --set out=outputs/train/grpo_1p7b_seed1 --set run_name=grpo_1p7b_seed1; }
step_a1()   { grpo_run a1 outputs/train/grpo_1p7b_a1_nokl --set kl_beta=0 --set out=outputs/train/grpo_1p7b_a1_nokl --set run_name=grpo_1p7b_a1_nokl; }
step_a2()   { grpo_run a2 outputs/train/grpo_1p7b_a2_dynamic --set dynamic_sampling=true --set epsilon_high=0.28 --set out=outputs/train/grpo_1p7b_a2_dynamic --set run_name=grpo_1p7b_a2_dynamic; }
step_a3()   { grpo_run a3 outputs/train/grpo_1p7b_a3_drgrpo --set loss_type=dr_grpo --set scale_rewards=none --set out=outputs/train/grpo_1p7b_a3_drgrpo --set run_name=grpo_1p7b_a3_drgrpo; }

step_evals() {
  # the main checkpoint: three templates (each config holds all its sets), gap, probe, the n = 32 trained-format samples
  for t in think_tag few_shot_cot zero_shot; do
    run_if_missing "outputs/eval/grpo_1.7b_${t}/metrics.json" rlvr-eval --config "configs/eval/grpo_1.7b_${t}.yaml"
  done
  run_if_missing outputs/eval/grpo_1.7b_think_tag/perturbation_gap.json python scripts/perturbation_gap.py outputs/eval/grpo_1.7b_think_tag
  run_if_missing outputs/probe/partial_prompt_grpo_1.7b/metrics.json rlvr-probe-partial --config configs/probe/partial_prompt_1.7b.yaml \
    --model "$MAIN/final" --out outputs/probe/partial_prompt_grpo_1.7b
  python scripts/phase4_preflight.py --subsets | tee -a outputs/phase4_preflight.log
  run_if_missing outputs/samples/grpo_1.7b_n32_think_tag_t06/metrics.json rlvr-eval --config configs/sample/grpo_1.7b_n32_think_tag_t06.yaml
  # every other checkpoint: the five-set greedy think-tag eval in one process
  for r in grpo_1p7b_seed1 grpo_1p7b_a1_nokl grpo_1p7b_a2_dynamic grpo_1p7b_a3_drgrpo; do
    [[ -e "outputs/train/$r/final/config.json" ]] || continue
    run_if_missing "outputs/eval/${r}_think_tag/metrics.json" rlvr-eval --config configs/eval/grpo_1.7b_think_tag.yaml \
      --model "outputs/train/$r/final" --out "outputs/eval/${r}_think_tag"
  done
}

step_report() {
  mkdir -p outputs/reports/figures
  python scripts/compare_checkpoints.py --ref outputs/eval/rft_1.7b_think_tag --new outputs/eval/grpo_1.7b_think_tag \
    --label-ref rft --label-new grpo --out outputs/reports/phase-4-compare-think_tag.md
  for t in few_shot_cot zero_shot; do
    python scripts/compare_checkpoints.py --ref "outputs/eval/rft_1.7b_${t}" --new "outputs/eval/grpo_1.7b_${t}" \
      --label-ref rft --label-new grpo --out "outputs/reports/phase-4-compare-${t}.md" || true
  done
  python scripts/compare_checkpoints.py --ref outputs/eval/sft_1.7b_think_tag --new outputs/eval/grpo_1.7b_think_tag \
    --label-ref sft --label-new grpo --out outputs/reports/phase-4-compare-sft-think_tag.md || true
  for pair in "grpo_1p7b_seed1 seed1 grpo_1p7b main" "grpo_1p7b_a1_nokl a1 grpo_1p7b main" "grpo_1p7b_a2_dynamic a2 grpo_1p7b main" "grpo_1p7b_a3_drgrpo a3 grpo_1p7b main"; do
    set -- $pair
    [[ -e "outputs/eval/$1_think_tag/metrics.json" ]] || continue
    python scripts/compare_checkpoints.py --ref outputs/eval/grpo_1.7b_think_tag --new "outputs/eval/$1_think_tag" \
      --label-ref "$4" --label-new "$2" --out "outputs/reports/phase-4-compare-$2-vs-main.md"
  done
  if [[ -e outputs/samples/grpo_1.7b_n32_think_tag_t06/metrics.json ]]; then
    python scripts/compare_checkpoints.py --ref outputs/samples/rft_1.7b_n32_think_tag_t06 --new outputs/samples/grpo_1.7b_n32_think_tag_t06 \
      --all-samples --label-ref rft --label-new grpo --out outputs/reports/phase-4-compare-think_tag_n32_t06.md
  fi
  for r in rft_1p7b_nofilter rft_1p7b_random2; do
    [[ -e "outputs/eval/${r/_1p7b/_1.7b}_think_tag/metrics.json" ]] || continue
    python scripts/compare_checkpoints.py --ref outputs/eval/rft_1.7b_think_tag --new "outputs/eval/${r/_1p7b/_1.7b}_think_tag" \
      --label-ref rft --label-new "${r#rft_1p7b_}" --out "outputs/reports/phase-3-compare-${r#rft_1p7b_}-vs-rft.md"
  done
  python scripts/push_to_hub.py model --run "$MAIN" --repo "$HF_REPO_GRPO" --base-model-id "$HF_REPO_RFT" \
    --evals outputs/eval/grpo_1.7b_think_tag outputs/eval/grpo_1.7b_few_shot_cot outputs/eval/grpo_1.7b_zero_shot \
    --compare outputs/reports/phase-4-compare-think_tag.md --card-out cards/grpo.md ${HF_DRY_RUN:+--dry-run}
  [[ -e outputs/samples/grpo_1.7b_n32_think_tag_t06/metrics.json ]] && python scripts/push_to_hub.py samples \
    --run outputs/samples/grpo_1.7b_n32_think_tag_t06 --repo "$HF_REPO_SAMPLES" --path-in-repo grpo_1.7b_n32_think_tag_t06 ${HF_DRY_RUN:+--dry-run}
  true
}

case "$STEP" in
  setup|smoke|p3ablations|debug|main|seed|a1|a2|a3|evals|report) "step_$STEP" ;;
  session1) for s in setup smoke p3ablations debug main seed evals; do "step_$s"; echo "== step done ($(date -u +%H:%M:%S) UTC): $s"; done ;;
  session2) for s in setup a1 a2 a3 evals report; do "step_$s"; echo "== step done ($(date -u +%H:%M:%S) UTC): $s"; done ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
