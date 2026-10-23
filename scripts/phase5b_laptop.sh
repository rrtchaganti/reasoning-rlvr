#!/usr/bin/env bash
# Phase 5B on the laptop (RTX 4060 8 GB, WSL2, venv ~/venvs/rlvr): EXP-021 (the "Wait" pass) and
# EXP-022 (the outcome reward model). Resumable: a step whose outputs exist is skipped. Every step
# runs on a clean tree (the harness checks). Detached launch, as for EXP-016:
#   setsid nohup bash -c 'source ~/venvs/rlvr/bin/activate && bash scripts/phase5b_laptop.sh all; echo "EXIT=$?"' > outputs/phase5b_all.log 2>&1 < /dev/null &
# Steps:
#   wait        EXP-021: sft and rft at budget 1,536 with "\nWait" + up to 512 tokens before the forced stop
#   orm-build   EXP-022: labelled pairs from the Phase 3 solve-rate samples (../hub_samples/sft_1.7b_rft_n8)
#   orm-train   the LoRA head on Qwen3-0.6B-Base (about 5 GPU-hours)
#   orm-score   scores for the n = 32 think-tag sample sets of sft, rft, grpo (written next to samples.jsonl)
#   orm-plot    the accuracy-against-tokens chart with best-of-k and weighted vote
#   all         wait, orm-build, orm-train, orm-score, orm-plot (a failed step is reported, the next runs)
set -uo pipefail
export HF_HUB_DISABLE_XET=1
# The reward model's steps need no network (Qwen3-0.6B-Base is cached): offline mode keeps the
# Hub out of the save path, where a hung DNS lookup cost the first training run (23 Oct 2026).
ORM_ENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
STEP="${1:-all}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
cd "$ROOT"
RLVR_MODELS="${RLVR_MODELS:-$HOME/models}"
HUB="${RLVR_HUB_SAMPLES:-$PARENT/hub_samples}"
P4="${RLVR_P4_ARCHIVE:-$PARENT/h100_p4_session2_archive/outputs}"
declare -A MODEL=([sft]="$RLVR_MODELS/reasoning-rlvr-sft" [rft]="$RLVR_MODELS/reasoning-rlvr-rft")
stamp() { date -u +%H:%M:%S; }

step_wait() {
  for c in sft rft; do
    local out="outputs/eval/${c}_force1536_wait512_think_tag"
    if [[ -e "$out/metrics.json" ]]; then echo "== skip (exists): $out"; continue; fi
    [[ -e "${MODEL[$c]}/config.json" ]] || { echo "missing checkpoint ${MODEL[$c]} (run scripts/phase5_laptop.sh download)"; return 1; }
    echo "== $(stamp) UTC: $out"
    rlvr-eval --model "${MODEL[$c]}" --evals gsm8k math500 --template think_tag --n 1 --temperature 0 \
      --max-tokens 4096 --max-model-len 6144 --gpu-memory-utilization 0.85 \
      --force-stop-at 1536 --force-stop-continue 128 --force-continue '\nWait' --force-continue-tokens 512 \
      --out "$out" || return 1
  done
}

step_orm_build() {
  if [[ -e outputs/data/orm/manifest.json ]]; then echo "== skip (exists): outputs/data/orm"; return 0; fi
  echo "== $(stamp) UTC: orm-build"
  python -m rlvr.train.orm build --samples "$HUB/sft_1.7b_rft_n8" --evals gsm8k_train math_train \
    --out outputs/data/orm --per-problem 2 --heldout-frac 0.1 --max-examples 20000 --seed 0
}

step_orm_train() {
  if [[ -e outputs/train/orm_0p6b/metrics.json && -e outputs/train/orm_0p6b/head.pt ]]; then echo "== skip (exists): outputs/train/orm_0p6b"; return 0; fi
  echo "== $(stamp) UTC: orm-train"
  env $ORM_ENV python -m rlvr.train.orm train --data outputs/data/orm --model Qwen/Qwen3-0.6B-Base --out outputs/train/orm_0p6b \
    --lora-r 16 --lora-alpha 32 --lr 1e-4 --epochs 1 --batch 4 --grad-accum 4 --max-len 2048 --bf16 --log-every 10 --count-truncated
}

step_orm_score() {
  [[ -e outputs/train/orm_0p6b/head.pt ]] || { echo "no trained ORM"; return 1; }
  for pair in "sft $HUB/sft_1.7b_n32_think_tag_t06" "rft $HUB/rft_1.7b_n32_think_tag_t06" "grpo $P4/samples/grpo_1.7b_n32_think_tag_t06"; do
    set -- $pair
    if [[ -e "$2/gsm8k/scores.jsonl" && -e "$2/math500/scores.jsonl" ]]; then echo "== skip (scored): $1"; continue; fi
    echo "== $(stamp) UTC: orm-score $1"
    env $ORM_ENV python -m rlvr.train.orm score --orm outputs/train/orm_0p6b --samples "$2" --evals gsm8k math500 --batch 16 --bf16 || return 1
  done
}

step_orm_plot() {
  echo "== $(stamp) UTC: orm-plot"
  python scripts/plot_accuracy_vs_tokens.py \
    --samples "sft=$HUB/sft_1.7b_n32_think_tag_t06" "rft=$HUB/rft_1.7b_n32_think_tag_t06" "grpo=$P4/samples/grpo_1.7b_n32_think_tag_t06" \
    --greedy "sft=$PARENT/h100_session_archive/outputs/eval/sft_1.7b_think_tag" "rft=$PARENT/h100_session_archive/outputs/eval/rft_1.7b_think_tag" "grpo=$P4/eval/grpo_1.7b_think_tag" \
    --reference-greedy grpo --evals gsm8k math500 --ks 1 2 4 8 16 32 \
    --out outputs/reports/figures/phase-5-accuracy-vs-tokens-orm.png \
    --title "EXP-022: majority vote, best-of-k and reward-weighted vote (ORM: LoRA head on Qwen3-0.6B-Base), think-tag, T = 0.6, n = 32"
}

case "$STEP" in
  wait) step_wait ;;
  orm-build) step_orm_build ;;
  orm-train) step_orm_train ;;
  orm-score) step_orm_score ;;
  orm-plot) step_orm_plot ;;
  all)
    for s in wait orm-build orm-train orm-score orm-plot; do
      "step_${s//-/_}" && echo "== step done ($(stamp) UTC): $s" || echo "== STEP FAILED ($(stamp) UTC): $s (continuing)"
    done ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
echo "== done ($(stamp) UTC): $STEP"
