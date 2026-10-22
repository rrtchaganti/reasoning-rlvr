#!/usr/bin/env bash
# Phase 5 option A on the laptop (RTX 4060 8 GB, WSL2, venv ~/venvs/rlvr): EXP-016 budget forcing.
# Usage (from PowerShell, so the job survives the Claude Code session):
#   wsl -d Ubuntu -- bash -lc "source ~/venvs/rlvr/bin/activate && cd /mnt/c/Users/chaga/Documents/Job_shift/Independent_research_projects/reasoning_rlvr && bash scripts/phase5_laptop.sh exp016 2>&1 | tee outputs/phase5_exp016.log"
# Steps:
#   exp016        the forced-stop evals of sft and rft at the budgets below (skips finished runs)
#   exp016-ctrl   the unforced sft and rft think-tag evals at 4,096 on this machine (the same-machine
#                 control for the archive's H100 greedy runs; optional, about the longest step)
# Every run is `rlvr-eval` greedy, think-tag, GSM8K + MATH-500, batch-invariant vLLM; the harness
# refuses a dirty tree. Outputs: outputs/eval/<ckpt>_force<B>_think_tag/{gsm8k,math500}/samples.jsonl
# and metrics.json. The checkpoints come from the Hub (public; about 7 GB each, cached once).
set -euo pipefail
export HF_HUB_DISABLE_XET=1
STEP="${1:-exp016}"
BUDGETS="${RLVR_FORCE_BUDGETS:-1536 1024 2048 768 3072}"
CKPTS="${RLVR_FORCE_CKPTS:-sft rft}"
# The Hub repos hold seed, control and ablation checkpoints in subfolders; vLLM's Hub loader pulls
# every model.safetensors in a repo (six extra 7 GB files for the SFT repo, found 21 Oct 2026), so
# the `download` step fetches the root checkpoint only, as scripts/phase4_rental.sh does, and the
# runs use the local copies under $RLVR_MODELS (default ~/models).
RLVR_MODELS="${RLVR_MODELS:-$HOME/models}"
declare -A REPO=([sft]=Raviteja546/reasoning-rlvr-sft [rft]=Raviteja546/reasoning-rlvr-rft [grpo]=Raviteja546/reasoning-rlvr-grpo)
declare -A MODEL=([sft]="$RLVR_MODELS/reasoning-rlvr-sft" [rft]="$RLVR_MODELS/reasoning-rlvr-rft" [grpo]="$RLVR_MODELS/reasoning-rlvr-grpo")
download() {  # <ckpt>: the root checkpoint of its repo, nothing from the subfolders
  local c="$1" dir="${MODEL[$c]}"
  if [[ -e "$dir/config.json" && -e "$dir/model.safetensors" ]]; then echo "== have $dir"; return; fi
  echo "== $(date -u +%H:%M:%S) UTC: download ${REPO[$c]} -> $dir"
  hf download "${REPO[$c]}" --local-dir "$dir" --exclude "ablations/*" --exclude "controls/*" --exclude "seeds/*" --exclude "*.md"
}
mkdir -p outputs/eval "$RLVR_MODELS"
run_eval() {  # <out dir> <model> <extra flags...>
  local out="$1" model="$2"; shift 2
  if [[ -e "$out/metrics.json" ]]; then echo "== skip (exists): $out"; return; fi
  echo "== $(date -u +%H:%M:%S) UTC: $out"
  rlvr-eval --model "$model" --evals gsm8k math500 --template think_tag --n 1 --temperature 0 \
    --max-tokens 4096 --max-model-len 6144 --gpu-memory-utilization 0.85 --out "$out" "$@"
}
case "$STEP" in
  smoke)  # the forced-stop path end to end on the 0.6B base, 8 problems, about 3 minutes
    rm -rf outputs/eval/smoke_force_stop
    rlvr-eval --model Qwen/Qwen3-0.6B-Base --evals gsm8k --limit 8 --template think_tag --n 1 --temperature 0 \
      --max-tokens 4096 --max-model-len 4096 --gpu-memory-utilization 0.85 --force-stop-at 64 --force-stop-continue 32 \
      --out outputs/eval/smoke_force_stop
    python - <<'EOF'
import json
m = json.load(open("outputs/eval/smoke_force_stop/gsm8k/metrics.json"))
print("force_stop:", m["force_stop"], "| forced_stop_rate:", m["forced_stop_rate"])
rows = [json.loads(l) for l in open("outputs/eval/smoke_force_stop/gsm8k/samples.jsonl", encoding="utf-8")]
forced = [r for r in rows if r.get("forced_stop")]
print(len(forced), "forced of", len(rows))
for r in forced[:2]:
    print("---", r["finish_reason"], r["num_tokens"], "tokens; tail:", repr(r["text"][-120:]))
EOF
    ;;
  download)
    for c in $CKPTS; do download "$c"; done ;;
  exp016)
    # checkpoint-outer: on a slow link (0.4-0.6 MB/s to the Hub on the night of 21 Oct 2026) the
    # first checkpoint's runs finish while the second downloads, instead of waiting for both
    for c in $CKPTS; do
      download "$c"
      for B in $BUDGETS; do
        run_eval "outputs/eval/${c}_force${B}_think_tag" "${MODEL[$c]}" --force-stop-at "$B" --force-stop-continue 128
      done
    done ;;
  exp016-ctrl)
    for c in $CKPTS; do download "$c"; run_eval "outputs/eval/${c}_laptop_think_tag" "${MODEL[$c]}"; done ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
echo "== done ($(date -u +%H:%M:%S) UTC): $STEP"
