#!/usr/bin/env bash
# EXP-024: reproduce the Qwen3-1.7B-Base greedy baseline table (EXP-001) from a fresh clone, and pair
# every completion with the archived run.
#
# Usage (Linux or WSL2 with a CUDA GPU; uv on the PATH; no token needed, the Hub repos are public):
#   bash reproduce_baselines.sh [REPO_URL_OR_PATH] [REV]
#     REPO  default https://github.com/rrtchaganti/reasoning-rlvr.git
#     REV   default main
#   RLVR_REPRO_DIR   work directory (default ~/rlvr_repro): repo/ and ref/ go here
#   RLVR_REPRO_VENV  the new environment (default ~/venvs/rlvr_repro)
#
# Steps, each skipped when its output exists (rerun the same command after a drop):
#   clone -> setup_env.sh into a new venv -> the archived reference from the Hub (sha256 checked)
#   -> perturbed GSM8K -> the four base_1.7b_* greedy configs -> the table -> the per-problem check.
# The Hugging Face cache (model weights, datasets) is whatever the machine already has.
set -euo pipefail
REPO=${1:-https://github.com/rrtchaganti/reasoning-rlvr.git}
REV=${2:-main}
WORK=${RLVR_REPRO_DIR:-$HOME/rlvr_repro}
export RLVR_VENV=${RLVR_REPRO_VENV:-$HOME/venvs/rlvr_repro}
DATASET=Raviteja546/reasoning-rlvr-samples
ARCHIVE=archives/laptop_archive_2026-10-24.tar.gz
SHA256=8cbe1bdde8ab93976a2f46707bd11c7f53999402138e0025ba5d3005bc5606f1
TEMPLATES="zero_shot few_shot few_shot_cot think_tag"
step() { echo "== $(date -u +%H:%M:%S) UTC: $*"; }

mkdir -p "$WORK"
step "clone $REPO at $REV"
if [ ! -d "$WORK/repo/.git" ]; then git clone -q "$REPO" "$WORK/repo"; fi
cd "$WORK/repo"
git checkout -q "$REV"
echo "commit $(git rev-parse HEAD)"

step "environment $RLVR_VENV"
if [ ! -x "$RLVR_VENV/bin/rlvr-eval" ]; then bash scripts/setup_env.sh; fi
# shellcheck disable=SC1091
source "$RLVR_VENV/bin/activate"

step "reference: $ARCHIVE from $DATASET"
if [ ! -d "$WORK/ref/outputs/eval/base_1.7b_think_tag" ]; then
  mkdir -p "$WORK/ref"
  python - "$DATASET" "$ARCHIVE" "$WORK/ref" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
p = hf_hub_download(sys.argv[1], sys.argv[2], repo_type="dataset")
shutil.copy(p, f"{sys.argv[3]}/archive.tar.gz")
PY
  echo "$SHA256  $WORK/ref/archive.tar.gz" | sha256sum -c -
  tar -xzf "$WORK/ref/archive.tar.gz" -C "$WORK/ref" --wildcards 'outputs/eval/base_1.7b_*'
  rm "$WORK/ref/archive.tar.gz"
fi

step "perturbed GSM8K (seed 0)"
rlvr-perturb-gsm8k --seed 0
# The step rewrites the tracked manifest; a deterministic build leaves it byte-identical.
if ! git diff --quiet -- data/manifests/gsm8k_perturbed.json; then
  echo "the perturbed set differs from the committed manifest (data/manifests/gsm8k_perturbed.json):"
  git diff --stat
  exit 1
fi

for t in $TEMPLATES; do
  step "EXP-001 greedy: base_1.7b_$t"
  if [ ! -f "outputs/eval/base_1.7b_$t/metrics.json" ]; then
    rlvr-eval --config "configs/eval/base_1.7b_$t.yaml"
  fi
done

step "the table"
python scripts/collect_baselines.py outputs/eval --out outputs/reports/phase-1-baselines-repro.md
echo "compare with the committed outputs/reports/phase-1-baselines.md (Qwen3-1.7B-Base rows)"

step "per-problem check against the archive"
python scripts/check_reproduction.py --ref "$WORK/ref/outputs/eval" --new outputs/eval \
  --runs $(for t in $TEMPLATES; do printf 'base_1.7b_%s ' "$t"; done) \
  --out outputs/reports/reproduction-check.md
step "done"
