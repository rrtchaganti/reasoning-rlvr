# E2E node session: state and next steps (written 17 Oct 2026, laptop session)

For the Claude Code session running on the E2E node. The laptop session keeps the journal, the
ledger and the commits; this file is how it hands work over. Read `CLAUDE.md` first.

## Where things stand

- Node: E2E Networks L4 (24 GB), Ubuntu 24.04.3, driver 570.133.20 (CUDA 12.8), 25 vCPU, 100 GB RAM,
  233 GB root disk. Billed INR 49/h while it exists; powering off does not stop billing.
- No block-storage volume yet (ticket open). Everything lives on the root disk at the volume's future
  paths: `/mnt/rlvr/reasoning-rlvr` (repo, branch `phase-2-sft`), `/mnt/rlvr/venv`, `/mnt/rlvr/hf`.
  `source scripts/e2e_env.sh` sets the environment (it also unsets the image's `LD_LIBRARY_PATH`).
- Environment built and verified: torch 2.13.0+cu129, vLLM 0.31.0+cu129, transformers 5.17.0,
  trl 1.15.0, Triton compile and launch OK. No flash-attn on this box (no wheel for torch 2.13 and the
  image's 12.6 toolkit cannot build it): every training run here needs `--set packing=false`.
- Not yet done on the node: the data step, the model downloads, the smoke eval, the copy of the base
  reference from the laptop (`outputs/eval/base_1.7b_think_tag_4096/`; the user runs that `scp`).

## What this node is for

Cheap, long, memory-light work while the volume ticket is open. The 80 GB card (A100) does the full
fine-tunes later (`sweep`, `main`, `scaling` need about 27 GB of fp32 master weights and do not fit
here). Do not start those steps on this node.

## Steps, in order

1. Data, downloads, smoke (about 20 min):

   ```bash
   cd /mnt/rlvr/reasoning-rlvr && source scripts/e2e_env.sh && git pull
   bash scripts/phase2_rental.sh data 2>&1 | tee outputs/e2e_data.log
   hf download Qwen/Qwen3-1.7B-Base && hf download Qwen/Qwen3-0.6B-Base
   rlvr-eval --model Qwen/Qwen3-1.7B-Base --evals gsm8k --template few_shot_cot --limit 20 --max-tokens 256 --out outputs/eval/e2e_smoke
   ```

   The data step must print content hashes `b5afb83e6aacafcf` (perturbed GSM8K) and
   `45b3fa4862b11e2a` (SFT set). If either differs, stop and report; nothing downstream is valid.

2. Confirm `outputs/eval/base_1.7b_think_tag_4096/metrics.json` exists (the user copies it from the
   laptop). Without it, `exp002` would spend about 10 L4-hours regenerating it.

3. EXP-002, the deferred base sampling (n = 64), inside tmux. Expect 8-12 h on the L4 (about 60 M
   generated tokens at 1.5-2.5 k tokens/s); it costs about the same as 2-3 h on the A100.

   ```bash
   tmux new -s phase2
   cd /mnt/rlvr/reasoning-rlvr && source scripts/e2e_env.sh
   bash scripts/phase2_rental.sh exp002 2>&1 | tee outputs/e2e_exp002.log
   ```

   Detach with Ctrl-b then d; `tmux attach -t phase2` to return. Check progress with
   `tail -3 outputs/e2e_exp002.log` and `nvidia-smi`.

4. Optional after EXP-002, if the volume has still not arrived and the user agrees to the hours:
   EXP-007 (LoRA r 64) fits 24 GB with the smaller micro-batch and no packing, about 8-10 h here.

   ```bash
   RLVR_TRAIN_EXTRA="--set packing=false --set per_device_batch=2 --set grad_accum=32" \
     bash scripts/phase2_rental.sh lora 2>&1 | tee outputs/e2e_lora.log
   ```

   Watch the first lines: `[rlvr-sft] tag token '</think>'` must say `"untrained": true,
   `"initialised": true`; `run_info.json` must end with `train_delta_norm` above 0 for both tags.
   Before this step, one cheap try of the kernels-hub flash attention is allowed
   (`uv pip install kernels`, then add `--set attn_implementation=kernels-community/flash-attn` and
   drop `--set packing=false`); if it errors, revert to the line above and note it in the log.

## When the volume arrives (the user will say so)

As root, with the volume attached (it appears as `/dev/vdb`; check with `lsblk`):

```bash
mkdir -p /mnt/rlvr_vol && mkfs.ext4 -q -L rlvr /dev/vdb && mount /dev/vdb /mnt/rlvr_vol
rsync -aH --info=progress2 /mnt/rlvr/ /mnt/rlvr_vol/
umount /mnt/rlvr_vol && mv /mnt/rlvr /mnt/rlvr_rootdisk && bash /mnt/rlvr_rootdisk/reasoning-rlvr/scripts/e2e_attach.sh
```

Do this between steps, never while a step is writing to `outputs/`. Paths do not change, so the venv
keeps working. Until that copy exists, everything on the node is lost if the node is terminated.

## Handing results back

- Keep `outputs/e2e_session_log.md` (where / why / ran / result / undo) and `outputs/e2e_session_notes.md`
  (observations the journal should carry: throughput, memory, surprises). The user copies both, plus
  every `metrics.json`, `report.md`, `run_info.json` and `train_log.jsonl`, to the laptop with `scp`.
- Samples (`outputs/samples/*`) are uploaded to the Hub by the `report` step on the A100 session; do not
  delete them here.
- Do not edit tracked files, commit, or push from the node.
