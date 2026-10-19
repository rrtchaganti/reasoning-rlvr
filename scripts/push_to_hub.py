"""Upload a checkpoint (with a generated model card) or a sample set to the Hugging Face Hub.

Checkpoint (D4):
    python scripts/push_to_hub.py model --run outputs/train/sft_1p7b --repo Raviteja546/reasoning-rlvr-sft \
        --evals outputs/eval/sft_1.7b_think_tag outputs/eval/sft_1.7b_few_shot_cot outputs/eval/sft_1.7b_zero_shot \
        --compare outputs/reports/phase-2-compare-think_tag.md --card-out cards/sft.md

    Uploads ``<run>/final`` (weights + tokenizer) plus ``run_info.json``, ``train_log.jsonl`` and the
    generated ``README.md`` (the model card: training config, data manifest hash, compute, eval
    table with CIs, known failure modes, commit). The card is also written to ``--card-out`` so it is
    committed with the phase.

Samples (D2, the n = 64 sets):
    python scripts/push_to_hub.py samples --run outputs/samples/base_1.7b_n64_few_shot_cot \
        --repo Raviteja546/reasoning-rlvr-samples --path-in-repo base_1.7b_n64_few_shot_cot

Needs ``HF_TOKEN`` in the environment (never in the repo). ``--dry-run`` writes the card and prints
what would be uploaded without touching the Hub.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

EVAL_ORDER = ("gsm8k", "gsm8k_perturbed", "math500", "countdown", "aime24")


def _f(e: dict | None) -> str:
    if not e:
        return "-"
    return f"{100 * e['mean']:.1f} [{100 * e['lo']:.1f}, {100 * e['hi']:.1f}]"


def eval_table(eval_dirs: list[Path]) -> str:
    rows = [
        "| template | " + " | ".join(EVAL_ORDER) + " | trunc % | extract-fail % |",
        "|---|" + "---|" * (len(EVAL_ORDER) + 2),
    ]
    for d in eval_dirs:
        m = json.loads((Path(d) / "metrics.json").read_text())
        template = next((v.get("template") for v in m.values() if isinstance(v, dict)), d.name)
        cells = [_f(m[e]["pass@1"]) if e in m else "-" for e in EVAL_ORDER]
        trunc = max(
            (v.get("truncation_rate", 0.0) for v in m.values() if isinstance(v, dict)), default=0.0
        )
        fail = max(
            (v.get("extraction_failure_rate", 0.0) for v in m.values() if isinstance(v, dict)),
            default=0.0,
        )
        rows.append(
            f"| {template} | " + " | ".join(cells) + f" | {100 * trunc:.1f} | {100 * fail:.1f} |"
        )
    return "\n".join(rows) + "\n"


HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")


def card_base_model(trained_from: str, base_model_id: str | None) -> str | None:
    """The `base_model` the card header may carry: the Hub validates it as a model id, so a
    checkpoint trained from a local path (the RFT run starts from outputs/train/sft_1p7b/final)
    needs the id of that checkpoint's own repo, or no header line at all. The Phase 3 report step
    on the H100 node (19 Oct 2026) failed with HTTP 400 on exactly this."""
    if base_model_id:
        return base_model_id
    if HUB_ID.match(trained_from) and not Path(trained_from).exists():
        return trained_from
    return None


def model_card(
    run_dir: Path,
    repo: str,
    eval_dirs: list[Path],
    compare_md: str | None,
    base_model_id: str | None = None,
) -> str:
    info = json.loads((Path(run_dir) / "run_info.json").read_text())
    cfg = info.get("config", {})
    base = cfg.get("model", "?")
    lora = cfg.get("lora")
    hours = (info.get("train_runtime_s") or 0) / 3600
    header_base = card_base_model(str(base), base_model_id)
    lines = [
        "---",
        "license: apache-2.0",
        *([f"base_model: {header_base}"] if header_base else []),
        "tags: [reasoning, math, sft, qwen3, rlvr-study]",
        "datasets: [open-r1/OpenR1-Math-220k]",
        "---",
        "",
        f"# {repo.split('/')[-1]}",
        "",
        f"Checkpoint from the reasoning-rlvr study (https://github.com/rrtchaganti/reasoning-rlvr), "
        f"commit `{info.get('git_commit', '?')}` on branch `{info.get('git_branch', '?')}`. "
        f"Prompt format: the think-tag template (`<think> ... </think> <answer> ... </answer>`) used by "
        "every eval and by the RL phases of the study; see the repo's `src/rlvr/eval/prompts.py`.",
        "",
        "## Training",
        "",
        f"- Base: `{base}`; {'LoRA ' + json.dumps(lora) + ', merged' if lora else 'full fine-tune'}; "
        f"bf16={cfg.get('bf16')}, packing={cfg.get('packing')} ({cfg.get('attn_implementation')}).",
        f"- Data: `{cfg.get('data')}` built by `rlvr-build-sft` (manifest hash `{info.get('data_hash')}`, "
        f"{info.get('num_examples_used')} of {info.get('num_examples_total')} examples, "
        f"subset fraction {cfg.get('subset_fraction')}, {info.get('tokens_per_epoch')} tokens per epoch).",
        f"- Optimiser: AdamW ({cfg.get('adam_beta1')}, {cfg.get('adam_beta2')}), lr {cfg.get('learning_rate')}, "
        f"{cfg.get('lr_scheduler')} to {cfg.get('min_lr_rate')} of peak, warmup {cfg.get('warmup_ratio')}, "
        f"weight decay {cfg.get('weight_decay')}, grad clip {cfg.get('max_grad_norm')}, "
        f"effective batch {cfg.get('per_device_batch', 0) * cfg.get('grad_accum', 0)} sequences of "
        f"{cfg.get('max_length')} tokens, {cfg.get('epochs')} epochs ({info.get('global_steps')} steps), seed {cfg.get('seed')}.",
        f"- Loss: cross-entropy on completion tokens only; final train loss {info.get('train_loss')}.",
        f"- Compute: {hours:.2f} GPU-hours on {info.get('gpu', '?')}, peak memory {info.get('peak_memory_gb')} GB.",
        "",
        "## Evaluation (greedy, batch-invariant vLLM, 95 % bootstrap CI over problems)",
        "",
    ]
    if eval_dirs:
        lines.append(eval_table(eval_dirs))
    if compare_md:
        lines += ["", "Paired against the base model on the same problems:", "", compare_md]
    lines += [
        "",
        "## Known failure modes",
        "",
        "- Trained on olympiad-style traces capped at 4096 tokens; longer reasoning is cut at the cap "
        "(truncation rate in the table).",
        "- Answers are graded with math-verify on the `<answer>` block; prose answers without a box "
        "count as extraction failures, reported separately, never as wrong.",
        "- Three MATH-500 problems and no GSM8K problem were found in the SFT source and removed before "
        "training (`outputs/reports/phase-2-data.md`); the base model's own pretraining contamination is "
        "measured in the repo's Phase 1 report and is not addressed by this checkpoint.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "rlvr-build-sft                                   # regenerates the data, checks the manifest hash",
        f"rlvr-sft --config {cfg.get('_config_path', 'configs/train/sft_1p7b.yaml')}",
        "rlvr-eval --config configs/eval/sft_1.7b_think_tag.yaml",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Upload a checkpoint or a sample set to the HF Hub.")
    sub = ap.add_subparsers(dest="what", required=True)
    m = sub.add_parser("model")
    m.add_argument("--run", type=Path, required=True)
    m.add_argument("--repo", required=True)
    m.add_argument("--evals", type=Path, nargs="*", default=[])
    m.add_argument(
        "--compare", type=Path, default=None, help="Markdown from compare_checkpoints.py"
    )
    m.add_argument("--card-out", type=Path, default=None)
    m.add_argument(
        "--base-model-id",
        default=None,
        help="Hub id for the card's base_model when the run trained from a local path "
        "(e.g. the SFT repo for the RFT checkpoint); otherwise that header line is omitted",
    )
    m.add_argument("--private", action="store_true")
    m.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("samples")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--repo", required=True)
    s.add_argument("--path-in-repo", required=True)
    s.add_argument("--private", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.what == "model":
        final = args.run / "final"
        if not (final / "config.json").exists():
            raise FileNotFoundError(f"no checkpoint at {final}")
        compare = args.compare.read_text(encoding="utf-8") if args.compare else None
        card = model_card(args.run, args.repo, args.evals, compare, args.base_model_id)
        (final / "README.md").write_text(card, encoding="utf-8", newline="\n")
        for extra in ("run_info.json", "train_log.jsonl", "train_ids.json"):
            if (args.run / extra).exists():
                (final / extra).write_bytes((args.run / extra).read_bytes())
        if args.card_out:
            args.card_out.parent.mkdir(parents=True, exist_ok=True)
            args.card_out.write_text(card, encoding="utf-8", newline="\n")
        print(f"card written ({len(card)} chars); folder {final}")
        if args.dry_run:
            return
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
        info = api.upload_folder(folder_path=str(final), repo_id=args.repo, repo_type="model")
        print(f"uploaded: {info}")
    else:
        if args.dry_run:
            print(f"would upload {args.run} -> {args.repo}/{args.path_in_repo}")
            return
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
        info = api.upload_folder(
            folder_path=str(args.run),
            repo_id=args.repo,
            repo_type="dataset",
            path_in_repo=args.path_in_repo,
        )
        print(f"uploaded: {info}")


if __name__ == "__main__":
    main()
