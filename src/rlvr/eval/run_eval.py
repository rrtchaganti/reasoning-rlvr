"""``rlvr-eval``: score a checkpoint on one or more eval sets and write the D1 report.

Example (greedy, all evals, local 4060):

    rlvr-eval --model Qwen/Qwen3-0.6B-Base --evals gsm8k math500 countdown \\
              --template few_shot_cot --out outputs/eval/qwen3-0.6b-base

Sampled pass@k (rented GPU):

    rlvr-eval --model Qwen/Qwen3-1.7B-Base --evals math500 --template think_tag \\
              --n 64 --temperature 0.6 --top-p 0.95 --out outputs/eval/base-math500-n64

Outputs per eval: ``<out>/<eval>/samples.jsonl`` (git-ignored), ``metrics.json``,
``manifest.json``; and ``<out>/report.md`` + ``run_info.json`` for the whole run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from rlvr.eval.datasets import load_eval_set, write_manifest
from rlvr.eval.force_stop import force_stop
from rlvr.eval.prompts import TEMPLATES, build_prompt
from rlvr.eval.report import (
    MARKDOWN_HEADER,
    grade_completions,
    markdown_row,
    summarize,
    write_samples_jsonl,
)
from rlvr.eval.sampler import Sampler, SamplingConfig
from rlvr.utils.run_info import finish_run_info, write_run_info


def given_dests(p: argparse.ArgumentParser, argv: Sequence[str]) -> set[str]:
    """Destinations of the options that appear in ``argv`` (``--max-tokens 4096`` and
    ``--max-tokens=4096`` alike). argparse itself cannot tell "given with the default's value"
    from "absent", and the config merge below must: on the H100 node (19 Oct 2026) ``--max-tokens
    4096`` lost to a config's 2,048 because 4096 is the parser default (EXP-009 ran at the wrong
    cap), and ``--evals gsm8k math500`` lost to a config's five sets for the same reason."""
    words = {str(a).split("=", 1)[0] for a in argv}
    return {a.dest for a in p._actions if any(s in words for s in a.option_strings)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # given_dests matches option strings exactly
    )
    p.add_argument("--config", type=Path, help="YAML with any of the flags below (flags override)")
    p.add_argument("--model", help="HF id or local path")
    p.add_argument("--revision", default=None)
    p.add_argument(
        "--evals",
        nargs="+",
        default=["gsm8k", "math500"],
        help="gsm8k math500 countdown aime24 gsm8k_perturbed",
    )
    p.add_argument("--template", choices=TEMPLATES, default="few_shot_cot")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--force-stop-at",
        type=int,
        default=None,
        help="budget forcing (EXP-016): generate at most this many tokens, then give every "
        "completion that did not close its think block the template's closing tags and "
        "--force-stop-continue more tokens to answer; think_tag only",
    )
    p.add_argument("--force-stop-continue", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="subset size for smoke runs (never for reported numbers)",
    )
    p.add_argument("--countdown-n", type=int, default=500)
    p.add_argument(
        "--levels",
        default=None,
        help="training pool: keep these MATH levels, e.g. 1,2,3 (evals with level metadata only)",
    )
    p.add_argument(
        "--exclude-dev",
        action="store_true",
        help="training pool: drop the dev-split problems from gsm8k_train / math_train",
    )
    p.add_argument(
        "--pool-limit",
        action="append",
        default=None,
        metavar="EVAL=N",
        help="training pool: seeded cap of N problems for that eval (repeatable)",
    )
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=6144)
    p.add_argument(
        "--no-batch-invariant",
        dest="batch_invariant",
        action="store_false",
        help="disable VLLM_BATCH_INVARIANT (on by default since EXP-000; off = faster, non-reproducible)",
    )
    p.add_argument("--out", type=Path, required=False)
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit uncommitted code; report is marked -dirty",
    )
    args = p.parse_args(argv)
    if args.config:
        # Flags override the config: a config value applies only to options absent from argv,
        # whatever value the flag carries (see given_dests).
        given = given_dests(p, argv)
        cfg = yaml.safe_load(args.config.read_text()) or {}
        for k, v in cfg.items():
            key = k.replace("-", "_")
            if key not in given:
                setattr(args, key, v)
    if not args.model or not args.out:
        p.error("--model and --out are required (via flags or --config)")
    return args


LEVEL_EVALS = ("math_train", "math500", "math_dev")
POOL_EVALS = ("gsm8k_train", "math_train")


def parse_levels(value: Any) -> list[int] | None:
    if value in (None, "", []):
        return None
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(x) for x in str(value).split(",") if x.strip()]


def parse_pool_limits(value: Any) -> dict[str, int]:
    """``--pool-limit gsm8k_train=4000`` (repeatable) or a YAML mapping / list of such strings."""
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    out: dict[str, int] = {}
    for item in value:
        if "=" not in str(item):
            raise ValueError(f"--pool-limit expects EVAL=N, got {item!r}")
        k, v = str(item).split("=", 1)
        out[k.strip()] = int(v)
    return out


def make_sampler(args: argparse.Namespace) -> Sampler:
    if args.backend == "vllm":
        from rlvr.eval.sampler import VLLMSampler

        return VLLMSampler(
            args.model,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            revision=args.revision,
            seed=args.seed,
            batch_invariant=args.batch_invariant,
        )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rlvr.eval.sampler import HFSampler

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision)
    return HFSampler(model, tok)


def run(args: argparse.Namespace, sampler: Sampler | None = None) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    info = write_run_info(out, config=config, seed=args.seed, allow_dirty=args.allow_dirty)
    t_start = time.time()  # the compute ledger counts engine start-up as GPU time too
    sampler = sampler or make_sampler(args)
    engine_startup_s = time.time() - t_start
    cfg_base = SamplingConfig(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    rows = [MARKDOWN_HEADER]
    all_metrics: dict[str, dict] = {}
    levels = parse_levels(getattr(args, "levels", None))
    pool_limits = parse_pool_limits(getattr(args, "pool_limit", None))
    exclude_dev = bool(getattr(args, "exclude_dev", False))
    t0 = time.time()
    for name in args.evals:
        es = load_eval_set(
            name,
            revision=args.revision,
            limit=args.limit,
            seed=args.seed,
            countdown_n=args.countdown_n,
            levels=levels if name in LEVEL_EVALS else None,
            exclude_dev=exclude_dev and name in POOL_EVALS,
            pool_limit=pool_limits.get(name),
        )
        if levels or pool_limits or exclude_dev:
            print(f"[rlvr-eval] {name}: pool of {len(es)} problems ({es.source})", flush=True)
        template = "think_tag" if name == "countdown" else args.template
        prompts = [build_prompt(p, template, name) for p in es.problems]
        stop = prompts[0].stop if prompts else ()
        force_at = getattr(args, "force_stop_at", None)
        if force_at and prompts and not prompts[0].force_suffix:
            raise ValueError(
                f"--force-stop-at is defined for the think_tag template only; {template} has no "
                "closing tags to force"
            )
        cfg = SamplingConfig(
            n=cfg_base.n,
            temperature=cfg_base.temperature,
            top_p=cfg_base.top_p,
            top_k=cfg_base.top_k,
            max_tokens=int(force_at) if force_at else cfg_base.max_tokens,
            seed=cfg_base.seed,
            stop=tuple(stop),
        )
        budget = sampler.context_budget([p.text for p in prompts], cfg.max_tokens)
        if budget and budget["capped_by_context"]:
            print(
                f"[rlvr-eval] {name}: max_model_len {budget['max_model_len']} caps new tokens at "
                f"{budget['effective_max_tokens_min']}-{budget['effective_max_tokens_max']} "
                f"(max_tokens {budget['max_tokens']}, prompts {budget['prompt_tokens_min']}-"
                f"{budget['prompt_tokens_max']} tokens); 'length' means the context, not max_tokens",
                flush=True,
            )
        t_eval = time.time()
        completions = sampler.generate([p.text for p in prompts], cfg)
        force_stats = None
        if force_at and prompts:
            completions, force_stats = force_stop(
                sampler,
                [p.text for p in prompts],
                completions,
                cfg,
                suffix=prompts[0].force_suffix,
                continue_tokens=int(getattr(args, "force_stop_continue", 128)),
            )
            print(
                f"[rlvr-eval] {name}: budget {force_at}, forced {force_stats['num_forced']} of "
                f"{force_stats['num_completions']} completions, {force_stats['num_forced_finished']} "
                f"of them finished within {force_stats['continue_tokens']} more tokens",
                flush=True,
            )
        results = grade_completions(
            es, completions, completion_prefix=prompts[0].completion_prefix if prompts else ""
        )
        summary = summarize(results)
        summary.update(
            {
                "eval": name,
                "template": template,
                "model": args.model,
                "run_tag": info["run_tag"],
                "sampling": cfg.as_dict(),
                "manifest": es.manifest(),
                "wall_clock_s": time.time() - t_eval,
                "backend": sampler.describe(),
                "context_budget": budget,
                "force_stop": force_stats,
            }
        )
        eval_dir = out / name
        eval_dir.mkdir(exist_ok=True)
        write_samples_jsonl(results, eval_dir / "samples.jsonl")
        write_manifest(es, eval_dir / "manifest.json")
        (eval_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        (eval_dir / "prompt_example.txt").write_text(prompts[0].text if prompts else "")
        rows.append(markdown_row(f"{name} ({template})", summary))
        all_metrics[name] = summary
        if len(rows) == 2:
            print(MARKDOWN_HEADER)
        print(rows[-1])
    report = (
        f"# {args.model} @ {info['run_tag']}\n\n"
        f"template={args.template}, n={args.n}, T={args.temperature}, top_p={args.top_p}, "
        f"max_tokens={args.max_tokens}, seed={args.seed}, limit={args.limit}\n\n"
        + "\n".join(rows)
        + "\n"
    )
    (out / "report.md").write_text(report)
    (out / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    finish_run_info(
        out,
        wall_clock_s=time.time() - t_start,
        engine_startup_s=engine_startup_s,
        sampling_wall_clock_s=time.time() - t0,
        evals=list(all_metrics),
    )
    return all_metrics


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
