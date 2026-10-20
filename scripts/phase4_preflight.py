"""Phase 4 preflight: print the resolved run config and refuse a launch that does not match the plan.

    python scripts/phase4_preflight.py --run main --config configs/train/grpo_1p7b.yaml
    python scripts/phase4_preflight.py --run a2 --config ... --set dynamic_sampling=true --set epsilon_high=0.28
    python scripts/phase4_preflight.py --subsets            # the pinned eval subsets (needs the datasets)

Checks, each a line of output and a non-zero exit on failure (docs/phase-4-plan.md, sections 3, 6, 7):
  1. the resolved knobs against the plan's table for the named run (shape B, and the one setting
     the ablation changes);
  2. the prompt file against the committed manifest (count and learnable_hash), and the GSM8K slice;
  3. the start checkpoint's run_info.json: commit, kind and run name of the RFT run;
  4. with --subsets: the seeded 300-problem subsets reproduce the manifest hashes of the RFT runs.
The user's rule of 19 Oct 2026: no run starts on a config nobody compared with the plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlvr.train import grpo

SHAPE_B: dict[str, Any] = {
    "num_prompts_per_step": 32,
    "num_generations": 8,
    "max_new_tokens": 1536,
    "temperature": 1.0,
    "top_p": 1.0,
    "learning_rate": 1e-6,
    "warmup_steps": 10,
    "max_steps": 200,
    "kl_beta": 0.04,
    "epsilon": 0.2,
    "epsilon_high": None,
    "loss_type": "grpo",
    "scale_rewards": "group",
    "dynamic_sampling": False,
    "gsm8k_fraction": 0.2,
    "prompt_seed": 0,
    "seed": 0,
    "model": "outputs/train/rft_1p7b/final",
    "limit_prompts": None,  # the prompt file and manifest are checked by content below, not by path
}

RUNS: dict[str, dict[str, Any]] = {
    "main": {**SHAPE_B, "out": "outputs/train/grpo_1p7b"},
    "seed1": {**SHAPE_B, "seed": 1, "out": "outputs/train/grpo_1p7b_seed1"},
    "a1": {**SHAPE_B, "kl_beta": 0.0, "out": "outputs/train/grpo_1p7b_a1_nokl"},
    "a2": {
        **SHAPE_B,
        "dynamic_sampling": True,
        "epsilon_high": 0.28,
        "oversample": 2,
        "out": "outputs/train/grpo_1p7b_a2_dynamic",
    },
    "a3": {
        **SHAPE_B,
        "loss_type": "dr_grpo",
        "scale_rewards": "none",
        "out": "outputs/train/grpo_1p7b_a3_drgrpo",
    },
    "debug": {  # PLAN.md step 2: GSM8K-heavy, 50 steps, not a result
        **SHAPE_B,
        "max_steps": 50,
        "gsm8k_fraction": 1.0,
        "out": "outputs/train/grpo_1p7b_debug",
    },
}

START_COMMIT = "bb6f2a6"  # the RFT run (EXP-010) on the H100, 19 Oct 2026
START_RUN_NAME = "rft_1p7b"

PINNED_SUBSETS = {  # (eval, limit, seed) -> content_hash of the manifest the RFT runs wrote
    ("gsm8k", 300, 0): "58c8ded1c644f332",
    ("math500", 300, 0): "5ca9795b45227236",
    ("countdown", 300, 0): "9698ed78ea4e0491",
}


def check_knobs(cfg: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    problems = []
    for key, want in expected.items():
        got = cfg.get(key)
        same = (
            abs(float(got) - float(want)) < 1e-12
            if isinstance(want, (int, float)) and not isinstance(want, bool) and got is not None
            else got == want
        )
        if not same:
            problems.append(f"{key}: config has {got!r}, plan says {want!r}")
    return problems


def check_prompts(cfg: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    facts: dict[str, Any] = {}
    path = Path(cfg["prompts"])
    if not path.exists():
        return [f"prompt file missing: {path}"], facts
    rows = grpo.read_prompt_rows(path)
    try:
        facts["prompt_file_hash"] = grpo.check_prompt_manifest(rows, cfg["manifest"])
    except (RuntimeError, FileNotFoundError, KeyError) as e:
        problems.append(str(e))
    chosen = grpo.select_prompts(
        rows,
        gsm8k_fraction=cfg["gsm8k_fraction"],
        seed=cfg["prompt_seed"],
        limit=cfg["limit_prompts"],
    )
    facts["prompts_used"] = len(chosen)
    facts["prompts_by_source"] = {
        s: sum(1 for r in chosen if r["source"] == s) for s in sorted({r["source"] for r in chosen})
    }
    return problems, facts


def check_start_checkpoint(
    model_dir: str | Path, *, commit: str, run_name: str
) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    facts: dict[str, Any] = {}
    final = Path(model_dir)
    if not (final / "config.json").exists():
        return [f"start checkpoint missing: {final}/config.json"], facts
    info_path = final / "run_info.json"
    if not info_path.exists():
        info_path = final.parent / "run_info.json"
    if not info_path.exists():
        return [f"no run_info.json next to {final}"], facts
    info = json.loads(info_path.read_text(encoding="utf-8"))
    facts["start_commit"] = info.get("git_commit")
    facts["start_run_name"] = (info.get("config") or {}).get("run_name")
    facts["start_kind"] = info.get("kind")
    if not str(info.get("git_commit", "")).startswith(commit):
        problems.append(
            f"start checkpoint commit {info.get('git_commit')!r} does not start with {commit!r}"
        )
    if facts["start_run_name"] != run_name:
        problems.append(
            f"start checkpoint run_name {facts['start_run_name']!r}, plan says {run_name!r}"
        )
    return problems, facts


def check_subsets(pinned: dict[tuple[str, int, int], str] = PINNED_SUBSETS) -> list[str]:
    from rlvr.eval.datasets import load_eval_set

    problems = []
    for (name, limit, seed), want in pinned.items():
        es = load_eval_set(name, limit=limit, seed=seed, countdown_n=500)
        got = es.content_hash()
        line = f"subset {name} limit={limit} seed={seed}: {len(es)} problems, hash {got} (pinned {want})"
        print(("ok   " if got == want else "FAIL ") + line)
        if got != want:
            problems.append(line)
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", choices=sorted(RUNS), default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--subsets", action="store_true", help="also check the pinned eval subsets")
    ap.add_argument("--start-commit", default=START_COMMIT)
    ap.add_argument(
        "--skip-checkpoint", action="store_true", help="do not look for the start checkpoint"
    )
    args = ap.parse_args(argv)
    problems: list[str] = []
    if args.run:
        cfg = grpo.load_config(args.config, args.set)
        print("resolved config:")
        print(json.dumps(cfg, indent=2, default=str))
        knob_problems = check_knobs(cfg, RUNS[args.run])
        problems += knob_problems
        print(
            ("ok   " if not knob_problems else "FAIL ")
            + f"knobs against the plan's row {args.run!r}"
        )
        for p in knob_problems:
            print("     " + p)
        pp, facts = check_prompts(cfg)
        problems += pp
        print(("ok   " if not pp else "FAIL ") + f"prompts: {json.dumps(facts)}")
        for p in pp:
            print("     " + p)
        if not args.skip_checkpoint:
            cp, facts = check_start_checkpoint(
                cfg["model"], commit=args.start_commit, run_name=START_RUN_NAME
            )
            problems += cp
            print(("ok   " if not cp else "FAIL ") + f"start checkpoint: {json.dumps(facts)}")
            for p in cp:
                print("     " + p)
    if args.subsets:
        problems += check_subsets()
    if problems:
        print(f"PREFLIGHT FAILED: {len(problems)} problem(s); nothing may launch")
        return 1
    print("PREFLIGHT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
