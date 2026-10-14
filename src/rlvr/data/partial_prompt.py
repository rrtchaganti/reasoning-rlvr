"""Partial-prompt completion probe (PLAN.md section 4.3, probe 2).

Feed the model the first ``frac`` of a test problem's text and let it continue greedily. If the
continuation reproduces the held-out remainder near-verbatim, the model has seen that problem
(or a close copy) in training. Reported per checkpoint, so a rise after SFT/RL is visible.

Scoring: normalised Indel similarity (rapidfuzz) between the held-out text and the continuation
truncated to the held-out length, both whitespace-normalised and lower-cased. A problem counts as
"completed" at similarity >= 0.9; the rate at 0.75 is reported too. A chance baseline pairs each
continuation with a *different* problem's held-out text, so the reader can see what the metric
gives when nothing is memorised (templated problems share phrasing).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz.distance import Indel

from rlvr.eval.datasets import load_eval_set
from rlvr.eval.sampler import Completion, Sampler, SamplingConfig

_WS = re.compile(r"\s+")


def split_prefix(text: str, frac: float = 0.6) -> tuple[str, str]:
    """Split at a word boundary so that about ``frac`` of the words are in the prefix."""
    words = text.split()
    if len(words) < 4:
        raise ValueError("problem too short to split")
    k = max(1, min(len(words) - 1, math.ceil(frac * len(words))))
    prefix = " ".join(words[:k])
    holdout = " ".join(words[k:])
    return prefix, holdout


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def continuation_similarity(continuation: str, holdout: str) -> float:
    """Similarity in [0, 1] of the continuation's head (cut to the held-out length) against
    the held-out text. Cutting to the same length means text the model adds *after* a
    verbatim continuation does not lower the score."""
    h = _norm(holdout)
    c = _norm(continuation)[: len(h)]
    if not h:
        return 0.0
    return float(Indel.normalized_similarity(c, h))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def completion_report(
    holdouts: list[str],
    continuations: list[str],
    *,
    thresholds: tuple[float, ...] = (0.9, 0.75),
    seed: int = 0,
) -> dict[str, Any]:
    if len(holdouts) != len(continuations):
        raise ValueError("holdouts and continuations must align")
    sims = np.array(
        [continuation_similarity(c, h) for c, h in zip(continuations, holdouts, strict=True)]
    )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(holdouts))
    # a derangement-ish shuffle: make sure no item is paired with itself when n > 1
    if len(perm) > 1:
        same = np.nonzero(perm == np.arange(len(perm)))[0]
        for i in same:
            j = (i + 1) % len(perm)
            perm[i], perm[j] = perm[j], perm[i]
    chance = np.array(
        [continuation_similarity(continuations[i], holdouts[perm[i]]) for i in range(len(holdouts))]
    )
    n = len(holdouts)
    out: dict[str, Any] = {
        "num_problems": n,
        "mean_similarity": float(sims.mean()) if n else 0.0,
        "chance_mean_similarity": float(chance.mean()) if n else 0.0,
        "rates": {},
    }
    for t in thresholds:
        k = int((sims >= t).sum())
        kc = int((chance >= t).sum())
        lo, hi = wilson_ci(k, n)
        out["rates"][str(t)] = {
            "count": k,
            "rate": k / n if n else 0.0,
            "wilson_lo": lo,
            "wilson_hi": hi,
            "chance_count": kc,
            "chance_rate": kc / n if n else 0.0,
        }
    out["similarities"] = sims.tolist()
    return out


def run_probe(
    sampler: Sampler,
    eval_name: str,
    *,
    frac: float = 0.6,
    max_tokens: int = 192,
    limit: int | None = None,
    seed: int = 0,
    revision: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    es = load_eval_set(eval_name, revision=revision, limit=limit, seed=seed)
    prefixes, holdouts, ids = [], [], []
    for p in es.problems:
        try:
            pre, hold = split_prefix(p.question, frac)
        except ValueError:
            continue
        prefixes.append(pre)
        holdouts.append(hold)
        ids.append(p.id)
    cfg = SamplingConfig(
        n=1,
        temperature=0.0,
        max_tokens=max_tokens,
        seed=seed,
        stop=("\n\n", "\nQuestion", "\nProblem"),
    )
    comps: list[list[Completion]] = sampler.generate(prefixes, cfg)
    continuations = [c[0].text for c in comps]
    report = completion_report(holdouts, continuations, seed=seed)
    report.update(
        {"eval": eval_name, "frac": frac, "max_tokens": max_tokens, "manifest": es.manifest()}
    )
    rows = [
        {
            "problem_id": i,
            "prefix": pre,
            "holdout": hold,
            "continuation": cont,
            "similarity": sim,
            "finish_reason": c[0].finish_reason,
        }
        for i, pre, hold, cont, sim, c in zip(
            ids, prefixes, holdouts, continuations, report["similarities"], comps, strict=True
        )
    ]
    return report, rows


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    import yaml

    from rlvr.eval.run_eval import make_sampler
    from rlvr.utils.run_info import finish_run_info, write_run_info

    p = argparse.ArgumentParser(description="Partial-prompt completion probe.")
    p.add_argument("--config", type=Path)
    p.add_argument("--model")
    p.add_argument("--revision", default=None)
    p.add_argument("--evals", nargs="+", default=["gsm8k", "math500"])
    p.add_argument("--frac", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--no-batch-invariant", dest="batch_invariant", action="store_false")
    p.add_argument("--out", type=Path)
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args(argv)
    if args.config:
        for k, v in (yaml.safe_load(args.config.read_text()) or {}).items():
            key = k.replace("-", "_")
            if getattr(args, key, None) in (None, p.get_default(key)):
                setattr(args, key, v)
    if not args.model or not args.out:
        p.error("--model and --out are required (via flags or --config)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    write_run_info(out, config=config, seed=args.seed, allow_dirty=args.allow_dirty)
    t0 = time.time()
    sampler = make_sampler(args)
    summary: dict[str, Any] = {}
    for name in args.evals:
        report, rows = run_probe(
            sampler,
            name,
            frac=args.frac,
            max_tokens=args.max_tokens,
            limit=args.limit,
            seed=args.seed,
            revision=args.revision,
        )
        d = out / name
        d.mkdir(exist_ok=True)
        with (d / "samples.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sims = report.pop("similarities")
        (d / "similarities.json").write_text(json.dumps(sims))
        (d / "metrics.json").write_text(json.dumps(report, indent=2))
        summary[name] = report
        r9 = report["rates"]["0.9"]
        print(
            f"{name}: completed@0.9 = {r9['count']}/{report['num_problems']} "
            f"({100 * r9['rate']:.1f} %, Wilson [{100 * r9['wilson_lo']:.1f}, {100 * r9['wilson_hi']:.1f}]), "
            f"chance {100 * r9['chance_rate']:.1f} %; mean similarity {report['mean_similarity']:.3f} "
            f"vs chance {report['chance_mean_similarity']:.3f}"
        )
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    finish_run_info(out, wall_clock_s=time.time() - t0, evals=list(summary))


if __name__ == "__main__":
    main()
