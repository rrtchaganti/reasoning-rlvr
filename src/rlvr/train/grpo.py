"""``rlvr-grpo``: GRPO with a verifiable reward on the learnable prompts of EXP-009 (Phase 4).

A thin wrapper around TRL's ``GRPOTrainer`` in the style of ``rlvr-sft``: a YAML config with
``--set`` overrides, the prompt set rebuilt with the shared think-tag template and checked against
the committed manifest hash, the reward of ``docs/phase-4-plan.md`` section 1 (1 only when the
completion finished, has the format and is correct; the three parts logged separately), the
ablation switches of PLAN.md Phase 4 step 5 (``kl_beta``, ``loss_type`` + ``scale_rewards``,
``dynamic_sampling``), a reward-hacking watch, and ``run_info.json`` + ``train_log.jsonl`` +
``reward_log.jsonl`` for the ledger and the report.

    rlvr-grpo --config configs/train/grpo_1p7b.yaml                       # EXP-012
    rlvr-grpo --config configs/train/grpo_1p7b.yaml --set seed=1 --set out=outputs/train/grpo_1p7b_seed1
    rlvr-grpo --config configs/train/grpo_1p7b.yaml --set kl_beta=0 ...   # EXP-014a
    rlvr-grpo ... --set dynamic_sampling=true --set epsilon_high=0.28    # EXP-014b
    rlvr-grpo ... --set loss_type=dr_grpo --set scale_rewards=none       # EXP-014c

What TRL's names mean here (trl 1.15): ``loss_type='grpo'`` is the published per-sequence
normalisation (the main run), ``'dr_grpo'`` the constant normalisation of Liu et al. 2025,
``'dapo'`` the token-level one; ``scale_rewards='group'`` divides advantages by the group std,
``'none'`` does not (Dr. GRPO). With ``num_iterations=1`` and one optimiser step per generation
the importance ratio is exactly 1 and the clip (``epsilon``, ``epsilon_high``) never activates;
``clip_ratio`` is logged so this is visible. Dynamic sampling (DAPO) is not in TRL; it is
implemented below by over-generating ``oversample`` x the prompts and keeping the first
``num_prompts_per_step`` groups whose rewards differ, so every optimiser step sees the same number
of completions and the extra work is sampling and scoring only. The dev-split evaluation every 50
steps is done on the saved checkpoints by ``rlvr-eval`` (``scripts/phase4_rental.sh``), not inside
the trainer, so the GPU runs one job at a time.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from rlvr.eval.datasets import Problem
from rlvr.eval.extraction import extract_answer, grade, normalize_for_vote, tag_format_ok
from rlvr.eval.prompts import build_prompt

DEFAULTS: dict[str, Any] = {
    "model": "outputs/train/rft_1p7b/final",  # the policy's start; the reference is a frozen copy
    "tokenizer": None,  # defaults to model
    "prompts": "outputs/data/rft/learnable_prompts.jsonl",
    "manifest": "data/manifests/rft_iter1.json",  # its learnable_hash must match the file
    "skip_manifest_check": False,
    "gsm8k_fraction": 0.2,  # PLAN.md Phase 4 step 3: MATH L1-3 in full + a 20 % slice of GSM8K
    "prompt_seed": 0,
    "limit_prompts": None,  # smoke runs only
    "num_prompts_per_step": 32,
    "num_generations": 8,
    "max_new_tokens": 1536,
    "temperature": 1.0,
    "top_p": 1.0,
    "learning_rate": 1.0e-6,
    "lr_scheduler": "constant_with_warmup",
    "warmup_steps": 10,
    "weight_decay": 0.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "max_grad_norm": 1.0,
    "max_steps": 200,
    "kl_beta": 0.04,  # 0 = ablation A1 (the reference forward still runs so the KL is logged)
    "epsilon": 0.2,
    "epsilon_high": None,  # 0.28 = DAPO clip-higher; inert at one update per batch
    "loss_type": "grpo",  # grpo | dr_grpo | dapo
    "scale_rewards": "group",  # group | none
    "dynamic_sampling": False,  # ablation A2
    "oversample": 2,  # prompts generated per step as a multiple of num_prompts_per_step (A2 only)
    "per_device_batch": 32,  # micro-batch of completions; grad_accum = completions_per_step / this
    "use_vllm": True,
    "vllm_gpu_memory_utilization": 0.35,
    "attn_implementation": "flash_attention_2",
    "load_dtype": "float32",  # fp32 master weights under bf16 autocast, as every run of the study
    "bf16": True,
    "gradient_checkpointing": True,
    "lora": None,  # {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"}: local checks only
    "save_steps": 50,
    "logging_steps": 1,
    "seed": 0,
    "report_to": "wandb",
    "run_name": None,
    "out": "outputs/train/grpo_1p7b",
    "use_cpu": False,
}

ANSWER_TYPES = {"gsm8k_train": "numeric", "math_train": "latex"}
LOSS_TYPES = ("grpo", "dr_grpo", "dapo")
SCALE_REWARDS = ("group", "none")
WAIT_RE = re.compile(r"\bwait\b", re.IGNORECASE)


# --------------------------------------------------------------------------- config


def parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"override must look like key=value, got {item!r}")
    k, v = item.split("=", 1)
    return k.strip(), yaml.safe_load(v)


def coerce_like_default(key: str, value: Any) -> Any:
    """Numeric knobs written as strings (YAML 1.1's `1e-6`) become the number the default is."""
    default = DEFAULTS.get(key)
    if (
        isinstance(value, str)
        and isinstance(default, (int, float))
        and not isinstance(default, bool)
    ):
        try:
            return int(value) if isinstance(default, int) else float(value)
        except ValueError as e:
            raise ValueError(f"{key} expects a number, got {value!r}") from e
    if key in ("epsilon_high", "limit_prompts") and isinstance(value, str):
        return None if value.lower() in ("none", "null", "") else float(value)
    return value


def load_config(path: str | Path | None, overrides: Sequence[str] = ()) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(loaded) - set(DEFAULTS)
        if unknown:
            raise KeyError(f"unknown config keys {sorted(unknown)}; known: {sorted(DEFAULTS)}")
        cfg.update({k: coerce_like_default(k, v) for k, v in loaded.items()})
    for item in overrides:
        k, v = parse_override(item)
        if k not in DEFAULTS:
            raise KeyError(f"unknown override {k!r}")
        cfg[k] = coerce_like_default(k, v)
    if cfg["tokenizer"] is None:
        cfg["tokenizer"] = cfg["model"]
    if cfg["epsilon_high"] is not None:
        cfg["epsilon_high"] = float(cfg["epsilon_high"])
    if cfg["limit_prompts"] is not None:
        cfg["limit_prompts"] = int(cfg["limit_prompts"])
    validate_config(cfg)
    return cfg


def completions_per_step(cfg: dict[str, Any]) -> int:
    return int(cfg["num_prompts_per_step"]) * int(cfg["num_generations"])


def validate_config(cfg: dict[str, Any]) -> None:
    if cfg["loss_type"] not in LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got {cfg['loss_type']!r}")
    if cfg["scale_rewards"] not in SCALE_REWARDS:
        raise ValueError(f"scale_rewards must be one of {SCALE_REWARDS}")
    if int(cfg["num_generations"]) < 2:
        raise ValueError("num_generations must be at least 2 (a group needs a baseline)")
    n = completions_per_step(cfg)
    if n % int(cfg["per_device_batch"]) != 0:
        raise ValueError(
            f"completions per step {n} must be a multiple of per_device_batch {cfg['per_device_batch']}"
        )
    if int(cfg["per_device_batch"]) % int(cfg["num_generations"]) != 0 and n != int(
        cfg["per_device_batch"]
    ):
        # TRL splits the generation batch into micro-batches; keeping whole groups per micro-batch
        # is not required for the loss but keeps the logged per-micro-batch numbers readable
        pass
    if cfg["dynamic_sampling"] and int(cfg["oversample"]) < 2:
        raise ValueError("dynamic_sampling needs oversample >= 2")
    if not 0.0 <= float(cfg["gsm8k_fraction"]) <= 1.0:
        raise ValueError("gsm8k_fraction must be in [0, 1]")


# --------------------------------------------------------------------------- prompts


def read_prompt_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def check_prompt_manifest(rows: Sequence[dict[str, Any]], manifest_path: str | Path) -> str:
    """The file must be the one the manifest describes: same count, same learnable_hash."""
    from rlvr.data.rft_data import rows_hash

    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    got = rows_hash(rows, ("id", "question", "answer"))
    want = m["learnable_hash"]
    if got != want or len(rows) != int(m["num_learnable"]):
        raise RuntimeError(
            f"learnable prompts do not match {manifest_path}: hash {got} vs {want}, "
            f"{len(rows)} rows vs {m['num_learnable']}"
        )
    return got


def select_prompts(
    rows: Sequence[dict[str, Any]], *, gsm8k_fraction: float, seed: int, limit: int | None = None
) -> list[dict[str, Any]]:
    """MATH rows in full plus a seeded fraction of the GSM8K rows, in a deterministic order."""
    math_rows = [r for r in rows if r["source"] == "math_train"]
    gsm_rows = [r for r in rows if r["source"] == "gsm8k_train"]
    other = [r for r in rows if r["source"] not in ANSWER_TYPES]
    if other:
        raise ValueError(f"unknown prompt sources: {sorted({r['source'] for r in other})}")
    k = round(gsm8k_fraction * len(gsm_rows))
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(gsm_rows)), k)) if k < len(gsm_rows) else range(len(gsm_rows))
    chosen = math_rows + [gsm_rows[i] for i in idx]
    rng.shuffle(chosen)  # the trainer shuffles again per epoch; this fixes the file order's effect
    if limit is not None:
        chosen = chosen[:limit]
    return chosen


def prompt_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dataset rows for TRL: the think-tag prompt text plus what the reward needs."""
    out = []
    for r in rows:
        problem = Problem(r["id"], r["question"], str(r["answer"]), {"level": r.get("level")})
        p = build_prompt(problem, "think_tag", r["source"])
        out.append(
            {
                "prompt": p.text,
                "prefix": p.completion_prefix,
                "answer": str(r["answer"]),
                "answer_type": ANSWER_TYPES[r["source"]],
                "problem_id": r["id"],
                "source": r["source"],
                "level": -1 if r.get("level") is None else int(r["level"]),
            }
        )
    return out


def stop_strings() -> list[str]:
    return list(build_prompt(Problem("x", "q", "1"), "think_tag", "gsm8k_train").stop)


# --------------------------------------------------------------------------- reward


def score_completion(
    text: str,
    completion_ids: Sequence[int],
    *,
    prefix: str,
    answer: str,
    answer_type: str,
    eos_ids: set[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    """The three parts of the reward for one completion, plus what the hacking watch reads."""
    finished = not (len(completion_ids) >= max_new_tokens and completion_ids[-1] not in eos_ids)
    full = prefix + text
    fmt = tag_format_ok(full)
    ext = extract_answer(full, answer_type)  # type: ignore[arg-type]
    correct = grade(ext.answer, answer, answer_type)  # type: ignore[arg-type]
    return {
        "finished": finished,
        "format_ok": fmt,
        "correct": bool(correct),
        "reward": 1.0 if (finished and fmt and correct) else 0.0,
        "extracted": ext.answer,
        "num_tokens": len(completion_ids),
        "wait_count": len(WAIT_RE.findall(text)),
    }


def group_stats(records: Sequence[dict[str, Any]], group_size: int) -> dict[str, float]:
    """Per-call statistics for the log: rates, length, and the reward-hacking signals."""
    n = max(len(records), 1)
    rate = lambda key: sum(1 for r in records if r[key]) / n  # noqa: E731
    repeated = 0.0
    zero_var = 0.0
    groups = [records[i : i + group_size] for i in range(0, len(records), group_size)]
    groups = [g for g in groups if len(g) == group_size]
    if groups:
        for g in groups:
            answers = Counter(normalize_for_vote(r["extracted"] or "") for r in g)
            top = answers.most_common(1)[0][1]
            repeated += 1.0 if top >= group_size - 1 else 0.0
            rewards = {r["reward"] for r in g}
            zero_var += 1.0 if len(rewards) == 1 else 0.0
        repeated /= len(groups)
        zero_var /= len(groups)
    return {
        "reward": sum(r["reward"] for r in records) / n,
        "finished_rate": rate("finished"),
        "format_rate": rate("format_ok"),
        "correct_rate": rate("correct"),
        "format_only_rate": sum(
            1 for r in records if r["finished"] and r["format_ok"] and not r["correct"]
        )
        / n,
        "mean_tokens": sum(r["num_tokens"] for r in records) / n,
        "p90_tokens": sorted(r["num_tokens"] for r in records)[int(0.9 * (len(records) - 1))]
        if records
        else 0.0,
        "wait_per_completion": sum(r["wait_count"] for r in records) / n,
        "repeated_answer_group_rate": repeated,
        "zero_variance_group_rate": zero_var,
    }


def make_reward_fn(
    *, eos_ids: set[int], max_new_tokens: int, group_size: int, sink: list[dict[str, Any]]
) -> Callable[..., list[float]]:
    """TRL calls ``fn(prompts=, completions=, completion_ids=, **dataset columns)`` once per
    generation batch; every call appends one statistics record to ``sink``."""

    def reward_fn(prompts, completions, completion_ids, prefix, answer, answer_type, **kwargs):
        records = [
            score_completion(
                c,
                ids,
                prefix=pf,
                answer=a,
                answer_type=t,
                eos_ids=eos_ids,
                max_new_tokens=max_new_tokens,
            )
            for c, ids, pf, a, t in zip(
                completions, completion_ids, prefix, answer, answer_type, strict=True
            )
        ]
        stats = group_stats(records, group_size)
        stats["num_completions"] = len(records)
        state = kwargs.get("trainer_state")
        stats["global_step"] = getattr(state, "global_step", None)
        sink.append(stats)
        log_metric = kwargs.get("log_metric")
        if callable(log_metric):
            for k, v in stats.items():
                if k in ("num_completions", "global_step"):
                    continue
                try:
                    log_metric(f"rlvr/{k}", float(v))
                except Exception:
                    break
        return [r["reward"] for r in records]

    reward_fn.__name__ = "finished_format_correct"
    return reward_fn


def hacking_flags(stats: Sequence[dict[str, Any]], *, window: int = 20) -> dict[str, Any]:
    """PLAN.md Phase 4 step 6, read off the per-step reward statistics at the end of a run."""
    if len(stats) < 2 * window:
        return {"evaluated": False, "reason": f"fewer than {2 * window} logged steps"}
    first, last = stats[:window], stats[-window:]
    mean = lambda rows, k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
    len0, len1 = mean(first, "mean_tokens"), mean(last, "mean_tokens")
    acc0, acc1 = mean(first, "correct_rate"), mean(last, "correct_rate")
    flags = {
        "length_growth_without_accuracy": len1 > 1.3 * len0 and acc1 < acc0 + 0.01,
        "format_only_outputs": max(r["format_only_rate"] for r in stats) > 0.2,
        "repeated_answers": max(r["repeated_answer_group_rate"] for r in stats) > 0.5,
        "wait_spam": mean(last, "wait_per_completion")
        > 3 * max(mean(first, "wait_per_completion"), 0.5),
        "truncation_rose": (1 - mean(last, "finished_rate"))
        > (1 - mean(first, "finished_rate")) + 0.05,
    }
    return {
        "evaluated": True,
        "any": any(flags.values()),
        "flags": flags,
        "first_window": {"mean_tokens": len0, "correct_rate": acc0},
        "last_window": {"mean_tokens": len1, "correct_rate": acc1},
    }


# --------------------------------------------------------------------------- dynamic sampling


def select_groups(advantages, group_size: int, num_keep: int) -> tuple[list[int], dict[str, Any]]:
    """Indices of the completions to keep: the first ``num_keep`` groups (in order) whose
    advantages are not all zero, filled with zero-advantage groups from the front when there are
    not enough. ``advantages`` is a flat sequence over consecutive groups of ``group_size``."""
    adv = [float(a) for a in advantages]
    n_groups = len(adv) // group_size
    informative = [
        g
        for g in range(n_groups)
        if any(abs(a) > 1e-8 for a in adv[g * group_size : (g + 1) * group_size])
    ]
    chosen = informative[:num_keep]
    filled = 0
    if len(chosen) < num_keep:
        for g in range(n_groups):
            if g not in chosen:
                chosen.append(g)
                filled += 1
                if len(chosen) == num_keep:
                    break
    chosen.sort()
    idx = [g * group_size + j for g in chosen for j in range(group_size)]
    stats = {
        "groups_generated": n_groups,
        "groups_informative": len(informative),
        "informative_fraction": len(informative) / max(n_groups, 1),
        "groups_filled_with_zero_advantage": filled,
    }
    return idx, stats


def make_trainer_class(dynamic: bool, oversample: int, num_prompts: int, group_size: int):
    """The trainer class: TRL's, or a subclass that over-generates and keeps informative groups."""
    from trl import GRPOTrainer

    if not dynamic:
        return GRPOTrainer

    class DynamicSamplingGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._dyn_rng = random.Random(0)
            self.dyn_log: list[dict[str, Any]] = []

        def _generate_and_score_completions(self, inputs):
            import torch

            if not self.model.training or len(inputs) != num_prompts * group_size:
                return super()._generate_and_score_completions(inputs)
            # the sampler repeats each prompt group_size times consecutively; add more groups
            extra_prompts = (oversample - 1) * num_prompts
            pool = self.train_dataset
            seen = {inp["prompt"] for inp in inputs}
            picks: list[int] = []
            tries = 0
            while len(picks) < extra_prompts and tries < 20 * extra_prompts:
                i = self._dyn_rng.randrange(len(pool))
                tries += 1
                if pool[i]["prompt"] in seen:
                    continue
                seen.add(pool[i]["prompt"])
                picks.append(i)
            extended = list(inputs) + [dict(pool[i]) for i in picks for _ in range(group_size)]
            out = super()._generate_and_score_completions(extended)
            n_ext = len(extended)
            idx, stats = select_groups(
                out["advantages"].detach().cpu().tolist(), group_size, num_prompts
            )
            index = torch.tensor(idx, device=out["advantages"].device)
            kept: dict[str, Any] = {}
            for k, v in out.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == n_ext:
                    kept[k] = v.index_select(0, index)
                elif isinstance(v, list) and len(v) == n_ext:
                    kept[k] = [v[i] for i in idx]
                else:
                    kept[k] = v
            if "completion_mask" in kept and "num_items_in_batch" in kept:
                kept["num_items_in_batch"] = kept["completion_mask"].sum()
            stats["extra_prompts_sampled"] = len(picks)
            stats["global_step"] = self.state.global_step
            self.dyn_log.append(stats)
            for k, v in stats.items():
                if k != "global_step":
                    self._metrics["train"][f"dyn/{k}"].append(float(v))
            return kept

    return DynamicSamplingGRPOTrainer


# --------------------------------------------------------------------------- trainer config


def make_grpo_config(cfg: dict[str, Any], out_dir: Path):
    from trl import GRPOConfig

    n = completions_per_step(cfg)
    grad_accum = n // int(cfg["per_device_batch"])
    kwargs: dict[str, Any] = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": int(cfg["per_device_batch"]),
        "gradient_accumulation_steps": grad_accum,
        "num_generations": int(cfg["num_generations"]),
        "max_completion_length": int(cfg["max_new_tokens"]),
        "temperature": float(cfg["temperature"]),
        "top_p": float(cfg["top_p"]),
        "learning_rate": float(cfg["learning_rate"]),
        "lr_scheduler_type": cfg["lr_scheduler"],
        "warmup_steps": int(cfg["warmup_steps"]),
        "weight_decay": float(cfg["weight_decay"]),
        "adam_beta1": float(cfg["adam_beta1"]),
        "adam_beta2": float(cfg["adam_beta2"]),
        "max_grad_norm": float(cfg["max_grad_norm"]),
        "max_steps": int(cfg["max_steps"]),
        "num_train_epochs": 1000,  # max_steps governs; the prompt set is re-shuffled per pass
        "beta": float(cfg["kl_beta"]),
        "epsilon": float(cfg["epsilon"]),
        "epsilon_high": None if cfg["epsilon_high"] is None else float(cfg["epsilon_high"]),
        "loss_type": cfg["loss_type"],
        "scale_rewards": cfg["scale_rewards"],
        "num_iterations": 1,
        "use_vllm": bool(cfg["use_vllm"]),
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": float(cfg["vllm_gpu_memory_utilization"]),
        "bf16": bool(cfg["bf16"]),
        "gradient_checkpointing": bool(cfg["gradient_checkpointing"]),
        "save_steps": int(cfg["save_steps"]),
        "save_strategy": "steps",
        "save_only_model": True,  # weights only: the dev evals read them; no resume by design
        "logging_steps": int(cfg["logging_steps"]),
        "report_to": cfg["report_to"],
        "run_name": cfg["run_name"] or out_dir.name,
        "seed": int(cfg["seed"]),
        "remove_unused_columns": False,  # answer, answer_type, prefix reach the reward function
        "log_completions": True,
        "num_completions_to_print": 2,
        "mask_truncated_completions": False,  # a truncated completion earns 0 instead
        "shuffle_dataset": True,
        "use_cpu": bool(cfg["use_cpu"]),
        "dataloader_num_workers": 0,
    }
    if cfg["use_vllm"]:
        kwargs["generation_kwargs"] = {"stop": stop_strings()}
    return GRPOConfig(**kwargs)


# --------------------------------------------------------------------------- run


def run(cfg: dict[str, Any], *, allow_dirty: bool = False) -> dict[str, Any]:
    import torch

    from rlvr.train.sft import load_model_and_tokenizer, make_peft_config
    from rlvr.utils.run_info import finish_run_info, write_run_info

    out_dir = Path(cfg["out"])
    rows = read_prompt_rows(cfg["prompts"])
    prompt_hash = (
        None if cfg["skip_manifest_check"] else check_prompt_manifest(rows, cfg["manifest"])
    )
    chosen = select_prompts(
        rows,
        gsm8k_fraction=cfg["gsm8k_fraction"],
        seed=cfg["prompt_seed"],
        limit=cfg["limit_prompts"],
    )
    records = prompt_records(chosen)
    info = write_run_info(
        out_dir,
        config=cfg,
        seed=cfg["seed"],
        allow_dirty=allow_dirty,
        extra={
            "kind": "grpo",
            "prompt_file_hash": prompt_hash,
            "num_prompts_total": len(rows),
            "num_prompts_used": len(records),
            "prompts_by_source": dict(Counter(r["source"] for r in records)),
            "completions_per_step": completions_per_step(cfg),
            "clip_active": False,  # one update per generation batch: ratio == 1
        },
    )
    (out_dir / "prompt_ids.json").write_text(json.dumps([r["problem_id"] for r in records]))

    model, tok = load_model_and_tokenizer(cfg)
    eos_ids = {tok.eos_token_id}
    if tok.pad_token_id is not None:
        eos_ids.add(tok.pad_token_id)
    sink: list[dict[str, Any]] = []
    reward_fn = make_reward_fn(
        eos_ids=eos_ids,
        max_new_tokens=int(cfg["max_new_tokens"]),
        group_size=int(cfg["num_generations"]),
        sink=sink,
    )
    from datasets import Dataset

    ds = Dataset.from_list(records)
    trainer_cls = make_trainer_class(
        bool(cfg["dynamic_sampling"]),
        int(cfg["oversample"]),
        int(cfg["num_prompts_per_step"]),
        int(cfg["num_generations"]),
    )
    trainer = trainer_cls(
        model=model,
        reward_funcs=[reward_fn],
        args=make_grpo_config(cfg, out_dir / "checkpoints"),
        train_dataset=ds,
        processing_class=tok,
        peft_config=make_peft_config(cfg["lora"]),
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0

    final_dir = out_dir / "final"
    if cfg["lora"]:
        trainer.model.save_pretrained(out_dir / "adapter")
        trainer.model.merge_and_unload().save_pretrained(final_dir)
    else:
        trainer.save_model(str(final_dir))
    tok.save_pretrained(final_dir)
    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for rec in trainer.state.log_history:
            f.write(json.dumps(rec) + "\n")
    with (out_dir / "reward_log.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for rec in sink:
            f.write(json.dumps(rec) + "\n")
    dyn_log = getattr(trainer, "dyn_log", None)
    if dyn_log:
        with (out_dir / "dynamic_sampling_log.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as f:
            for rec in dyn_log:
                f.write(json.dumps(rec) + "\n")
    summary = {
        "train_runtime_s": round(wall, 1),
        "train_loss": result.training_loss,
        "global_steps": trainer.state.global_step,
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if torch.cuda.is_available()
        else None,
        "final_model": str(final_dir),
        "reward_first_step": sink[0]["reward"] if sink else None,
        "reward_last_step": sink[-1]["reward"] if sink else None,
        "finished_rate_first_step": sink[0]["finished_rate"] if sink else None,
        "finished_rate_last_step": sink[-1]["finished_rate"] if sink else None,
        "mean_tokens_first_step": sink[0]["mean_tokens"] if sink else None,
        "mean_tokens_last_step": sink[-1]["mean_tokens"] if sink else None,
        "zero_variance_group_rate_mean": sum(s["zero_variance_group_rate"] for s in sink)
        / len(sink)
        if sink
        else None,
        "hacking_watch": hacking_flags(sink),
        "dynamic_sampling_summary": {
            "informative_fraction_mean": sum(d["informative_fraction"] for d in dyn_log)
            / len(dyn_log),
            "steps_filled": sum(1 for d in dyn_log if d["groups_filled_with_zero_advantage"]),
        }
        if dyn_log
        else None,
    }
    finish_run_info(out_dir, **summary)
    return {**info, **summary}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="GRPO with a verifiable reward (TRL) on the learnable prompts."
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.print_config:
        print(json.dumps(cfg, indent=2, default=str))
        return
    summary = run(cfg, allow_dirty=args.allow_dirty)
    print(json.dumps({k: summary[k] for k in summary if k != "pip_freeze"}, indent=2, default=str))


if __name__ == "__main__":
    main()
