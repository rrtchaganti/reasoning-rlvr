"""Supervised fine-tuning on the Phase 2 traces with TRL's ``SFTTrainer`` (``rlvr-sft``).

What this wrapper adds on top of TRL:

- a YAML config with ``key=value`` overrides, hashed into ``run_info.json`` like every eval;
- a **manifest check**: the training file's content hash must match the committed manifest, so
  a run can never silently train on a different data version (the Phase 1 lesson);
- **nested subsets** for the LR sweep (10 %) and the data-scaling ablation (25 / 50 %): one
  seeded shuffle, prefixes of it, so each subset is contained in the next;
- **loss on completion tokens only** (``completion_only_loss``) in the prompt/completion format
  written by ``rlvr-build-sft``; the prompt is the think-tag template shared with eval and RL;
- **packing refused without flash attention**: TRL only warns when best-fit packing runs on an
  attention backend that cannot read the per-sequence boundaries; with SDPA or eager the packed
  sequences attend to each other (``tests/test_ref_sft_loss.py`` shows the effect). Here it is
  an error unless ``allow_unsafe_packing: true`` is set on purpose;
- optional LoRA (PEFT), merged into the base weights at the end so vLLM evaluates a plain
  checkpoint;
- the trainer log history, wall clock, peak memory and token count written next to the model.

Usage::

    rlvr-sft --config configs/train/sft_1p7b.yaml
    rlvr-sft --config configs/train/sft_1p7b.yaml --set learning_rate=2e-5 --set out=outputs/train/x
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

FLASH_VARIANTS = {"flash_attention_2", "flash_attention_3", "kernels-community/flash-attn"}

DEFAULTS: dict[str, Any] = {
    "model": "Qwen/Qwen3-1.7B-Base",
    "tokenizer": None,  # defaults to model
    "data": "outputs/data/sft/train.jsonl",
    "manifest": "data/manifests/sft_openr1.json",
    "skip_manifest_check": False,
    "subset_fraction": 1.0,
    "subset_seed": 0,
    "limit": None,  # hard cap on examples after the subset (smoke runs)
    "max_length": 4096,
    "epochs": 2,
    "max_steps": -1,
    "learning_rate": 1.0e-5,
    "lr_scheduler": "cosine_with_min_lr",
    "min_lr_rate": 0.1,
    "warmup_ratio": 0.03,
    "weight_decay": 0.1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "max_grad_norm": 1.0,
    "per_device_batch": 4,
    "grad_accum": 16,
    "packing": True,
    "allow_unsafe_packing": False,
    "attn_implementation": "flash_attention_2",
    "gradient_checkpointing": True,
    "bf16": True,
    "lora": None,  # or {"r": 64, "alpha": 128, "dropout": 0.05, "target_modules": "all-linear", "merge": true}
    "save_steps": 500,
    "save_total_limit": 3,
    "logging_steps": 10,
    "seed": 0,
    "dataset_num_proc": 4,
    "report_to": "none",
    "run_name": None,
    "out": "outputs/train/sft_1p7b",
    "use_cpu": False,
}


# --------------------------------------------------------------------------- config


def parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"override must look like key=value, got {item!r}")
    k, v = item.split("=", 1)
    return k.strip(), yaml.safe_load(v)


def load_config(path: str | Path | None, overrides: Sequence[str] = ()) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(loaded) - set(DEFAULTS)
        if unknown:
            raise KeyError(f"unknown config keys {sorted(unknown)}; known: {sorted(DEFAULTS)}")
        cfg.update(loaded)
    for item in overrides:
        k, v = parse_override(item)
        if k not in DEFAULTS:
            raise KeyError(f"unknown override {k!r}")
        cfg[k] = v
    if cfg["tokenizer"] is None:
        cfg["tokenizer"] = cfg["model"]
    return cfg


# --------------------------------------------------------------------------- data


def nested_subset(ids: Sequence[str], fraction: float, seed: int) -> list[str]:
    """Prefix of one seeded shuffle: ``nested_subset(ids, 0.25, s)`` is contained in
    ``nested_subset(ids, 0.5, s)`` for the same seed, so the data-scaling points are nested."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    order = list(ids)
    random.Random(seed).shuffle(order)
    k = math.ceil(fraction * len(order))
    return order[:k]


def check_manifest(rows: list[dict[str, Any]], manifest_path: str | Path) -> dict[str, Any]:
    """Recompute the content hash of the training file and compare with the committed manifest."""
    from rlvr.data.sft_data import SftExample, content_hash

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    exs = [SftExample(**{k: r[k] for k in SftExample.__dataclass_fields__ if k in r}) for r in rows]
    got, want = content_hash(exs), manifest["content_hash"]
    if got != want or len(rows) != manifest["num_examples"]:
        raise RuntimeError(
            f"training file does not match {manifest_path}: hash {got} vs {want}, "
            f"{len(rows)} vs {manifest['num_examples']} examples. Rebuild with rlvr-build-sft."
        )
    return manifest


def select_rows(
    rows: list[dict[str, Any]], *, fraction: float, seed: int, limit: int | None
) -> list[dict[str, Any]]:
    by_id = {r["id"]: r for r in rows}
    chosen = nested_subset(list(by_id), fraction, seed)
    if limit is not None:
        chosen = chosen[:limit]
    return [by_id[i] for i in chosen]


def to_hf_dataset(rows: list[dict[str, Any]]):
    from datasets import Dataset

    return Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])


# --------------------------------------------------------------------------- model / trainer


def packing_is_safe(cfg: dict[str, Any]) -> bool:
    return (not cfg["packing"]) or cfg["attn_implementation"] in FLASH_VARIANTS


def make_sft_config(cfg: dict[str, Any], out_dir: Path):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg["epochs"],
        max_steps=cfg["max_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler"],
        lr_scheduler_kwargs={"min_lr_rate": cfg["min_lr_rate"]}
        if "min_lr" in cfg["lr_scheduler"]
        else None,
        warmup_steps=cfg["warmup_ratio"],  # float in [0, 1) = ratio of total steps
        weight_decay=cfg["weight_decay"],
        adam_beta1=cfg["adam_beta1"],
        adam_beta2=cfg["adam_beta2"],
        max_grad_norm=cfg["max_grad_norm"],
        per_device_train_batch_size=cfg["per_device_batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        bf16=cfg["bf16"],
        max_length=cfg["max_length"],
        packing=cfg["packing"],
        packing_strategy="bfd",
        completion_only_loss=True,
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        logging_steps=cfg["logging_steps"],
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        dataset_num_proc=cfg["dataset_num_proc"],
        report_to=cfg["report_to"],
        run_name=cfg["run_name"],
        use_cpu=cfg["use_cpu"],
        dataloader_pin_memory=not cfg["use_cpu"],
    )


def make_peft_config(lora: dict[str, Any] | None):
    if not lora:
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=int(lora.get("r", 64)),
        lora_alpha=int(lora.get("alpha", 128)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=lora.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(cfg: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: dict[str, Any] = {"attn_implementation": cfg["attn_implementation"]}
    if cfg["bf16"]:
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **kwargs)
    return model, tok


def run(cfg: dict[str, Any], *, allow_dirty: bool = False) -> dict[str, Any]:
    """Train and save; returns the summary written to ``run_info.json``."""
    import torch
    from trl import SFTTrainer

    from rlvr.data.sft_data import read_jsonl
    from rlvr.utils.run_info import finish_run_info, write_run_info

    if not packing_is_safe(cfg) and not cfg["allow_unsafe_packing"]:
        raise RuntimeError(
            f"packing=true with attn_implementation={cfg['attn_implementation']!r}: packed "
            "sequences would attend to each other. Use a flash-attention variant, set "
            "packing: false, or set allow_unsafe_packing: true knowingly."
        )

    out_dir = Path(cfg["out"])
    rows = read_jsonl(Path(cfg["data"]))
    manifest = None if cfg["skip_manifest_check"] else check_manifest(rows, cfg["manifest"])
    chosen = select_rows(
        rows, fraction=cfg["subset_fraction"], seed=cfg["subset_seed"], limit=cfg["limit"]
    )
    tokens_per_epoch = int(sum(int(r.get("n_tokens", 0)) for r in chosen))
    info = write_run_info(
        out_dir,
        config=cfg,
        seed=cfg["seed"],
        allow_dirty=allow_dirty,
        extra={
            "kind": "sft",
            "data_hash": manifest["content_hash"] if manifest else None,
            "num_examples_total": len(rows),
            "num_examples_used": len(chosen),
            "tokens_per_epoch": tokens_per_epoch,
            "packing_safe": packing_is_safe(cfg),
        },
    )
    (out_dir / "train_ids.json").write_text(json.dumps([r["id"] for r in chosen]))

    model, tok = load_model_and_tokenizer(cfg)
    trainer = SFTTrainer(
        model=model,
        args=make_sft_config(cfg, out_dir / "checkpoints"),
        train_dataset=to_hf_dataset(chosen),
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
        if cfg["lora"].get("merge", True):
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(final_dir)
    else:
        trainer.save_model(str(final_dir))
    tok.save_pretrained(final_dir)

    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for rec in trainer.state.log_history:
            f.write(json.dumps(rec) + "\n")
    summary = {
        "train_runtime_s": round(wall, 1),
        "train_loss": result.training_loss,
        "global_steps": trainer.state.global_step,
        "epochs_done": trainer.state.epoch,
        "tokens_per_epoch": tokens_per_epoch,
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if torch.cuda.is_available()
        else None,
        "final_model": str(final_dir),
    }
    finish_run_info(out_dir, **summary)
    return {**info, **summary}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="SFT with TRL on the Phase 2 traces.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    summary = run(cfg, allow_dirty=args.allow_dirty)
    print(json.dumps({k: summary[k] for k in summary if k != "pip_freeze"}, indent=2, default=str))


if __name__ == "__main__":
    main()
