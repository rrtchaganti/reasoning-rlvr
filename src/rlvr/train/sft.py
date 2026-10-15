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
- **format tags the base has never trained**: ``<think>`` and ``</think>`` are single added tokens
  in the Qwen3 tokeniser whose embedding rows in the *Base* checkpoints are the untouched filler
  vector shared by every unused row (15 Oct 2026 probe: 94 identical rows in 1.7B, 291 in 0.6B).
  A LoRA run cannot change them, so the smoke checkpoint never emitted ``</think>``. Here each such
  row is initialised to the mean of its plain-text pieces (``</``, ``think``, ``>``) before training;
  LoRA runs train exactly those rows as well (the embedding matrix gets gradients, a hook zeroes
  every other row, the matrix is excluded from weight decay; PEFT's ``trainable_token_indices`` is
  bypassed by TRL's fused LM head, which reads ``lm_head.weight`` directly); ``run_info.json``
  records how far each row moved;
- **fp32 master weights for full fine-tunes** (``load_dtype: float32`` with ``bf16: true`` autocast):
  a bf16 weight of magnitude 0.02 has a spacing of 1.2e-4, so pure-bf16 AdamW at LR 1e-5 rounds
  most updates to nothing. LoRA runs load the frozen base in bf16 (PEFT keeps adapters in fp32);
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
import re
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
    "bf16": True,  # autocast; the weights' own dtype is load_dtype
    "load_dtype": "float32",  # full fine-tunes keep fp32 master weights; LoRA configs use bfloat16
    # or {"r": 64, "alpha": 128, "dropout": 0.05, "target_modules": "all-linear", "merge": true,
    #     "train_tags": true}; train_tags also trains the embedding rows of the tag tokens below
    "lora": None,
    "tag_tokens": ["<think>", "</think>"],  # format tags that may be single, never-trained tokens
    "init_tag_tokens": True,  # initialise an untrained tag row from its plain-text pieces
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


def decay_parameter_names(names: Sequence[str], exclude_suffixes: Sequence[str]) -> list[str]:
    """Drop parameters whose name ends with one of ``exclude_suffixes`` from the weight-decay group
    (the embedding matrix when only a few of its rows are trained: decaying the frozen rows would
    silently shrink the whole vocabulary)."""
    return [n for n in names if not any(n.endswith(s) for s in exclude_suffixes)]


EMBED_SUFFIXES = ("embed_tokens.weight", "lm_head.weight")


# --------------------------------------------------------------------------- format tag tokens

_TAG_RE = re.compile(r"^(</?)([A-Za-z_]+)(>)$")


def single_token_ids(tok, tags: Sequence[str]) -> dict[str, int]:
    """The tags that the tokeniser maps to exactly one token (``<answer>`` is three pieces and
    needs nothing; ``<think>`` and ``</think>`` are one added token each in Qwen3)."""
    out: dict[str, int] = {}
    for tag in tags:
        ids = tok(tag, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            out[tag] = int(ids[0])
    return out


def plain_piece_ids(tok, tag: str) -> list[int]:
    """Token ids of the tag written as ordinary text: ``</think>`` -> pieces of ``</``, ``think``,
    ``>``. These are what the base model would have seen for the tag in plain text."""
    m = _TAG_RE.match(tag)
    parts = list(m.groups()) if m else [tag[:1], tag[1:-1], tag[-1:]]
    ids: list[int] = []
    for part in parts:
        ids.extend(int(i) for i in tok(part, add_special_tokens=False)["input_ids"])
    return ids


def duplicate_row_count(weight, idx: int) -> int:
    """How many *other* rows of an embedding matrix are bit-identical to row ``idx``. Trained rows
    are never identical; the never-trained rows of Qwen3-Base all equal one filler vector."""
    return int((weight == weight[idx]).all(dim=1).sum().item()) - 1


def init_tag_embeddings(model, tok, tags: Sequence[str], *, apply: bool = True) -> dict[str, Any]:
    """Report every single-token tag whose embedding row is untrained (identical to other rows)
    and, with ``apply``, set that row (and the output-head row if untied) to the mean of the rows of
    its plain-text pieces. Returns ``{tag: {id, duplicate_rows, untrained, initialised, ...}}``."""
    import torch

    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings()
    tied = head is None or head.weight.data_ptr() == emb.data_ptr()
    report: dict[str, Any] = {}
    with torch.no_grad():
        for tag, idx in single_token_ids(tok, tags).items():  # detect first, on untouched rows
            dups = duplicate_row_count(emb.data, idx)
            report[tag] = {
                "id": idx,
                "duplicate_rows": dups,
                "untrained": dups > 0,
                "initialised": False,
                "norm_before": round(float(emb.data[idx].float().norm()), 4),
            }
        for tag, rec in report.items():
            idx = rec["id"]
            if rec["untrained"] and apply:
                pieces = plain_piece_ids(tok, tag)
                vec = emb.data[pieces].float().mean(0).to(emb.dtype)
                emb.data[idx] = vec
                if not tied:
                    head.weight.data[idx] = vec
                rec.update(
                    initialised=True,
                    init_from=pieces,
                    norm_after_init=round(float(vec.float().norm()), 4),
                )
    return report


def enable_tag_row_training(model, ids: Sequence[int]) -> None:
    """Train only the embedding rows ``ids`` of an otherwise frozen (LoRA) model: the embedding
    matrix gets ``requires_grad`` and a hook that zeroes every other row's gradient. The tied output
    head shares the storage, so the model learns to *emit* the tags as well as read them; an untied
    head is treated the same way. Works with TRL's fused LM head because that kernel reads
    ``lm_head.weight`` directly (which is why PEFT's ``trainable_token_indices`` cannot be used)."""
    import torch

    idx = torch.as_tensor(sorted(set(int(i) for i in ids)))
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings()
    params = [emb]
    if head is not None and head.weight.data_ptr() != emb.data_ptr():
        params.append(head.weight)
    for p in params:
        p.data = p.data.float()  # fp32 master rows: bf16 would round LR-sized updates away
        mask = torch.zeros(p.shape[0], 1, dtype=p.dtype, device=p.device)
        mask[idx.to(p.device)] = 1

        def _keep_rows(grad, mask=mask):
            return grad * mask.to(device=grad.device, dtype=grad.dtype)

        p.requires_grad_(True)
        p.register_hook(_keep_rows)


def snapshot_rows(model, ids: Sequence[int]) -> dict[int, Any]:
    emb = model.get_input_embeddings().weight.data
    return {int(i): emb[int(i)].detach().float().cpu().clone() for i in ids}


def tag_training_delta(model, snapshot: dict[int, Any]) -> dict[int, float]:
    """L2 distance each tag row moved during training (0.0 means the row was frozen)."""
    emb = model.get_input_embeddings().weight.data
    return {
        i: round(float((emb[i].detach().float().cpu() - row).norm()), 4)
        for i, row in snapshot.items()
    }


def load_model_and_tokenizer(cfg: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: dict[str, Any] = {
        "attn_implementation": cfg["attn_implementation"],
        "dtype": getattr(torch, cfg["load_dtype"]),
    }
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
    tag_report = init_tag_embeddings(
        model, tok, cfg["tag_tokens"] or [], apply=bool(cfg["init_tag_tokens"])
    )
    tag_ids = [rec["id"] for rec in tag_report.values()]
    untrained = [t for t, rec in tag_report.items() if rec["untrained"]]
    if untrained and cfg["lora"] and not cfg["lora"].get("train_tags", True):
        raise RuntimeError(
            f"tag tokens {untrained} have untrained embedding rows and LoRA leaves embeddings "
            "frozen, so the model could never emit them. Set lora.train_tags: true (default)."
        )
    rows0 = snapshot_rows(model, tag_ids)
    peft_config = make_peft_config(cfg["lora"])
    train_rows = bool(cfg["lora"]) and cfg["lora"].get("train_tags", True) and bool(tag_ids)
    no_decay: tuple[str, ...] = ()
    if train_rows:
        from peft import get_peft_model

        model = get_peft_model(model, peft_config)  # adapters in fp32 (PEFT default)
        peft_config = None
        enable_tag_row_training(model, tag_ids)
        no_decay = EMBED_SUFFIXES
    for rec in tag_report.values():
        rec["trained_via"] = "masked_rows" if train_rows else ("frozen" if cfg["lora"] else "full")
    for tag, rec in tag_report.items():
        print(f"[rlvr-sft] tag token {tag!r}: {json.dumps(rec)}")

    class _Trainer(SFTTrainer):
        def get_decay_parameter_names(self, model):
            return decay_parameter_names(super().get_decay_parameter_names(model), no_decay)

    trainer = _Trainer(
        model=model,
        args=make_sft_config(cfg, out_dir / "checkpoints"),
        train_dataset=to_hf_dataset(chosen),
        processing_class=tok,
        peft_config=peft_config,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0

    final_dir = out_dir / "final"
    final_model = trainer.model
    if cfg["lora"]:
        trainer.model.save_pretrained(out_dir / "adapter")  # adapters only; tag rows live in final/
        if cfg["lora"].get("merge", True):
            final_model = trainer.model.merge_and_unload()
            if train_rows:  # back to the checkpoint dtype after fp32 row training
                emb = final_model.get_input_embeddings().weight
                emb.data = emb.data.to(getattr(torch, cfg["load_dtype"]))
                head = final_model.get_output_embeddings()
                if head is not None and head.weight.data_ptr() != emb.data_ptr():
                    head.weight.data = head.weight.data.to(emb.dtype)
            final_model.save_pretrained(final_dir)
    else:
        trainer.save_model(str(final_dir))
    tok.save_pretrained(final_dir)
    deltas = tag_training_delta(final_model, rows0)
    for rec in tag_report.values():
        rec["train_delta_norm"] = deltas[rec["id"]]
        rec["duplicate_rows_final"] = duplicate_row_count(
            final_model.get_input_embeddings().weight.data, rec["id"]
        )

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
        "tag_tokens": tag_report,
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
