"""An outcome reward model on the laptop (EXP-022): build labelled pairs from a solve-rate sample
set, train a LoRA head on a small base model, score sample sets for best-of-N and weighted vote.

    python -m rlvr.train.orm build --samples ../hub_samples/sft_1.7b_rft_n8 --evals gsm8k_train math_train \\
        --out outputs/data/orm --per-problem 2 --heldout-frac 0.1 --max-examples 20000
    python -m rlvr.train.orm train --data outputs/data/orm --model Qwen/Qwen3-0.6B-Base --out outputs/train/orm_0p6b
    python -m rlvr.train.orm score --orm outputs/train/orm_0p6b --samples ../hub_samples/rft_1.7b_n32_think_tag_t06 \\
        --evals gsm8k math500

Build: for every problem of the sample set (8 samples at T = 1.0 from the Phase 3 solve-rate pass,
each graded), up to ``--per-problem`` correct and ``--per-problem`` incorrect samples, the think-tag
prompt rebuilt from the problem text, a seeded held-out split by problem, a manifest with the ids'
hash. Train: the base decoder (``AutoModel``) with a LoRA adapter on every linear layer and a scalar
head on the last token's hidden state, binary cross-entropy on correct/incorrect, bf16 autocast on
CUDA, gradient checkpointing, inputs longer than ``--max-len`` keep their tail (the answer end);
held-out accuracy and AUROC after each epoch; the adapter, the head and ``run_info.json`` saved.
Score: one logit per sample into ``<samples>/<eval>/scores.jsonl`` next to ``samples.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rlvr.eval.prompts import build_prompt

# ----------------------------------------------------------------------------- build


def read_samples(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_examples(
    rows: Sequence[dict[str, Any]],
    prompt_for: Callable[[str], str | None],
    *,
    eval_name: str,
    per_problem: int = 2,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Up to ``per_problem`` correct and ``per_problem`` incorrect samples per problem, each with
    its prompt; problems without a prompt (not in the loaded set) are skipped and counted."""
    by_pid: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_pid.setdefault(r["problem_id"], []).append(r)
    rng = random.Random(seed)
    out = []
    for pid in sorted(by_pid):
        prompt = prompt_for(pid)
        if prompt is None:
            continue
        for label in (True, False):
            cands = [r for r in by_pid[pid] if bool(r["correct"]) is label]
            rng.shuffle(cands)
            for r in cands[:per_problem]:
                out.append(
                    {
                        "problem_id": pid,
                        "eval": eval_name,
                        "sample_idx": r["sample_idx"],
                        "prompt": prompt,
                        "completion": r["text"],
                        "label": int(label),
                        "num_tokens": r.get("num_tokens"),
                        "finish_reason": r.get("finish_reason"),
                    }
                )
    return out


def split_heldout(examples: list[dict[str, Any]], frac: float, seed: int) -> tuple[list, list]:
    pids = sorted({e["problem_id"] for e in examples})
    rng = random.Random(seed)
    rng.shuffle(pids)
    held = set(pids[: round(frac * len(pids))])
    return [e for e in examples if e["problem_id"] not in held], [
        e for e in examples if e["problem_id"] in held
    ]


def ids_hash(examples: Sequence[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for e in examples:
        h.update(f"{e['eval']}|{e['problem_id']}|{e['sample_idx']}|{e['label']}\n".encode())
    return h.hexdigest()[:16]


def write_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_build(args: argparse.Namespace) -> None:
    from rlvr.eval.datasets import load_eval_set

    train_all, held_all = [], []
    counts: dict[str, Any] = {}
    for ev in args.evals:
        es = load_eval_set(ev, revision=args.revision)
        prompts = {p.id: build_prompt(p, "think_tag", ev).text for p in es.problems}
        rows = read_samples(Path(args.samples) / ev / "samples.jsonl")
        ex = build_examples(
            rows, prompts.get, eval_name=ev, per_problem=args.per_problem, seed=args.seed
        )
        tr, he = split_heldout(ex, args.heldout_frac, args.seed)
        counts[ev] = {
            "samples": len(rows),
            "examples": len(ex),
            "train": len(tr),
            "heldout": len(he),
            "positives": sum(e["label"] for e in ex),
        }
        train_all += tr
        held_all += he
    rng = random.Random(args.seed)
    rng.shuffle(train_all)
    if args.max_examples and len(train_all) > args.max_examples:
        train_all = train_all[: args.max_examples]
    out = Path(args.out)
    write_jsonl(train_all, out / "train.jsonl")
    write_jsonl(held_all, out / "heldout.jsonl")
    manifest = {
        "samples": str(args.samples),
        "evals": list(args.evals),
        "per_problem": args.per_problem,
        "heldout_frac": args.heldout_frac,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "counts": counts,
        "train": len(train_all),
        "train_positives": sum(e["label"] for e in train_all),
        "heldout": len(held_all),
        "heldout_positives": sum(e["label"] for e in held_all),
        "train_hash": ids_hash(train_all),
        "heldout_hash": ids_hash(held_all),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest, indent=2))


# ----------------------------------------------------------------------------- model


def tail_ids(tok, text: str, max_len: int) -> list[int]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids[-max_len:] if len(ids) > max_len else ids


def collate(tok, batch_ids: list[list[int]], device):
    import torch

    width = max(len(x) for x in batch_ids)
    pad = tok.pad_token_id
    input_ids = torch.full((len(batch_ids), width), pad, dtype=torch.long)
    attn = torch.zeros((len(batch_ids), width), dtype=torch.long)
    for i, ids in enumerate(batch_ids):  # left padding: the last position is the last real token
        input_ids[i, width - len(ids) :] = torch.tensor(ids)
        attn[i, width - len(ids) :] = 1
    return input_ids.to(device), attn.to(device)


class RewardModel:
    """A decoder with a LoRA adapter and a scalar head on the last token's hidden state."""

    def __init__(self, base, head, tok, max_len: int):
        self.base, self.head, self.tok, self.max_len = base, head, tok, max_len

    @classmethod
    def new(
        cls,
        model_id: str,
        *,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        max_len: int,
        dtype,
        device,
        gradient_checkpointing: bool,
    ):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModel.from_pretrained(model_id, dtype=dtype)
        if gradient_checkpointing:
            base.gradient_checkpointing_enable()
            base.config.use_cache = False
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()
        base = get_peft_model(
            base,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules="all-linear",
            ),
        )
        head = torch.nn.Linear(base.config.hidden_size, 1)
        base.to(device)
        head.to(device)
        return cls(base, head, tok, max_len)

    @classmethod
    def load(cls, orm_dir: Path, *, dtype, device):
        import torch
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer

        cfg = json.loads((Path(orm_dir) / "orm_config.json").read_text(encoding="utf-8"))
        tok = AutoTokenizer.from_pretrained(cfg["model"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModel.from_pretrained(cfg["model"], dtype=dtype)
        adapter_dir = Path(orm_dir) / "adapter"
        if (adapter_dir / "adapter_config.json").exists() and any(
            adapter_dir.glob("adapter_model.*")
        ):
            base = PeftModel.from_pretrained(base, str(adapter_dir))
        else:  # the plain state dict written before save_pretrained (see ``save``)
            from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

            base = get_peft_model(
                base,
                LoraConfig(
                    r=int(cfg["lora_r"]),
                    lora_alpha=int(cfg["lora_alpha"]),
                    lora_dropout=0.0,
                    target_modules="all-linear",
                ),
            )
            state = torch.load(Path(orm_dir) / "adapter_state.pt", map_location="cpu")
            set_peft_model_state_dict(base, state)
        head = torch.nn.Linear(base.config.hidden_size, 1)
        head.load_state_dict(torch.load(Path(orm_dir) / "head.pt", map_location="cpu"))
        base.to(device).eval()
        head.to(device).eval()
        return cls(base, head, tok, int(cfg["max_len"]))

    def logits(self, input_ids, attn):
        hidden = self.base(input_ids=input_ids, attention_mask=attn).last_hidden_state
        return self.head(hidden[:, -1, :].float()).squeeze(-1)

    def trainable(self):
        return [p for p in self.base.parameters() if p.requires_grad] + list(self.head.parameters())

    def save(self, out: Path, model_id: str, extra: dict[str, Any]) -> None:
        """Local files first, the PEFT folder last. On 23 Oct 2026 PEFT's ``save_pretrained`` made a
        Hub lookup after writing only its README, the DNS step hung inside WSL (a netlink dump in
        uninterruptible sleep) and 6.4 hours of training were lost with the weights in memory. Now
        the head and the adapter's state dict are written with plain ``torch.save`` before anything
        that can touch the network, and ``load`` falls back to them."""
        import torch
        from peft import get_peft_model_state_dict

        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), out / "head.pt")
        torch.save(
            {k: v.detach().cpu() for k, v in get_peft_model_state_dict(self.base).items()},
            out / "adapter_state.pt",
        )
        cfg = {"model": model_id, "max_len": self.max_len, **extra}
        (out / "orm_config.json").write_text(
            json.dumps(cfg, indent=2), encoding="utf-8", newline="\n"
        )
        self.base.save_pretrained(out / "adapter")


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-based AUROC (the Mann-Whitney statistic), ties averaged."""
    pairs = sorted(zip(scores, labels, strict=True), key=lambda x: x[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs, strict=True) if y)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def evaluate(
    rm: RewardModel, examples: Sequence[dict[str, Any]], *, batch: int, device, autocast
) -> dict[str, Any]:
    import torch

    scores, labels, losses = [], [], []
    bce = torch.nn.BCEWithLogitsLoss()
    rm.base.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch):
            chunk = examples[start : start + batch]
            ids = [tail_ids(rm.tok, e["prompt"] + e["completion"], rm.max_len) for e in chunk]
            input_ids, attn = collate(rm.tok, ids, device)
            with autocast():
                lg = rm.logits(input_ids, attn)
            y = torch.tensor([float(e["label"]) for e in chunk], device=lg.device)
            losses.append(float(bce(lg, y)))
            scores += lg.float().cpu().tolist()
            labels += [int(e["label"]) for e in chunk]
    rm.base.train()
    acc = sum(int((s > 0) == bool(y)) for s, y in zip(scores, labels, strict=True)) / max(
        1, len(scores)
    )
    return {
        "n": len(scores),
        "accuracy": acc,
        "auroc": auroc(scores, labels),
        "loss": sum(losses) / max(1, len(losses)),
        "positives": sum(labels),
    }


def cmd_train(args: argparse.Namespace) -> None:
    import torch

    from rlvr.utils.run_info import finish_run_info, write_run_info

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"}
    write_run_info(out, config=cfg, seed=args.seed, allow_dirty=args.allow_dirty)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    use_bf16 = device == "cuda" and args.bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if use_bf16
        else (lambda: torch.autocast("cpu", enabled=False))
    )
    train = read_samples(Path(args.data) / "train.jsonl")
    held = read_samples(Path(args.data) / "heldout.jsonl")
    if args.limit:
        train, held = train[: args.limit], held[: max(1, args.limit // 5)]
    torch.manual_seed(args.seed)
    rm = RewardModel.new(
        args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_len=args.max_len,
        dtype=dtype,
        device=device,
        gradient_checkpointing=args.gradient_checkpointing and device == "cuda",
    )
    params = rm.trainable()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / (args.batch * args.grad_accum))
    total = steps_per_epoch * args.epochs
    warmup = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    bce = torch.nn.BCEWithLogitsLoss()
    rng = random.Random(args.seed)
    log: list[dict[str, Any]] = []
    t0 = time.time()
    step = 0
    n_trunc = (
        sum(
            1
            for e in train
            if len(rm.tok(e["prompt"] + e["completion"], add_special_tokens=False)["input_ids"])
            > args.max_len
        )
        if args.count_truncated
        else None
    )
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        rm.base.train()
        micro = 0
        running = 0.0
        for start in range(0, len(order), args.batch):
            chunk = [train[i] for i in order[start : start + args.batch]]
            ids = [tail_ids(rm.tok, e["prompt"] + e["completion"], rm.max_len) for e in chunk]
            input_ids, attn = collate(rm.tok, ids, device)
            with autocast():
                lg = rm.logits(input_ids, attn)
            y = torch.tensor([float(e["label"]) for e in chunk], device=lg.device)
            loss = bce(lg, y) / args.grad_accum
            loss.backward()
            running += float(loss.detach()) * args.grad_accum
            micro += 1
            if micro % args.grad_accum == 0 or start + args.batch >= len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0 or step == total:
                    rec = {
                        "step": step,
                        "epoch": epoch,
                        "loss": running / max(1, micro),
                        "lr": sched.get_last_lr()[0],
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    log.append(rec)
                    print(json.dumps(rec), flush=True)
                    running, micro = 0.0, 0
                if args.max_steps and step >= args.max_steps:
                    break
        # weights to disk before the held-out pass, so nothing after training can lose them
        rm.save(
            out,
            args.model,
            {"lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "data": str(args.data)},
        )
        ev = evaluate(rm, held, batch=args.batch, device=device, autocast=autocast)
        ev.update({"epoch": epoch, "step": step})
        log.append({"heldout": ev})
        print("heldout:", json.dumps(ev), flush=True)
        if args.max_steps and step >= args.max_steps:
            break
    with (out / "train_log.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for rec in log:
            f.write(json.dumps(rec) + "\n")
    final = next((r["heldout"] for r in reversed(log) if "heldout" in r), None)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "heldout": final,
                "train_examples": len(train),
                "heldout_examples": len(held),
                "steps": step,
                "truncated_inputs": n_trunc,
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    finish_run_info(
        out,
        train_runtime_s=time.time() - t0,
        global_steps=step,
        heldout=final,
        train_examples=len(train),
        peak_memory_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if device == "cuda"
        else None,
    )
    print(f"saved {out}; heldout {final}")


def cmd_score(args: argparse.Namespace) -> None:
    import torch

    from rlvr.eval.datasets import load_eval_set

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    use_bf16 = device == "cuda" and args.bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if use_bf16
        else (lambda: torch.autocast("cpu", enabled=False))
    )
    rm = RewardModel.load(Path(args.orm), dtype=dtype, device=device)
    for ev in args.evals:
        rows = read_samples(Path(args.samples) / ev / "samples.jsonl")
        es = load_eval_set(ev, revision=args.revision, countdown_n=args.countdown_n)
        prompts = {p.id: build_prompt(p, "think_tag", ev).text for p in es.problems}
        missing = [r["problem_id"] for r in rows if r["problem_id"] not in prompts]
        if missing:
            raise SystemExit(
                f"{ev}: {len(set(missing))} problem ids not in the loaded set, e.g. {missing[0]!r}"
            )
        out_path = Path(args.samples) / ev / args.out_name
        t0 = time.time()
        with out_path.open("w", encoding="utf-8", newline="\n") as f, torch.no_grad():
            for start in range(0, len(rows), args.batch):
                chunk = rows[start : start + args.batch]
                ids = [
                    tail_ids(rm.tok, prompts[r["problem_id"]] + r["text"], rm.max_len)
                    for r in chunk
                ]
                input_ids, attn = collate(rm.tok, ids, device)
                with autocast():
                    lg = rm.logits(input_ids, attn).float().cpu().tolist()
                for r, s in zip(chunk, lg, strict=True):
                    f.write(
                        json.dumps(
                            {
                                "problem_id": r["problem_id"],
                                "sample_idx": r["sample_idx"],
                                "score": s,
                                "prob": 1 / (1 + math.exp(-s)),
                                "correct": bool(r["correct"]),
                            }
                        )
                        + "\n"
                    )
        print(
            f"{ev}: scored {len(rows)} samples -> {out_path} in {time.time() - t0:.0f} s",
            flush=True,
        )


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--samples", type=Path, required=True)
    b.add_argument("--evals", nargs="+", default=["gsm8k_train", "math_train"])
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--per-problem", type=int, default=2)
    b.add_argument("--heldout-frac", type=float, default=0.1)
    b.add_argument("--max-examples", type=int, default=20000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--revision", default=None)
    b.set_defaults(func=cmd_build)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--lora-alpha", type=int, default=32)
    t.add_argument("--lora-dropout", type=float, default=0.05)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--grad-accum", type=int, default=4)
    t.add_argument("--max-len", type=int, default=2048)
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--bf16", action="store_true")
    t.add_argument("--cpu", action="store_true")
    t.add_argument(
        "--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false"
    )
    t.add_argument("--count-truncated", action="store_true")
    t.add_argument("--allow-dirty", action="store_true")
    t.set_defaults(func=cmd_train)
    s = sub.add_parser("score")
    s.add_argument("--orm", type=Path, required=True)
    s.add_argument("--samples", type=Path, required=True)
    s.add_argument("--evals", nargs="+", default=["gsm8k", "math500"])
    s.add_argument("--out-name", default="scores.jsonl")
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--bf16", action="store_true")
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--revision", default=None)
    s.add_argument("--countdown-n", type=int, default=500)
    s.set_defaults(func=cmd_score)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
