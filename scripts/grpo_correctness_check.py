"""PLAN.md Phase 4 step 1: the own reference GRPO loss against TRL's, on the same completions.

    python scripts/grpo_correctness_check.py --model Qwen/Qwen3-0.6B-Base --steps 3 --out outputs/grpo_check

What it does, per generation batch of a small GRPOTrainer (HF generate, no vLLM; LoRA so it fits
the laptop's 8 GB): takes the batch TRL generated and scored (completion ids, masks, rewards and
advantages, old and reference log-probs), recomputes with the code in ``src/rlvr/ref``
  (a) the group advantages from the per-completion rewards (`group_advantages`, with and without
      std scaling) and compares them with TRL's `advantages`;
  (b) the per-token log-probs of the policy with a plain forward pass, and the loss with
      `grpo_loss` for the aggregation TRL is configured with, against TRL's `_compute_loss`;
  (c) the per-token KL (k3) against TRL's logged `kl`.
Three TRL settings are checked: the main run's (`loss_type='grpo'`, group scaling, beta 0.04),
A3's (`dr_grpo`, no scaling) and A1's (beta 0). The table it prints and writes is the evidence
the engineering notes cite before any node time. Dr. GRPO's constant: TRL divides by
B x max_completion_length; `grpo_loss` divides by B x T_padded, so the reference value is rescaled
by T_padded / max_completion_length before the comparison (both are "a constant", as the paper
intends; the difference is which constant).

Rewards come from the project's own reward (`rlvr.train.grpo.make_reward_fn`) so the completions'
correctness is graded the same way as in the real runs. 50 prompts from a seeded slice of the
GSM8K learnable prompts if the file exists, else 50 GSM8K-train problems from the loader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from rlvr.ref.advantages import group_advantages, zero_advantage_fraction
from rlvr.ref.grpo_loss import grpo_loss
from rlvr.ref.kl import kl_k3
from rlvr.train import grpo

SETTINGS = {
    "main": {"loss_type": "grpo", "scale_rewards": "group", "kl_beta": 0.04},
    "a3_drgrpo": {"loss_type": "dr_grpo", "scale_rewards": "none", "kl_beta": 0.04},
    "a1_nokl": {"loss_type": "grpo", "scale_rewards": "group", "kl_beta": 0.0},
}
AGG = {"grpo": "sequence", "dr_grpo": "drgrpo", "dapo": "token"}


def prompt_rows(n: int, seed: int) -> list[dict]:
    path = Path("outputs/data/rft/learnable_prompts.jsonl")
    if path.exists():
        rows = [r for r in grpo.read_prompt_rows(path) if r["source"] == "gsm8k_train"]
    else:
        from rlvr.eval.datasets import load_eval_set

        es = load_eval_set("gsm8k_train", limit=4 * n, seed=seed)
        rows = [
            {
                "id": p.id,
                "source": "gsm8k_train",
                "question": p.question,
                "answer": p.answer,
                "level": None,
            }
            for p in es.problems
        ]
    import random

    random.Random(seed).shuffle(rows)
    return rows[:n]


@torch.no_grad()
def policy_logps(model, input_ids, attention_mask, logits_to_keep, temperature, chunk=4):
    """Per-token log-probs of the completion tokens under ``model``: a plain forward pass, a
    few sequences at a time (the full-vocabulary logits of 32 sequences are about 10 GB)."""
    pieces = []
    for i in range(0, input_ids.size(0), chunk):
        out = model(
            input_ids=input_ids[i : i + chunk], attention_mask=attention_mask[i : i + chunk]
        )
        logits = out.logits[:, -logits_to_keep - 1 : -1, :] / temperature
        targets = input_ids[i : i + chunk, -logits_to_keep:]
        logp = torch.log_softmax(logits.float(), dim=-1)
        pieces.append(torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1))
        del out, logits, logp
    return torch.cat(pieces, dim=0)


@torch.no_grad()
def trl_loss_chunked(trainer, inputs, chunk=4) -> float:
    """TRL's own loss on the same batch, a few sequences at a time. For `loss_type='grpo'`
    (mean of per-sequence means) and `'dr_grpo'` (token sum over B x max_len) the mean over
    equal-sized chunks equals the full-batch value; the `kl` metric TRL appends per call is
    averaged the same way by the caller."""
    n = inputs["completion_ids"].size(0)
    vals = []
    for i in range(0, n, chunk):
        part = {
            k: (
                v[i : i + chunk]
                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.size(0) == n
                else v
            )
            for k, v in inputs.items()
        }
        vals.append(float(trainer._compute_loss(trainer.model, part).detach()))
        torch.cuda.empty_cache()
    return sum(vals) / len(vals)


def check_one(name, setting, args, rows) -> dict:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = grpo.load_config(
        None,
        [
            f"model={args.model}",
            "use_vllm=false",
            "report_to=none",
            f"num_prompts_per_step={args.prompts_per_step}",
            f"num_generations={args.group}",
            f"max_new_tokens={args.max_new_tokens}",
            f"per_device_batch={args.prompts_per_step * args.group}",
            f"max_steps={args.steps}",
            "gradient_checkpointing=false",
            "attn_implementation=sdpa",
            "load_dtype=bfloat16",
            f"loss_type={setting['loss_type']}",
            f"scale_rewards={setting['scale_rewards']}",
            f"kl_beta={setting['kl_beta']}",
            f"out={args.out}/{name}",
            "skip_manifest_check=true",
            f"seed={args.seed}",
        ],
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    sink: list[dict] = []
    reward_fn = grpo.make_reward_fn(
        eos_ids={tok.eos_token_id, tok.pad_token_id},
        max_new_tokens=args.max_new_tokens,
        group_size=args.group,
        sink=sink,
    )
    ds = Dataset.from_list(grpo.prompt_records(rows))
    trainer_cls = grpo.make_trainer_class(False, 2, args.prompts_per_step, args.group)
    lora = {"r": 16, "alpha": 32, "dropout": 0.0, "target_modules": "all-linear"}
    from rlvr.train.sft import make_peft_config

    trainer = trainer_cls(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo.make_grpo_config(cfg, Path(args.out) / name / "checkpoints"),
        train_dataset=ds,
        processing_class=tok,
        peft_config=make_peft_config(lora),
    )
    torch.manual_seed(args.seed)
    results = []
    loader = trainer.get_train_dataloader()
    it = iter(loader)
    trainer.model.train()
    for step in range(args.steps):
        batch = next(
            it
        )  # one generation batch: prompts_per_step prompts, each repeated group times
        inputs = trainer._generate_and_score_completions(batch)
        # the per-completion rewards, re-scored from the generated text with the project's own
        # reward (the same function TRL called), so the advantages can be recomputed independently
        completion_ids = inputs["completion_ids"]
        mask = inputs["completion_mask"]
        per = []
        for ids, m, b in zip(completion_ids, mask, batch, strict=True):
            kept_ids = ids[: int(m.sum())].tolist()
            text = tok.decode(kept_ids, skip_special_tokens=True)
            per.append(
                grpo.score_completion(
                    text,
                    kept_ids,
                    prefix=b["prefix"],
                    answer=b["answer"],
                    answer_type=b["answer_type"],
                    eos_ids={tok.eos_token_id, tok.pad_token_id},
                    max_new_tokens=args.max_new_tokens,
                )["reward"]
            )
        rewards = torch.tensor(per, dtype=torch.float32).view(-1, args.group)
        ref_adv = group_advantages(
            rewards, normalize_std=setting["scale_rewards"] == "group"
        ).flatten()
        trl_adv = inputs["advantages"].detach().float().cpu()
        adv_max_diff = float((ref_adv - trl_adv).abs().max())
        # the loss on the same completions
        torch.cuda.empty_cache()
        input_ids = torch.cat([inputs["prompt_ids"], completion_ids], dim=1)
        attn = torch.cat([inputs["prompt_mask"], mask], dim=1)
        logp = policy_logps(
            trainer.model,
            input_ids,
            attn,
            completion_ids.size(1),
            args.temperature,
            chunk=args.chunk,
        )
        logp_ref = inputs.get("ref_per_token_logps")
        beta = setting["kl_beta"]
        ref = grpo_loss(
            logp,
            logp.detach(),
            trl_adv.to(logp.device),
            mask.to(logp.dtype),
            logp_ref=None if logp_ref is None else logp_ref.to(logp.device),
            beta=beta,
            eps_low=cfg["epsilon"],
            eps_high=cfg["epsilon"],
            loss_agg=AGG[setting["loss_type"]],
        )
        ref_loss = float(ref.loss)
        if setting["loss_type"] == "dr_grpo":
            ref_loss *= completion_ids.size(1) / args.max_new_tokens
        n_kl_before = len(trainer._metrics["train"].get("kl", []))
        trl_loss = trl_loss_chunked(trainer, inputs, chunk=args.chunk)
        kl_ref = None
        if logp_ref is not None:
            kl_ref = float(
                (kl_k3(logp.detach(), logp_ref.to(logp.device)) * mask).sum() / mask.sum()
            )
        trl_logged_kl = None
        kls = trainer._metrics["train"].get("kl", [])
        if len(kls) > n_kl_before:
            new = [float(k) for k in kls[n_kl_before:]]
            trl_logged_kl = sum(new) / len(new)
        results.append(
            {
                "step": step,
                "completions": int(completion_ids.size(0)),
                "zero_advantage_fraction": zero_advantage_fraction(rewards),
                "mean_reward": float(rewards.mean()),
                "advantage_max_abs_diff": adv_max_diff,
                "ref_loss": ref_loss,
                "trl_loss": trl_loss,
                "loss_abs_diff": abs(ref_loss - trl_loss),
                "ref_kl": kl_ref,
                "trl_logged_kl": trl_logged_kl,
                "ref_clip_fraction": ref.clip_fraction,
            }
        )
        print(json.dumps({"setting": name, **results[-1]}))
    return {"setting": setting, "steps": results}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument(
        "--chunk",
        type=int,
        default=4,
        help="sequences per forward pass in the checks (8 GB laptop: 4)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num-prompts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    ap.add_argument("--out", type=Path, default=Path("outputs/grpo_check"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = prompt_rows(args.num_prompts, args.seed)
    report = {}
    for name in args.settings:
        report[name] = check_one(name, SETTINGS[name], args, rows)
    (args.out / "grpo_correctness_check.json").write_text(json.dumps(report, indent=2))
    fmt = lambda v: "-" if v is None else f"{v:.6f}"  # noqa: E731
    lines = [
        "| setting | step | completions | zero-adv groups | mean reward | adv max diff | ref loss | TRL loss | diff | ref KL | TRL KL |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, rep in report.items():
        for r in rep["steps"]:
            lines.append(
                f"| {name} | {r['step']} | {r['completions']} | {r['zero_advantage_fraction']:.2f} | "
                f"{r['mean_reward']:.3f} | {r['advantage_max_abs_diff']:.2e} | {r['ref_loss']:.6f} | "
                f"{r['trl_loss']:.6f} | {r['loss_abs_diff']:.2e} | {fmt(r['ref_kl'])} | {fmt(r['trl_logged_kl'])} |"
            )
    (args.out / "grpo_correctness_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
