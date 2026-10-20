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
import hashlib
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


def synthetic_reward(text: str) -> float:
    """A deterministic 0/1 reward from the text itself (the last hex digit of its SHA-1), so a
    group of distinct completions carries mixed rewards whatever the model knows."""
    return float(int(hashlib.sha1(text.encode("utf-8")).hexdigest()[-1], 16) % 2)


def grads_of(trainer, params, loss):
    """The gradient of ``loss`` with respect to ``params`` as one flat fp32 vector."""
    trainer.model.zero_grad(set_to_none=True)
    loss.backward()
    g = torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten()
            for p in params
        ]
    )
    trainer.model.zero_grad(set_to_none=True)
    return g


def compare_gradients(trainer, inputs, setting, cfg, args) -> dict:
    """Backward through TRL's loss and through the reference loss on the same chunks of the
    same batch; the gradients with respect to the trainable (LoRA) parameters are compared."""
    params = [p for p in trainer.model.parameters() if p.requires_grad]
    n = inputs["completion_ids"].size(0)
    trainer.current_gradient_accumulation_steps = 1
    g_trl = None
    g_ref = None
    for i in range(0, n, args.chunk):
        part = {
            k: (
                v[i : i + args.chunk]
                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.size(0) == n
                else v
            )
            for k, v in inputs.items()
        }
        loss_trl = trainer._compute_loss(trainer.model, part)
        g = grads_of(trainer, params, loss_trl)
        g_trl = g if g_trl is None else g_trl + g
        del loss_trl
        torch.cuda.empty_cache()
        comp = part["completion_ids"]
        mask = part["completion_mask"]
        input_ids = torch.cat([part["prompt_ids"], comp], dim=1)
        attn = torch.cat([part["prompt_mask"], mask], dim=1)
        out = trainer.model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, -comp.size(1) - 1 : -1, :] / args.temperature
        logp = torch.gather(
            torch.log_softmax(logits.float(), dim=-1), 2, comp.unsqueeze(-1)
        ).squeeze(-1)
        logp_ref = part.get("ref_per_token_logps")
        ref = grpo_loss(
            logp,
            logp.detach(),
            part["advantages"].float(),
            mask.to(logp.dtype),
            logp_ref=None if logp_ref is None else logp_ref.float(),
            beta=setting["kl_beta"],
            eps_low=cfg["epsilon"],
            eps_high=cfg["epsilon"],
            loss_agg=AGG[setting["loss_type"]],
        )
        loss_ref = ref.loss
        if setting["loss_type"] == "dr_grpo":
            loss_ref = loss_ref * (comp.size(1) / args.max_new_tokens)
        g = grads_of(trainer, params, loss_ref)
        g_ref = g if g_ref is None else g_ref + g
        del out, logits, logp, loss_ref
        torch.cuda.empty_cache()
    norm_trl = float(g_trl.norm())
    norm_ref = float(g_ref.norm())
    return {
        "grad_cosine": float((g_trl @ g_ref) / max(norm_trl * norm_ref, 1e-30)),
        "grad_max_abs_diff": float((g_trl - g_ref).abs().max()),
        "grad_rel_norm_diff": float((g_trl - g_ref).norm() / max(norm_trl, 1e-30)),
        "grad_norm_trl": norm_trl,
        "grad_norm_ref": norm_ref,
    }


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
    # TRL divides each micro-batch loss by the accumulation count, which only the training loop
    # sets; outside it the attribute does not exist. With 1 every chunk returns its own mean and
    # the caller's average over equal chunks is the batch value.
    trainer.current_gradient_accumulation_steps = 1
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


def build_trainer(name, setting, args, rows, *, dynamic=False):
    """A small GRPOTrainer (HF generate, LoRA) for ``setting``; the dynamic-sampling subclass
    when asked. Returns (trainer, tokenizer, reward sink, config)."""
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
    real_reward_fn = grpo.make_reward_fn(
        eos_ids={tok.eos_token_id, tok.pad_token_id},
        max_new_tokens=args.max_new_tokens,
        group_size=args.group,
        sink=sink,
    )
    if args.reward == "real":
        reward_fn = real_reward_fn
    else:
        # The base 0.6B never writes a finished, well-formed, correct think-tag answer in 256
        # tokens, so the real reward is 0 everywhere and every advantage is 0: the policy term
        # would go untested. A deterministic mixed reward (a hash bit of the text) exercises the
        # advantages, the aggregation and the gradients; the real reward's rate is still recorded.
        def reward_fn(prompts, completions, completion_ids, prefix, answer, answer_type, **kwargs):
            real_reward_fn(
                prompts, completions, completion_ids, prefix, answer, answer_type, **kwargs
            )
            return [synthetic_reward(c) for c in completions]

        reward_fn.__name__ = "synthetic_hash_bit"
    ds = Dataset.from_list(grpo.prompt_records(rows))
    trainer_cls = grpo.make_trainer_class(dynamic, 2, args.prompts_per_step, args.group)
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
    return trainer, tok, sink, cfg


def check_dynamic(args, rows) -> dict:
    """The dynamic-sampling subclass for a few generation batches: it must return exactly
    prompts_per_step x group rows, every tensor subselected alike, and log how many of the
    over-generated groups were informative."""
    trainer, _tok, _sink, _cfg = build_trainer(
        "dynamic", SETTINGS["main"], args, rows, dynamic=True
    )
    torch.manual_seed(args.seed)
    trainer.model.train()
    it = iter(trainer.get_train_dataloader())
    want = args.prompts_per_step * args.group
    results = []
    for step in range(args.steps):
        batch = next(it)
        out = trainer._generate_and_score_completions(batch)
        n = int(out["completion_ids"].size(0))
        # every row-indexed tensor must have been subselected to n rows (none left at 2n)
        tensors_ok = not any(
            isinstance(v, torch.Tensor) and v.dim() >= 1 and v.size(0) == 2 * n
            for v in out.values()
        )
        rows_ok = all(
            v.size(0) == n
            for k, v in out.items()
            if k in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "advantages")
        )
        stats = trainer.dyn_log[-1]
        rec = {
            "step": step,
            "rows_returned": n,
            "rows_expected": want,
            "shapes_consistent": bool(tensors_ok and rows_ok and n == want),
            "groups_generated": stats["groups_generated"],
            "groups_informative": stats["groups_informative"],
            "groups_filled_with_zero_advantage": stats["groups_filled_with_zero_advantage"],
            "extra_prompts_sampled": stats["extra_prompts_sampled"],
            "kept_zero_advantage_groups": int(
                sum(
                    1
                    for g in range(n // args.group)
                    if float(out["advantages"][g * args.group : (g + 1) * args.group].abs().max())
                    < 1e-8
                )
            ),
        }
        results.append(rec)
        print(json.dumps({"setting": "dynamic", **rec}))
        torch.cuda.empty_cache()
    return {"setting": {"dynamic_sampling": True, "oversample": 2}, "steps": results}


def check_one(name, setting, args, rows) -> dict:
    trainer, tok, _sink, cfg = build_trainer(name, setting, args, rows)
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
        real = []
        for ids, m, b in zip(completion_ids, mask, batch, strict=True):
            kept_ids = ids[: int(m.sum())].tolist()
            text = tok.decode(kept_ids, skip_special_tokens=True)
            real.append(
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
            per.append(real[-1] if args.reward == "real" else synthetic_reward(text))
        real_rate = sum(real) / len(real)
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
        del logp
        torch.cuda.empty_cache()
        grads = compare_gradients(trainer, inputs, setting, cfg, args)
        results.append(
            {
                "step": step,
                "completions": int(completion_ids.size(0)),
                "reward_source": args.reward,
                "real_reward_rate": real_rate,
                "zero_advantage_fraction": zero_advantage_fraction(rewards),
                "mean_reward": float(rewards.mean()),
                "advantage_max_abs_diff": adv_max_diff,
                "ref_loss": ref_loss,
                "trl_loss": trl_loss,
                "loss_abs_diff": abs(ref_loss - trl_loss),
                "ref_kl": kl_ref,
                "trl_logged_kl": trl_logged_kl,
                "ref_clip_fraction": ref.clip_fraction,
                **grads,
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
    ap.add_argument(
        "--reward",
        choices=("synthetic", "real"),
        default="synthetic",
        help="the reward TRL trains on in the check: a deterministic hash bit (mixed groups, the "
        "default) or the project's real reward (all zero on the base 0.6B at 256 tokens)",
    )
    ap.add_argument(
        "--dynamic",
        action="store_true",
        help="also run the dynamic-sampling subclass for --steps generation batches (shapes and group counts)",
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/grpo_check"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = prompt_rows(args.num_prompts, args.seed)
    report = {}
    for name in args.settings:
        report[name] = check_one(name, SETTINGS[name], args, rows)
    if args.dynamic:
        report["dynamic"] = check_dynamic(args, rows)
    (args.out / "grpo_correctness_check.json").write_text(json.dumps(report, indent=2))
    fmt = lambda v: "-" if v is None else f"{v:.6f}"  # noqa: E731
    lines = [
        "| setting | step | reward | real reward rate | zero-adv groups | mean reward | adv max diff | ref loss | TRL loss | loss diff | grad cosine | grad max diff | grad rel diff | ref KL | TRL KL |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, rep in report.items():
        if name == "dynamic":
            continue
        for r in rep["steps"]:
            lines.append(
                f"| {name} | {r['step']} | {r['reward_source']} | {r['real_reward_rate']:.3f} | "
                f"{r['zero_advantage_fraction']:.2f} | {r['mean_reward']:.3f} | {r['advantage_max_abs_diff']:.2e} | "
                f"{r['ref_loss']:.6f} | {r['trl_loss']:.6f} | {r['loss_abs_diff']:.2e} | {r['grad_cosine']:.6f} | "
                f"{r['grad_max_abs_diff']:.2e} | {r['grad_rel_norm_diff']:.2e} | {fmt(r['ref_kl'])} | {fmt(r['trl_logged_kl'])} |"
            )
    if "dynamic" in report:
        lines += [
            "",
            "| dynamic sampling | step | rows returned / expected | shapes consistent | groups generated | informative | filled with zero-advantage | kept zero-advantage |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in report["dynamic"]["steps"]:
            lines.append(
                f"| oversample 2 | {r['step']} | {r['rows_returned']} / {r['rows_expected']} | {r['shapes_consistent']} | "
                f"{r['groups_generated']} | {r['groups_informative']} | {r['groups_filled_with_zero_advantage']} | {r['kept_zero_advantage_groups']} |"
            )
    (args.out / "grpo_correctness_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
