"""The GRPO objective, written out so every ablation in PLAN.md Phase 4 is one switch.

For a batch of B completions (already flattened from [num_prompts, G]) with T padded token
positions, let

    ratio_t   = exp(logp_t - logp_old_t)                    importance ratio per token
    A         = advantage of the completion (one scalar per completion, broadcast over t)
    surr_t    = min(ratio_t * A, clip(ratio_t, 1 - eps_low, 1 + eps_high) * A)
    kl_t      = k3(logp_t, logp_ref_t)                       optional, weight beta
    obj_t     = surr_t - beta * kl_t

and the loss is the negative of an aggregate of obj_t over the completion tokens:

    "sequence"  mean_b [ (1/|y_b|) sum_t mask_bt obj_bt ]        GRPO as published
    "token"     (sum_b sum_t mask_bt obj_bt) / (sum_b |y_b|)      DAPO token-level
    "drgrpo"    (sum_b sum_t mask_bt obj_bt) / (B * T_max)        Dr. GRPO constant norm

Why these three differ: "sequence" gives every completion equal weight, so a token in a
short completion counts more than a token in a long one. With negative advantages that
penalises short wrong answers more than long wrong ones, which is the length-growth bias
Liu et al. 2025 identify. "token" pools all tokens, so long completions dominate the batch.
"drgrpo" divides by a constant, so no per-sequence or per-batch length enters the weighting.

Clip-higher (DAPO): eps_high > eps_low lets a low-probability token's ratio grow further
before being clipped, so rare correct tokens can be up-weighted; this is the mechanism
claimed to slow entropy collapse. With one gradient step per rollout batch ratio_t == 1
exactly, and clipping does nothing; the clip fraction is still reported so that this fact is
visible in the logs rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from rlvr.ref.kl import kl_k3

LossAgg = Literal["sequence", "token", "drgrpo"]


@dataclass
class GRPOLossOutput:
    """Loss plus the diagnostics PLAN.md says every run must log."""

    loss: torch.Tensor
    clip_fraction: float
    kl: float
    ratio_mean: float
    ratio_max: float
    policy_term: float
    kl_term: float


def grpo_loss(
    logp: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    logp_ref: torch.Tensor | None = None,
    beta: float = 0.0,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    loss_agg: LossAgg = "sequence",
) -> GRPOLossOutput:
    """Compute the clipped GRPO loss for a padded batch of completions.

    Args:
        logp: [B, T] log-prob of each completion token under the current policy (with grad).
        logp_old: [B, T] log-prob under the policy that generated the samples (no grad).
            Pass ``logp.detach()`` for the single-update-per-batch case.
        advantages: [B] one scalar advantage per completion.
        mask: [B, T] 1 for completion tokens, 0 for prompt and padding.
        logp_ref: [B, T] log-prob under the frozen reference; required if ``beta > 0``.
        beta: KL penalty weight (0 disables the reference model entirely: ablation A1).
        eps_low, eps_high: PPO clip range; ``eps_high > eps_low`` is DAPO clip-higher (A2).
        loss_agg: see module docstring (A3 flips "sequence" to "drgrpo").
    """
    if logp.shape != logp_old.shape or logp.shape != mask.shape:
        raise ValueError("logp, logp_old and mask must share the shape [B, T]")
    if advantages.shape != (logp.shape[0],):
        raise ValueError(f"advantages must be [B]={logp.shape[0]}, got {tuple(advantages.shape)}")
    if beta > 0 and logp_ref is None:
        raise ValueError("beta > 0 requires logp_ref")

    mask = mask.to(logp.dtype)
    adv = advantages.to(logp.dtype).unsqueeze(1)  # [B, 1]

    log_ratio = logp - logp_old.detach()
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
    surr = torch.minimum(ratio * adv, clipped * adv)

    if beta > 0:
        assert logp_ref is not None
        kl = kl_k3(logp, logp_ref.detach())
    else:
        kl = torch.zeros_like(logp)

    obj = surr - beta * kl  # maximise this

    if loss_agg == "sequence":
        per_seq_len = mask.sum(dim=1).clamp_min(1.0)
        per_seq = (obj * mask).sum(dim=1) / per_seq_len
        loss = -per_seq.mean()
    elif loss_agg == "token":
        loss = -(obj * mask).sum() / mask.sum().clamp_min(1.0)
    elif loss_agg == "drgrpo":
        loss = -(obj * mask).sum() / float(mask.shape[0] * mask.shape[1])
    else:
        raise ValueError(f"unknown loss_agg {loss_agg!r}")

    with torch.no_grad():
        n_tok = mask.sum().clamp_min(1.0)
        # a token is "clipped" when the clipped surrogate is the active (smaller) branch
        is_clipped = ((ratio < 1.0 - eps_low) | (ratio > 1.0 + eps_high)).to(logp.dtype)
        clip_fraction = float((is_clipped * mask).sum() / n_tok)
        kl_mean = float((kl * mask).sum() / n_tok)
        ratio_mean = float((ratio * mask).sum() / n_tok)
        ratio_max = float((ratio * mask + (1.0 - mask)).max())
        policy_term = float(-(surr * mask).sum() / n_tok)
        kl_term = float(beta * kl_mean)

    return GRPOLossOutput(
        loss=loss,
        clip_fraction=clip_fraction,
        kl=kl_mean,
        ratio_mean=ratio_mean,
        ratio_max=ratio_max,
        policy_term=policy_term,
        kl_term=kl_term,
    )
