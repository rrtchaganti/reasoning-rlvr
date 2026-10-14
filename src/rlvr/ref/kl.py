"""Monte-Carlo estimators of KL(pi || pi_ref) from per-token log-probabilities.

All three take ``logp`` (log pi(a|s) under the policy being trained) and ``logp_ref``
(log pi_ref(a|s)) for the *same* sampled tokens a ~ pi, and return a per-token estimate.
Let r = logp_ref - logp, so exp(r) = pi_ref / pi.

    k1 = -r                       unbiased, high variance, can be negative per token
    k2 = r^2 / 2                  biased (it is an f-divergence estimate), low variance, >= 0
    k3 = exp(r) - r - 1           unbiased, low variance, >= 0 (Schulman 2020, "Approximating KL")

Why k3 is the GRPO default: it is unbiased like k1 but, because exp(r) - r - 1 >= 0 with
equality at r = 0, every term is non-negative, so the estimate never swings negative on a
small batch. Its variance is lower than k1 because it is a control-variate version of k1
(adds exp(r) - 1, whose expectation under pi is zero). The cost is that its gradient with
respect to logp is (1 - exp(r)), not -1, so the penalty pushes harder where the policy has
moved far below the reference.
"""

from __future__ import annotations

import torch


def kl_k1(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Per-token k1 = log pi - log pi_ref. Unbiased for KL(pi || pi_ref)."""
    return logp - logp_ref


def kl_k2(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Per-token k2 = (log pi_ref - log pi)^2 / 2. Biased, low variance, non-negative."""
    r = logp_ref - logp
    return 0.5 * r * r


def kl_k3(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Per-token k3 = exp(r) - r - 1 with r = log pi_ref - log pi. Unbiased, non-negative."""
    r = logp_ref - logp
    return torch.exp(r) - r - 1.0


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` over positions where ``mask`` is 1 (all positions pooled together)."""
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)
