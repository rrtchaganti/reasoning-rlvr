"""Group-relative advantages (GRPO, Shao et al. 2024).

For each prompt we sample a group of G completions and score them r_1..r_G. The advantage
of completion i is its reward relative to the group:

    A_i = (r_i - mean(r)) / (std(r) + eps)        (GRPO)
    A_i =  r_i - mean(r)                           (Dr. GRPO, no std normalisation)

Why the group mean is a valid baseline: it does not depend on the action taken by sample i
(it depends on the other samples only through an average that i contributes 1/G of; the
bias this introduces is O(1/G) and vanishes as G grows). Subtracting it reduces variance
without changing the expected gradient. Dividing by std rescales every prompt to unit
reward spread, which Liu et al. 2025 (Dr. GRPO) show over-weights prompts that are nearly
always right or nearly always wrong (small std, huge advantages). The switch is here so
that ablation A3 flips exactly one thing.
"""

from __future__ import annotations

import torch


def group_advantages(
    rewards: torch.Tensor,
    *,
    normalize_std: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-sample advantages from a [B, G] matrix of rewards.

    Args:
        rewards: shape [num_prompts, group_size]; one row per prompt.
        normalize_std: divide by the group's (unbiased) standard deviation. ``False`` is the
            Dr. GRPO variant.
        eps: added to the std to avoid division by zero for constant groups.

    Returns:
        Tensor with the same shape as ``rewards``. A group with identical rewards gets all
        zero advantages (no learning signal: this is why dynamic sampling exists).
    """
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be [B, G], got shape {tuple(rewards.shape)}")
    if rewards.shape[1] < 2:
        raise ValueError("group size must be at least 2 to form a baseline")
    rewards = rewards.to(torch.float32)
    centred = rewards - rewards.mean(dim=1, keepdim=True)
    if not normalize_std:
        return centred
    std = rewards.std(dim=1, keepdim=True, unbiased=True)
    return centred / (std + eps)


def zero_advantage_fraction(rewards: torch.Tensor, atol: float = 1e-8) -> float:
    """Fraction of groups whose rewards are all equal, i.e. that contribute no gradient.

    Logged every step: on an easy prompt set this climbs toward 1 and the run is wasting
    most of its rollouts (PLAN.md Phase 4 step 2).
    """
    spread = rewards.max(dim=1).values - rewards.min(dim=1).values
    return float((spread <= atol).float().mean())
