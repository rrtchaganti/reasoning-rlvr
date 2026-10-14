"""From-scratch reference implementations.

Everything here is written without TRL/verl so that each formula can be read, tested and
defended line by line. The library implementations are checked against these on a toy run
(PLAN.md Phase 4 step 1). Nothing in this package is optimised for speed.
"""

from rlvr.ref.advantages import group_advantages
from rlvr.ref.grpo_loss import GRPOLossOutput, grpo_loss
from rlvr.ref.kl import kl_k1, kl_k2, kl_k3
from rlvr.ref.pass_at_k import pass_at_k, pass_at_k_mean

__all__ = [
    "GRPOLossOutput",
    "group_advantages",
    "grpo_loss",
    "kl_k1",
    "kl_k2",
    "kl_k3",
    "pass_at_k",
    "pass_at_k_mean",
]
