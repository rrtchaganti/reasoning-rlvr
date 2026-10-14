import pytest
import torch

from rlvr.ref.grpo_loss import grpo_loss


def _batch(seed=0, n_seq=4, n_pos=6):
    g = torch.Generator().manual_seed(seed)
    logp = -torch.rand(n_seq, n_pos, generator=g) * 3 - 0.1
    logp.requires_grad_(True)
    lengths = torch.tensor([6, 3, 5, 2])[:n_seq]
    mask = (torch.arange(n_pos).unsqueeze(0) < lengths.unsqueeze(1)).float()
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])[:n_seq]
    return logp, mask, adv


def test_single_update_ratio_is_one_and_nothing_clipped():
    logp, mask, adv = _batch()
    out = grpo_loss(logp, logp.detach(), adv, mask)
    assert out.clip_fraction == 0.0
    assert out.ratio_mean == pytest.approx(1.0)
    assert out.kl == 0.0


def test_gradient_equals_reinforce_with_baseline_when_ratio_is_one():
    # d/dlogp of -(ratio*A) at ratio=1 is -A per token (token-level agg, divided by n_tokens)
    logp, mask, adv = _batch()
    out = grpo_loss(logp, logp.detach(), adv, mask, loss_agg="token")
    out.loss.backward()
    expected = -(adv.unsqueeze(1) * mask) / mask.sum()
    assert torch.allclose(logp.grad, expected, atol=1e-6)


def test_sequence_agg_weights_short_completions_more():
    logp, mask, adv = _batch()
    out_seq = grpo_loss(logp, logp.detach(), adv, mask, loss_agg="sequence")
    out_seq.loss.backward()
    g_seq = logp.grad.clone()
    logp.grad = None
    out_tok = grpo_loss(logp, logp.detach(), adv, mask, loss_agg="token")
    out_tok.loss.backward()
    g_tok = logp.grad.clone()
    # per-token gradient magnitude of the length-2 sequence (row 3) relative to the length-6 one (row 0)
    ratio_seq = g_seq[3, 0].abs() / g_seq[0, 0].abs()
    ratio_tok = g_tok[3, 0].abs() / g_tok[0, 0].abs()
    assert ratio_seq > ratio_tok  # sequence-level inflates short sequences
    assert ratio_tok == pytest.approx(0.5)  # token-level: only the |A| ratio (0.5 / 1.0)


def test_drgrpo_divides_by_constant():
    logp, mask, adv = _batch()
    out = grpo_loss(logp, logp.detach(), adv, mask, loss_agg="drgrpo")
    out_tok = grpo_loss(logp, logp.detach(), adv, mask, loss_agg="token")
    n_seq, n_pos = mask.shape
    assert out.loss.item() == pytest.approx(
        out_tok.loss.item() * float(mask.sum()) / (n_seq * n_pos)
    )


def test_clipping_bounds_the_update():
    logp, mask, adv = _batch()
    # old policy much less likely on every token -> ratio >> 1 + eps everywhere
    logp_old = logp.detach() - 2.0
    out = grpo_loss(logp, logp_old, adv, mask, eps_low=0.2, eps_high=0.2)
    assert out.clip_fraction == 1.0
    out.loss.backward()
    # positive-advantage rows are clipped (constant branch) -> zero grad; negative rows still flow
    assert torch.allclose(logp.grad[0], torch.zeros_like(logp.grad[0]))
    assert logp.grad[1].abs().sum() > 0


def test_clip_higher_lets_more_positive_ratios_through():
    logp, mask, adv = _batch()
    logp_old = logp.detach() - 0.22  # ratio ~ 1.246: above 1.20, below 1.28
    std = grpo_loss(logp, logp_old, adv, mask, eps_low=0.2, eps_high=0.2)
    dapo = grpo_loss(logp, logp_old, adv, mask, eps_low=0.2, eps_high=0.28)
    assert std.clip_fraction == 1.0 and dapo.clip_fraction == 0.0


def test_kl_penalty_requires_ref_and_is_zero_at_ref():
    logp, mask, adv = _batch()
    with pytest.raises(ValueError):
        grpo_loss(logp, logp.detach(), adv, mask, beta=0.04)
    out = grpo_loss(logp, logp.detach(), adv, mask, beta=0.04, logp_ref=logp.detach())
    assert out.kl == pytest.approx(0.0)
    out2 = grpo_loss(logp, logp.detach(), adv, mask, beta=0.04, logp_ref=logp.detach() - 1.0)
    assert out2.kl > 0 and out2.kl_term == pytest.approx(0.04 * out2.kl)


def test_shape_errors():
    logp, mask, adv = _batch()
    with pytest.raises(ValueError):
        grpo_loss(logp, logp.detach(), adv[:2], mask)
    with pytest.raises(ValueError):
        grpo_loss(logp, logp.detach()[:, :3], adv, mask)
