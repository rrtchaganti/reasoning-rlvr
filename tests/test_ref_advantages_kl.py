import pytest
import torch

from rlvr.ref.advantages import group_advantages, zero_advantage_fraction
from rlvr.ref.kl import kl_k1, kl_k2, kl_k3, masked_mean


def test_advantages_zero_mean_unit_std():
    torch.manual_seed(0)
    r = torch.rand(5, 8)
    a = group_advantages(r)
    assert torch.allclose(a.mean(dim=1), torch.zeros(5), atol=1e-6)
    assert torch.allclose(a.std(dim=1, unbiased=True), torch.ones(5), atol=1e-3)


def test_constant_group_gives_zero_signal():
    r = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    a = group_advantages(r)
    assert torch.all(a[0] == 0) and torch.all(a[1] == 0)
    assert zero_advantage_fraction(r) == pytest.approx(2 / 3)


def test_drgrpo_variant_only_centres():
    r = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    a = group_advantages(r, normalize_std=False)
    assert torch.allclose(a, torch.tensor([[-0.5, -0.5, 0.5, 0.5]]))


def test_std_normalisation_inflates_low_variance_groups():
    # same centred advantage magnitude pattern, but the near-unanimous group gets scaled up
    mixed = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
    rare = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    assert group_advantages(rare).max() > group_advantages(mixed).max()


def test_shape_validation():
    with pytest.raises(ValueError):
        group_advantages(torch.zeros(8))
    with pytest.raises(ValueError):
        group_advantages(torch.zeros(3, 1))


def _discrete_kl_setup():
    torch.manual_seed(1)
    p = torch.softmax(torch.randn(16), dim=0)
    q = torch.softmax(torch.randn(16), dim=0)
    true_kl = float((p * (p.log() - q.log())).sum())
    idx = torch.multinomial(p, 200_000, replacement=True)
    return p.log()[idx], q.log()[idx], true_kl


def test_k1_and_k3_unbiased_k2_biased_on_discrete_toy():
    logp, logq, true_kl = _discrete_kl_setup()
    assert float(kl_k1(logp, logq).mean()) == pytest.approx(true_kl, abs=0.02)
    assert float(kl_k3(logp, logq).mean()) == pytest.approx(true_kl, abs=0.02)
    # k2 is a different quantity in general; just check it is finite and non-negative
    k2 = kl_k2(logp, logq)
    assert torch.all(k2 >= 0) and torch.isfinite(k2).all()


def test_k3_nonnegative_and_lower_variance_than_k1():
    logp, logq, _ = _discrete_kl_setup()
    assert torch.all(kl_k3(logp, logq) >= -1e-6)
    assert float(kl_k3(logp, logq).var()) < float(kl_k1(logp, logq).var())


def test_masked_mean():
    x = torch.tensor([[1.0, 2.0, 100.0], [3.0, 100.0, 100.0]])
    m = torch.tensor([[1, 1, 0], [1, 0, 0]])
    assert float(masked_mean(x, m)) == pytest.approx(2.0)
