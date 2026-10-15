"""The SFT objective and packing masks, checked exactly on a tiny random Qwen3 model (CPU)."""

import pytest
import torch

from rlvr.ref.sft_loss import (
    IGNORE,
    additive_mask,
    block_diagonal_causal_mask,
    completion_labels,
    masked_cross_entropy,
    pack_sequences,
    packed_cross_entropy,
)

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


def test_completion_labels_mask_prompt_and_padding():
    ids = torch.tensor([[5, 6, 7, 8, 9], [1, 2, 3, 0, 0]])
    am = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    labels = completion_labels(ids, [2, 1], am)
    assert labels.tolist() == [[IGNORE, IGNORE, 7, 8, 9], [IGNORE, 2, 3, IGNORE, IGNORE]]


def test_masked_cross_entropy_matches_manual_token_mean():
    torch.manual_seed(1)
    logits = torch.randn(1, 5, 11)
    labels = torch.tensor([[IGNORE, IGNORE, 3, 4, 5]])  # targets at positions 2, 3, 4
    logp = torch.log_softmax(logits[0], -1)
    manual = -(logp[1, 3] + logp[2, 4] + logp[3, 5]) / 3  # logits[t] predicts labels[t + 1]
    assert torch.allclose(masked_cross_entropy(logits, labels), manual)
    assert masked_cross_entropy(logits, labels, reduction="none").shape == (3,)
    assert torch.allclose(masked_cross_entropy(logits, labels, reduction="sum"), manual * 3)


def test_additive_mask_is_zero_where_allowed_and_min_elsewhere():
    allowed = torch.tensor([[True, False]])
    add = additive_mask(allowed)
    assert add[0, 0].item() == 0.0 and add[0, 1].item() == torch.finfo(torch.float32).min


def test_block_diagonal_mask_shape_and_structure():
    m = block_diagonal_causal_mask([2, 3])[0, 0]
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(m, expected)


def test_pack_sequences_first_fit_and_length_guard():
    seqs = [[1] * 6, [2] * 3, [3] * 4, [4] * 2]
    packed = pack_sequences(seqs, [1, 1, 1, 1], max_length=9)
    assert [p.seq_lens for p in packed] == [[6, 3], [4, 2]]
    assert packed[0].position_ids.tolist() == [[0, 1, 2, 3, 4, 5, 0, 1, 2]]
    with pytest.raises(ValueError):
        pack_sequences([[1] * 10], [1], max_length=9)


def _per_sequence_nll(model, seqs, prompt_lens):
    """Reference: run every sequence on its own, collect its labelled per-token NLLs."""
    out = []
    for s, pl in zip(seqs, prompt_lens, strict=True):
        ids = torch.tensor([s])
        labels = completion_labels(ids, [pl])
        logits = model(input_ids=ids).logits
        out.append(masked_cross_entropy(logits, labels, reduction="none"))
    return torch.cat(out)


def test_packed_with_block_mask_equals_unpacked_and_plain_causal_does_not(tiny):
    torch.manual_seed(2)
    seqs = [torch.randint(1, 128, (n,)).tolist() for n in (7, 5, 9)]
    prompt_lens = [3, 2, 4]
    ref = _per_sequence_nll(tiny, seqs, prompt_lens)

    (packed,) = pack_sequences(seqs, prompt_lens, max_length=32)
    with torch.no_grad():
        logits = tiny(
            input_ids=packed.input_ids,
            position_ids=packed.position_ids,
            attention_mask=additive_mask(packed.attention_mask),
        ).logits
    ours = packed_cross_entropy(logits, packed)
    assert ours.shape == ref.shape
    assert torch.allclose(ours, ref, atol=1e-4), (ours - ref).abs().max()

    # Same packed row, plain causal attention (what a naive concatenation does): sequences 2
    # and 3 see sequence 1's tokens, and the per-token losses change. This is the
    # cross-contamination PLAN.md warns about.
    with torch.no_grad():
        naive = tiny(input_ids=packed.input_ids, position_ids=packed.position_ids).logits
    leaked = packed_cross_entropy(naive, packed)
    first_len = packed.seq_lens[0] - prompt_lens[0]  # sequence 1 is unaffected either way
    assert torch.allclose(leaked[:first_len], ref[:first_len], atol=1e-4)
    assert not torch.allclose(leaked[first_len:], ref[first_len:], atol=1e-3)
