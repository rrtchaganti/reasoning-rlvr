import pytest
import torch

from rlvr.ref.sampling import _filter_logits, sample_sequence

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_model():
    # a random, tiny Qwen2-architecture model: same cache plumbing as Qwen3, runs on CPU in ms
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).eval()


def test_greedy_loop_matches_hf_generate(tiny_model):
    prompt = torch.tensor([[5, 17, 42, 7]])
    ours = sample_sequence(
        tiny_model, prompt, max_new_tokens=12, eos_token_id=None, temperature=0.0
    )
    ref = tiny_model.generate(
        prompt, max_new_tokens=12, do_sample=False, pad_token_id=0, eos_token_id=None
    )[0, prompt.shape[1] :]
    assert torch.equal(ours, ref)


def test_greedy_loop_matches_no_cache_forward(tiny_model):
    # the KV-cache path must reproduce recomputing the full sequence each step
    prompt = torch.tensor([[1, 2, 3]])
    ours = sample_sequence(tiny_model, prompt, max_new_tokens=6, eos_token_id=None)
    seq = prompt.clone()
    for _ in range(6):
        logits = tiny_model(input_ids=seq, use_cache=False).logits[0, -1]
        seq = torch.cat([seq, logits.argmax().view(1, 1)], dim=1)
    assert torch.equal(ours, seq[0, 3:])


def test_stops_at_eos(tiny_model):
    prompt = torch.tensor([[1, 2, 3]])
    full = sample_sequence(tiny_model, prompt, max_new_tokens=10, eos_token_id=None)
    eos = int(full[2])
    stopped = sample_sequence(tiny_model, prompt, max_new_tokens=10, eos_token_id=eos)
    # a random model may repeat tokens, so stop at the *first* occurrence of the chosen id
    first = full.tolist().index(eos)
    assert stopped.shape[0] == first + 1 and int(stopped[-1]) == eos


def test_sampling_is_reproducible_with_generator(tiny_model):
    prompt = torch.tensor([[9, 8, 7]])
    a = sample_sequence(
        tiny_model,
        prompt,
        max_new_tokens=8,
        eos_token_id=None,
        temperature=1.0,
        top_p=0.9,
        generator=torch.Generator().manual_seed(3),
    )
    b = sample_sequence(
        tiny_model,
        prompt,
        max_new_tokens=8,
        eos_token_id=None,
        temperature=1.0,
        top_p=0.9,
        generator=torch.Generator().manual_seed(3),
    )
    assert torch.equal(a, b)


def test_filter_logits_top_k_and_top_p():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    k2 = _filter_logits(logits, top_k=2, top_p=None)
    assert torch.isfinite(k2[:2]).all() and torch.isinf(k2[2:]).all()
    p = _filter_logits(logits, top_k=None, top_p=0.7)
    # 0.5 alone is below 0.7, so the second token (cum 0.8) is kept, the rest dropped
    assert torch.isfinite(p[:2]).all() and torch.isinf(p[2:]).all()
