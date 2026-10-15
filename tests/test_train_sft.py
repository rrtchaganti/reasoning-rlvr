"""rlvr-sft plumbing: config, nested subsets, manifest check, packing guard; plus a two-step
training smoke on a tiny model with a tiny tokenizer. The smoke needs a GPU: TRL 1.15 computes
the SFT loss with a fused Triton LM-head kernel that has no CPU path (15 Oct 2026)."""

import json

import pytest

from rlvr.train.sft import (
    DEFAULTS,
    check_manifest,
    load_config,
    nested_subset,
    packing_is_safe,
    parse_override,
    select_rows,
)


def test_load_config_defaults_overrides_and_unknown_keys(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("learning_rate: 2.0e-5\nlora:\n  r: 16\n", encoding="utf-8")
    cfg = load_config(cfg_path, ["epochs=1", "packing=false", "out=outputs/train/x"])
    assert cfg["learning_rate"] == 2e-5 and cfg["lora"] == {"r": 16}
    assert cfg["epochs"] == 1 and cfg["packing"] is False and cfg["out"] == "outputs/train/x"
    assert cfg["tokenizer"] == cfg["model"] == DEFAULTS["model"]
    cfg_path.write_text("learnig_rate: 1\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(cfg_path)
    with pytest.raises(KeyError):
        load_config(None, ["nope=1"])
    with pytest.raises(ValueError):
        parse_override("novalue")


def test_nested_subsets_are_prefixes_of_one_shuffle():
    ids = [f"id{i}" for i in range(1000)]
    q = nested_subset(ids, 0.25, seed=0)
    h = nested_subset(ids, 0.5, seed=0)
    full = nested_subset(ids, 1.0, seed=0)
    assert len(q) == 250 and len(h) == 500 and len(full) == 1000
    assert set(q) < set(h) < set(full)
    assert h[:250] == q  # literally a prefix
    assert nested_subset(ids, 0.25, seed=1) != q
    assert sorted(full) == sorted(ids)  # a permutation, nothing dropped
    with pytest.raises(ValueError):
        nested_subset(ids, 0.0, 0)


def _rows():
    return [
        {
            "id": f"u{i}",
            "source": "s",
            "problem_type": "t",
            "prompt": f"User: q{i}\nAssistant:\n<think>\n",
            "completion": f"r{i}\n</think>\n<answer> {i} </answer>",
            "answer": str(i),
            "n_tokens": 10 + i,
        }
        for i in range(8)
    ]


def test_check_manifest_accepts_matching_and_rejects_changed_data(tmp_path):
    from rlvr.data.sft_data import SftExample, content_hash

    rows = _rows()
    exs = [SftExample(**r) for r in rows]
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"content_hash": content_hash(exs), "num_examples": len(rows)}))
    assert check_manifest(rows, man)["num_examples"] == 8
    rows[3]["completion"] += "!"
    with pytest.raises(RuntimeError):
        check_manifest(rows, man)
    with pytest.raises(RuntimeError):
        check_manifest(_rows()[:7], man)


def test_select_rows_applies_fraction_then_limit():
    rows = _rows()
    half = select_rows(rows, fraction=0.5, seed=0, limit=None)
    assert len(half) == 4 and all(r["id"] in {x["id"] for x in rows} for r in half)
    assert [r["id"] for r in select_rows(rows, fraction=0.5, seed=0, limit=2)] == [
        r["id"] for r in half[:2]
    ]


def test_packing_guard():
    assert packing_is_safe({"packing": False, "attn_implementation": "sdpa"})
    assert packing_is_safe({"packing": True, "attn_implementation": "flash_attention_2"})
    assert packing_is_safe({"packing": True, "attn_implementation": "kernels-community/flash-attn"})
    assert not packing_is_safe({"packing": True, "attn_implementation": "sdpa"})
    assert not packing_is_safe({"packing": True, "attn_implementation": "eager"})


# --------------------------------------------------------------------------- CPU smoke


def _tiny_model_and_tokenizer(path):
    """A random 2-layer Qwen3 and a word-level tokenizer over a tiny vocabulary, saved to
    ``path`` so the trainer loads them like any checkpoint; no network."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    words = ["<|endoftext|>", "<unk>", "User:", "Assistant:", "<think>", "</think>", "<answer>"]
    words += ["</answer>", "q", "r", "a", "b", "c"] + [str(i) for i in range(20)]
    vocab = {w: i for i, w in enumerate(words)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>", unk_token="<unk>"
    )
    cfg = Qwen3Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    model.save_pretrained(path)
    tok.save_pretrained(path)


@pytest.mark.slow
@pytest.mark.gpu
def test_two_training_steps_on_gpu(tmp_path):
    pytest.importorskip("trl")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device: TRL's fused LM-head loss is a Triton kernel")
    from rlvr.data.sft_data import SftExample, content_hash, write_jsonl
    from rlvr.train.sft import run

    mdl = tmp_path / "tiny"
    _tiny_model_and_tokenizer(mdl)
    rows = [
        {
            **r,
            "prompt": f"User: q {i} Assistant: <think>",
            "completion": f"r a b c {i} </think> <answer> {i} </answer>",
        }
        for i, r in enumerate(_rows())
    ]
    exs = [SftExample(**r) for r in rows]
    write_jsonl(exs, tmp_path / "train.jsonl")
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"content_hash": content_hash(exs), "num_examples": len(exs)}))
    cfg = load_config(
        None,
        [
            f"model={mdl}",
            f"data={tmp_path / 'train.jsonl'}",
            f"manifest={man}",
            f"out={tmp_path / 'run'}",
            "packing=false",
            "attn_implementation=sdpa",
            "bf16=false",
            "gradient_checkpointing=false",
            "per_device_batch=4",
            "grad_accum=1",
            "max_steps=2",
            "save_steps=1000",
            "logging_steps=1",
            "max_length=64",
            "dataset_num_proc=1",
            "warmup_ratio=0",
            "lr_scheduler=constant",
        ],
    )
    summary = run(cfg, allow_dirty=True)
    assert summary["global_steps"] == 2 and summary["num_examples_used"] == 8
    assert (tmp_path / "run" / "final" / "config.json").exists()
    assert (tmp_path / "run" / "run_info.json").exists()
    log = [json.loads(line) for line in (tmp_path / "run" / "train_log.jsonl").open()]
    assert any("loss" in rec for rec in log)
    assert summary["tokens_per_epoch"] == sum(r["n_tokens"] for r in rows)


@pytest.mark.slow
def test_unsafe_packing_is_refused(tmp_path):
    pytest.importorskip("trl")
    from rlvr.train.sft import run

    cfg = load_config(None, ["packing=true", "attn_implementation=sdpa", f"out={tmp_path}"])
    with pytest.raises(RuntimeError, match="attend to each other"):
        run(cfg, allow_dirty=True)
