"""rlvr-sft plumbing: config, nested subsets, manifest check, packing guard, the untrained
format-tag rows; plus two-step training smokes (full and LoRA) on a tiny model with a tiny
tokenizer. The smokes need a GPU: TRL 1.15 computes the SFT loss with a fused Triton LM-head
kernel that has no CPU path (15 Oct 2026)."""

import json

import pytest

from rlvr.train.sft import (
    DEFAULTS,
    EMBED_SUFFIXES,
    check_manifest,
    decay_parameter_names,
    duplicate_row_count,
    enable_tag_row_training,
    init_tag_embeddings,
    load_config,
    make_peft_config,
    nested_subset,
    packing_is_safe,
    parse_override,
    plain_piece_ids,
    select_rows,
    single_token_ids,
    snapshot_rows,
    tag_training_delta,
)


def test_load_config_defaults_overrides_and_unknown_keys(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("learning_rate: 2.0e-5\nlora:\n  r: 16\n", encoding="utf-8")
    cfg = load_config(cfg_path, ["epochs=1", "packing=false", "out=outputs/train/x"])
    assert cfg["learning_rate"] == 2e-5 and cfg["lora"] == {"r": 16}
    assert cfg["epochs"] == 1 and cfg["packing"] is False and cfg["out"] == "outputs/train/x"
    assert cfg["tokenizer"] == cfg["model"] == DEFAULTS["model"]
    assert cfg["tag_tokens"] == ["<think>", "</think>"] and cfg["init_tag_tokens"] is True
    assert cfg["load_dtype"] == "float32" and cfg["bf16"] is True
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


# --------------------------------------------------------------------------- tiny model


def _tiny_model_and_tokenizer(path=None, *, tie=False, filler_tags=False):
    """A random 2-layer Qwen3 and a word-level tokenizer over a tiny vocabulary. ``tie`` ties the
    output head to the embeddings like Qwen3; ``filler_tags`` sets the rows of ``<think>``,
    ``</think>`` and three unused words to one shared vector, as in Qwen3-Base. Saved to ``path``
    when given so the trainer loads them like any checkpoint; no network."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    words = ["<|endoftext|>", "<unk>", "User:", "Assistant:", "<think>", "</think>", "<answer>"]
    words += ["</answer>", "q", "r", "a", "b", "c", "<", "</", ">", "think"]
    words += [str(i) for i in range(20)]
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
        tie_word_embeddings=tie,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    if filler_tags:
        emb = model.get_input_embeddings().weight.data
        filler = torch.full_like(emb[0], 0.01)
        for w in ["<think>", "</think>", "17", "18", "19"]:
            emb[vocab[w]] = filler
    if path is not None:
        model.save_pretrained(path)
        tok.save_pretrained(path)
    return model, tok, vocab


def test_tag_token_rows_detected_and_initialised_from_plain_pieces():
    import torch

    model, tok, vocab = _tiny_model_and_tokenizer(tie=True, filler_tags=True)
    tags = ["<think>", "</think>", "<answer>"]
    assert single_token_ids(tok, tags) == {t: vocab[t] for t in tags}
    assert plain_piece_ids(tok, "</think>") == [vocab["</"], vocab["think"], vocab[">"]]
    assert plain_piece_ids(tok, "<think>") == [vocab["<"], vocab["think"], vocab[">"]]
    emb = model.get_input_embeddings().weight.data
    assert duplicate_row_count(emb, vocab["</think>"]) == 4
    assert duplicate_row_count(emb, vocab["<answer>"]) == 0
    # tied head shares the storage, as in Qwen3
    head = model.get_output_embeddings().weight
    assert head.data_ptr() == model.get_input_embeddings().weight.data_ptr()

    rep = init_tag_embeddings(model, tok, tags)
    assert rep["</think>"]["untrained"] and rep["</think>"]["initialised"]
    assert rep["</think>"]["duplicate_rows"] == 4 and rep["</think>"]["id"] == vocab["</think>"]
    assert rep["</think>"]["init_from"] == [vocab["</"], vocab["think"], vocab[">"]]
    assert rep["<think>"]["initialised"]
    assert not rep["<answer>"]["untrained"] and not rep["<answer>"]["initialised"]
    expect = emb[[vocab["</"], vocab["think"], vocab[">"]]].float().mean(0)
    assert torch.allclose(emb[vocab["</think>"]].float(), expect)
    assert duplicate_row_count(emb, vocab["</think>"]) == 0
    assert duplicate_row_count(emb, vocab["17"]) == 2  # the unused filler rows are left alone
    assert torch.equal(head.data[vocab["</think>"]], emb[vocab["</think>"]])

    # untied model: apply=False only reports; apply writes the head row too
    m2, tok2, _ = _tiny_model_and_tokenizer(tie=False, filler_tags=True)
    rep2 = init_tag_embeddings(m2, tok2, ["</think>"], apply=False)
    assert rep2["</think>"]["untrained"] and not rep2["</think>"]["initialised"]
    init_tag_embeddings(m2, tok2, ["</think>"])
    e2 = m2.get_input_embeddings().weight.data
    h2 = m2.get_output_embeddings().weight.data
    assert torch.equal(e2[vocab["</think>"]], h2[vocab["</think>"]])
    snap = snapshot_rows(m2, [vocab["</think>"]])
    assert tag_training_delta(m2, snap) == {vocab["</think>"]: 0.0}
    e2[vocab["</think>"]] += 1.0
    assert tag_training_delta(m2, snap)[vocab["</think>"]] > 0


def test_make_peft_config():
    pytest.importorskip("peft")
    c = make_peft_config({"r": 4, "alpha": 8})
    assert c.r == 4 and c.lora_alpha == 8 and c.target_modules == "all-linear"
    assert make_peft_config(None) is None


def test_masked_row_training_touches_only_the_tag_rows():
    import torch

    model, _tok, vocab = _tiny_model_and_tokenizer(tie=True, filler_tags=True)
    for p in model.parameters():
        p.requires_grad_(False)  # what PEFT does to the base model
    ids = [vocab["<think>"], vocab["</think>"]]
    enable_tag_row_training(model, ids)
    x = torch.tensor(
        [[vocab["User:"], vocab["q"], vocab["<think>"], vocab["r"], vocab["</think>"]]]
    )
    model(input_ids=x, labels=x).loss.backward()
    g = model.get_input_embeddings().weight.grad
    assert g is not None
    nonzero = g.abs().sum(1).nonzero().flatten().tolist()
    assert sorted(nonzero) == sorted(ids)  # every other row (incl. the tied output side) is zero
    assert model.get_output_embeddings().weight.grad is g  # tied: same storage, one hook

    names = ["model.layers.0.self_attn.q_proj.lora_A.default.weight", "model.embed_tokens.weight"]
    assert decay_parameter_names(names, EMBED_SUFFIXES) == names[:1]
    assert decay_parameter_names(names, ()) == names


# --------------------------------------------------------------------------- GPU smokes


def _tiny_run_overrides(tmp_path):
    """Tiny tied model with filler tag rows, eight examples, a matching manifest; returns the
    ``--set`` list for a two-step run writing to ``tmp_path / 'run'``."""
    from rlvr.data.sft_data import SftExample, content_hash, write_jsonl

    mdl = tmp_path / "tiny"
    _tiny_model_and_tokenizer(mdl, tie=True, filler_tags=True)
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
    return rows, [
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
    ]


def _needs_gpu():
    pytest.importorskip("trl")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device: TRL's fused LM-head loss is a Triton kernel")


@pytest.mark.slow
@pytest.mark.gpu
def test_two_training_steps_on_gpu(tmp_path):
    _needs_gpu()
    from rlvr.train.sft import run

    rows, overrides = _tiny_run_overrides(tmp_path)
    overrides += ["bf16=true", "load_dtype=float32"]  # the main run's precision setup
    summary = run(load_config(None, overrides), allow_dirty=True)
    assert summary["global_steps"] == 2 and summary["num_examples_used"] == 8
    assert (tmp_path / "run" / "final" / "config.json").exists()
    assert (tmp_path / "run" / "run_info.json").exists()
    log = [json.loads(line) for line in (tmp_path / "run" / "train_log.jsonl").open()]
    assert any("loss" in rec for rec in log)
    assert summary["tokens_per_epoch"] == sum(r["n_tokens"] for r in rows)
    tt = summary["tag_tokens"]
    assert tt["</think>"]["untrained"] and tt["</think>"]["initialised"]
    assert tt["</think>"]["train_delta_norm"] > 0  # full fine-tune moves the row
    assert tt["</think>"]["duplicate_rows_final"] == 0
    info = json.loads((tmp_path / "run" / "run_info.json").read_text())
    assert info["tag_tokens"]["<think>"]["initialised"]


@pytest.mark.slow
@pytest.mark.gpu
def test_lora_trains_and_merges_the_tag_rows_on_gpu(tmp_path):
    _needs_gpu()
    import torch
    from transformers import AutoModelForCausalLM

    from rlvr.train.sft import run

    _, overrides = _tiny_run_overrides(tmp_path)
    overrides += ["bf16=true", "load_dtype=bfloat16"]  # the LoRA configs' precision setup
    overrides.append("lora={r: 4, alpha: 8, dropout: 0.0, target_modules: all-linear, merge: true}")
    summary = run(load_config(None, overrides), allow_dirty=True)
    tt = summary["tag_tokens"]["</think>"]
    assert tt["trained_via"] == "masked_rows"
    assert tt["initialised"] and tt["train_delta_norm"] > 0  # the row trained under LoRA
    assert summary["tag_tokens"]["<think>"]["train_delta_norm"] > 0
    assert (tmp_path / "run" / "adapter").exists()
    merged = AutoModelForCausalLM.from_pretrained(tmp_path / "run" / "final")
    emb = merged.get_input_embeddings().weight.data
    head = merged.get_output_embeddings().weight.data
    assert torch.equal(emb[tt["id"]], head[tt["id"]])  # tie survived merge and save
    assert duplicate_row_count(emb, tt["id"]) == 0


@pytest.mark.slow
def test_unsafe_packing_is_refused(tmp_path):
    pytest.importorskip("trl")
    from rlvr.train.sft import run

    cfg = load_config(None, ["packing=true", "attn_implementation=sdpa", f"out={tmp_path}"])
    with pytest.raises(RuntimeError, match="attend to each other"):
        run(cfg, allow_dirty=True)


@pytest.mark.slow
def test_lora_with_frozen_untrained_tags_is_refused(tmp_path):
    pytest.importorskip("trl")
    from rlvr.train.sft import run

    _, overrides = _tiny_run_overrides(tmp_path)
    overrides.append("lora={r: 4, train_tags: false}")
    with pytest.raises(RuntimeError, match="could never emit"):
        run(load_config(None, overrides), allow_dirty=True)
