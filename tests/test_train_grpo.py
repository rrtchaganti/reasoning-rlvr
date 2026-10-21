"""rlvr-grpo plumbing on the CPU: config, the prompt set and its manifest check, the reward's
three parts, the per-step statistics and the hacking watch, the dynamic-sampling group selection,
and the TRL config it builds. The trainer itself is exercised by scripts/grpo_correctness_check.py
on a GPU (the correctness check of PLAN.md Phase 4 step 1)."""

import json

import pytest

from rlvr.data.rft_data import rows_hash
from rlvr.train import grpo as g


def test_load_config_defaults_overrides_and_validation(tmp_path):
    cfg = g.load_config(None, ["kl_beta=0", "epsilon_high=0.28", "seed=1", "loss_type=dr_grpo"])
    assert cfg["kl_beta"] == 0.0 and cfg["epsilon_high"] == 0.28 and cfg["seed"] == 1
    assert cfg["tokenizer"] == cfg["model"] and g.completions_per_step(cfg) == 256
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "learning_rate: 1e-6\nnum_prompts_per_step: '16'\nepsilon_high: none\n", encoding="utf-8"
    )
    cfg = g.load_config(cfg_path)
    assert cfg["learning_rate"] == 1e-6 and cfg["num_prompts_per_step"] == 16
    assert cfg["epsilon_high"] is None and g.completions_per_step(cfg) == 128
    with pytest.raises(KeyError):
        g.load_config(None, ["nope=1"])
    with pytest.raises(ValueError):
        g.load_config(None, ["loss_type=ppo"])
    with pytest.raises(ValueError):
        g.load_config(None, ["per_device_batch=48"])  # 256 is not a multiple of 48
    with pytest.raises(ValueError):
        g.load_config(None, ["dynamic_sampling=true", "oversample=1"])


def _rows():
    rows = []
    for i in range(10):
        rows.append(
            {
                "id": f"math-train-{i}",
                "source": "math_train",
                "question": f"What is {i}+{i}?",
                "answer": str(2 * i),
                "level": 1 + i % 3,
                "rate": 0.5,
            }
        )
    for i in range(10):
        rows.append(
            {
                "id": f"gsm8k-train-{i}",
                "source": "gsm8k_train",
                "question": f"Tom has {i} apples. How many?",
                "answer": str(i),
                "level": None,
                "rate": 0.25,
            }
        )
    return rows


def test_prompt_file_is_checked_against_the_manifest_and_sliced(tmp_path):
    rows = _rows()
    path = tmp_path / "learnable_prompts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"num_learnable": 20, "learnable_hash": rows_hash(rows, ("id", "question", "answer"))}
        )
    )
    read = g.read_prompt_rows(path)
    assert g.check_prompt_manifest(read, manifest) == rows_hash(rows, ("id", "question", "answer"))
    manifest.write_text(json.dumps({"num_learnable": 20, "learnable_hash": "0000000000000000"}))
    with pytest.raises(RuntimeError):
        g.check_prompt_manifest(read, manifest)
    chosen = g.select_prompts(read, gsm8k_fraction=0.2, seed=0)
    assert sum(r["source"] == "math_train" for r in chosen) == 10  # MATH in full
    assert sum(r["source"] == "gsm8k_train" for r in chosen) == 2  # 20 % of GSM8K
    assert [r["id"] for r in chosen] == [
        r["id"] for r in g.select_prompts(read, gsm8k_fraction=0.2, seed=0)
    ]
    assert len(g.select_prompts(read, gsm8k_fraction=0.2, seed=0, limit=5)) == 5
    recs = g.prompt_records(chosen)
    assert recs[0]["prompt"].endswith("Assistant:\n<think>\n") and recs[0]["prefix"] == "<think>\n"
    assert {r["answer_type"] for r in recs} == {"latex", "numeric"}
    assert all(r["level"] in (-1, 1, 2, 3) for r in recs)
    assert g.stop_strings() == ["</answer>", "\nUser:"]


def _ids(n, eos=False, cap=10):
    ids = list(range(5, 5 + n))
    if eos:
        ids[-1] = 0
    return ids[:cap]


def test_reward_is_one_only_for_finished_well_formed_correct_completions():
    kw = dict(
        prefix="<think>\n", answer="42", answer_type="numeric", eos_ids={0}, max_new_tokens=10
    )
    good = "2 + 40.\n</think>\n<answer> 42 </answer>"
    s = g.score_completion(good, _ids(6, eos=True), **kw)
    assert s == {**s, "finished": True, "format_ok": True, "correct": True, "reward": 1.0}
    wrong = g.score_completion("x\n</think>\n<answer> 41 </answer>", _ids(6, eos=True), **kw)
    assert wrong["correct"] is False and wrong["reward"] == 0.0 and wrong["format_ok"]
    truncated = g.score_completion(good, _ids(10), **kw)  # hit the cap without EOS
    assert truncated["finished"] is False and truncated["reward"] == 0.0 and truncated["correct"]
    at_cap_with_eos = g.score_completion(good, _ids(10, eos=True), **kw)
    assert at_cap_with_eos["finished"] is True and at_cap_with_eos["reward"] == 1.0
    no_format = g.score_completion("the answer is 42", _ids(4, eos=True), **kw)
    assert no_format["format_ok"] is False and no_format["reward"] == 0.0
    assert (
        g.score_completion(
            "Wait, wait. WAIT.\n</think>\n<answer> 42 </answer>", _ids(6, eos=True), **kw
        )["wait_count"]
        == 3
    )


def test_group_stats_and_hacking_flags():
    kw = dict(prefix="<think>\n", answer="7", answer_type="numeric", eos_ids={0}, max_new_tokens=50)
    good = g.score_completion("s\n</think>\n<answer> 7 </answer>", _ids(6, eos=True, cap=50), **kw)
    bad = g.score_completion("s\n</think>\n<answer> 8 </answer>", _ids(6, eos=True, cap=50), **kw)
    trunc = g.score_completion("s s s", _ids(50, cap=50), **kw)
    stats = g.group_stats([good, good, bad, bad, bad, bad, trunc, trunc], group_size=4)
    assert (
        stats["reward"] == 0.25 and stats["correct_rate"] == 0.25 and stats["finished_rate"] == 0.75
    )
    assert stats["format_only_rate"] == 0.5  # four finished, well-formed, wrong
    assert stats["zero_variance_group_rate"] == 0.5  # the second group is all zero
    # answers 7,7,8,8 and 8,8,none,none: no group has group_size - 1 identical answers
    assert stats["repeated_answer_group_rate"] == 0.0
    # the explicit check of the repeated-answer rule: three identical answers out of four
    same = g.group_stats([good, good, good, bad], group_size=4)
    assert same["repeated_answer_group_rate"] == 1.0
    # agreeing on the right answer is accuracy, not collapse: only a repeated wrong answer counts
    assert same["repeated_incorrect_answer_group_rate"] == 0.0
    same_wrong = g.group_stats([bad, bad, bad, good], group_size=4)
    assert same_wrong["repeated_answer_group_rate"] == 1.0
    assert same_wrong["repeated_incorrect_answer_group_rate"] == 1.0
    base = {
        "mean_tokens": 100.0,
        "correct_rate": 0.5,
        "format_only_rate": 0.1,
        "repeated_answer_group_rate": 0.9,  # a healthy run agrees on right answers
        "repeated_incorrect_answer_group_rate": 0.1,
        "wait_per_completion": 0.5,
        "finished_rate": 0.9,
    }
    flat = [dict(base) for _ in range(40)]
    assert g.hacking_flags(flat)["any"] is False
    grown = [dict(base) for _ in range(20)] + [{**base, "mean_tokens": 150.0} for _ in range(20)]
    assert g.hacking_flags(grown)["flags"]["length_growth_without_accuracy"] is True
    better = [dict(base) for _ in range(20)] + [
        {**base, "mean_tokens": 150.0, "correct_rate": 0.6} for _ in range(20)
    ]
    assert g.hacking_flags(better)["flags"]["length_growth_without_accuracy"] is False
    assert g.hacking_flags(flat[:10])["evaluated"] is False
    assert g.hacking_flags(flat)["flags"]["repeated_answers"] is False
    collapsed = [{**base, "repeated_incorrect_answer_group_rate": 0.6} for _ in range(40)]
    assert g.hacking_flags(collapsed)["flags"]["repeated_answers"] is True
    old_log = [{k: v for k, v in base.items() if k != "repeated_incorrect_answer_group_rate"}]
    assert g.hacking_flags(old_log * 40)["flags"]["repeated_answers"] is True  # falls back


def test_select_groups_keeps_informative_groups_first_and_fills_from_the_front():
    # 6 groups of 2: groups 0, 3, 5 are zero-advantage
    adv = [0, 0, 1, -1, 0.5, -0.5, 0, 0, 2, -2, 0, 0]
    idx, stats = g.select_groups(adv, group_size=2, num_keep=2)
    assert (
        idx == [2, 3, 4, 5]
        and stats["groups_informative"] == 3
        and stats["groups_filled_with_zero_advantage"] == 0
    )
    idx, stats = g.select_groups(adv, group_size=2, num_keep=4)
    assert idx == [0, 1, 2, 3, 4, 5, 8, 9] and stats["groups_filled_with_zero_advantage"] == 1
    assert stats["informative_fraction"] == pytest.approx(0.5)
    idx, _ = g.select_groups([0.0] * 8, group_size=4, num_keep=1)
    assert idx == [0, 1, 2, 3]


def test_grpo_config_matches_shape_b(tmp_path):
    cfg = g.load_config(None, ["use_vllm=false", "report_to=none"])
    args = g.make_grpo_config(cfg, tmp_path)
    assert args.per_device_train_batch_size == 16 and args.gradient_accumulation_steps == 16
    assert (
        args.num_generations == 8 and args.max_completion_length == 1536 and args.max_steps == 200
    )
    assert args.beta == 0.04 and args.loss_type == "grpo" and args.scale_rewards == "group"
    assert args.num_iterations == 1 and args.save_only_model and args.save_steps == 50
    assert args.remove_unused_columns is False and args.mask_truncated_completions is False
    a3 = g.make_grpo_config(
        g.load_config(None, ["loss_type=dr_grpo", "scale_rewards=none", "use_vllm=false"]), tmp_path
    )
    assert a3.loss_type == "dr_grpo" and a3.scale_rewards == "none"
    a2 = g.make_grpo_config(g.load_config(None, ["epsilon_high=0.28", "use_vllm=false"]), tmp_path)
    assert a2.epsilon_high == 0.28
    assert g.make_trainer_class(False, 2, 32, 8).__name__ == "GRPOTrainer"
    assert g.make_trainer_class(True, 2, 32, 8).__name__ == "DynamicSamplingGRPOTrainer"
