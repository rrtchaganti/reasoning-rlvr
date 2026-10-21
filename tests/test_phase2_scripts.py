"""Phase 2 reporting scripts on fake run directories: data-scaling figure, LR choice, paired
checkpoint comparison, model-card generation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compare_checkpoints
import pick_lr
import plot_data_scaling
import project_run_time
import push_to_hub


def _est(mean, n=100):
    return {"mean": mean, "lo": mean - 0.05, "hi": mean + 0.05, "n": n}


def _metrics_dir(tmp_path, name, accs, template="think_tag"):
    d = tmp_path / name
    d.mkdir()
    m = {
        ev: {
            "num_problems": 100,
            "pass@1": _est(a),
            "eval": ev,
            "template": template,
            "truncation_rate": 0.1,
            "extraction_failure_rate": 0.0,
            "tag_format_rate": 0.98,
        }
        for ev, a in accs.items()
    }
    (d / "metrics.json").write_text(json.dumps(m))
    return d


def test_data_scaling_collects_sorted_points_and_plots(tmp_path):
    d25 = _metrics_dir(tmp_path, "d25", {"gsm8k": 0.70, "math500": 0.45})
    d50 = _metrics_dir(tmp_path, "d50", {"gsm8k": 0.74, "math500": 0.50})
    d100 = _metrics_dir(tmp_path, "d100", {"gsm8k": 0.76, "math500": 0.54})
    base = _metrics_dir(tmp_path, "base", {"gsm8k": 0.61, "math500": 0.43})
    pts = plot_data_scaling.parse_points([f"1.0={d100}", f"0.25={d25}", f"0.5={d50}"])
    assert [f for f, _ in pts] == [0.25, 0.5, 1.0]
    curves = plot_data_scaling.collect(pts)
    assert [f for f, _ in curves["math500"]] == [0.25, 0.5, 1.0]
    table = plot_data_scaling.markdown_table(curves, plot_data_scaling.load_metrics(base))
    assert "| math500 | 43.0 [38.0, 48.0] | 45.0 [40.0, 50.0] | 50.0" in table
    out = tmp_path / "fig.png"
    plot_data_scaling.plot(curves, plot_data_scaling.load_metrics(base), out)
    assert out.exists() and out.stat().st_size > 1000
    with pytest.raises(ValueError):
        plot_data_scaling.parse_points(["nodir"])


def test_pick_lr_rule_prefers_default_within_two_points(tmp_path):
    assert pick_lr.choose({"5e-6": 70.0, "1e-5": 72.0, "2e-5": 73.5}) == "1e-5"  # tie within 2
    assert pick_lr.choose({"5e-6": 70.0, "1e-5": 72.0, "2e-5": 74.5}) == "2e-5"
    assert pick_lr.choose({"5e-6": 75.0, "1e-5": 72.0}) == "5e-6"
    d = _metrics_dir(tmp_path, "dev", {"gsm8k_dev": 0.80, "math_dev": 0.40})
    acc, n = pick_lr.pooled_accuracy(json.loads((d / "metrics.json").read_text()))
    assert n == 200 and abs(acc - 60.0) < 1e-9


def _samples_dir(tmp_path, name, correct_by_id, level_by_id=None):
    d = tmp_path / name / "math500"
    d.mkdir(parents=True)
    with (d / "samples.jsonl").open("w") as f:
        for pid, ok in correct_by_id.items():
            meta = {"level": (level_by_id or {}).get(pid)}
            f.write(
                json.dumps({"problem_id": pid, "sample_idx": 0, "correct": ok, "meta": meta}) + "\n"
            )
    return tmp_path / name


def test_compare_checkpoints_pairs_by_id_and_reports_flips(tmp_path):
    ids = [f"p{i}" for i in range(40)]
    levels = {pid: 1 + (i % 2) for i, pid in enumerate(ids)}
    ref = {pid: i % 4 == 0 for i, pid in enumerate(ids)}  # 10 correct
    new = {pid: i % 2 == 0 for i, pid in enumerate(ids)}  # 20 correct, superset of ref
    rd = _samples_dir(tmp_path, "ref", ref, levels)
    nd = _samples_dir(tmp_path, "new", new, levels)
    results, by_level = compare_checkpoints.compare_runs(rd, nd)
    g = results["math500"]
    assert g["num_pairs"] == 40 and g["only_ref"] == 0 and g["only_new"] == 10
    assert abs(g["diff_new_minus_ref"]["mean"] - 0.25) < 1e-9
    assert set(by_level) == {1, 2} and by_level[1]["num_pairs"] == 20
    md = compare_checkpoints.markdown(results, by_level, "base", "sft")
    assert "| math500 | 40 |" in md and "| 1 | 20 |" in md
    with pytest.raises(ValueError):
        compare_checkpoints.paired({"a": (True, {})}, {"b": (True, {})})


def test_model_card_renders_from_run_info(tmp_path):
    run = tmp_path / "run"
    (run / "final").mkdir(parents=True)
    (run / "final" / "config.json").write_text("{}")
    info = {
        "git_commit": "abc1234",
        "git_branch": "phase-2-sft",
        "gpu": "A100",
        "config": {
            "model": "Qwen/Qwen3-1.7B-Base",
            "data": "outputs/data/sft/train.jsonl",
            "learning_rate": 1e-5,
            "epochs": 2,
            "per_device_batch": 4,
            "grad_accum": 16,
            "max_length": 4096,
            "lora": None,
            "bf16": True,
            "packing": True,
            "attn_implementation": "flash_attention_2",
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "lr_scheduler": "cosine_with_min_lr",
            "min_lr_rate": 0.1,
            "warmup_ratio": 0.03,
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,
            "subset_fraction": 1.0,
            "seed": 0,
        },
        "data_hash": "45b3fa4862b11e2a",
        "num_examples_used": 23000,
        "num_examples_total": 23000,
        "tokens_per_epoch": 61796025,
        "train_runtime_s": 12600,
        "train_loss": 0.42,
        "global_steps": 720,
        "peak_memory_gb": 61.2,
    }
    (run / "run_info.json").write_text(json.dumps(info))
    ev = _metrics_dir(tmp_path, "ev", {"gsm8k": 0.78, "math500": 0.52})
    card = push_to_hub.model_card(run, "Raviteja546/reasoning-rlvr-sft", [ev], "| x |\n")
    assert "base_model: Qwen/Qwen3-1.7B-Base" in card and "full fine-tune" in card
    assert "45b3fa4862b11e2a" in card and "3.50 GPU-hours" in card
    assert "| think_tag | 78.0 [73.0, 83.0] | - | 52.0 [47.0, 57.0] | - | - |" in card
    push_to_hub.main(["model", "--run", str(run), "--repo", "x/y", "--evals", str(ev), "--dry-run"])
    assert (run / "final" / "README.md").exists()
    # a checkpoint trained from a local path (the RFT run): the Hub rejects a path as base_model
    # (H100 node, 19 Oct 2026), so the header carries the given repo id or no base_model line at all
    info["config"]["model"] = "outputs/train/sft_1p7b/final"
    (run / "run_info.json").write_text(json.dumps(info))
    card = push_to_hub.model_card(run, "x/rft", [ev], None, "Raviteja546/reasoning-rlvr-sft")
    assert "base_model: Raviteja546/reasoning-rlvr-sft" in card
    assert "- Base: `outputs/train/sft_1p7b/final`" in card  # the body still names the real start
    card = push_to_hub.model_card(run, "x/rft", [ev], None)
    assert "base_model:" not in card
    push_to_hub.main(
        ["model", "--run", str(run), "--repo", "x/rft", "--base-model-id", "a/b", "--dry-run"]
    )
    assert "base_model: a/b" in (run / "final" / "README.md").read_text(encoding="utf-8")


def test_model_card_for_a_grpo_run_uses_its_own_fields(tmp_path):
    # The card the H100 node generated for the GRPO final on 21 Oct 2026 printed the SFT fields
    # (data hash None, epochs None, "cross-entropy"); a run_info of kind "grpo" gets its own section.
    run = tmp_path / "grpo"
    (run / "final").mkdir(parents=True)
    (run / "final" / "config.json").write_text("{}")
    info = {
        "kind": "grpo",
        "git_commit": "a8b9f69",
        "git_branch": "phase-4-grpo",
        "gpu": "H100",
        "config": {
            "model": "outputs/train/rft_1p7b/final",
            "prompts": "outputs/data/rft/learnable_prompts.jsonl",
            "manifest": "data/manifests/rft_iter1.json",
            "num_prompts_per_step": 32,
            "num_generations": 8,
            "max_new_tokens": 1536,
            "temperature": 1.0,
            "learning_rate": 1e-6,
            "lr_scheduler": "constant_with_warmup",
            "warmup_steps": 10,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.99,
            "max_grad_norm": 1.0,
            "kl_beta": 0.04,
            "epsilon": 0.2,
            "epsilon_high": None,
            "loss_type": "grpo",
            "scale_rewards": "group",
            "dynamic_sampling": False,
            "per_device_batch": 16,
            "bf16": True,
            "attn_implementation": "flash_attention_2",
            "seed": 0,
        },
        "prompt_file_hash": "281bdfe31176411b",
        "num_prompts_total": 4606,
        "num_prompts_used": 2735,
        "prompts_by_source": {"gsm8k_train": 468, "math_train": 2267},
        "completions_per_step": 256,
        "clip_active": False,
        "train_runtime_s": 11253.0,
        "train_loss": -0.0346,
        "global_steps": 200,
        "peak_memory_gb": 70.35,
        "reward_first_step": 0.34375,
        "reward_last_step": 0.68359375,
        "finished_rate_first_step": 0.3515625,
        "finished_rate_last_step": 0.75390625,
        "mean_tokens_first_step": 1406.4,
        "mean_tokens_last_step": 1200.7,
        "zero_variance_group_rate_mean": 0.30046875,
        "hacking_watch": {"evaluated": True, "any": True},
    }
    (run / "run_info.json").write_text(json.dumps(info))
    ev = _metrics_dir(tmp_path, "ev2", {"gsm8k": 0.823, "math500": 0.634})
    card = push_to_hub.model_card(
        run, "Raviteja546/reasoning-rlvr-grpo", [ev], "| x |\n", "Raviteja546/reasoning-rlvr-rft"
    )
    assert "base_model: Raviteja546/reasoning-rlvr-rft" in card
    assert "32 prompts x 8 samples per step (256 completions), 1536-token cap" in card
    assert "reward 0.344 -> 0.684, finished rate 35 % -> 75 %, mean completion 1406 -> 1201" in card
    assert "281bdfe31176411b" in card and "3.13 GPU-hours" in card
    assert "rlvr-grpo --config configs/train/grpo_1p7b.yaml" in card
    assert "Paired on the same problems against the checkpoint it was trained from" in card
    assert "None" not in card and "cross-entropy" not in card and "rlvr-build-sft" not in card


def _short_run_info(**over):
    info = {
        "gpu": "NVIDIA L40S",
        "config": {"subset_fraction": 0.1, "epochs": 2, "limit": None},
        "tokens_per_epoch": 6.18e6,
        "epochs_done": 2.0,
        "train_runtime_s": 1800.0,
    }
    info.update(over)
    return info


def test_project_run_time_rule():
    p = project_run_time.project(_short_run_info(), main_epochs=2, max_hours=8, rate=102)
    assert abs(p["tokens_per_s"] - 2 * 6.18e6 / 1800) < 1e-6
    assert abs(p["main_tokens"] - 123.6e6) < 1
    assert 4.9 < p["projected_hours"] < 5.1 and p["stay"]
    assert abs(p["projected_cost"] - p["projected_hours"] * 102) < 1e-6
    assert "STAY" in project_run_time.render(p)
    slow = project_run_time.project(
        _short_run_info(train_runtime_s=3600.0), main_epochs=2, max_hours=8, rate=102
    )
    assert not slow["stay"] and "MOVE" in project_run_time.render(slow)
    with pytest.raises(ValueError):
        project_run_time.project(
            _short_run_info(config={"subset_fraction": 1.0, "limit": 400}),
            main_epochs=2,
            max_hours=8,
            rate=102,
        )


def test_project_run_time_cli(tmp_path, capsys):
    f = tmp_path / "run_info.json"
    f.write_text(json.dumps(_short_run_info()))
    project_run_time.main([str(f), "--json", "--rate", "189"])
    out = json.loads(capsys.readouterr().out)
    assert out["stay"] and abs(out["rate_per_hour"] - 189) < 1e-9


def _samples_dir_n(tmp_path, name, counts_by_id, n):
    d = tmp_path / name / "gsm8k"
    d.mkdir(parents=True)
    with (d / "samples.jsonl").open("w") as f:
        for pid, c in counts_by_id.items():
            for j in range(n):
                f.write(json.dumps({"problem_id": pid, "sample_idx": j, "correct": j < c}) + "\n")
    return tmp_path / name


def test_compare_checkpoints_all_samples_gives_paired_pass_at_k(tmp_path):
    n = 4
    ref = {"a": 0, "b": 1, "c": 4, "d": 0}
    new = {"a": 2, "b": 4, "c": 4, "d": 0, "e": 4}  # e has no pair
    rd = _samples_dir_n(tmp_path, "ref", ref, n)
    nd = _samples_dir_n(tmp_path, "new", new, n)
    results, levels = compare_checkpoints.compare_runs(rd, nd, all_samples=True, ks=(1, 4, 8))
    g = results["gsm8k"]
    assert levels == {} and g["num_pairs"] == 4 and g["unpaired"] == 1 and set(g["by_k"]) == {1, 4}
    assert g["by_k"][1]["diff_new_minus_ref"]["mean"] == pytest.approx((10 - 5) / 4 / 4)
    assert g["by_k"][4]["ref"]["mean"] == pytest.approx(2 / 4)
    assert g["by_k"][4]["new"]["mean"] == pytest.approx(3 / 4)
    assert g["ever_solved"] == {"both": 2, "only_ref": 0, "only_new": 1, "neither": 1}
    md = compare_checkpoints.markdown_samples(results, "base", "sft")
    assert "| gsm8k | 4 | 1 |" in md and "| gsm8k | 4 | 2 | 0 | 1 | 1 |" in md
    bad = _samples_dir_n(tmp_path, "bad", {"a": 1}, 2)
    with pytest.raises(ValueError):
        compare_checkpoints.compare_runs(rd, bad, all_samples=True)
