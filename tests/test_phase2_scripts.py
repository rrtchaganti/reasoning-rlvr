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
    card = push_to_hub.model_card(run, "rrtchaganti/reasoning-rlvr-sft", [ev], "| x |\n")
    assert "base_model: Qwen/Qwen3-1.7B-Base" in card and "full fine-tune" in card
    assert "45b3fa4862b11e2a" in card and "3.50 GPU-hours" in card
    assert "| think_tag | 78.0 [73.0, 83.0] | - | 52.0 [47.0, 57.0] | - | - |" in card
    push_to_hub.main(["model", "--run", str(run), "--repo", "x/y", "--evals", str(ev), "--dry-run"])
    assert (run / "final" / "README.md").exists()
