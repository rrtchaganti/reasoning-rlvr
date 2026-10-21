"""The Phase 4 preflight refuses a config that differs from the plan's row, a prompt file that does
not match its manifest, and a start checkpoint from the wrong run; it passes the real shape-B
config file for every run it knows."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import phase4_preflight as pf

from rlvr.data.rft_data import rows_hash
from rlvr.train import grpo

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train" / "grpo_1p7b.yaml"

ABLATION_SETS = {
    "main": [],
    "seed1": ["seed=1", "out=outputs/train/grpo_1p7b_seed1"],
    "a1": ["kl_beta=0", "out=outputs/train/grpo_1p7b_a1_nokl"],
    "a2": ["dynamic_sampling=true", "epsilon_high=0.28", "out=outputs/train/grpo_1p7b_a2_dynamic"],
    "a3": ["loss_type=dr_grpo", "scale_rewards=none", "out=outputs/train/grpo_1p7b_a3_drgrpo"],
    "cap": ["max_new_tokens=2048", "out=outputs/train/grpo_1p7b_cap2048"],
    "debug": ["max_steps=50", "gsm8k_fraction=1.0", "out=outputs/train/grpo_1p7b_debug"],
}


def test_the_committed_config_matches_every_plan_row_with_its_overrides():
    for run, sets in ABLATION_SETS.items():
        cfg = grpo.load_config(CONFIG, sets)
        assert pf.check_knobs(cfg, pf.RUNS[run]) == [], run
    # and a wrong knob is named
    cfg = grpo.load_config(CONFIG, ["max_new_tokens=2048", "kl_beta=0"])
    problems = pf.check_knobs(cfg, pf.RUNS["main"])
    assert (
        len(problems) == 2
        and any("max_new_tokens" in p for p in problems)
        and any("kl_beta" in p for p in problems)
    )
    a1 = grpo.load_config(CONFIG, ["kl_beta=0", "out=outputs/train/grpo_1p7b_a1_nokl"])
    assert pf.check_knobs(a1, pf.RUNS["a1"]) == []
    assert (
        pf.check_knobs(a1, pf.RUNS["main"]) != []
    )  # an ablation config never passes as the main run
    # the output directory is part of the row: an ablation may not overwrite the main run's folder
    assert any(
        "out" in p for p in pf.check_knobs(grpo.load_config(CONFIG, ["kl_beta=0"]), pf.RUNS["a1"])
    )


def _prompt_files(tmp_path, *, bad_hash=False):
    rows = [
        {
            "id": f"math-train-{i}",
            "source": "math_train",
            "question": f"q{i}",
            "answer": str(i),
            "level": 1,
            "rate": 0.5,
        }
        for i in range(5)
    ] + [
        {
            "id": f"gsm8k-train-{i}",
            "source": "gsm8k_train",
            "question": f"g{i}",
            "answer": str(i),
            "level": None,
            "rate": 0.5,
        }
        for i in range(5)
    ]
    p = tmp_path / "learnable_prompts.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    m = tmp_path / "manifest.json"
    h = "0" * 16 if bad_hash else rows_hash(rows, ("id", "question", "answer"))
    m.write_text(json.dumps({"num_learnable": len(rows), "learnable_hash": h}))
    return p, m


def test_prompt_and_checkpoint_checks(tmp_path):
    p, m = _prompt_files(tmp_path)
    cfg = grpo.load_config(CONFIG, [f"prompts={p}", f"manifest={m}"])
    problems, facts = pf.check_prompts(cfg)
    assert (
        problems == []
        and facts["prompts_used"] == 6
        and facts["prompts_by_source"] == {"gsm8k_train": 1, "math_train": 5}
    )
    p, m = _prompt_files(tmp_path, bad_hash=True)
    cfg = grpo.load_config(CONFIG, [f"prompts={p}", f"manifest={m}"])
    problems, _ = pf.check_prompts(cfg)
    assert problems and "do not match" in problems[0]
    # the start checkpoint: run_info next to final/ (as rlvr-sft writes it) or inside it (as the Hub copy has it)
    run = tmp_path / "rft_1p7b"
    (run / "final").mkdir(parents=True)
    (run / "final" / "config.json").write_text("{}")
    (run / "run_info.json").write_text(
        json.dumps(
            {"git_commit": "bb6f2a6e01b0", "kind": "sft", "config": {"run_name": "rft_1p7b"}}
        )
    )
    problems, facts = pf.check_start_checkpoint(
        run / "final", commit="bb6f2a6", run_name="rft_1p7b"
    )
    assert problems == [] and facts["start_commit"].startswith("bb6f2a6")
    (run / "run_info.json").write_text(
        json.dumps({"git_commit": "deadbeef", "kind": "sft", "config": {"run_name": "sft_1p7b"}})
    )
    problems, _ = pf.check_start_checkpoint(run / "final", commit="bb6f2a6", run_name="rft_1p7b")
    assert len(problems) == 2
    assert pf.check_start_checkpoint(tmp_path / "nope", commit="x", run_name="y")[0]


def test_main_exit_codes(tmp_path, capsys):
    p, m = _prompt_files(tmp_path)
    rc = pf.main(
        [
            "--run",
            "main",
            "--config",
            str(CONFIG),
            "--set",
            f"prompts={p}",
            "--set",
            f"manifest={m}",
            "--skip-checkpoint",
        ]
    )
    assert rc == 0 and "PREFLIGHT OK" in capsys.readouterr().out
    rc = pf.main(
        [
            "--run",
            "main",
            "--config",
            str(CONFIG),
            "--set",
            f"prompts={p}",
            "--set",
            f"manifest={m}",
            "--set",
            "max_steps=500",
            "--skip-checkpoint",
        ]
    )
    assert rc == 1 and "PREFLIGHT FAILED" in capsys.readouterr().out
