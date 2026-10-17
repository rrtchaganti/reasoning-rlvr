"""The sampling projection (scripts/project_sampling.py): parsing vLLM's progress line and the
remaining-hours rule that the Countdown stop on the L4 (17 Oct 2026) made necessary."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import project_sampling as ps

# Verbatim from outputs/e2e_exp002.log on the L4 (carriage-return separated updates, as tee wrote them).
L4_LOG = (
    "== run: rlvr-eval --config configs/sample/base_1.7b_n64_countdown.yaml\n"
    "Processed prompts:   0%|          | 0/32000 [00:00<?, ?it/s]\r"
    "Processed prompts:   7%|▋         | 2304/32000 [1:32:54<19:28:48,  2.36s/it, "
    "est. speed input: 55.08 toks/s, output: 824.81 toks/s]\r"
    "Processed prompts:   7%|▋         | 2368/32000 [1:36:05<20:58:17,  2.55s/it, "
    "est. speed input: 54.73 toks/s, output: 815.97 toks/s]"
)


def test_parse_takes_the_last_update():
    p = ps.parse_progress(L4_LOG)
    assert p["done"] == 2368 and p["total"] == 32000
    assert p["elapsed_s"] == 1 * 3600 + 36 * 60 + 5
    assert p["tqdm_eta_s"] == 20 * 3600 + 58 * 60 + 17
    assert p["output_tps"] == pytest.approx(815.97)


def test_parse_handles_no_update_and_unknown_eta():
    assert ps.parse_progress("engine starting\nno progress yet") is None
    p = ps.parse_progress("Processed prompts:   0%|          | 0/32000 [00:00<?, ?it/s]")
    assert p == {"done": 0, "total": 32000, "elapsed_s": 0.0, "tqdm_eta_s": None, "output_tps": 0.0}


def test_projection_uses_remaining_times_mean_over_the_measured_rate():
    p = ps.project(ps.parse_progress(L4_LOG), mean_tokens=2700, rate=49, max_hours=8)
    remaining = 32000 - 2368
    assert p["remaining_sequences"] == remaining
    assert p["remaining_tokens"] == remaining * 2700
    assert p["projected_hours"] == pytest.approx(remaining * 2700 / 815.97 / 3600)
    assert p["projected_hours"] > 25  # the L4 reality: a day of sampling for 500 problems
    assert p["tqdm_eta_hours"] == pytest.approx(20.97, abs=0.01)  # tqdm said 21: it underestimates
    assert p["mean_tokens_finished_so_far"] == pytest.approx(815.97 * 5765 / 2368)
    assert p["projected_cost"] == pytest.approx(p["projected_hours"] * 49)
    assert p["stay"] is False
    assert "STOP" in ps.render(p)


def test_projection_stays_under_the_limit_on_a_fast_card():
    fast = dict(ps.parse_progress(L4_LOG), output_tps=4000.0)
    p = ps.project(fast, mean_tokens=2700, rate=189, max_hours=8)
    assert p["projected_hours"] == pytest.approx((32000 - 2368) * 2700 / 4000 / 3600)
    assert p["stay"] is True
    text = ps.render(p)
    assert "CONTINUE" in text and "INR 189/h" in text


def test_projection_refuses_a_zero_rate():
    with pytest.raises(ValueError):
        ps.project(
            {"done": 0, "total": 10, "elapsed_s": 0.0, "output_tps": 0.0},
            mean_tokens=1,
            rate=1,
            max_hours=1,
        )


def test_mean_tokens_from_metrics_reads_eval_or_run_level(tmp_path):
    ev = tmp_path / "metrics.json"
    ev.write_text(json.dumps({"tokens": {"mean": 2715.6, "median": 3962, "p90": 3964}}))
    assert ps.mean_tokens_from_metrics(ev) == pytest.approx(2715.6)
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps({"countdown": {"tokens": {"mean": 2715.6}}, "gsm8k": {"tokens": {"mean": 154}}})
    )
    assert ps.mean_tokens_from_metrics(run, "countdown") == pytest.approx(2715.6)
    with pytest.raises(KeyError):
        ps.mean_tokens_from_metrics(run)


def test_cli_end_to_end(tmp_path, capsys):
    log = tmp_path / "exp002.log"
    log.write_text(L4_LOG, encoding="utf-8")
    ps.main([str(log), "--mean-tokens", "2700", "--rate", "49", "--max-hours", "8"])
    out = capsys.readouterr().out
    assert "2,368 / 32,000" in out and "verdict: STOP" in out
    ps.main([str(log), "--mean-tokens", "2700", "--json"])
    assert json.loads(capsys.readouterr().out)["remaining_sequences"] == 29632
