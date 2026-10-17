import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "perturbation_gap.py"
spec = importlib.util.spec_from_file_location("perturbation_gap", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["perturbation_gap"] = mod
spec.loader.exec_module(mod)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_gap_pairs_by_original_id_and_counts_flips(tmp_path):
    orig = [{"problem_id": f"g{i}", "sample_idx": 0, "correct": i % 2 == 0} for i in range(100)]
    # perturbed: 80 paired problems; of the 40 originally-correct ones 10 flip to wrong,
    # of the 40 originally-wrong ones 2 flip to right; 1 unpaired perturbed id
    pert = []
    for i in range(80):
        ok = i % 2 == 0
        if ok and i < 20:
            ok = False
        if not ok and i in (1, 3):
            ok = True
        pert.append(
            {
                "problem_id": f"g{i}-p",
                "sample_idx": 0,
                "correct": ok,
                "meta": {"original_id": f"g{i}"},
            }
        )
    pert.append(
        {"problem_id": "zz-p", "sample_idx": 0, "correct": True, "meta": {"original_id": "nope"}}
    )
    _write(tmp_path / "gsm8k" / "samples.jsonl", orig)
    _write(tmp_path / "gsm8k_perturbed" / "samples.jsonl", pert)

    o = mod.first_sample_correct(tmp_path / "gsm8k" / "samples.jsonl")
    p = mod.first_sample_correct(tmp_path / "gsm8k_perturbed" / "samples.jsonl")
    g = mod.perturbation_gap(o, p)
    assert g["num_pairs"] == 80 and g["unpaired_perturbed"] == 1
    assert g["table"] == {"both": 30, "only_original": 10, "only_perturbed": 2, "neither": 38}
    assert g["acc_original_on_pairs"]["mean"] == pytest.approx(0.5)
    assert g["acc_perturbed"]["mean"] == pytest.approx(32 / 80)
    assert g["gap_original_minus_perturbed"]["mean"] == pytest.approx(8 / 80)
    assert g["gap_original_minus_perturbed"]["lo"] < 0.1 < g["gap_original_minus_perturbed"]["hi"]
    assert g["flip_rate"] == pytest.approx(12 / 80)
    row = mod.markdown_row("r", g)
    assert row.startswith("| r | 80 |") and "10 / 2" in row


def test_gap_requires_pairs(tmp_path):
    with pytest.raises(ValueError):
        mod.perturbation_gap({"a": (True, {})}, {"b": (True, {"original_id": "zzz"})})


def _write_n(path: Path, counts: dict[str, int], n: int, original_of=None) -> None:
    rows = []
    for pid, c in counts.items():
        for j in range(n):
            r = {"problem_id": pid, "sample_idx": j, "correct": j < c}
            if original_of is not None:
                r["meta"] = {"original_id": original_of[pid]}
            rows.append(r)
    _write(path, rows)


def test_pass_at_k_gap_pairs_counts_and_reports_per_k(tmp_path):
    n = 4
    orig = {f"g{i}": c for i, c in enumerate([4, 4, 2, 0, 1])}
    pert = {f"g{i}-p": c for i, c in enumerate([4, 0, 1, 0, 3])}  # g4 flips up, g1 collapses
    _write_n(tmp_path / "gsm8k" / "samples.jsonl", orig, n)
    _write_n(
        tmp_path / "gsm8k_perturbed" / "samples.jsonl",
        pert,
        n,
        {f"g{i}-p": f"g{i}" for i in range(5)},
    )
    o = mod.sample_counts(tmp_path / "gsm8k" / "samples.jsonl")
    p = mod.sample_counts(tmp_path / "gsm8k_perturbed" / "samples.jsonl")
    assert o["g2"][:2] == (4, 2)
    g = mod.perturbation_gap_pass_at_k(o, p, ks=(1, 4, 64))
    assert g["n"] == 4 and g["num_pairs"] == 5 and set(g["by_k"]) == {1, 4}
    r1 = g["by_k"][1]
    assert r1["pass_at_k_original"]["mean"] == pytest.approx(11 / 20)
    assert r1["pass_at_k_perturbed"]["mean"] == pytest.approx(8 / 20)
    assert r1["gap_original_minus_perturbed"]["mean"] == pytest.approx(3 / 20)
    assert g["by_k"][4]["gap_original_minus_perturbed"]["mean"] == pytest.approx((4 - 3) / 5)
    assert g["ever_solved"] == {"both": 3, "only_original": 1, "only_perturbed": 0, "neither": 1}
    md = mod.markdown_pass_at_k("base", g)
    assert "| base | 5 | 1 |" in md and "only original 1" in md
    with pytest.raises(ValueError):
        mod.perturbation_gap_pass_at_k(o, {"x": (8, 1, {"original_id": "g0"})})


def test_pass_at_k_gap_cli(tmp_path, capsys):
    n = 2
    _write_n(tmp_path / "gsm8k" / "samples.jsonl", {"a": 2, "b": 0}, n)
    _write_n(
        tmp_path / "gsm8k_perturbed" / "samples.jsonl",
        {"a-p": 1, "b-p": 0},
        n,
        {"a-p": "a", "b-p": "b"},
    )
    out = tmp_path / "rep" / "gap.json"
    mod.main([str(tmp_path), "--all-samples", "--out", str(out), "--ks", "1,2"])
    assert out.exists() and out.with_suffix(".md").exists()
    assert json.loads(out.read_text())["by_k"]["1"]["gap_original_minus_perturbed"][
        "mean"
    ] == pytest.approx(0.25)
