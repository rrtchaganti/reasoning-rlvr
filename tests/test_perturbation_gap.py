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
