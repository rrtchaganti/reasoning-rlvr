"""The evidence index: from every quoted claim to the problems and completions behind it.

Usage (repo root, venv active):
    python scripts/evidence.py --claims configs/evidence/claims.yaml \
        --root P1=outputs --root P23=../h100_session_archive/outputs \
        --root P4=../h100_p4_session2_archive/outputs --root A2=../h100_p4_session2_addendum2/outputs \
        --root HUB=../hub_samples --root LAP=outputs \
        --out outputs/reports/evidence

``claims.yaml`` lists the claims the reports and README make, each with its experiment id, the
sentence as quoted, where it is quoted, the runs it rests on (a root name plus a path inside it),
the eval set, a selection rule, and the number as quoted (``expect``). For each claim the script
re-derives the number from the per-sample files, compares it with the quoted one, lists every
problem the rule selects, and writes ``<out>/<claim id>.md`` with the counts, the ids and short
excerpts of the first few completions (the full texts stay in the sample files, whose Hub location
each root declares). ``<out>/README.md`` is the index: one row per claim with the quoted and
re-derived numbers and whether they agree. A mismatch is reported, never hidden: the index is the
check that every sentence has a file behind it.

Rules (``rule:`` in the YAML):
  paired_flips       two greedy runs on shared ids: accuracies, the paired difference in points,
                     the problems only the reference got and only the new run got
  decomposition      the Phase 4 split of a paired gain by the reference's truncation
                     (scripts/decompose_gain.py's cells), with the problems in each cell
  truncation_rate    one run: the share of first samples cut at the window, and those completions
  truncated_right    one run: accuracy of truncated versus finished completions, the extraction
                     source of the truncated-right ones
  forced             one forced-stop run: the forced share, the forced answers' accuracy, examples
                     of forced-right and forced-wrong
  best_of_k          one scored sampled run against a greedy reference: the verifier's pick of k
                     seeded draws per problem, with each drawn sample's score (EXP-022)
  maj_k              one sampled run against a greedy reference: maj@k re-derived with the seeded
                     draws of scripts/plot_accuracy_vs_tokens.py, and per problem the drawn sample
                     indices, their answers, the vote and the hit; the paired difference
  pass_k_newly_solved  two sampled runs: problems solved at least once by one and never by the other
  perturbation_gap   one run's original eval against its perturbed (or names-only) eval, paired by
                     meta.original_id: the gap and the problems that flipped
  reward_log         a training run's reward_log.jsonl: first/last-window means of reward, finished
                     rate, tokens, zero-variance rate, format-only rate
  file_pointer       no computation: the files a claim rests on, with a note
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rlvr.eval.extraction import normalize_for_vote
from rlvr.eval.metrics import bootstrap_ci, majority_vote, paired_bootstrap_diff, pass_at_k

EXCERPT = 240


# ----------------------------------------------------------------------------- loading


def load_first(run_dir: Path, ev: str) -> dict[str, dict[str, Any]]:
    """problem_id -> first-sample record."""
    p = Path(run_dir) / ev / "samples.jsonl"
    if not p.exists():
        raise FileNotFoundError(p)
    out: dict[str, dict[str, Any]] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("sample_idx", 0) == 0:
                out[r["problem_id"]] = r
    return out


def load_all(run_dir: Path, ev: str) -> dict[str, list[dict[str, Any]]]:
    """problem_id -> all sample records sorted by sample_idx."""
    p = Path(run_dir) / ev / "samples.jsonl"
    if not p.exists():
        raise FileNotFoundError(p)
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["problem_id"]].append(r)
    for pid in out:
        out[pid].sort(key=lambda r: r["sample_idx"])
    return dict(out)


def excerpt(text: str, n: int = EXCERPT) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    t = t[-n:] if len(t) > n else t
    return t.replace("|", "\\|")


def finished(r: dict[str, Any]) -> bool:
    return r.get("finish_reason") != "length"


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def ci(e: dict[str, Any]) -> str:
    return f"{100 * e['mean']:.1f} [{100 * e['lo']:.1f}, {100 * e['hi']:.1f}]"


def sample_row(label: str, r: dict[str, Any]) -> str:
    return (
        f"| {label} | {'right' if r.get('correct') else 'wrong'} | `{excerpt(str(r.get('extracted')), 40)}` | "
        f"{r.get('finish_reason')} | {r.get('num_tokens')} | {'forced' if r.get('forced_stop') else ''} | {excerpt(r.get('text', ''))} |"
    )


SAMPLE_HEADER = "| run | graded | extracted | finish | tokens | note | tail of the completion |\n|---|---|---|---|---|---|---|"


# ----------------------------------------------------------------------------- rules


def rule_paired_flips(c: dict[str, Any], roots: dict[str, Path], ev: str) -> dict[str, Any]:
    ref = load_first(roots[c["ref"]["root"]] / c["ref"]["run"], ev)
    new = load_first(roots[c["new"]["root"]] / c["new"]["run"], ev)
    ids = sorted(set(ref) & set(new))
    if c.get("restrict_to"):
        rr = c["restrict_to"]
        keep = set(load_first(roots[rr["root"]] / rr["run"], rr.get("eval", ev)))
        ids = [i for i in ids if i in keep]
    a = [float(ref[i]["correct"]) for i in ids]
    b = [float(new[i]["correct"]) for i in ids]
    diff = paired_bootstrap_diff(b, a)
    only_ref = [i for i in ids if ref[i]["correct"] and not new[i]["correct"]]
    only_new = [i for i in ids if new[i]["correct"] and not ref[i]["correct"]]
    lr, ln = c["ref"].get("label", "ref"), c["new"].get("label", "new")
    lines = [
        f"Paired on {len(ids)} shared problems: {lr} {pct(np.mean(a))}, {ln} {pct(np.mean(b))}, "
        f"{ln} - {lr} = **{100 * diff.mean:+.1f} [{100 * diff.lo:+.1f}, {100 * diff.hi:+.1f}]** points; "
        f"only {lr} right: {len(only_ref)}; only {ln} right: {len(only_new)}.",
        "",
        f"**Problems only {ln} got right** ({len(only_new)}): "
        + ", ".join(f"`{i}`" for i in only_new[:60])
        + (" ..." if len(only_new) > 60 else ""),
        "",
        f"**Problems only {lr} got right** ({len(only_ref)}): "
        + ", ".join(f"`{i}`" for i in only_ref[:60])
        + (" ..." if len(only_ref) > 60 else ""),
        "",
        f"**Examples** (first {c.get('examples', 5)} of each kind; gold in the first column):",
        "",
    ]
    for kind, lst in (("only " + ln, only_new), ("only " + lr, only_ref)):
        for i in lst[: c.get("examples", 5)]:
            lines += [
                f"- `{i}` (gold `{excerpt(str(ref[i].get('gold')), 40)}`, {kind})",
                "",
                SAMPLE_HEADER,
                sample_row(lr, ref[i]),
                sample_row(ln, new[i]),
                "",
            ]
    return {
        "derived": round(100 * diff.mean, 1),
        "n": len(ids),
        "only_ref": len(only_ref),
        "only_new": len(only_new),
        "lines": lines,
    }


def rule_decomposition(c: dict[str, Any], roots: dict[str, Path], ev: str) -> dict[str, Any]:
    ref = load_first(roots[c["ref"]["root"]] / c["ref"]["run"], ev)
    new = load_first(roots[c["new"]["root"]] / c["new"]["run"], ev)
    ids = sorted(set(ref) & set(new))
    n = len(ids)
    lr, ln = c["ref"].get("label", "ref"), c["new"].get("label", "new")
    gain = lambda i: float(new[i]["correct"]) - float(ref[i]["correct"])  # noqa: E731
    on_tr = [i for i in ids if not finished(ref[i])]
    on_fi = [i for i in ids if finished(ref[i])]
    both = [i for i in ids if finished(ref[i]) and finished(new[i])]
    g_tr = [gain(i) if i in set(on_tr) else 0.0 for i in ids]
    g_fi = [gain(i) if i in set(on_fi) else 0.0 for i in ids]
    e_tr, e_fi = bootstrap_ci(g_tr), bootstrap_ci(g_fi)
    e_both = (
        paired_bootstrap_diff(
            [float(new[i]["correct"]) for i in both], [float(ref[i]["correct"]) for i in both]
        )
        if both
        else None
    )
    tr_new_right = [i for i in on_tr if new[i]["correct"] and not ref[i]["correct"]]
    both_to_new = [i for i in both if new[i]["correct"] and not ref[i]["correct"]]
    both_to_ref = [i for i in both if ref[i]["correct"] and not new[i]["correct"]]
    net = 100 * (np.mean(g_tr) + np.mean(g_fi))
    lines = [
        f"{n} pairs; net {ln} - {lr} = {net:+.1f} points, of which on problems {lr} truncated **{100 * e_tr.mean:+.1f} [{100 * e_tr.lo:+.1f}, {100 * e_tr.hi:+.1f}]** "
        f"({len(on_tr)} problems) and on problems {lr} finished {100 * e_fi.mean:+.1f} [{100 * e_fi.lo:+.1f}, {100 * e_fi.hi:+.1f}] ({len(on_fi)}); "
        + (
            f"both finished: {len(both)} problems, paired difference **{100 * e_both.mean:+.1f} [{100 * e_both.lo:+.1f}, {100 * e_both.hi:+.1f}]** ({len(both_to_new)} flipped to {ln}, {len(both_to_ref)} to {lr})."
            if e_both
            else "no problem both finished."
        ),
        "",
        f"**{lr} truncated, {ln} right, {lr} wrong** ({len(tr_new_right)}): "
        + ", ".join(f"`{i}`" for i in tr_new_right[:60])
        + (" ..." if len(tr_new_right) > 60 else ""),
        "",
        f"**Both finished, flipped to {ln}** ({len(both_to_new)}): "
        + ", ".join(f"`{i}`" for i in both_to_new[:40]),
        "",
        f"**Both finished, flipped to {lr}** ({len(both_to_ref)}): "
        + ", ".join(f"`{i}`" for i in both_to_ref[:40]),
        "",
        "**Examples:**",
        "",
    ]
    k = c.get("examples", 4)
    for kind, lst in (
        (f"{lr} truncated, {ln} right", tr_new_right),
        (f"both finished, flipped to {ln}", both_to_new),
        (f"both finished, flipped to {lr}", both_to_ref),
    ):
        for i in lst[:k]:
            lines += [
                f"- `{i}` (gold `{excerpt(str(ref[i].get('gold')), 40)}`, {kind})",
                "",
                SAMPLE_HEADER,
                sample_row(lr, ref[i]),
                sample_row(ln, new[i]),
                "",
            ]
    return {
        "derived": round(100 * e_tr.mean, 1),
        "derived_both_finished": round(100 * e_both.mean, 1) if e_both else None,
        "n": n,
        "lines": lines,
    }


def rule_truncation_rate(c: dict[str, Any], roots: dict[str, Path], ev: str) -> dict[str, Any]:
    run = load_first(roots[c["run"]["root"]] / c["run"]["run"], ev)
    ids = sorted(run)
    tr = [i for i in ids if not finished(run[i])]
    rate = len(tr) / len(ids)
    lines = [
        f"{len(tr)} of {len(ids)} first samples cut at the window: **{pct(rate)} %**; of those, right: {pct(np.mean([run[i]['correct'] for i in tr]) if tr else 0)} %.",
        "",
        "**Truncated problems:** "
        + ", ".join(f"`{i}`" for i in tr[:80])
        + (" ..." if len(tr) > 80 else ""),
        "",
        "**Examples** (the tail of the cut completion):",
        "",
        SAMPLE_HEADER,
    ]
    lines += [sample_row(c["run"].get("label", "run"), run[i]) for i in tr[: c.get("examples", 6)]]
    return {"derived": round(100 * rate, 1), "n": len(ids), "lines": lines}


def rule_truncated_right(c: dict[str, Any], roots: dict[str, Path], ev: str) -> dict[str, Any]:
    run = load_first(roots[c["run"]["root"]] / c["run"]["run"], ev)
    ids = sorted(run)
    tr = [i for i in ids if not finished(run[i])]
    fi = [i for i in ids if finished(run[i])]
    acc_tr = np.mean([run[i]["correct"] for i in tr]) if tr else float("nan")
    acc_fi = np.mean([run[i]["correct"] for i in fi]) if fi else float("nan")
    right = [i for i in tr if run[i]["correct"]]
    src = Counter(run[i]["extraction_source"] for i in right)
    lines = [
        f"Truncated: {len(tr)} of {len(ids)}, right **{pct(acc_tr)} %**; finished: {len(fi)}, right {pct(acc_fi)} %. The truncated-right answers were extracted by: {dict(src)}.",
        "",
        "**Truncated and right:** "
        + ", ".join(f"`{i}`" for i in right[:80])
        + (" ..." if len(right) > 80 else ""),
        "",
        "**Examples (truncated and right; the extractor's source in the note column is in the table above):**",
        "",
        SAMPLE_HEADER,
    ]
    lines += [
        sample_row(c["run"].get("label", "run"), run[i]) for i in right[: c.get("examples", 6)]
    ]
    return {"derived": round(100 * acc_tr, 1), "n": len(tr), "lines": lines}


def rule_forced(c: dict[str, Any], roots: dict[str, Path], ev: str) -> dict[str, Any]:
    run = load_first(roots[c["run"]["root"]] / c["run"]["run"], ev)
    ids = sorted(run)
    forced = [i for i in ids if run[i].get("forced_stop")]
    acc_f = np.mean([run[i]["correct"] for i in forced]) if forced else float("nan")
    right = [i for i in forced if run[i]["correct"]]
    wrong = [i for i in forced if not run[i]["correct"]]
    lines = [
        f"Forced: {len(forced)} of {len(ids)} (**{pct(len(forced) / len(ids))} %**), right **{pct(acc_f)} %**; the run's accuracy {pct(np.mean([run[i]['correct'] for i in ids]))} %.",
        "",
        "**Forced and right:** "
        + ", ".join(f"`{i}`" for i in right[:60])
        + (" ..." if len(right) > 60 else ""),
        "",
        "**Examples** (the tail shows the appended `</think>` `<answer>` and the model's continuation):",
        "",
        SAMPLE_HEADER,
    ]
    k = c.get("examples", 4)
    lines += [sample_row("forced, right", run[i]) for i in right[:k]] + [
        sample_row("forced, wrong", run[i]) for i in wrong[:k]
    ]
    return {
        "derived": round(100 * acc_f, 1),
        "derived_forced_rate": round(100 * len(forced) / len(ids), 1),
        "n": len(ids),
        "lines": lines,
    }


def rule_best_of_k(c: dict[str, Any], roots: dict[str, Path], ev: str) -> dict[str, Any]:
    """EXP-022: best-of-k with the outcome reward model's scores (``scores.jsonl`` next to the
    samples), over the same seeded draws as maj@k, against a greedy reference; per problem the drawn
    indices, their scores and answers, the pick and whether it was right."""
    samples = load_all(roots[c["run"]["root"]] / c["run"]["run"], ev)
    ref = load_first(roots[c["ref"]["root"]] / c["ref"]["run"], ev)
    scores: dict[str, dict[int, float]] = defaultdict(dict)
    sp = roots[c["run"]["root"]] / c["run"]["run"] / ev / "scores.jsonl"
    if not sp.exists():
        raise FileNotFoundError(sp)
    with sp.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                scores[r["problem_id"]][int(r["sample_idx"])] = float(r["score"])
    ids = sorted(set(samples) & set(ref))
    k, repeats = int(c["k"]), int(c.get("repeats", 20))
    per_problem, audit = [], []
    for i in ids:
        rows = samples[i]
        n = len(rows)
        rng = np.random.default_rng(0)
        n_rep = 1 if k >= n else repeats
        hits, first = 0.0, None
        for _ in range(n_rep):
            idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
            best = max(idx, key=lambda j: scores[i].get(int(rows[j]["sample_idx"]), float("-inf")))
            hit = bool(rows[best]["correct"])
            hits += float(hit)
            if first is None:
                first = (sorted(int(j) for j in idx), int(best), hit)
        per_problem.append(hits / n_rep)
        audit.append((i, first, hits / n_rep))
    refc = [float(ref[i]["correct"]) for i in ids]
    e = bootstrap_ci(per_problem, seed=k)
    d = paired_bootstrap_diff(per_problem, refc, seed=k)
    lr = c["ref"].get("label", "ref")
    lines = [
        f"best-of-{k} of `{c['run'].get('label', 'run')}` (the verifier's highest-scored of {k} seeded draws) on {len(ids)} shared problems: **{ci(e.as_dict())}**; "
        f"{lr} greedy {pct(np.mean(refc))}; paired difference **{100 * d.mean:+.1f} [{100 * d.lo:+.1f}, {100 * d.hi:+.1f}]**.",
        "",
        f"**Per problem, the first seeded draw** (sample index: verifier score, answer, right?; the pick; then {lr} greedy):",
        "",
        f"| problem | drawn samples (score, answer, right) | pick | hit (mean over draws) | {lr} greedy |",
        "|---|---|---|---|---|",
    ]
    for i, (idx, best, hit), mean_hit in audit[: c.get("examples", 25)]:
        cells = "; ".join(
            f"{j}: {scores[i].get(int(samples[i][j]['sample_idx']), float('nan')):+.2f}, `{excerpt(str(samples[i][j]['extracted']), 20)}`, {'right' if samples[i][j]['correct'] else 'wrong'}"
            for j in idx
        )
        lines.append(
            f"| `{i}` | {cells} | {best} | {'yes' if hit else 'no'} ({mean_hit:.2f}) | {'right' if ref[i]['correct'] else 'wrong'} |"
        )
    lines += [
        "",
        "The scores are in the run's `scores.jsonl`; every sample's full text in its `samples.jsonl`.",
    ]
    return {"derived": round(100 * d.mean, 1), "n": len(ids), "lines": lines}


def rule_maj_k(c: dict[str, Any], roots: dict[str, Path], ev: str) -> dict[str, Any]:
    samples = load_all(roots[c["run"]["root"]] / c["run"]["run"], ev)
    ref = load_first(roots[c["ref"]["root"]] / c["ref"]["run"], ev)
    ids = sorted(set(samples) & set(ref))
    if c.get("restrict_to"):
        rr = c["restrict_to"]
        keep = set(load_first(roots[rr["root"]] / rr["run"], rr.get("eval", ev)))
        ids = [i for i in ids if i in keep]
    k, repeats = int(c["k"]), int(c.get("repeats", 20))
    per_problem, audit = [], []
    for i in ids:
        rows = samples[i]
        n = len(rows)
        rng = np.random.default_rng(0)
        correct_keys = {
            normalize_for_vote(r["extracted"]) for r in rows if r["correct"] and r["extracted"]
        }
        n_rep = 1 if k >= n else repeats
        hits, draws = 0.0, []
        for _ in range(n_rep):
            idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
            voted = majority_vote([rows[j]["extracted"] for j in idx], normalize=normalize_for_vote)
            hit = voted is not None and voted in correct_keys
            hits += float(hit)
            draws.append((sorted(int(j) for j in idx), voted, hit))
        per_problem.append(hits / n_rep)
        audit.append((i, draws[0], hits / n_rep))
    refc = [float(ref[i]["correct"]) for i in ids]
    e = bootstrap_ci(per_problem, seed=k)
    d = paired_bootstrap_diff(per_problem, refc, seed=k)
    lr = c["ref"].get("label", "ref")
    tokens = float(np.mean([k * np.mean([r["num_tokens"] for r in samples[i]]) for i in ids]))
    lines = [
        f"maj@{k} of `{c['run'].get('label', 'run')}` on {len(ids)} shared problems: **{ci(e.as_dict())}** (hit rate averaged over {repeats} seeded draws of {k} of the {len(samples[ids[0]])} samples per problem); "
        f"{lr} greedy {pct(np.mean(refc))}; paired difference **{100 * d.mean:+.1f} [{100 * d.lo:+.1f}, {100 * d.hi:+.1f}]**; expected tokens per problem {tokens:.0f}.",
        "",
        f"**Per problem, the first seeded draw** (sample indices, each one's answer, the vote, hit; then {lr} greedy):",
        "",
        f"| problem | drawn samples | answers | vote | hit (mean over draws) | {lr} greedy |",
        "|---|---|---|---|---|---|",
    ]
    for i, (idx, voted, hit), mean_hit in audit[: c.get("examples", 25)]:
        answers = ", ".join(f"`{excerpt(str(samples[i][j]['extracted']), 24)}`" for j in idx)
        lines.append(
            f"| `{i}` | {idx} | {answers} | `{excerpt(str(voted), 24)}` | {'yes' if hit else 'no'} ({mean_hit:.2f}) | {'right' if ref[i]['correct'] else 'wrong'} |"
        )
    lines += [
        "",
        f"The other {max(0, len(ids) - c.get('examples', 25))} problems follow the same draws (seed 0 per problem); every sample's full text is in the run's `samples.jsonl`.",
    ]
    return {
        "derived": round(100 * d.mean, 1),
        "derived_maj": round(100 * e.mean, 1),
        "n": len(ids),
        "lines": lines,
    }


def rule_pass_k_newly_solved(c: dict[str, Any], roots: dict[str, Path], ev: str) -> dict[str, Any]:
    ref = load_all(roots[c["ref"]["root"]] / c["ref"]["run"], ev)
    new = load_all(roots[c["new"]["root"]] / c["new"]["run"], ev)
    ids = sorted(set(ref) & set(new))
    n = len(ref[ids[0]])
    k = int(c.get("k", 16))
    cr = [sum(r["correct"] for r in ref[i]) for i in ids]
    cn = [sum(r["correct"] for r in new[i]) for i in ids]
    d = paired_bootstrap_diff(
        [pass_at_k(n, x, k) for x in cn], [pass_at_k(n, x, k) for x in cr], seed=k
    )
    only_new = [i for i, a, b in zip(ids, cr, cn, strict=True) if b and not a]
    only_ref = [i for i, a, b in zip(ids, cr, cn, strict=True) if a and not b]
    lr, ln = c["ref"].get("label", "ref"), c["new"].get("label", "new")
    lines = [
        f"{len(ids)} shared problems, n = {n} samples each: pass@{k} {ln} - {lr} = **{100 * d.mean:+.1f} [{100 * d.lo:+.1f}, {100 * d.hi:+.1f}]**; solved at least once in {n} tries by {ln} and never by {lr}: **{len(only_new)}**; the reverse: **{len(only_ref)}**.",
        "",
        f"**Only {ln} ever solved:** "
        + ", ".join(f"`{i}` ({cn[ids.index(i)]} of {n})" for i in only_new),
        "",
        f"**Only {lr} ever solved:** "
        + ", ".join(f"`{i}` ({cr[ids.index(i)]} of {n})" for i in only_ref),
        "",
        f"**Examples** (for the first problems only {ln} solved: one right sample of {ln} and one sample of {lr}):",
        "",
        SAMPLE_HEADER,
    ]
    for i in only_new[: c.get("examples", 3)]:
        right = next(r for r in new[i] if r["correct"])
        lines += [sample_row(ln, right), sample_row(lr, ref[i][0])]
    return {
        "derived": round(100 * d.mean, 1),
        "only_new": len(only_new),
        "only_ref": len(only_ref),
        "n": len(ids),
        "lines": lines,
    }


def rule_perturbation_gap(c: dict[str, Any], roots: dict[str, Path], ev: str) -> dict[str, Any]:
    orig = load_first(roots[c["run"]["root"]] / c["run"]["run"], c.get("original_eval", "gsm8k"))
    pert_root = c.get("perturbed_run", c["run"])
    pert = load_first(roots[pert_root["root"]] / pert_root["run"], ev)
    pairs = [
        (r["meta"]["original_id"], pid)
        for pid, r in pert.items()
        if r.get("meta", {}).get("original_id") in orig
    ]
    a = [float(orig[o]["correct"]) for o, _ in pairs]
    b = [float(pert[p]["correct"]) for _, p in pairs]
    gap = paired_bootstrap_diff(a, b)
    lost = [(o, p) for o, p in pairs if orig[o]["correct"] and not pert[p]["correct"]]
    gained = [(o, p) for o, p in pairs if pert[p]["correct"] and not orig[o]["correct"]]
    lines = [
        f"{len(pairs)} pairs: original {pct(np.mean(a))} %, {ev} {pct(np.mean(b))} %, gap (original - perturbed) **{100 * gap.mean:+.1f} [{100 * gap.lo:+.1f}, {100 * gap.hi:+.1f}]**; right on the original and wrong on the rewrite: {len(lost)}; the reverse: {len(gained)}.",
        "",
        "**Lost on the rewrite:** "
        + ", ".join(f"`{p}`" for _, p in lost[:60])
        + (" ..." if len(lost) > 60 else ""),
        "",
        "**Examples** (the original and its rewrite; the rewrite's substitutions in the note):",
        "",
        SAMPLE_HEADER,
    ]
    for o, p in lost[: c.get("examples", 4)]:
        lines += [
            sample_row("original", orig[o]),
            sample_row(
                f"rewrite {excerpt(json.dumps(pert[p].get('meta', {}).get('substitutions', pert[p].get('meta', {}).get('name_map', {}))), 60)}",
                pert[p],
            ),
        ]
    return {"derived": round(100 * gap.mean, 1), "n": len(pairs), "lines": lines}


def rule_reward_log(c: dict[str, Any], roots: dict[str, Path], ev: str) -> dict[str, Any]:
    p = roots[c["run"]["root"]] / c["run"]["run"] / "reward_log.jsonl"
    rows = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
    w = int(c.get("window", 20))
    first, last = rows[:w], rows[-w:]
    mean = lambda rs, k: float(np.mean([r[k] for r in rs if k in r]))  # noqa: E731
    keys = [
        "reward",
        "finished_rate",
        "mean_tokens",
        "zero_variance_group_rate",
        "format_only_rate",
    ]
    lines = [
        f"`{p.name}` of `{c['run']['run']}`: {len(rows)} steps; window means over the first and last {w} steps:",
        "",
        "| statistic | first window | last window | mean over the run |",
        "|---|---|---|---|",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        if k in rows[0]:
            f, la, m = mean(first, k), mean(last, k), mean(rows, k)
            out[k] = (f, la, m)
            lines.append(f"| {k} | {f:.3f} | {la:.3f} | {m:.3f} |")
    derived = out.get(c.get("expect_key", "reward"), (None, None, None))[1]
    return {
        "derived": round(100 * derived, 1)
        if derived is not None and c.get("expect_key", "reward") != "mean_tokens"
        else derived,
        "n": len(rows),
        "lines": lines,
    }


def rule_file_pointer(c: dict[str, Any], roots: dict[str, Path], ev: str) -> dict[str, Any]:
    lines = [c.get("note", ""), "", "**Files:**"] + [
        f"- `{roots[f['root']] / f['path']}` ({f.get('what', '')})" for f in c.get("files", [])
    ]
    return {"derived": None, "n": None, "lines": lines}


RULES = {
    "paired_flips": rule_paired_flips,
    "decomposition": rule_decomposition,
    "truncation_rate": rule_truncation_rate,
    "truncated_right": rule_truncated_right,
    "forced": rule_forced,
    "maj_k": rule_maj_k,
    "best_of_k": rule_best_of_k,
    "pass_k_newly_solved": rule_pass_k_newly_solved,
    "perturbation_gap": rule_perturbation_gap,
    "reward_log": rule_reward_log,
    "file_pointer": rule_file_pointer,
}


# ----------------------------------------------------------------------------- driver


def agree(expect: float | None, derived: float | None, tol: float) -> str:
    if expect is None or derived is None:
        return "n/a"
    return "yes" if abs(float(expect) - float(derived)) <= tol else "**NO**"


def run_claim(
    c: dict[str, Any], roots: dict[str, Path], hub: dict[str, str]
) -> list[dict[str, Any]]:
    """One result per eval of the claim (a claim may span several evals with one expect each)."""
    evals = c.get("evals") or {c.get("eval", "-"): c.get("expect")}
    results = []
    for ev, expect in evals.items():
        exp_main = expect.get("value") if isinstance(expect, dict) else expect
        res = RULES[c["rule"]](c, roots, ev)
        tol = float(c.get("tolerance", 0.2))
        ok = agree(exp_main, res["derived"], tol)
        extra = ""
        if (
            isinstance(expect, dict)
            and "both_finished" in expect
            and res.get("derived_both_finished") is not None
        ):
            ok2 = agree(expect["both_finished"], res["derived_both_finished"], tol)
            extra = f"; both-finished quoted {expect['both_finished']:+.1f}, re-derived {res['derived_both_finished']:+.1f}, agree: {ok2}"
            if ok2 != "yes":
                ok = "**NO**"
        results.append(
            {
                "claim": c,
                "eval": ev,
                "expect": exp_main,
                "derived": res["derived"],
                "agree": ok,
                "extra": extra,
                "res": res,
            }
        )
    return results


def write_claim(
    out_dir: Path,
    c: dict[str, Any],
    results: list[dict[str, Any]],
    roots: dict[str, Path],
    hub: dict[str, str],
) -> Path:
    path = out_dir / f"{c['id']}.md"
    lines = [
        f"# {c['id']}: {c['title']}",
        "",
        f"**Experiment.** {c.get('experiment', '-')}. **Quoted in.** {c.get('where', '-')}.",
        "",
        f"**The claim.** {c['statement']}",
        "",
    ]
    runs = [c.get(k) for k in ("run", "ref", "new") if c.get(k)]
    if runs:
        lines.append("**Runs** (root: path; the root's Hub location is in the index):")
        lines += [
            f"- {r.get('label', k)}: `{r['root']}: {r['run']}`"
            for k, r in zip(("run", "ref", "new"), runs, strict=False)
        ]
        lines.append("")
    for r in results:
        head = f"## {r['eval']}" if r["eval"] != "-" else "## Evidence"
        check = (
            f"Quoted **{r['expect']}**, re-derived **{r['derived']}**, agree: {r['agree']}{r['extra']}."
            if r["expect"] is not None
            else ""
        )
        lines += [head, "", check, ""] + r["res"]["lines"] + [""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--claims", type=Path, required=True)
    ap.add_argument(
        "--root",
        action="append",
        default=[],
        help="NAME=PATH for each root named in the claims file",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="claim ids to run (default all)")
    args = ap.parse_args(argv)
    spec = yaml.safe_load(args.claims.read_text(encoding="utf-8"))
    roots = {}
    for item in args.root:
        name, p = item.split("=", 1)
        roots[name] = Path(p)
    hub = {k: v.get("hub", "") for k, v in spec.get("roots", {}).items()}
    args.out.mkdir(parents=True, exist_ok=True)
    index = [
        "# Evidence index: every quoted claim, the number re-derived from the sample files, and the problems behind it",
        "",
        'Generated by `scripts/evidence.py` from `configs/evidence/claims.yaml`. A claim\'s file lists the problems its rule selects and excerpts of their completions; the full texts are in the sample files at the roots below. "agree" compares the number as quoted in the report with the number re-derived here (tolerance 0.2 points unless the claim says otherwise); a **NO** is a sentence to fix.',
        "",
        "**Roots:**",
        "",
    ]
    index += [
        f"- `{k}`: {v.get('desc', '')} ({v.get('hub', '')})"
        for k, v in spec.get("roots", {}).items()
    ]
    index += [
        "",
        "| id | experiment | eval | claim | quoted | re-derived | agree | file |",
        "|---|---|---|---|---|---|---|---|",
    ]
    n_bad = 0
    for c in spec["claims"]:
        if args.only and c["id"] not in args.only:
            continue
        missing = [k for k in ("run", "ref", "new") if c.get(k) and c[k]["root"] not in roots]
        if missing:
            index.append(
                f"| {c['id']} | {c.get('experiment', '')} | - | {c['title']} | - | - | root missing | - |"
            )
            continue
        try:
            results = run_claim(c, roots, hub)
        except FileNotFoundError as e:
            index.append(
                f"| {c['id']} | {c.get('experiment', '')} | - | {c['title']} | - | - | file missing: `{e}` | - |"
            )
            continue
        path = write_claim(args.out, c, results, roots, hub)
        for r in results:
            if r["agree"] == "**NO**":
                n_bad += 1
            index.append(
                f"| {c['id']} | {c.get('experiment', '')} | {r['eval']} | {c['title']} | {r['expect'] if r['expect'] is not None else '-'} | {r['derived'] if r['derived'] is not None else '-'} | {r['agree']} | [`{path.name}`]({path.name}) |"
            )
        print(
            f"{c['id']}: "
            + "; ".join(
                f"{r['eval']} quoted {r['expect']} derived {r['derived']} agree {r['agree']}"
                for r in results
            ),
            flush=True,
        )
    (args.out / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out / 'README.md'}; disagreements: {n_bad}")


if __name__ == "__main__":
    main()
