"""Training-data overlap probe (PLAN.md section 4.3, probe 1).

Two detectors, both over whitespace-normalised lower-cased text:

- **13-gram overlap**: an eval problem is "hit" if any of its 13-token windows appears in any
  training item (the GPT-3 / Brown et al. 2020 convention, also used by the Qwen reports).
  Implemented with a hash set of eval n-grams, so a pass over millions of training items is
  one dictionary lookup per window.
- **Fuzzy match**: normalised indel similarity (rapidfuzz) above a threshold between an eval
  problem and a training item, with a length pre-filter so only comparable items are
  compared. Catches paraphrases and number-swapped copies that 13-grams miss.

Output: for each eval set, the list of hit eval ids with the training ids that hit them, and
the training ids to drop. Dropping is done by the caller; this module only reports.
"""

from __future__ import annotations

import bisect
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_text(s: str) -> str:
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def ngrams(tokens: Sequence[str], n: int) -> Iterable[tuple[str, ...]]:
    if len(tokens) < n:
        # short items: use the whole item as a single "gram" so they are still comparable
        yield tuple(tokens)
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


@dataclass
class OverlapReport:
    eval_name: str
    n: int
    num_eval: int
    num_train: int
    ngram_hits: dict[str, list[str]] = field(default_factory=dict)  # eval_id -> train ids
    fuzzy_hits: dict[str, list[tuple[str, float]]] = field(
        default_factory=dict
    )  # eval_id -> (train id, score)
    # n-gram hits whose shared windows cover at least ``min_coverage`` of the eval problem;
    # None when the coverage filter was not run
    ngram_hits_covered: dict[str, list[str]] | None = None
    min_coverage: float | None = None

    @property
    def train_ids_to_drop(self) -> set[str]:
        """Conservative: every training item that hit by any detector, boilerplate included."""
        ids = {t for ts in self.ngram_hits.values() for t in ts}
        ids |= {t for ts in self.fuzzy_hits.values() for t, _ in ts}
        return ids

    @property
    def contaminated_eval_ids(self) -> set[str]:
        """Eval problems counted as present in the training data: an n-gram hit that covers
        at least ``min_coverage`` of the problem (when the filter ran, else any n-gram hit)
        or a fuzzy hit."""
        grams = self.ngram_hits if self.ngram_hits_covered is None else self.ngram_hits_covered
        return set(grams) | set(self.fuzzy_hits)

    def summary(self) -> dict:
        return {
            "eval": self.eval_name,
            "n": self.n,
            "num_eval": self.num_eval,
            "num_train": self.num_train,
            "eval_hit_ngram": len(self.ngram_hits),
            "eval_hit_ngram_covered": None
            if self.ngram_hits_covered is None
            else len(self.ngram_hits_covered),
            "min_coverage": self.min_coverage,
            "eval_hit_fuzzy": len(self.fuzzy_hits),
            "eval_hit_any": len(set(self.ngram_hits) | set(self.fuzzy_hits)),
            "eval_contaminated": len(self.contaminated_eval_ids),
            "train_to_drop": len(self.train_ids_to_drop),
        }


def ngram_overlap(
    eval_items: dict[str, str],
    train_items: dict[str, str],
    *,
    n: int = 13,
    max_hits_per_eval: int = 20,
    min_coverage: float | None = None,
) -> dict[str, list[str]]:
    """Map eval id -> training ids sharing at least one n-gram with it.

    ``min_coverage``: count a (eval, train) pair only if the shared n-gram windows cover at
    least this fraction of the eval problem's tokens. A copied problem is covered end to end;
    a template sentence ("where m and n are relatively prime positive integers, find m + n",
    in many variants across AIME-style problems) covers one line. Found 15 Oct 2026, when one
    unrelated training item made 8 of 30 AIME24 problems "hit"; a document-frequency cutoff
    did not fix it because each template variant is itself rare. None keeps every pair.
    """
    index: dict[tuple[str, ...], list[str]] = defaultdict(list)
    positions: dict[str, dict[tuple[str, ...], list[int]]] = {}
    lengths: dict[str, int] = {}
    for eid, text in eval_items.items():
        toks = normalize_text(text).split()
        lengths[eid] = max(1, len(toks))
        pos: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i, g in enumerate(ngrams(toks, n)):
            pos[g].append(i)
        positions[eid] = pos
        for g in pos:
            index[g].append(eid)
    hits: dict[str, list[str]] = defaultdict(list)
    for tid, text in train_items.items():
        shared: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for g in ngrams(normalize_text(text).split(), n):
            for eid in index.get(g, ()):
                shared[eid].add(g)
        for eid, gs in shared.items():
            if (
                min_coverage is not None
                and ngram_coverage(gs, positions[eid], n, lengths[eid]) < min_coverage
            ):
                continue
            if len(hits[eid]) < max_hits_per_eval:
                hits[eid].append(tid)
    return dict(hits)


def ngram_coverage(
    shared: set[tuple[str, ...]],
    positions: dict[tuple[str, ...], list[int]],
    n: int,
    length: int,
) -> float:
    """Fraction of an eval problem's tokens inside at least one shared n-gram window."""
    covered: set[int] = set()
    for g in shared:
        width = len(g)  # a short item is one gram of its own length
        for start in positions.get(g, ()):
            covered.update(range(start, start + width))
    return len(covered) / length


def fuzzy_overlap(
    eval_items: dict[str, str],
    train_items: dict[str, str],
    *,
    threshold: float = 0.9,
    length_tolerance: float = 0.3,
    max_hits_per_eval: int = 20,
) -> dict[str, list[tuple[str, float]]]:
    """Map eval id -> (training id, similarity) with normalised indel similarity >= threshold.

    Items are bucketed by length so each eval problem is compared only against training items
    within ±``length_tolerance`` of its length; a 0.9 similarity is impossible beyond that.
    """
    from rapidfuzz.distance import Indel

    norm_train = [(tid, normalize_text(t)) for tid, t in train_items.items()]
    norm_train.sort(key=lambda x: len(x[1]))
    lengths = [len(t) for _, t in norm_train]
    hits: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for eid, text in eval_items.items():
        q = normalize_text(text)
        lo = bisect.bisect_left(lengths, int(len(q) * (1 - length_tolerance)))
        hi = bisect.bisect_right(lengths, int(len(q) * (1 + length_tolerance)) + 1)
        for tid, t in norm_train[lo:hi]:
            score = Indel.normalized_similarity(q, t, score_cutoff=threshold)
            if score >= threshold:
                hits[eid].append((tid, float(score)))
                if len(hits[eid]) >= max_hits_per_eval:
                    break
    return dict(hits)


def overlap_report(
    eval_name: str,
    eval_items: dict[str, str],
    train_items: dict[str, str],
    *,
    n: int = 13,
    fuzzy_threshold: float = 0.9,
    run_fuzzy: bool = True,
    min_coverage: float | None = None,
) -> OverlapReport:
    rep = OverlapReport(eval_name, n, len(eval_items), len(train_items), min_coverage=min_coverage)
    rep.ngram_hits = ngram_overlap(eval_items, train_items, n=n)
    if min_coverage is not None:
        rep.ngram_hits_covered = ngram_overlap(
            eval_items, train_items, n=n, min_coverage=min_coverage
        )
    if run_fuzzy:
        rep.fuzzy_hits = fuzzy_overlap(eval_items, train_items, threshold=fuzzy_threshold)
    return rep


def write_report(rep: OverlapReport, path: str | Path, *, examples: int = 10) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": rep.summary(),
        "ngram_hits": dict(list(rep.ngram_hits.items())[:examples]),
        "ngram_hits_covered": None
        if rep.ngram_hits_covered is None
        else dict(list(rep.ngram_hits_covered.items())[:examples]),
        "fuzzy_hits": dict(list(rep.fuzzy_hits.items())[:examples]),
        "contaminated_eval_ids": sorted(rep.contaminated_eval_ids),
        "train_ids_to_drop": sorted(rep.train_ids_to_drop),
    }
    path.write_text(json.dumps(payload, indent=2))


def read_jsonl_texts(
    path: str | Path, *, text_field: str, id_field: str | None = None
) -> dict[str, str]:
    """Load ``{id: text}`` from a JSONL file; ids default to the line number."""
    items: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row[id_field]) if id_field else f"{Path(path).stem}-{i}"
            items[key] = str(row[text_field])
    return items


def main(argv: list[str] | None = None) -> None:
    """``python -m rlvr.data.contamination --evals gsm8k math500 --train data.jsonl --text-field problem``"""
    import argparse

    from rlvr.eval.datasets import load_eval_set

    p = argparse.ArgumentParser(description="Training-data overlap probe (13-gram + fuzzy).")
    p.add_argument("--evals", nargs="+", required=True)
    p.add_argument(
        "--train", type=Path, required=True, help="JSONL with one training item per line"
    )
    p.add_argument("--text-field", required=True)
    p.add_argument("--id-field", default=None)
    p.add_argument("--n", type=int, default=13)
    p.add_argument("--fuzzy-threshold", type=float, default=0.9)
    p.add_argument("--no-fuzzy", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)

    train = read_jsonl_texts(args.train, text_field=args.text_field, id_field=args.id_field)
    for name in args.evals:
        es = load_eval_set(name)
        eval_items = {pr.id: pr.question for pr in es.problems}
        rep = overlap_report(
            name,
            eval_items,
            train,
            n=args.n,
            fuzzy_threshold=args.fuzzy_threshold,
            run_fuzzy=not args.no_fuzzy,
        )
        out = args.out_dir / f"{args.train.stem}__{name}.json"
        write_report(rep, out)
        print(json.dumps(rep.summary()))


if __name__ == "__main__":
    main()
