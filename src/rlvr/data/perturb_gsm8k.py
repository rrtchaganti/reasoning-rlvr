"""Perturbed GSM8K (GSM-Symbolic / GSM-Plus style), built from GSM8K's calculator annotations.

PLAN.md section 4.3, probe 3: if a model's GSM8K accuracy drops when the numbers (and names) of a
problem change but the reasoning does not, part of its original score was recall. This module
makes that perturbed set *with recomputed gold answers*, deterministically, without an LLM.

How the gold is recomputed. Every GSM8K solution carries calculator annotations such as
``<<16-3=13>>``; read in order they form a small arithmetic program whose last result is the
answer. We classify each literal in that program as

- a **reference** to an earlier step's result (same value as a previous ``<<...=r>>``),
- a **symbol** (same value as a perturbable number written in the question), or
- a **constant** (anything else: "twice", "a dozen", "60 minutes", small numbers).

Then we resample the symbols and re-run the program. A problem is kept only if

1. the program evaluated with the *original* numbers reproduces every annotated result and the
   official ``#### gold`` (this validates our reading of the problem),
2. no literal is ambiguous (equal to both a question number and a previous result),
3. every literal in the program is explained: a reference, a symbol, a number written in the
   question (as digits or as a word such as "twenty" or "dozen"), a unit constant, or a small
   multiplier below 13 ("twice", "a week"). An unexplained 120 means the solution did a step in
   prose without an annotation, and perturbing around it would corrupt the gold,
4. the resampled program has strictly positive intermediates with money-like denominators, a
   positive integer answer, and that answer differs from the original.

Problems that fail any check are dropped and counted by reason, so the coverage is reported, not
hidden. Numbers below 3 and a short list of unit constants (7 days, 12 months, 60 minutes, 100 %)
are never perturbed because they are usually structure, not quantity.

Names are swapped within gender lists so pronouns stay consistent. The perturbed problem keeps
``meta.original_id`` so the paired comparison (original vs perturbed, same problem) is exact.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from rlvr.eval.datasets import EvalSet, Problem, gsm8k_gold, output_dir, write_manifest

STEP_RE = re.compile(r"<<([^<>=]+)=([^<>]+)>>")
# digits in a question: optional $, comma groups, optional decimals; not part of a word or id
QNUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w])")
LITERAL_RE = re.compile(r"\d+(?:\.\d+)?")

MIN_PERTURBABLE = 3
SMALL_CONSTANT_MAX = 12  # literals below this are allowed unexplained ("twice", "half")
WORD_NUMBERS: dict[str, Fraction] = {
    **{
        w: Fraction(i)
        for i, w in enumerate(
            [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "eleven",
                "twelve",
                "thirteen",
                "fourteen",
                "fifteen",
                "sixteen",
                "seventeen",
                "eighteen",
                "nineteen",
                "twenty",
            ]
        )
    },
    **{
        w: Fraction(10 * (i + 3))
        for i, w in enumerate(["thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"])
    },
    "hundred": Fraction(100),
    "thousand": Fraction(1000),
    "dozen": Fraction(12),
    "half": Fraction(1, 2),
    "quarter": Fraction(1, 4),
    "third": Fraction(1, 3),
    "twice": Fraction(2),
    "double": Fraction(2),
    "triple": Fraction(3),
    "thrice": Fraction(3),
}
PERCENT_RE = re.compile(r"^\s*(%|percent)")
UNIT_CONSTANTS = frozenset({7, 12, 24, 30, 31, 52, 60, 100, 180, 360, 365, 1000})
MONEY_DENOMINATORS = frozenset({1, 2, 4, 5, 10, 20, 25, 50, 100})

FEMALE_EXTENDED = [
    "Janet",
    "Mary",
    "Sarah",
    "Emily",
    "Lisa",
    "Anna",
    "Carla",
    "Jenna",
    "Olivia",
    "Sophia",
    "Ava",
    "Mia",
    "Isabella",
    "Grace",
    "Hannah",
    "Megan",
    "Rachel",
    "Laura",
    "Nancy",
    "Karen",
    "Betty",
    "Alice",
    "Claire",
    "Julia",
    "Diana",
    "Ella",
    "Zoe",
    "Lily",
    "Ruby",
    "Chloe",
    "Sally",
    "Martha",
    "Sharon",
    "Amy",
    "Anne",
    "Jane",
    "Dora",
    "Samantha",
    "Jean",
    "Cindy",
    "Maria",
    "Jenny",
    "Madeline",
    "Kelly",
    "Sue",
    "Erin",
    "Ashley",
    "Julie",
    "Susan",
    "Jill",
    "Vanessa",
    "Naomi",
    "Brianna",
    "Erica",
    "Bernadette",
    "Jana",
    "Kate",
    "Emma",
    "Sofie",
    "Polly",
    "Beatrice",
    "Brittany",
    "Sandra",
    "Harriet",
    "Jessica",
    "Jennifer",
    "Linda",
    "Patricia",
    "Barbara",
    "Elizabeth",
    "Michelle",
    "Amanda",
    "Melissa",
    "Stephanie",
    "Nicole",
    "Heather",
    "Rebecca",
    "Katie",
    "Lucy",
    "Sophie",
    "Natalie",
    "Lauren",
    "Rose",
    "Helen",
]
MALE_EXTENDED = [
    "John",
    "James",
    "Tom",
    "Mike",
    "Mark",
    "Carlos",
    "Jack",
    "Jim",
    "Bob",
    "Peter",
    "Paul",
    "George",
    "Henry",
    "Ben",
    "Sam",
    "Alex",
    "Daniel",
    "David",
    "Josh",
    "Kevin",
    "Brian",
    "Ryan",
    "Eric",
    "Adam",
    "Luke",
    "Noah",
    "Ethan",
    "Leo",
    "Oscar",
    "Max",
    "Robert",
    "Jason",
    "Billy",
    "Charlie",
    "Harry",
    "Fred",
    "Bobby",
    "Bill",
    "Joe",
    "Ali",
    "Raymond",
    "Tim",
    "Jake",
    "Marcus",
    "Jamal",
    "Carl",
    "Tony",
    "Joey",
    "Richard",
    "Stanley",
    "Boris",
    "Igor",
    "Aaron",
    "Stephen",
    "Jackson",
    "Gerald",
    "Joseph",
    "Jerry",
    "Chris",
    "Sean",
    "Timothy",
    "Oliver",
    "Wayne",
    "Andy",
    "Mitchell",
    "Liam",
    "Gary",
    "Gunther",
    "Anthony",
    "Stan",
    "Kory",
    "Allen",
    "Tommy",
    "Greg",
    "Corey",
    "Johnny",
    "Jimmy",
    "Anakin",
    "Michael",
    "William",
    "Steve",
    "Matt",
    "Matthew",
    "Andrew",
    "Nick",
    "Frank",
    "Jeff",
    "Larry",
    "Scott",
    "Ken",
    "Roger",
    "Dan",
    "Edward",
    "Charles",
    "Arthur",
    "Walter",
    "Nathan",
    "Isaac",
]
# The first 30 of each list are the original pools; the numbers+names set (data/manifests/gsm8k_perturbed.json)
# was generated with them and must keep regenerating byte-for-byte, so the default swap uses only
# these. The extended pools serve the names-only control (EXP-003b), where coverage matters more.
FEMALE = FEMALE_EXTENDED[:30]
MALE = MALE_EXTENDED[:30]


def _name_tables(extended: bool) -> tuple[dict[str, str], re.Pattern[str]]:
    fem, mal = (FEMALE_EXTENDED, MALE_EXTENDED) if extended else (FEMALE, MALE)
    group = {n: "f" for n in fem} | {n: "m" for n in mal}
    pattern = re.compile(r"\b(" + "|".join(sorted(group, key=len, reverse=True)) + r")\b")
    return group, pattern


_NAME_GROUP, NAME_RE = _name_tables(False)


# --------------------------------------------------------------------------- arithmetic


def eval_arith(expr: str) -> Fraction:
    """Exact evaluation of ``+ - * / ( )`` over decimal literals; raises on anything else."""
    tree = ast.parse(expr.strip(), mode="eval")

    def ev(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                if b == 0:
                    raise ZeroDivisionError(expr)
                return a / b
        raise ValueError(f"unsupported expression: {expr!r}")

    return ev(tree)


def _to_fraction(text: str) -> Fraction | None:
    t = text.replace(",", "").replace("$", "").strip()
    try:
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return None


def _fmt(value: Fraction, like: str) -> str:
    """Format ``value`` the way the original literal ``like`` was written (commas, decimals)."""
    if value.denominator == 1:
        s = str(value.numerator)
        if "," in like or (value.numerator >= 10_000 and len(like.replace(",", "")) >= 4):
            s = f"{value.numerator:,}"
        return s
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------- program model


@dataclass
class Step:
    expr: str
    result: Fraction


@dataclass
class Program:
    steps: list[Step]
    gold: Fraction

    @classmethod
    def parse(cls, solution: str) -> Program:
        steps = []
        for expr, res in STEP_RE.findall(solution):
            r = _to_fraction(res)
            if r is None:
                raise ValueError(f"unreadable result {res!r}")
            steps.append(Step(expr.strip(), r))
        if not steps:
            raise ValueError("no_steps")
        gold = _to_fraction(gsm8k_gold(solution))
        if gold is None:
            raise ValueError("unreadable gold")
        return cls(steps, gold)

    def check_original(self) -> None:
        """Every annotated result must follow from its expression; the last must be the gold."""
        for s in self.steps:
            if eval_arith(s.expr) != s.result:
                raise ValueError("step_mismatch")
        if self.steps[-1].result != self.gold:
            raise ValueError("last_step_not_gold")

    def run(
        self, symbols: dict[Fraction, Fraction], allowed: set[Fraction] | None = None
    ) -> list[Fraction]:
        """Re-run the program with question numbers remapped through ``symbols``.

        Literals equal to an earlier *original* result are references and get that step's new
        result; literals in ``symbols`` get their new value; the rest are constants.
        """
        old_results: list[Fraction] = []
        new_results: list[Fraction] = []
        for s in self.steps:

            def sub(m: re.Match[str]) -> str:
                v = Fraction(m.group(0))
                is_ref = v in old_results
                is_sym = v in symbols
                if is_ref and is_sym:
                    raise ValueError("ambiguous_literal")
                if is_ref:
                    return f"({new_results[len(old_results) - 1 - old_results[::-1].index(v)]})"
                if is_sym:
                    return f"({symbols[v]})"
                if allowed is not None and v > SMALL_CONSTANT_MAX and v not in allowed:
                    raise ValueError("unexplained_constant")
                return m.group(0)

            new_expr = LITERAL_RE.sub(sub, s.expr)
            new_results.append(eval_arith(new_expr))
            old_results.append(s.result)
        return new_results


# --------------------------------------------------------------------------- perturbation


@dataclass
class Perturbed:
    id: str
    question: str
    answer: str
    meta: dict[str, Any] = field(default_factory=dict)


def question_numbers(question: str) -> list[tuple[int, int, str, Fraction]]:
    """(start, end, text, value) for every numeric literal in the question."""
    out = []
    for m in QNUM_RE.finditer(question):
        text = m.group(0)
        val = _to_fraction(text)
        if val is not None:
            out.append((m.start(), m.end(), text, val))
    return out


def word_numbers(question: str) -> set[Fraction]:
    """Values written as words ("twenty-five", "a dozen", "half"); never perturbed, but they
    explain constants in the solution's arithmetic."""
    out: set[Fraction] = set()
    for tok in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", question.lower()):
        if tok in WORD_NUMBERS:
            out.add(WORD_NUMBERS[tok])
        elif "-" in tok:
            a, b = tok.split("-", 1)
            if (
                a in WORD_NUMBERS
                and b in WORD_NUMBERS
                and WORD_NUMBERS[a] >= 20
                and WORD_NUMBERS[b] < 10
            ):
                out.add(WORD_NUMBERS[a] + WORD_NUMBERS[b])
    return out


def perturbable_values(
    nums: list[tuple[int, int, str, Fraction]], question: str = ""
) -> list[Fraction]:
    vals = []
    for _, end, _, v in nums:
        if v.denominator != 1 or v < MIN_PERTURBABLE or int(v) in UNIT_CONSTANTS:
            continue
        if question and PERCENT_RE.match(question[end:]):
            continue  # "20%" hides in the chain as 0.2 or 20/100; leave percentages alone
        if v not in vals:
            vals.append(v)
    return vals


def _sample_value(rng: np.random.Generator, old: Fraction) -> Fraction:
    lo, hi = max(2, int(old) // 2), 2 * int(old) + 1
    for _ in range(20):
        v = int(rng.integers(lo, hi))
        if v != int(old):
            return Fraction(v)
    return Fraction(int(old) + 1)


def _valid_run(results: list[Fraction], gold: Fraction, *, allow_cents: bool) -> bool:
    # strictly positive: a zero intermediate makes a degenerate story ("she had 0 notebooks");
    # fractional intermediates only when the story is about money ("8.5 students" is not a story)
    denoms = MONEY_DENOMINATORS if allow_cents else frozenset({1})
    if any(r <= 0 or r.denominator not in denoms for r in results):
        return False
    final = results[-1]
    return final.denominator == 1 and final > 0 and final != gold


def swap_names(
    question: str, rng: np.random.Generator, *, extended: bool = False
) -> tuple[str, dict[str, str]]:
    group, pattern = _name_tables(extended)
    fem, mal = (FEMALE_EXTENDED, MALE_EXTENDED) if extended else (FEMALE, MALE)
    found = []
    for m in pattern.finditer(question):
        if m.group(1) not in found:
            found.append(m.group(1))
    if not found:
        return question, {}
    mapping: dict[str, str] = {}
    taken = set(found)
    for name in found:
        pool = [n for n in (fem if group[name] == "f" else mal) if n not in taken]
        new = pool[int(rng.integers(len(pool)))]
        mapping[name] = new
        taken.add(new)
    out = pattern.sub(lambda m: mapping.get(m.group(1), m.group(1)), question)
    return out, mapping


def perturb_problem(
    pid: str,
    question: str,
    solution: str,
    rng: np.random.Generator,
    *,
    tries: int = 40,
    swap_names_too: bool = True,
    names_only: bool = False,
) -> Perturbed:
    """Return the perturbed problem or raise ``ValueError(reason)``.

    ``names_only`` keeps every number and only swaps names: the control for EXP-003b. Its
    gold is the original gold; a problem without a swappable name is dropped (``no_names``).
    """
    prog = Program.parse(solution)
    prog.check_original()
    if names_only:
        new_q, name_map = swap_names(question, rng, extended=True)
        if not name_map:
            raise ValueError("no_names")
        return Perturbed(
            id=f"{pid}-names",
            question=new_q,
            answer=str(prog.gold.numerator) if prog.gold.denominator == 1 else str(prog.gold),
            meta={
                "original_id": pid,
                "original_answer": str(prog.gold.numerator)
                if prog.gold.denominator == 1
                else str(prog.gold),
                "substitutions": {},
                "name_map": name_map,
                "num_steps": len(prog.steps),
            },
        )
    nums = question_numbers(question)
    symbols = perturbable_values(nums, question)
    if not symbols:
        raise ValueError("nothing_to_perturb")
    allowed = (
        {v for _, _, _, v in nums} | word_numbers(question) | {Fraction(c) for c in UNIT_CONSTANTS}
    )
    # classify once with an identity map: ambiguity and unexplained constants are detected
    # independent of sampling
    prog.run({v: v for v in symbols}, allowed)

    all_values = {v for _, _, _, v in nums}
    allow_cents = (
        "$" in question
        or "cent" in question.lower()
        or any(v.denominator != 1 for v in all_values)
        or any(s.result.denominator != 1 for s in prog.steps)
    )
    for _ in range(tries):
        mapping: dict[Fraction, Fraction] = {}
        for v in symbols:
            new = _sample_value(rng, v)
            # a new value must not coincide with any other number in the question
            # (original or already resampled), or the text becomes ambiguous to read
            taken = (all_values - {v}) | set(mapping.values())
            for _ in range(10):
                if new not in taken:
                    break
                new = _sample_value(rng, v)
            mapping[v] = new
        if len(set(mapping.values()) | (all_values - set(symbols))) < len(all_values):
            continue
        try:
            results = prog.run(mapping, allowed)
        except (ZeroDivisionError, ValueError):
            continue
        if _valid_run(results, prog.gold, allow_cents=allow_cents):
            break
    else:
        raise ValueError("no_valid_perturbation")

    new_q = question
    for start, end, text, val in sorted(nums, key=lambda t: -t[0]):
        if val in mapping:
            new_q = new_q[:start] + _fmt(mapping[val], text) + new_q[end:]
    name_map: dict[str, str] = {}
    if swap_names_too:
        new_q, name_map = swap_names(new_q, rng)
    return Perturbed(
        id=f"{pid}-perturbed",
        question=new_q,
        answer=str(results[-1].numerator),
        meta={
            "original_id": pid,
            "original_answer": str(prog.gold.numerator)
            if prog.gold.denominator == 1
            else str(prog.gold),
            "substitutions": {str(k): str(v) for k, v in mapping.items()},
            "name_map": name_map,
            "num_steps": len(prog.steps),
        },
    )


def perturb_many(
    rows: list[tuple[str, str, str]],
    *,
    seed: int = 0,
    swap_names_too: bool = True,
    names_only: bool = False,
) -> tuple[list[Perturbed], Counter[str]]:
    """``rows`` are (id, question, solution). Returns kept problems and drop reasons."""
    rng = np.random.default_rng(seed)
    kept: list[Perturbed] = []
    reasons: Counter[str] = Counter()
    for pid, q, sol in rows:
        try:
            kept.append(
                perturb_problem(
                    pid, q, sol, rng, swap_names_too=swap_names_too, names_only=names_only
                )
            )
            reasons["kept"] += 1
        except (ValueError, ZeroDivisionError, SyntaxError) as e:
            reasons[str(e).split(":")[0] or type(e).__name__] += 1
    return kept, reasons


def write_jsonl(problems: list[Perturbed], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in problems:
            f.write(
                json.dumps(
                    {"id": p.id, "question": p.question, "answer": p.answer, "meta": p.meta},
                    ensure_ascii=False,
                )
                + "\n"
            )


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the perturbed GSM8K test set.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--no-names", action="store_true", help="numbers only, keep names")
    ap.add_argument(
        "--names-only", action="store_true", help="swap names, keep every number (EXP-003b control)"
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="default outputs/data/gsm8k_perturbed.jsonl"
    )
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/gsm8k_perturbed.json"))
    args = ap.parse_args(argv)

    from rlvr.eval.datasets import _hf_load

    ds = _hf_load("openai/gsm8k", "main", args.split, None)
    rows = [
        (f"gsm8k-{args.split}-{i}", r["question"].strip(), r["answer"]) for i, r in enumerate(ds)
    ]
    kept, reasons = perturb_many(
        rows, seed=args.seed, swap_names_too=not args.no_names, names_only=args.names_only
    )

    stem = "gsm8k_names_only" if args.names_only else "gsm8k_perturbed"
    out = args.out or output_dir() / "data" / f"{stem}.jsonl"
    if args.names_only and args.manifest == Path("data/manifests/gsm8k_perturbed.json"):
        args.manifest = Path("data/manifests/gsm8k_names_only.json")
    write_jsonl(kept, out)
    es = EvalSet(
        stem,
        "numeric",
        [Problem(p.id, p.question, p.answer, p.meta) for p in kept],
        f"generated:perturb_gsm8k:seed={args.seed}:names={not args.no_names}:names_only={args.names_only}",
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(es, args.manifest)
    report = {
        "seed": args.seed,
        "split": args.split,
        "total": len(rows),
        "kept": len(kept),
        "coverage": len(kept) / len(rows),
        "drop_reasons": {k: v for k, v in reasons.items() if k != "kept"},
        "content_hash": es.content_hash(),
    }
    (out.with_suffix(".report.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out} and {args.manifest}")


if __name__ == "__main__":
    main()
