"""Eval-set loaders and the Countdown generator.

Each loader returns an :class:`EvalSet` of :class:`Problem` with a gold answer string and an
``answer_type`` that tells the grader how to compare. Loaders pin a Hugging Face revision
when one is given and always record a content hash in the manifest, so a later run can prove
it scored the same problems (WORKFLOW.md section 4).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import random
from collections.abc import Callable, Collection, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

AnswerType = Literal["numeric", "latex", "countdown"]


@dataclass
class Problem:
    id: str
    question: str
    answer: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSet:
    name: str
    answer_type: AnswerType
    problems: list[Problem]
    source: str
    revision: str | None = None

    def __len__(self) -> int:
        return len(self.problems)

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for p in self.problems:
            h.update(p.id.encode())
            h.update(p.question.encode())
            h.update(p.answer.encode())
        return h.hexdigest()[:16]

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "revision": self.revision,
            "answer_type": self.answer_type,
            "num_problems": len(self.problems),
            "content_hash": self.content_hash(),
        }

    def subset(self, limit: int | None, seed: int = 0) -> EvalSet:
        """Deterministic subset for smoke runs; the full set is used for anything reported."""
        if limit is None or limit >= len(self.problems):
            return self
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(self.problems)), limit))
        return EvalSet(
            name=self.name,
            answer_type=self.answer_type,
            problems=[self.problems[i] for i in idx],
            source=self.source,
            revision=self.revision,
        )


def _hf_load(path: str, name: str | None, split: str, revision: str | None):
    from datasets import load_dataset  # heavy import kept local

    return load_dataset(path, name, split=split, revision=revision)


def _first_key(row: dict[str, Any], *candidates: str) -> Any:
    for c in candidates:
        if c in row:
            return row[c]
    raise KeyError(f"none of {candidates} in row keys {list(row)}")


# --------------------------------------------------------------------------- GSM8K


def gsm8k_gold(answer_field: str) -> str:
    """GSM8K stores 'reasoning #### 18'; the gold is the text after '####', commas removed."""
    gold = answer_field.split("####")[-1].strip()
    return gold.replace(",", "").replace("$", "").strip()


def load_gsm8k_test(revision: str | None = None) -> EvalSet:
    ds = _hf_load("openai/gsm8k", "main", "test", revision)
    problems = [
        Problem(
            id=f"gsm8k-test-{i}", question=row["question"].strip(), answer=gsm8k_gold(row["answer"])
        )
        for i, row in enumerate(ds)
    ]
    return EvalSet("gsm8k", "numeric", problems, "openai/gsm8k:main:test", revision)


def load_gsm8k_train(revision: str | None = None) -> EvalSet:
    ds = _hf_load("openai/gsm8k", "main", "train", revision)
    problems = [
        Problem(
            id=f"gsm8k-train-{i}",
            question=row["question"].strip(),
            answer=gsm8k_gold(row["answer"]),
            meta={"solution": row["answer"]},
        )
        for i, row in enumerate(ds)
    ]
    return EvalSet("gsm8k_train", "numeric", problems, "openai/gsm8k:main:train", revision)


# --------------------------------------------------------------------------- MATH-500 / AIME


def _level_int(level: Any) -> int | None:
    if level is None:
        return None
    s = str(level)
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else None


def load_math500(revision: str | None = None) -> EvalSet:
    ds = _hf_load("HuggingFaceH4/MATH-500", None, "test", revision)
    problems = []
    for i, row in enumerate(ds):
        problems.append(
            Problem(
                id=str(row.get("unique_id") or row.get("id") or f"math500-{i}"),
                question=row["problem"].strip(),
                answer=str(row["answer"]).strip(),
                meta={"level": _level_int(row.get("level")), "subject": row.get("subject")},
            )
        )
    return EvalSet("math500", "latex", problems, "HuggingFaceH4/MATH-500:test", revision)


def load_math_train(revision: str | None = None) -> EvalSet:
    """MATH train split (7500 problems, EleutherAI/hendrycks_math, all seven subjects); the gold
    is the last boxed expression of the reference solution."""
    from rlvr.eval.extraction import last_boxed

    subjects = (
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    )
    problems: list[Problem] = []
    for subj in subjects:
        ds = _hf_load("EleutherAI/hendrycks_math", subj, "train", revision)
        for i, row in enumerate(ds):
            gold = last_boxed(row["solution"])
            if gold is None:
                continue
            problems.append(
                Problem(
                    id=f"math-train-{subj}-{i}",
                    question=row["problem"].strip(),
                    answer=gold,
                    meta={
                        "level": _level_int(row.get("level")),
                        "subject": row.get("type") or subj,
                        "solution": row["solution"],
                    },
                )
            )
    return EvalSet("math_train", "latex", problems, "EleutherAI/hendrycks_math:*:train", revision)


DEV_SIZE_PER_SOURCE = 250
DEV_SEED = 20261015  # fixed once; the dev split must never move between phases


def _dev_split(es: EvalSet, name: str) -> EvalSet:
    """Seeded sample of held-out *training* problems used to pick checkpoints and learning
    rates (PLAN.md Phase 2 step 2: never select on a test set). Deterministic: a fixed seed and
    a sample over sorted ids, so it is identical on every machine and every phase."""
    rng = random.Random(DEV_SEED)
    order = sorted(range(len(es.problems)), key=lambda i: es.problems[i].id)
    chosen = sorted(rng.sample(order, DEV_SIZE_PER_SOURCE), key=lambda i: es.problems[i].id)
    return EvalSet(
        name, es.answer_type, [es.problems[i] for i in chosen], es.source + "#dev", es.revision
    )


def dev_problem_ids(train: EvalSet) -> set[str]:
    """Ids of the dev split drawn from this *full* train set: the same seeded choice as
    ``load_gsm8k_dev`` / ``load_math_dev``, so a training pool can exclude them (Phase 3+)."""
    return {p.id for p in _dev_split(train, train.name + "_dev").problems}


def filter_problems(
    es: EvalSet,
    *,
    levels: Sequence[int] | None = None,
    exclude_ids: Collection[str] = (),
    pool_limit: int | None = None,
    seed: int = 0,
) -> EvalSet:
    """A training pool: keep ``levels`` (evals with ``meta.level`` only), drop ``exclude_ids``,
    then a seeded cap of ``pool_limit`` problems. The source string records each step so the
    manifest says what the pool was."""
    problems = list(es.problems)
    source = es.source
    if levels is not None:
        want = {int(x) for x in levels}
        if not any(p.meta.get("level") is not None for p in problems):
            raise ValueError(f"{es.name} has no level metadata; --levels does not apply")
        problems = [p for p in problems if p.meta.get("level") in want]
        source += "#levels=" + ",".join(str(x) for x in sorted(want))
    if exclude_ids:
        drop = set(exclude_ids)
        problems = [p for p in problems if p.id not in drop]
        source += f"#excluded={len(drop)}"
    out = EvalSet(es.name, es.answer_type, problems, source, es.revision)
    if pool_limit is not None and pool_limit < len(out.problems):
        out = out.subset(pool_limit, seed)
        out.source += f"#pool={pool_limit},seed={seed}"
    return out


def load_gsm8k_dev(revision: str | None = None) -> EvalSet:
    return _dev_split(load_gsm8k_train(revision), "gsm8k_dev")


def load_math_dev(revision: str | None = None) -> EvalSet:
    return _dev_split(load_math_train(revision), "math_dev")


def load_aime24(revision: str | None = None) -> EvalSet:
    ds = _hf_load("HuggingFaceH4/aime_2024", None, "train", revision)
    problems = [
        Problem(
            id=str(row.get("id") or row.get("ID") or row.get("url") or f"aime24-{i}"),
            question=str(_first_key(row, "problem", "Problem")).strip(),
            answer=str(_first_key(row, "answer", "Answer")).strip(),
        )
        for i, row in enumerate(ds)
    ]
    return EvalSet("aime24", "numeric", problems, "HuggingFaceH4/aime_2024:train", revision)


# --------------------------------------------------------------------------- Countdown


@dataclass
class CountdownSpec:
    """Difficulty knobs; PLAN.md section 9 says to tune and then document them."""

    num_count: int = 4
    number_min: int = 1
    number_max: int = 25
    target_min: int = 10
    target_max: int = 100


def _random_solvable(rng: random.Random, spec: CountdownSpec) -> tuple[list[int], int, str]:
    """Draw numbers and build a random expression tree over all of them until the value is an
    integer inside the target range. Returns (numbers, target, one valid equation)."""
    while True:
        nums = [rng.randint(spec.number_min, spec.number_max) for _ in range(spec.num_count)]
        items: list[tuple[Fraction, str]] = [(Fraction(n), str(n)) for n in nums]
        rng.shuffle(items)
        ok = True
        while len(items) > 1:
            (a, sa), (b, sb) = items.pop(), items.pop()
            op = rng.choice("+-*/")
            if op == "/" and (b == 0 or a % b != 0):
                op = rng.choice("+-*")
            if op == "-" and a < b:
                a, sa, b, sb = b, sb, a, sa
            val = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op]
            if val is None or val < 0 or val > 10_000:
                ok = False
                break
            items.append((val, f"({sa} {op} {sb})"))
        if not ok:
            continue
        val, expr = items[0]
        if val.denominator == 1 and spec.target_min <= val <= spec.target_max:
            return nums, int(val), expr[1:-1] if expr.startswith("(") else expr


def generate_countdown(
    n_problems: int, *, seed: int = 0, spec: CountdownSpec | None = None
) -> EvalSet:
    spec = spec or CountdownSpec()
    rng = random.Random(seed)
    problems = []
    for i in range(n_problems):
        nums, target, solution = _random_solvable(rng, spec)
        problems.append(
            Problem(
                id=f"countdown-{seed}-{i}",
                question=json.dumps({"numbers": nums, "target": target}),
                answer=str(target),
                meta={
                    "numbers": nums,
                    "target": target,
                    "reference_solution": solution,
                    **asdict(spec),
                },
            )
        )
    return EvalSet("countdown", "countdown", problems, f"generated:seed={seed}:{asdict(spec)}")


_ALLOWED_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div)


def _eval_countdown_ast(node: ast.AST, used: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_countdown_ast(node.body, used)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        used.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_countdown_ast(node.operand, used)
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BIN):
        left = _eval_countdown_ast(node.left, used)
        right = _eval_countdown_ast(node.right, used)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ZeroDivisionError
        return left / right
    raise ValueError(f"disallowed syntax: {type(node).__name__}")


def verify_countdown(equation: str, numbers: list[int], target: int) -> bool:
    """True iff ``equation`` uses each of ``numbers`` exactly once with + - * / and equals target.

    The expression is parsed with ``ast`` and evaluated over Fractions; anything other than
    integer literals, the four operators, unary minus and parentheses is rejected, so model
    output can never execute code. A trailing ``= target`` is tolerated and ignored.
    """
    if not equation or len(equation) > 200:
        return False
    lhs = equation.split("=")[0].strip()
    try:
        tree = ast.parse(lhs, mode="eval")
        used: list[int] = []
        value = _eval_countdown_ast(tree, used)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError, MemoryError):
        return False
    return sorted(used) == sorted(numbers) and value == target


# --------------------------------------------------------------------------- perturbed GSM8K (file-based)


def output_dir() -> Path:
    return Path(os.environ.get("RLVR_OUTPUT_DIR", Path(__file__).resolve().parents[3] / "outputs"))


def load_jsonl_evalset(path: str | Path, name: str, answer_type: AnswerType) -> EvalSet:
    """Load an eval set written by one of our generators (one JSON object per line with
    id, question, answer, meta)."""
    path = Path(path)
    problems = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            problems.append(
                Problem(row["id"], row["question"], str(row["answer"]), row.get("meta", {}))
            )
    return EvalSet(name, answer_type, problems, f"file:{path.name}")


def load_gsm8k_perturbed(revision: str | None = None) -> EvalSet:
    """File-based; ``revision`` is accepted for signature parity with the HF loaders."""
    del revision
    path = output_dir() / "data" / "gsm8k_perturbed.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python -m rlvr.data.perturb_gsm8k` first (Phase 1)"
        )
    return load_jsonl_evalset(path, "gsm8k_perturbed", "numeric")


def load_gsm8k_names_only(revision: str | None = None) -> EvalSet:
    """Names swapped, numbers untouched (EXP-003b control); file-based like the perturbed set."""
    del revision
    path = output_dir() / "data" / "gsm8k_names_only.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `rlvr-perturb-gsm8k --names-only` first")
    return load_jsonl_evalset(path, "gsm8k_names_only", "numeric")


# --------------------------------------------------------------------------- registry

TRAIN_SPLITS = ("gsm8k_train", "math_train")  # the only splits with a dev draw inside them
# sets built problem for problem from another set; a `--limit` subset follows the parent's draw
DERIVED_FROM = {"gsm8k_perturbed": "gsm8k", "gsm8k_names_only": "gsm8k"}

LOADERS: dict[str, Callable[..., EvalSet]] = {
    "gsm8k": load_gsm8k_test,
    "gsm8k_train": load_gsm8k_train,
    "math500": load_math500,
    "aime24": load_aime24,
    "gsm8k_perturbed": load_gsm8k_perturbed,
    "gsm8k_names_only": load_gsm8k_names_only,
    "math_train": load_math_train,
    "gsm8k_dev": load_gsm8k_dev,
    "math_dev": load_math_dev,
}


def load_eval_set(
    name: str,
    *,
    revision: str | None = None,
    limit: int | None = None,
    seed: int = 0,
    countdown_n: int = 500,
    countdown_spec: CountdownSpec | None = None,
    levels: Sequence[int] | None = None,
    exclude_dev: bool = False,
    pool_limit: int | None = None,
) -> EvalSet:
    """``levels``, ``exclude_dev`` and ``pool_limit`` shape a *training pool* (Phase 3 sampling);
    ``limit`` is the smoke-run subset and is applied last."""
    if name == "countdown":
        es = generate_countdown(countdown_n, seed=seed, spec=countdown_spec)
    elif name in LOADERS:
        es = LOADERS[name](revision)
    else:
        raise KeyError(f"unknown eval set {name!r}; known: {[*sorted(LOADERS), 'countdown']}")
    if exclude_dev:
        if name not in TRAIN_SPLITS:
            raise ValueError(f"exclude_dev applies to {TRAIN_SPLITS}, not {name!r}")
        es = filter_problems(es, exclude_ids=dev_problem_ids(es))
    if levels is not None or pool_limit is not None:
        es = filter_problems(es, levels=levels, pool_limit=pool_limit, seed=seed)
    parent = DERIVED_FROM.get(name)
    if parent is not None and limit is not None:
        # A derived set exists to be paired with its parent problem for problem (meta.original_id).
        # Its subset is therefore the parent's seeded subset, not an independent draw: on the H100
        # node (19 Oct 2026) `--limit 300` drew 300 gsm8k and 300 gsm8k_perturbed problems with only
        # 67 pairs between them, which made the paired perturbation gap unusable.
        parent_ids = {
            p.id for p in load_eval_set(parent, revision=revision, limit=limit, seed=seed).problems
        }
        kept = [p for p in es.problems if p.meta.get("original_id") in parent_ids]
        return EvalSet(
            name=es.name,
            answer_type=es.answer_type,
            problems=kept,
            source=f"{es.source}#paired={parent}:limit={limit},seed={seed}",
            revision=es.revision,
        )
    return es.subset(limit, seed)


def write_manifest(es: EvalSet, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(es.manifest(), indent=2))
