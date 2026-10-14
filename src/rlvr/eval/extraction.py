"""Answer extraction and grading.

PLAN.md section 4.2: extraction failures are counted separately and never silently as wrong.
Extraction is a chain of increasingly permissive rules and the rule that fired is recorded,
so a shift from "tag" to "last_number" between checkpoints is visible in the report.

Order of rules:
1. ``<answer> ... </answer>`` block (last one); inside it, the last ``\\boxed{}`` if present.
2. Last ``\\boxed{...}`` anywhere (balanced braces, so nested fractions survive).
3. A final-answer phrase: "The answer is X", "#### X", "Final answer: X".
4. For numeric sets only: the last number in the text (the GSM8K convention for base models).

Grading uses ``math-verify`` when available (symbolic equivalence: 0.5 == 1/2 == \\frac12,
sets, intervals) with a numeric fallback that strips units, currency, percent and commas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from rlvr.eval.datasets import AnswerType, verify_countdown

ExtractionSource = str  # "tag" | "boxed" | "final_phrase" | "last_number" | "none"

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_TAG = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_FINAL_PHRASES = [
    re.compile(r"####\s*(.+?)\s*$", re.MULTILINE),
    re.compile(
        r"(?:final answer|the answer)\s*(?:is|:)\s*[:\s]*\$?\s*(.+?)\s*[.\n]?$",
        re.IGNORECASE | re.MULTILINE,
    ),
]
# Digits with optional comma groups, optional space-separated thousands ("400 000", as GSM8K
# questions sometimes write them), decimals and simple fractions.
_NUMBER = re.compile(r"-?\d[\d,]*(?: \d{3}(?!\d))*(?:\.\d+)?(?:/\d+)?")


def _num(s: str) -> str:
    return s.replace(" ", "")


@dataclass(frozen=True)
class Extraction:
    answer: str | None
    source: ExtractionSource

    @property
    def failed(self) -> bool:
        return self.answer is None


def last_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` (or ``\\fbox{...}``) with balanced braces."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        # \boxed 5 without braces
        m = re.match(r"\\(?:boxed|fbox)\s+(\S+)", text[idx:])
        return m.group(1) if m else None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None  # unbalanced: truncated output


def _clean_candidate(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = s.rstrip(".").strip()
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s)
    return s


def extract_answer(text: str, answer_type: AnswerType) -> Extraction:
    """Run the extraction chain on a model completion."""
    if not text:
        return Extraction(None, "none")

    tags = _ANSWER_TAG.findall(text)
    if tags:
        inner = tags[-1].strip()
        boxed = last_boxed(inner)
        cand = boxed if boxed is not None else inner
        cand = _clean_candidate(cand)
        if cand:
            return Extraction(cand, "tag")

    boxed = last_boxed(text)
    if boxed is not None:
        cand = _clean_candidate(boxed)
        if cand:
            return Extraction(cand, "boxed")

    if answer_type != "countdown":
        for pat in _FINAL_PHRASES:
            hits = pat.findall(text)
            if hits:
                cand = _clean_candidate(hits[-1])
                if answer_type == "numeric":
                    nums = _NUMBER.findall(cand)
                    cand = _num(nums[0]) if nums else cand
                if cand:
                    return Extraction(cand, "final_phrase")

    if answer_type == "numeric":
        nums = _NUMBER.findall(text)
        if nums:
            return Extraction(_num(nums[-1]), "last_number")

    return Extraction(None, "none")


def tag_format_ok(text: str) -> bool:
    """Exactly one <think>..</think> followed by exactly one <answer>..</answer>.

    The prompt template pre-fills ``<think>`` for the think-tag style, so a completion that
    starts mid-think is accepted when the prefix is prepended by the caller.
    """
    thinks = list(_THINK_TAG.finditer(text))
    answers = list(_ANSWER_TAG.finditer(text))
    if len(thinks) != 1 or len(answers) != 1:
        return False
    return thinks[0].end() <= answers[0].start()


# --------------------------------------------------------------------------- normalisation and grading

_UNITS = re.compile(
    r"\\?(?:text|mbox|mathrm)\{[^}]*\}|(?:dollars?|cents?|units?|degrees?|percent|miles?|hours?|minutes?|"
    r"seconds?|days?|weeks?|years?|meters?|metres?|cm|km|kg|mph|inches|inch|feet|foot|ft|sq\.?|square|"
    r"cubic|students?|people|apples?|eggs?)\b",
    re.IGNORECASE,
)


def normalize_numeric(s: str) -> Fraction | None:
    """Parse a numeric answer into an exact Fraction, or None if it is not numeric."""
    t = s.strip().lower()
    t = t.replace("$", "").replace("\\$", "").replace(",", "").replace("%", "")
    t = t.replace("\\%", "").replace("^\\circ", "").replace("°", "")
    t = _UNITS.sub("", t).strip()
    t = re.sub(r"^[a-z]\s*=\s*", "", t)  # x = 5
    t = t.rstrip(".").strip()
    m = re.fullmatch(r"\\?d?frac\{(-?\d+)\}\{(-?\d+)\}", t)
    if m:
        try:
            return Fraction(int(m.group(1)), int(m.group(2)))
        except ZeroDivisionError:
            return None
    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", t)
    if m:
        try:
            return Fraction(int(m.group(1)), int(m.group(2)))
        except ZeroDivisionError:
            return None
    try:
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return None


def normalize_for_vote(s: str) -> str:
    """Canonical string for majority voting: numeric answers as reduced fractions, LaTeX
    answers with cosmetic differences removed."""
    num = normalize_numeric(s)
    if num is not None:
        return str(num)
    t = s.strip().lower()
    t = t.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    t = t.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\s+", "", t)
    t = t.replace("$", "").rstrip(".")
    return t


def _math_verify_equal(pred: str, gold: str) -> bool | None:
    """Symbolic check via math-verify; None when the library is missing or cannot parse."""
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    try:
        gold_p = parse(f"${gold}$") or parse(gold)
        pred_p = parse(f"${pred}$") or parse(pred)
    except Exception:  # the parser raises many types on odd input
        return None
    if not gold_p or not pred_p:
        return None
    try:
        return bool(verify(gold_p, pred_p))
    except Exception:
        return None


def grade(
    pred: str | None, gold: str, answer_type: AnswerType, *, meta: dict | None = None
) -> bool:
    """Is ``pred`` equivalent to ``gold``? Deterministic in (pred, gold)."""
    if pred is None:
        return False
    if answer_type == "countdown":
        assert meta is not None, "countdown grading needs the problem's numbers and target"
        return verify_countdown(pred, list(meta["numbers"]), int(meta["target"]))

    if answer_type == "numeric":
        p, g = normalize_numeric(pred), normalize_numeric(gold)
        if p is not None and g is not None:
            return p == g
    mv = _math_verify_equal(pred, gold)
    if mv is not None:
        return mv
    return normalize_for_vote(pred) == normalize_for_vote(gold)
