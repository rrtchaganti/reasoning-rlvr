"""Prompt templates, fixed per model family (PLAN.md section 4.2).

Qwen3-*-Base is a *base* model: no chat template is applied; every prompt is plain text that
ends where the model should continue. The same templates are used for every checkpoint so a
template change is an experiment, never a tweak. Styles:

- ``zero_shot``   the question plus a one-line instruction to reason and box the answer
- ``few_shot``    4 worked exemplars with final answers only (no reasoning), then the question
- ``few_shot_cot`` 4 exemplars with chain-of-thought (Wei et al. 2022 for GSM8K, Minerva for MATH)
- ``think_tag``   the R1-Zero / TinyZero style "<think> ... </think> <answer> ... </answer>" format
                  that SFT and RL checkpoints are trained on; the completion starts inside <think>

Each template also defines its stop strings, because a few-shot base model will otherwise
keep generating new "Question:" blocks until the token limit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from rlvr.eval.datasets import Problem

TEMPLATES = ("zero_shot", "few_shot", "few_shot_cot", "think_tag")


@dataclass(frozen=True)
class Prompt:
    text: str
    stop: tuple[str, ...]
    # prefix the template already emitted that must be re-attached before format checks
    completion_prefix: str = ""


# --- GSM8K exemplars (Wei et al. 2022, chain-of-thought prompting; first four of the eight) ---
GSM8K_SHOTS: list[tuple[str, str, str]] = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.",
        "6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
        "5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
        "39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
        "8",
    ),
]

# --- MATH exemplars (Minerva 4-shot prompt, Lewkowycz et al. 2022) ---
MATH_SHOTS: list[tuple[str, str, str]] = [
    (
        "Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
        "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.",
        "[2,5)",
    ),
    (
        "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$",
        "24",
    ),
    (
        "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$: \\begin{align*} 30n&=480\\\\ \\Rightarrow\\qquad n&=480/30=\\boxed{16} \\end{align*}",
        "16",
    ),
    (
        "If the system of equations \\begin{align*} 6x-4y&=a,\\\\ 6y-9x &=b. \\end{align*} has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "If we multiply the first equation by $-\\frac{3}{2}$, we obtain $$6y-9x=-\\frac{3}{2}a.$$ Since we also know that $6y-9x=b$, we have $$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$",
        "-\\frac{2}{3}",
    ),
]

FEW_SHOT_STOP = ("\n\nProblem:", "\nProblem:", "\n\nQuestion:", "\nQuestion:")
BOX_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def _family(eval_name: str) -> str:
    if eval_name.startswith("gsm8k"):
        return "gsm8k"
    if eval_name == "countdown":
        return "countdown"
    return "math"  # math500, aime24


def countdown_question(problem: Problem) -> str:
    spec = json.loads(problem.question)
    nums, target = spec["numbers"], spec["target"]
    return (
        f"Using the numbers {nums}, create an equation that equals {target}. You can use basic "
        "arithmetic operations (+, -, *, /) and each number can only be used once."
    )


def _question_text(problem: Problem, eval_name: str) -> str:
    return countdown_question(problem) if _family(eval_name) == "countdown" else problem.question


def _few_shot_block(eval_name: str, cot: bool) -> str:
    fam = _family(eval_name)
    if fam == "gsm8k":
        shots, label = GSM8K_SHOTS, "Question"
        parts = []
        for q, sol, ans in shots:
            body = f"{sol} The answer is {ans}." if cot else f"The answer is {ans}."
            parts.append(f"{label}: {q}\nAnswer: {body}")
        return "\n\n".join(parts) + "\n\n"
    if fam == "math":
        parts = []
        for q, sol, ans in MATH_SHOTS:
            body = sol if cot else f"The final answer is $\\boxed{{{ans}}}$."
            parts.append(f"Problem:\n{q}\n\nSolution:\n{body}")
        return "\n\n".join(parts) + "\n\n"
    raise ValueError("few-shot prompting is not defined for countdown; use think_tag")


def build_prompt(problem: Problem, template: str, eval_name: str) -> Prompt:
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; known: {TEMPLATES}")
    fam = _family(eval_name)
    q = _question_text(problem, eval_name)

    if template == "think_tag":
        if fam == "countdown":
            task = (
                f"{q} Show your work in <think> </think> tags. And return the final equation in "
                "<answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
            )
        else:
            task = (
                f"{q} Show your work in <think> </think> tags. And return the final answer in "
                "<answer> </answer> tags, for example <answer> 42 </answer>."
            )
        text = (
            "A conversation between User and Assistant. The User asks a question, and the Assistant "
            "solves it. The Assistant first thinks about the reasoning process in the mind and then "
            "provides the User with the answer.\n"
            f"User: {task}\n"
            "Assistant:\n<think>\n"
        )
        # Qwen3 tokenises <think> as one added token that the *base* model has barely seen:
        # with <think> as the final prompt token, greedy output is garbage ("__(*", "⚐ ⚐ ⚐").
        # A newline after it restores normal text (probe, 14 Oct 2026, FAILURES_AND_LEARNINGS.md).
        return Prompt(text=text, stop=("</answer>", "\nUser:"), completion_prefix="<think>\n")

    if fam == "countdown":
        raise ValueError("countdown is evaluated with the think_tag template only")

    if template == "zero_shot":
        if fam == "gsm8k":
            return Prompt(text=f"Question: {q}\n{BOX_INSTRUCTION}\nAnswer:", stop=FEW_SHOT_STOP)
        return Prompt(text=f"Problem:\n{q}\n{BOX_INSTRUCTION}\n\nSolution:\n", stop=FEW_SHOT_STOP)

    cot = template == "few_shot_cot"
    block = _few_shot_block(eval_name, cot)
    if fam == "gsm8k":
        return Prompt(text=f"{block}Question: {q}\nAnswer:", stop=FEW_SHOT_STOP)
    # The prompt must end with "Solution:\n", not "Solution:". The Qwen3 tokeniser merges ":\n"
    # into one token, so the shots contain [Solution][:\n] and a prompt ending in a bare [:] is a
    # token state the shots never show; the 0.6B base then emits <|endoftext|> with p = 0.25-0.5
    # (53 % empty completions in the first smoke run). GSM8K is unaffected because its shots use
    # "Answer: The ..." where ":" stands alone anyway.
    return Prompt(text=f"{block}Problem:\n{q}\n\nSolution:\n", stop=FEW_SHOT_STOP)
