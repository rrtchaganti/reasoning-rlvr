"""The evaluation harness (PLAN.md D1).

One command scores any checkpoint on GSM8K-test, MATH-500, perturbed GSM8K, Countdown and
AIME24 and writes a JSON report with per-problem results, pass@1, unbiased pass@k, maj@k,
length statistics, truncation and extraction-failure rates, and bootstrap CIs.
"""
