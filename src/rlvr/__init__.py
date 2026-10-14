"""rlvr: a reproducible study of the post-training recipe for reasoning on a 1.7B base model.

Sub-packages:
- ``rlvr.ref``   from-scratch, tested reference implementations (pass@k, advantages, GRPO loss, KL, sampling)
- ``rlvr.eval``  the evaluation harness (datasets, prompts, sampler, extraction, metrics, report)
- ``rlvr.data``  dataset filtering, contamination probes, synthetic set generators
- ``rlvr.utils`` run metadata and reproducibility helpers
"""

__version__ = "0.0.1"
