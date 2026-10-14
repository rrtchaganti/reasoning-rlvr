"""Record what produced a run, so that every reported number can be traced to a commit.

WORKFLOW.md section 5: every entry point writes ``run_info.json`` into its output directory
with the commit hash, whether the tree was dirty, the resolved config and its hash, the seed,
``pip freeze``, library versions, host and GPU, and timestamps. Runs whose numbers will be
reported refuse a dirty tree unless ``--allow-dirty`` is passed, and the report then carries
the ``-dirty`` suffix so the reader knows.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_commit(short: bool = False) -> str | None:
    """Current HEAD hash, or None when not in a git checkout."""
    return _git("rev-parse", "--short", "HEAD") if short else _git("rev-parse", "HEAD")


def git_is_dirty() -> bool:
    """True when tracked files have uncommitted changes (untracked files are ignored)."""
    out = _git("status", "--porcelain", "--untracked-files=no")
    return bool(out)


def git_branch() -> str | None:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def config_hash(config: dict[str, Any]) -> str:
    """Stable short hash of a config dict (sorted keys), used to name and compare runs."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def require_clean_tree(allow_dirty: bool) -> str:
    """Return a tag for the run: the short commit, with ``-dirty`` appended when allowed.

    Raises when the tree is dirty and dirty runs are not allowed. Why a hard error: a result
    produced from uncommitted code cannot be reproduced by anyone, including you next week.
    """
    commit = git_commit(short=True) or "nogit"
    if git_is_dirty():
        if not allow_dirty:
            raise RuntimeError(
                "working tree has uncommitted changes; commit them or pass --allow-dirty "
                "(the report will then be marked -dirty)"
            )
        return f"{commit}-dirty"
    return commit


def _pip_freeze() -> list[str]:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip().splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # uv-managed venvs have no pip module; fall back to importlib metadata
        from importlib import metadata

        return sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # torch may be missing on CPU-only CI
        return None
    return None


def _versions() -> dict[str, str]:
    """Installed versions read from package metadata, so nothing heavy gets imported."""
    from importlib import metadata

    out: dict[str, str] = {"python": platform.python_version()}
    for dist in ("torch", "transformers", "vllm", "trl", "datasets", "math-verify", "peft"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = "not installed"
    return out


def write_run_info(
    out_dir: str | Path,
    *,
    config: dict[str, Any],
    seed: int | None,
    allow_dirty: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``run_info.json`` into ``out_dir`` and return its contents."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = require_clean_tree(allow_dirty)
    info: dict[str, Any] = {
        "run_tag": tag,
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "git_dirty": git_is_dirty(),
        "config": config,
        "config_hash": config_hash(config),
        "seed": seed,
        "started_at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "gpu": _gpu_name(),
        "versions": _versions(),
        "argv": sys.argv,
        "pip_freeze": _pip_freeze(),
    }
    if extra:
        info.update(extra)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    return info


def finish_run_info(out_dir: str | Path, **fields: Any) -> None:
    """Append end-of-run fields (duration, counts) to an existing ``run_info.json``."""
    path = Path(out_dir) / "run_info.json"
    info = json.loads(path.read_text()) if path.exists() else {}
    info["finished_at"] = datetime.now(UTC).isoformat()
    info.update(fields)
    path.write_text(json.dumps(info, indent=2, default=str))
