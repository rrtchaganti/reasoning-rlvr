# Failures and learnings

Every bug that changed a number, every diagnosis made from a plot or a log, every reward-hacking
incident, every design decision reversed. Written as it happens, not at the end. Same template as
the Spark-TTS project.

Template:

```
## YYYY-MM-DD  <one-line title>                                   [phase N] [tags]
**Symptom.** What was observed (number, plot, log line).
**Diagnosis.** What it turned out to be, and how it was found.
**Fix.** What changed (commit).
**Learning.** The general rule to carry forward.
```

---

## 2026-10-14  Git Bash converts `/mnt/c/...` paths before WSL sees them        [phase 0] [tooling]
**Symptom.** `wsl.exe -d Ubuntu -- bash /mnt/c/.../install.sh` failed with exit 127: bash looked for
`C:/Program Files/Git/mnt/c/...`.
**Diagnosis.** MSYS path conversion rewrites any argument that looks like a POSIX absolute path.
**Fix.** Pass the Linux path inside a quoted `bash -lc "..."` string and set `MSYS_NO_PATHCONV=1`.
**Learning.** Anything launched from Git Bash into WSL gets its arguments rewritten; wrap them.

## 2026-10-14  Ubuntu in WSL has no pip, no venv module, no passwordless sudo     [phase 0] [env]
**Symptom.** `python3 -m pip` and `ensurepip` both missing; `sudo apt install` needs a password.
**Diagnosis.** Stock Ubuntu 24.04 image ships Python without pip; the WSL user is not NOPASSWD.
**Fix.** Use `uv` installed into `~/.local/bin`; `uv venv` and `uv pip` need neither pip nor sudo.
**Learning.** Build the environment with `uv` everywhere (local and rental) so the lock file is the
same artefact on both machines.

## 2026-10-14  WSL given 12 GB starved Windows; install job killed                  [phase 0] [env]
**Symptom.** Background environment build was stopped by Claude Code for critically low system memory
during the wheel-download stage; WSL reported 11.9 GB total, 7.2 GB landed in the uv cache.
**Diagnosis.** `.wslconfig memory=12GB` on a 15.7 GB laptop leaves under 4 GB for Windows, the IDE and
the browser; the download stage fills WSL's page cache and WSL2 returns it to Windows only slowly.
**Fix.** `memory=9GB` and `autoMemoryReclaim=gradual` in `.wslconfig`; rerun is user-triggered.
**Learning.** On a shared laptop, give WSL at most ~60 % of RAM and turn on gradual reclaim; the same
low-memory kill hit the Spark-TTS install on 8 Oct, so this is the machine's standing constraint.

## 2026-10-14  vLLM's torch needs CUDA 13 driver; laptop driver is CUDA 12.7                [phase 0] [env]
**Symptom.** `torch.cuda.get_device_name` -> "The NVIDIA driver on your system is too old (found version 12070)".
**Diagnosis.** vLLM 0.31 pins torch 2.13.0+cu130; the Windows driver 566.36 exposes CUDA 12.7, and WSL2 has no
driver of its own. The CUDA 12.6 toolkit directory seen earlier in WSL is irrelevant; only the driver matters.
**Fix.** Update the Windows NVIDIA driver to the 580 series or newer (user action). Lock file written afterwards.
**Learning.** Check `nvidia-smi` "CUDA Version" against the wheel's `+cuXXX` tag before a multi-GB install, and
pin the rental image to the same CUDA major so the lock file transfers.

## 2026-10-14  `.gitignore` rule `data/` silently excluded the source package `src/rlvr/data`   [phase 0] [tooling]
**Symptom.** `git add src/rlvr/data` refused: "paths are ignored by one of your .gitignore files".
**Diagnosis.** An unanchored directory pattern matches at every depth; the rule meant for the top-level
data folder also caught the Python package of the same name.
**Fix.** Anchor top-level-only rules with a leading slash (`/data/`, `/outputs/`, `/models/`, ...).
**Learning.** Always run `git check-ignore -v <path>` on a new package directory; an ignored source
folder would have shipped a repo that cannot import itself.

## 2026-10-14  Triton cannot compile: `Python.h: No such file or directory` (WSL, no sudo)
**Symptom.** First vLLM engine start on the 0.6B model died after model load with
`torch._inductor.exc.InductorError: CalledProcessError: ['/usr/bin/gcc', '.../cuda_utils.c', ...]`;
the first line of the gcc output was `fatal error: Python.h: No such file or directory`.
**Cause.** Triton builds a tiny CPython extension (`cuda_utils`) on first use with the include path
from `sysconfig` (`/usr/include/python3.12`). Ubuntu's `python3` package does not ship headers; they
live in `python3.12-dev`, which needs `sudo apt install`. Same class of problem as the missing `pip`
and `venv`: a distro Python is not a development Python.
**Fix.** Rebuilt the venv on a uv-managed CPython 3.12 (`uv python install 3.12`), which ships
`include/python3.12/Python.h`. Pinned with `uv python pin 3.12` (`.python-version` committed) so the
environment no longer depends on the distro interpreter at all. Verified with a 4-line Triton kernel
before paying for another vLLM startup.
**Lesson.** Check `python -c "import sysconfig,os;print(os.path.exists(sysconfig.get_paths()['include']+'/Python.h'))"`
as part of environment setup whenever Triton or torch.compile is in the stack. Added to the install
script's verify step.
