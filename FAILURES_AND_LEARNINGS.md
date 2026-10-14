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
