"""Compile and launch a four-line Triton kernel. Fails fast if the environment cannot build
Triton's runtime extension (missing CPython headers, broken driver), which otherwise only shows
up fifteen minutes into a vLLM engine start. Run: python scripts/triton_check.py"""

import os
import sys
import sysconfig

import torch
import triton
import triton.language as tl


@triton.jit
def _add_one(x_ptr, n, BLOCK: tl.constexpr):  # noqa: N803  (Triton constexpr convention)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + i, tl.load(x_ptr + i, mask=i < n) + 1, mask=i < n)


def main() -> int:
    inc = sysconfig.get_paths()["include"]
    have_h = os.path.exists(os.path.join(inc, "Python.h"))
    print(
        f"python {sys.version.split()[0]}  headers: {inc}  Python.h {'found' if have_h else 'MISSING'}"
    )
    if not torch.cuda.is_available():
        print("no CUDA device; skipping kernel launch")
        return 1
    x = torch.zeros(1024, device="cuda")
    _add_one[(4,)](x, 1024, BLOCK=256)
    ok = x.sum().item() == 1024.0
    print("triton compile + launch:", "ok" if ok else "WRONG RESULT", torch.cuda.get_device_name(0))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
