# tri_math_submodule.py - nested Triton math namespace coverage.
import triton
import triton.language as tl

@triton.jit
def tri_math_submodule(in_ptr, out_ptr):
    pid = tl.program_id(axis=0)
    x = tl.load(in_ptr + pid)
    e = tl.math.exp(x)
    l = tl.math.log(e)
    f = tl.extra.libdevice.fast_expf(l)
    q = tl.extra.libdevice.sqrtf(f)
    tl.store(out_ptr + pid, q)
