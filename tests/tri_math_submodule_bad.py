# tri_math_submodule_bad.py - unknown nested tl.math name should fail sema.
import triton
import triton.language as tl

@triton.jit
def tri_math_submodule_bad(in_ptr, out_ptr):
    pid = tl.program_id(axis=0)
    x = tl.load(in_ptr + pid)
    y = tl.math.not_real(x)
    tl.store(out_ptr + pid, y)
