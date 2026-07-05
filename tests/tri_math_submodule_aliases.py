# tri_math_submodule_aliases.py - supported tl.math/libdevice aliases.
import triton
import triton.language as tl

@triton.jit
def tri_math_submodule_aliases(in_ptr, out_ptr):
    pid = tl.program_id(axis=0)
    x = tl.load(in_ptr + pid)
    p = tl.extra.libdevice.fabsf(x) + 1.25
    q = p + 0.5

    a = tl.math.exp(x) + tl.math.exp2(x)
    b = tl.math.log(p) + tl.math.log2(p)
    c = tl.math.sin(x) + tl.math.cos(x)
    d = tl.math.tan(x) + tl.math.tanh(x)
    e = tl.math.sqrt(p) + tl.math.rsqrt(p)
    f = tl.math.abs(x) + tl.math.floor(p) + tl.math.ceil(p)
    g = tl.math.maximum(p, q) + tl.math.minimum(p, q)
    h = tl.math.fdiv(q, p)

    i = tl.extra.libdevice.exp(x) + tl.extra.libdevice.expf(x)
    j = tl.extra.libdevice.fast_expf(x)
    k = tl.extra.libdevice.exp2(x) + tl.extra.libdevice.exp2f(x)
    l = tl.extra.libdevice.log(p) + tl.extra.libdevice.logf(p)
    m = tl.extra.libdevice.log2(p) + tl.extra.libdevice.log2f(p)
    n = tl.extra.libdevice.sin(x) + tl.extra.libdevice.sinf(x)
    o = tl.extra.libdevice.cos(x) + tl.extra.libdevice.cosf(x)
    r = tl.extra.libdevice.tan(x) + tl.extra.libdevice.tanf(x)
    s = tl.extra.libdevice.tanh(x) + tl.extra.libdevice.tanhf(x)
    t = tl.extra.libdevice.sqrt(p) + tl.extra.libdevice.sqrtf(p)
    u = tl.extra.libdevice.rsqrt(p) + tl.extra.libdevice.rsqrtf(p)
    v = tl.extra.libdevice.abs(x) + tl.extra.libdevice.fabs(x)
    w = tl.extra.libdevice.floor(p) + tl.extra.libdevice.floorf(p)
    y = tl.extra.libdevice.ceil(p) + tl.extra.libdevice.ceilf(p)
    z = tl.extra.libdevice.fmaxf(p, q) + tl.extra.libdevice.fminf(p, q)
    out = a + b + c + d + e + f + g + h
    out = out + i + j + k + l + m + n + o + r
    out = out + s + t + u + v + w + y + z
    tl.store(out_ptr + pid, out)
