# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import cutlass.cute as cute
from cutlass import Float32, const_expr, range_constexpr
from cutlass._mlir.dialects import llvm
from cutlass.cute import TensorSSA
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def _tanh(x: Float32 | float, *, loc=None, ip=None):
    return cute.Float32(
        llvm.inline_asm(
            res=T.f32(),
            operands_=[Float32(x).ir_value(loc=loc, ip=ip)],
            asm_string="tanh.approx.f32 $0, $1;",
            constraints="=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def tanh(x: cute.Float32 | float | TensorSSA, output_dtype=None):
    if const_expr(isinstance(x, TensorSSA)):
        y = cute.make_fragment(x.shape, x.dtype)
        for i in range_constexpr(cute.size(x.shape)):
            _x = x[i].to(Float32)
            y[i] = _tanh(_x)
        y = y.load()
    else:
        y = _tanh(x.to(Float32))
        y = y.to(x.dtype)

    return y


def sigmoid(x: cute.Float32 | float):
    return 0.5 * tanh(0.5 * x) + 0.5
