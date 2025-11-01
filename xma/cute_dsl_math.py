# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.math import tanh
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def tanh(x: cute.Float32 | float, *, loc=None, ip=None):
    if output_dtype is None:
        output_dtype = x.dtype

    x = x.to(cute.Float32)

    x = llvm.inline_asm(
        res=T.f32(),
        operands_=[cute.Float32.ir_value(loc=loc, ip=ip)],
        asm_string="tanh.approx.f32 $0, $1;",
        constraints="=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    x = cute.Float32(x)

    x = x.to(output_dtype)

    return x


def sigmoid(x: cute.Float32 | float):
    return tanh(x)
