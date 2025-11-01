# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import cutlass.cute as cute
from cutlass import Float32
from cutlass._mlir.dialects import llvm
from cutlass.cute.math import tanh
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def tanh(x: Float32 | float, *, loc=None, ip=None):
    x = x.to(Float32)

    x = cute.Float32(
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

    return x


def sigmoid(x: cute.Float32 | float):
    return 0.5 * tanh(0.5 * x) + 0.5
