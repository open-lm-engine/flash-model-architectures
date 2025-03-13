# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

import operator

import torch
import torch._inductor


aten = torch.ops.aten
prims = torch.ops.prims

from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    CallFunctionVarArgs,
    CallMethod,
    CallMethodVarArgs,
    CallModule,
    CallModuleVarArgs,
    ExclusiveKeywordArg,
    Ignored,
    KeywordArg,
    ListOf,
    MultiOutputPattern,
    PatternExpr,
    RepeatedExpr,
    _TargetArgsExpr,
    _TargetExpr,
    _TargetExprVarArgs,
)


sigmoid_default = CallFunction(aten.sigmoid.default, KeywordArg("gate"))
mul_Tensor = CallFunction(aten.mul.Tensor, KeywordArg("gate"), sigmoid_default, _users=2)
mul_Tensor_1 = CallFunction(aten.mul.Tensor, KeywordArg("up"), mul_Tensor)
mul_Tensor_2 = CallFunction(aten.mul.Tensor, KeywordArg("tangents_1"), KeywordArg("up"))
sigmoid_default_1 = CallFunction(aten.sigmoid.default, KeywordArg("gate"), _users=2)
full_default = CallFunction(
    aten.full.default, Ignored(), Ignored(), dtype=Ignored(), layout=torch.strided, device=Ignored(), pin_memory=False
)
sub_Tensor = CallFunction(aten.sub.Tensor, full_default, sigmoid_default_1)
mul_Tensor_3 = CallFunction(aten.mul.Tensor, KeywordArg("gate"), sub_Tensor)
add_Scalar = CallFunction(aten.add.Scalar, mul_Tensor_3, Ignored())
mul_Tensor_4 = CallFunction(aten.mul.Tensor, sigmoid_default_1, add_Scalar)
mul_Tensor_5 = CallFunction(aten.mul.Tensor, mul_Tensor_2, mul_Tensor_4)
mul_Tensor_6 = CallFunction(aten.mul.Tensor, KeywordArg("tangents_1"), mul_Tensor)
swiglu_training = MultiOutputPattern([mul_Tensor_1, mul_Tensor_5, mul_Tensor_6])
