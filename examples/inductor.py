# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import functools
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.fx_passes.joint_graph import patterns
from torch._inductor.pattern_matcher import fwd_only, joint_fwd_bwd, register_replacement

from fma.inductor import partialize_and_update_signature
from fma.ops import rmsnorm, rmsnorm_torch


def search_function(x: torch.Tensor, w: torch.Tensor, kwk: float | None) -> torch.Tensor:
    y = x + 1
    y = rmsnorm_torch(y, weight=w, eps=kwk)
    y = F.silu(y)
    return y


def replacement_function(x: torch.Tensor, w: torch.Tensor, kwk: float | None) -> torch.Tensor:
    y = rmsnorm(x, w, eps=kwk)
    return y


class MyModule(nn.Module):
    def forward(self, x: torch.Tensor, w, kwk: float | None) -> torch.Tensor:
        x = search_function(x, w, kwk)
        x = search_function(x, w, kwk)
        return x


device = torch.cuda.current_device()

for trace_function in [joint_fwd_bwd, fwd_only]:
    register_replacement(
        search_fn=partialize_and_update_signature(search_function, kwk=None),
        replace_fn=partialize_and_update_signature(replacement_function, kwk=None),
        example_inputs=[
            torch.empty(1, device=device, requires_grad=True),
            torch.empty(1, device=device, requires_grad=True),
        ],
        trace_fn=trace_function,
        pass_dicts=patterns,
    )

m = MyModule()

print(
    "original value =",
    m(
        torch.tensor([1.0], device=device, requires_grad=True),
        torch.tensor([1.0], device=device, requires_grad=True),
        None,
    ),
)
print(
    "expected value =",
    replacement_function(
        replacement_function(
            torch.tensor([1.0], device=device, requires_grad=True),
            torch.tensor([1.0], device=device, requires_grad=True),
            None,
        ),
        torch.tensor([1.0], device=device, requires_grad=True),
        None,
    ),
)

m_compiled = torch.compile(m, fullgraph=True)
print(
    "value with compile =",
    m_compiled(
        torch.tensor([1.0], device=device, requires_grad=True),
        torch.tensor([1.0], device=device, requires_grad=True),
        None,
    ),
)
