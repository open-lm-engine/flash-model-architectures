# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.fx_passes.joint_graph import patterns
from torch._inductor.pattern_matcher import fwd_only, joint_fwd_bwd, register_replacement


def search_function(x: torch.Tensor) -> torch.Tensor:
    y = x + 1
    y = F.silu(y)
    return y


def replacement_function(x: torch.Tensor) -> torch.Tensor:
    y = x - 1
    return y


class MyModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = search_function(x)
        x = search_function(x)
        return x


device = torch.cuda.current_device()

for trace_function in [joint_fwd_bwd, fwd_only]:
    register_replacement(
        search_fn=search_function,
        replace_fn=replacement_function,
        example_inputs=[torch.empty(1, device=device, requires_grad=True)],
        trace_fn=trace_function,
        pass_dicts=patterns,
    )

m = MyModule()

print("original value =", m(torch.tensor(1.0, device=device, requires_grad=True)))
print(
    "expected value =",
    replacement_function(replacement_function(torch.tensor(1.0, device=device, requires_grad=True))),
)

m_compiled = torch.compile(m, fullgraph=True)
print("value with compile =", m_compiled(torch.tensor(1.0, device=device, requires_grad=True)))
