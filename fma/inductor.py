# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from functools import partial

import torch
from torch._inductor.fx_passes.joint_graph import patterns
from torch._inductor.pattern_matcher import fwd_only, joint_fwd_bwd, register_replacement

from .enums import Kernel
from .kernel_backend import KernelBackend
from .ops import rmsnorm, rmsnorm_torch


def init_inductor(cache_size_limit: int) -> None:
    torch._dynamo.config.cache_size_limit = cache_size_limit
    torch._dynamo.config.accumulated_cache_size_limit = 1024


def _rmsnorm_example_inputs(device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [torch.empty(1, device=device, requires_grad=True), torch.empty(1, device=device, requires_grad=True), None]


_MAPPING = {
    Kernel.rmsnorm: (rmsnorm_torch, partial(rmsnorm, kernel_backend=KernelBackend.triton), _rmsnorm_example_inputs)
}


@contextmanager
def enable_kernels(kernels: list[Kernel]):
    patterns_clone = deepcopy(patterns)

    device = torch.cuda.current_device()

    for kernel in kernels:
        search_function, replacement_function, example_inputs_function = _MAPPING[kernel]

        for trace_function in [joint_fwd_bwd, fwd_only]:
            register_replacement(
                search_fn=search_function,
                replace_fn=replacement_function,
                example_inputs=example_inputs_function(device),
                trace_fn=trace_function,
                pass_dicts=patterns,
            )

    yield

    torch._inductor.fx_passes.joint_graph.patterns = patterns_clone
