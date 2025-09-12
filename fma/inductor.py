# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from functools import partial

import torch
from torch._inductor.fx_passes.joint_graph import patterns
from torch._inductor.pattern_matcher import fwd_only, joint_fwd_bwd, register_replacement

from .constants import Kernel
from .kernel_backend import KernelBackend
from .ops import fused_residual_add_rmsnorm


def init_inductor(cache_size_limit: int) -> None:
    torch._dynamo.config.cache_size_limit = cache_size_limit
    torch._dynamo.config.accumulated_cache_size_limit = 1024


_REPLACEMENT_PATTERNS = {
    Kernel.fused_residual_add_rmsnorm: (
        partial(fused_residual_add_rmsnorm, kernel_backend=KernelBackend.torch),
        partial(fused_residual_add_rmsnorm, kernel_backend=KernelBackend.triton),
    )
}


def enable_kernels(kernels: list[Kernel]) -> None:
    for kernel in kernels:
        search_function, replacement_function, example_inputs = _REPLACEMENT_PATTERNS[kernel]

        for trace_function in [joint_fwd_bwd, fwd_only]:
            register_replacement(
                search_fn=search_function,
                replace_fn=replacement_function,
                example_inputs=example_inputs,
                trace_fn=trace_function,
                pass_dicts=patterns,
            )
