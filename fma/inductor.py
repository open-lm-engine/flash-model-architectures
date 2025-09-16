# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import inspect
from functools import partial
from typing import Callable

import torch
from torch._inductor.fx_passes.joint_graph import patterns
from torch._inductor.pattern_matcher import fwd_only, joint_fwd_bwd, register_replacement

from .enums import KernelBackend
from .ops import fused_residual_add_rmsnorm, fused_residual_add_rmsnorm_torch, rmsnorm, rmsnorm_torch


def init_inductor(cache_size_limit: int) -> None:
    torch._dynamo.config.cache_size_limit = cache_size_limit
    torch._dynamo.config.accumulated_cache_size_limit = cache_size_limit


def partialize_and_update_signature(func: Callable, **kwargs) -> Callable:
    original_sig = inspect.signature(func)
    parameters = original_sig.parameters

    new_parameters = {key: value for key, value in parameters.items() if key not in kwargs}
    new_signature = inspect.Signature(parameters=list(new_parameters.values()))

    partial_func = partial(func, **kwargs)

    def wrapper(*args, **kwargs):
        return partial_func(*args, **kwargs)

    wrapper.__signature__ = new_signature
    wrapper.__name__ = func.__name__

    return wrapper


def _rmsnorm_example_inputs_0(device: torch.device) -> list[torch.Tensor, torch.Tensor, None]:
    return [torch.empty(1, device=device, requires_grad=True), torch.empty(1, device=device, requires_grad=True), None]


def _fused_residual_add_rmsnorm_inputs_0(device: torch.device) -> list[torch.Tensor, torch.Tensor, None, None]:
    return [
        torch.empty(1, device=device, requires_grad=True),
        torch.empty(1, device=device, requires_grad=True),
        torch.empty(1, device=device, requires_grad=True),
        None,
        None,
    ]


def _fused_residual_add_rmsnorm_inputs_1(device: torch.device) -> list[torch.Tensor, torch.Tensor, None, float]:
    return [
        torch.empty(1, device=device, requires_grad=True),
        torch.empty(1, device=device, requires_grad=True),
        torch.empty(1, device=device, requires_grad=True),
        None,
        0.1,
    ]


_MAPPING = {
    rmsnorm.__name__: (
        rmsnorm_torch,
        partial(rmsnorm, kernel_backend=KernelBackend.triton),
        [_rmsnorm_example_inputs_0],
    ),
    fused_residual_add_rmsnorm.__name__: (
        fused_residual_add_rmsnorm_torch,
        partial(fused_residual_add_rmsnorm, kernel_backend=KernelBackend.triton),
        [_fused_residual_add_rmsnorm_inputs_0, _fused_residual_add_rmsnorm_inputs_1],
    ),
}


def enable_kernels(kernels: list[str]):
    device = torch.cuda.current_device()

    for kernel in kernels:
        search_function, replacement_function, example_inputs_functions = _MAPPING[kernel]

        for trace_function in [joint_fwd_bwd, fwd_only]:
            for example_inputs_function in example_inputs_functions:
                register_replacement(
                    search_fn=search_function,
                    replace_fn=replacement_function,
                    example_inputs=example_inputs_function(device),
                    trace_fn=trace_function,
                    pass_dicts=patterns,
                )
