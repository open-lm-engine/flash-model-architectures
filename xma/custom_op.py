# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Any

import torch

from .enums import KernelBackend


class CustomOp(torch.autograd.Function):
    @staticmethod
    def run(*args, **kwargs) -> Any:
        kernel_backend = CustomOp.infer_kernel_backend(*args, **kwargs)

        if kernel_backend == KernelBackend.torch:
            output = CustomOp.forward_backward_torch(*args, **kwargs)
        else:
            args = args + tuple(kwargs.values())
            output = CustomOp.apply(*args)

        return output

    @staticmethod
    def infer_kernel_backend(*args, **kwargs) -> KernelBackend:
        for tensor in args:
            if isinstance(tensor, torch.Tensor):
                return KernelBackend.get_kernel_backend_from_device(tensor)

        for tensor in kwargs.values():
            if isinstance(tensor, torch.Tensor):
                return KernelBackend.get_kernel_backend_from_device(tensor)

    @staticmethod
    def forward(ctx, *args, **kwargs) -> Any:
        kernel_backend = CustomOp.infer_kernel_backend(ctx, *args, **kwargs)
        assert kernel_backend != KernelBackend.torch

        forward_function, backward_function = CustomOp._registry

        ctx.kernel_backend = kernel_backend
        ctx.backward_function = backward_function

        return forward_function(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs) -> Any:
        return ctx.backward_function(ctx, *args, **kwargs)

    @staticmethod
    def forward_triton(ctx, *args, **kwargs) -> Any: ...

    @staticmethod
    def backward_triton(ctx, *args, **kwargs) -> Any: ...

    @staticmethod
    def forward_backward_torch(*args, **kwargs) -> Any: ...

    _registry = {
        KernelBackend.triton: (forward_triton, backward_triton),
        KernelBackend.torch: forward_backward_torch,
    }
