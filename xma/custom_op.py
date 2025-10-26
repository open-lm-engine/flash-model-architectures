# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Any

import torch

from .enums import KernelBackend


def ctx_needs_gradients(ctx) -> bool:
    return any(ctx.needs_input_grad)


def ctx_save_for_backward(ctx, *args) -> None:
    if ctx_needs_gradients(ctx):
        ctx.save_for_backward(*args)


class CustomOp(torch.autograd.Function):
    @classmethod
    def run(cls, *args, kernel_backend: KernelBackend | None = None, **kwargs) -> Any:
        if kernel_backend is None:
            # infer the kernel backend using args
            for tensor in args:
                if isinstance(tensor, torch.Tensor):
                    kernel_backend = KernelBackend.get_kernel_backend_from_device(tensor)
                    break

            # infer the kernel backend using kwargs if it can't be inferred from kwargs
            if kernel_backend is None:
                for tensor in kwargs.values():
                    if isinstance(tensor, torch.Tensor):
                        kernel_backend = KernelBackend.get_kernel_backend_from_device(tensor)
                        break

        if kernel_backend is None:
            raise ValueError("code is not supposed to reach here! kernel_backend was not inferrable")
        elif kernel_backend == KernelBackend.torch:
            output = cls.forward_backward_torch(*args, **kwargs)
        else:
            args = args + tuple(kwargs.values())

            if kernel_backend == KernelBackend.cuda:
                args += (cls.forward_cuda, cls.backward_cuda)
            elif kernel_backend == KernelBackend.triton:
                args += (cls.forward_triton, cls.backward_triton)
            else:
                raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

            output = cls.apply(*args)

        return output

    @staticmethod
    def forward(ctx, *args) -> Any:
        single_arg = len(args) == 3
        *args, forward_function, backward_function = args

        ctx.backward_function = backward_function
        ctx.single_arg = single_arg

        return forward_function(ctx, *args)

    @staticmethod
    def backward(ctx, *args) -> Any:
        args = ctx.backward_function(ctx, *args)

        if ctx.single_arg:
            return args, None, None

        return *args, None, None

    @classmethod
    def forward_cuda(cls, ctx, *args, **kwargs) -> Any:
        return cls.forward_triton(ctx, *args, **kwargs)

    @classmethod
    def backward_cuda(cls, ctx, *args, **kwargs) -> Any:
        return cls.backward_triton(ctx, *args, **kwargs)

    @classmethod
    def forward_triton(cls, ctx, *args, **kwargs) -> Any:
        raise NotImplementedError

    @classmethod
    def backward_triton(cls, ctx, *args, **kwargs) -> Any:
        raise NotImplementedError

    @classmethod
    def forward_backward_torch(cls, *args, **kwargs) -> Any:
        raise NotImplementedError
