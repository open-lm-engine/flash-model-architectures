# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Any

import torch

from .accelerator import KernelBackend
from .counters import increment_counter


def ctx_needs_gradients(ctx) -> bool:
    return any(ctx.needs_input_grad)


def ctx_save_for_backward(ctx, *args) -> None:
    if ctx_needs_gradients(ctx):
        ctx.save_for_backward(*args)


class CustomOp(torch.autograd.Function):
    @classmethod
    def run(cls, kernel_backend: KernelBackend | None = None, **kwargs) -> Any:
        if kernel_backend is None:
            for tensor in kwargs.values():
                if isinstance(tensor, torch.Tensor):
                    kernel_backend = KernelBackend.get_kernel_backend_from_device(tensor)
                    break
        else:
            assert kernel_backend.verify_accelerator()

        if kernel_backend is None:
            raise ValueError("code is not supposed to reach here! kernel_backend was not inferrable")

        increment_counter(cls._get_key(kernel_backend))

        if kernel_backend == KernelBackend.torch:
            output = cls.forward_backward_torch(**kwargs)
        else:
            function_map = {
                KernelBackend.cuda: (cls.forward_cuda, cls.backward_cuda),
                KernelBackend.rocm: (cls.forward_rocm, cls.backward_rocm),
                KernelBackend.pallas: (cls.forward_pallas, cls.backward_pallas),
                KernelBackend.nki: (cls.forward_nki, cls.backward_nki),
                KernelBackend.triton: (cls.forward_triton, cls.backward_triton),
            }

            forward_function, backward_function = function_map[kernel_backend]
            args = tuple(kwargs.values()) + (forward_function, backward_function)

            output = cls.apply(*args)

        return output

    @staticmethod
    def forward(ctx, *args) -> Any:
        *args, forward_function, backward_function = args

        ctx.backward_function = backward_function

        return forward_function(ctx, *args)

    @staticmethod
    def backward(ctx, *grad_outputs) -> Any:
        grads = ctx.backward_function(ctx, *grad_outputs)

        if not isinstance(grads, tuple):
            grads = (grads,)

        return grads + (None, None)

    @classmethod
    def forward_cuda(cls, ctx, *args, **kwargs) -> Any:
        return cls.forward_triton(ctx, *args, **kwargs)

    @classmethod
    def backward_cuda(cls, ctx, *args, **kwargs) -> Any:
        return cls.backward_triton(ctx, *args, **kwargs)

    @classmethod
    def forward_rocm(cls, ctx, *args, **kwargs) -> Any:
        return cls.forward_triton(ctx, *args, **kwargs)

    @classmethod
    def backward_rocm(cls, ctx, *args, **kwargs) -> Any:
        return cls.backward_triton(ctx, *args, **kwargs)

    @classmethod
    def forward_pallas(cls, ctx, *args, **kwargs) -> Any:
        raise NotImplementedError

    @classmethod
    def backward_pallas(cls, ctx, *args, **kwargs) -> Any:
        raise NotImplementedError

    @classmethod
    def forward_nki(cls, ctx, *args, **kwargs) -> Any:
        raise NotImplementedError

    @classmethod
    def backward_nki(cls, ctx, *args, **kwargs) -> Any:
        raise NotImplementedError

    @classmethod
    def forward_triton(cls, ctx, *args, **kwargs) -> Any:
        raise NotImplementedError

    @classmethod
    def backward_triton(cls, ctx, *args, **kwargs) -> Any:
        raise NotImplementedError

    @staticmethod
    def forward_backward_torch(*args, **kwargs) -> Any:
        raise NotImplementedError

    @classmethod
    def _get_key(cls, kernel_backend: KernelBackend) -> str:
        return f"{cls.__name__}-{kernel_backend.value}"
