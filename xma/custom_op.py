# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import functools
import inspect
from typing import Any, Callable, Iterable, Sequence

import torch

from .accelerator import Accelerator, KernelBackend
from .autotuner import AutotunedFunction
from .constants import LIBRARY_NAME
from .counters import increment_counter


def ctx_needs_gradients(ctx) -> bool:
    return any(ctx.needs_input_grad)


def ctx_save_for_backward(ctx, *args) -> None:
    if ctx_needs_gradients(ctx):
        ctx.save_for_backward(*args)


class _CustomOpMeta(type(torch.autograd.Function)):
    """lets `_Op[kernel_backend] = (forward_function, backward_function)` register an op's per-backend
    implementations directly on the class; assignment via `[]` requires a metaclass in Python, there's no
    per-class `__setitem__` equivalent."""

    def __setitem__(cls, kernel_backend: KernelBackend, funcs: tuple[Callable, Callable]) -> None:
        cls.functions[kernel_backend] = funcs

    def __getitem__(cls, kernel_backend: KernelBackend) -> tuple[Callable, Callable]:
        if kernel_backend not in cls.functions:
            raise NotImplementedError(f"{cls.__name__} has nothing registered for kernel_backend ({kernel_backend})")

        return cls.functions[kernel_backend]


class CustomOp(torch.autograd.Function, metaclass=_CustomOpMeta):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.functions: dict[KernelBackend, tuple[Callable, Callable]] = {}

    @classmethod
    def run(cls, kernel_backend: KernelBackend | None = None, **kwargs) -> Any:
        if kernel_backend is None:
            kernel_backend = Accelerator.get_kernel_backend()
        else:
            assert kernel_backend.verify_accelerator()

        if kernel_backend is None:
            raise ValueError("code is not supposed to reach here! kernel_backend was not inferrable")

        increment_counter(cls._get_key(kernel_backend))

        if kernel_backend in cls.functions:
            forward_function, backward_function = cls.functions[kernel_backend]
            if backward_function is None:
                return forward_function(**kwargs)

            return cls.apply(*tuple(kwargs.values()), forward_function, backward_function)

        # ops that haven't migrated to the `functions` registry yet keep the old calling convention,
        # so registering only some backends for an op doesn't break the rest of them
        if kernel_backend == KernelBackend.torch:
            return cls.forward_backward_torch(**kwargs)

        return cls.apply(*tuple(kwargs.values()), kernel_backend)

    @staticmethod
    def forward(ctx, *args) -> Any:
        """dispatches to the (forward_function, backward_function) pair `run()` resolved and appended as
        the trailing two positional args - `apply()` takes no kwargs, so they can't be named parameters
        after `*inputs` without becoming keyword-only."""

        *inputs, forward_function, backward_function = args
        ctx.backward_function = backward_function

        return forward_function(ctx, *inputs)

    @staticmethod
    def backward(ctx, *grad_outputs) -> Any:
        """calls the backward_function `forward()` stashed on `ctx`, padding its result with `None`s for
        the non-differentiable forward_function/backward_function slots `apply()` was given."""

        grads = ctx.backward_function(ctx, *grad_outputs)
        if not isinstance(grads, tuple):
            grads = (grads,)

        return (*grads, None, None)

    @classmethod
    def _get_key(cls, kernel_backend: KernelBackend) -> str:
        return f"{cls.__name__}-{kernel_backend.value}"


def xma_op(
    mutates_args: str | Iterable[str] = None,
    device_types: str | Sequence[str] | None = None,
    schema: str | None = None,
    fake_func: Callable | None = None,
) -> Callable:
    def _inner(func: Callable):
        # support for autotuned function with custom op
        if isinstance(func, AutotunedFunction):
            autotuned_function = func

            @functools.wraps(autotuned_function.function)
            def func(*args, **kwargs):
                return autotuned_function(*args, **kwargs)

            func.__signature__ = autotuned_function.exposed_signature

        custom_op = torch.library.custom_op(
            f"{LIBRARY_NAME}::{func.__name__}",
            func,
            mutates_args=mutates_args,
            device_types=device_types,
            schema=schema,
        )

        if fake_func is not None:
            custom_op.register_fake(fake_func)

        def _run(*args, **kwargs):
            return custom_op(*args, **kwargs)

        _run.__signature__ = inspect.signature(func)
        _run.__name__ = func.__name__

        return _run

    return _inner
