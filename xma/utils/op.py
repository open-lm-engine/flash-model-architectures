# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import inspect
from typing import Callable, Iterable, Sequence

import torch


def xma_op(
    mutates_args: str | Iterable[str] = None,
    device_types: str | Sequence[str] | None = None,
    schema: str | None = None,
    fake_func: Callable | None = None,
) -> Callable:
    def _inner(func: Callable):
        custom_op = torch.library.custom_op(
            func.__name__, func, mutates_args=mutates_args, device_types=device_types, schema=schema
        )

        if fake_func is not None:
            custom_op.register_fake(fake_func)

        def _run(*args, **kwargs):
            return custom_op(*args, **kwargs)

        _run.__signature__ = inspect.signature(func)
        _run.__name__ = func.__name__

        return _run

    return _inner
