# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Any, Callable

import torch


def _make_contiguous(x: Any) -> Any:
    return x.contiguous() if isinstance(x, torch.Tensor) else x


def ensure_contiguous(func: Callable, condition: bool | None = None) -> Callable:
    def inner(*args, **kwargs):
        if condition is None or condition():
            args = [_make_contiguous(i) for i in args]
            kwargs = {k: _make_contiguous(v) for k, v in kwargs.items()}

        return func(*args, **kwargs)

    return inner
