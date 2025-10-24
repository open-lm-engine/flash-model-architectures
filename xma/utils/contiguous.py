# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Any, Callable

import torch
from torch.utils._pytree import tree_map


def make_contiguous(x: torch.Tensor | tuple[torch.Tensor] | None) -> torch.Tensor | tuple[torch.Tensor] | None:
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        x = x.contiguous()
    elif isinstance(x, (tuple, list)):
        x = [make_contiguous(i) for i in x]

    return x


def ensure_contiguous(func: Callable) -> Callable:
    def inner(*args, **kwargs):
        args = tree_map(make_contiguous, args)
        kwargs = tree_map(make_contiguous, kwargs)
        return func(*args, **kwargs)

    return inner


def ensure_same_strides(*args, force_contiguous: bool = False) -> list[torch.Tensor]:
    if force_contiguous:
        output = tree_map(make_contiguous, args)
    else:
        mismatch = False
        expected_stride = None

        for arg in args:
            if isinstance(arg, torch.Tensor):
                if expected_stride is None:
                    expected_stride = arg.stride()
                elif arg.stride() != expected_stride:
                    mismatch = True
                    break

        output = [make_contiguous(arg) for arg in args] if mismatch else args

    return output
