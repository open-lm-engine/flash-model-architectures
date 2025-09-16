# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial
from typing import Any, Callable

import torch
from torch.utils._pytree import tree_map


def make_contiguous(x: Any, only_last_stride: bool = False) -> Any:
    if isinstance(x, torch.Tensor):
        if (only_last_stride and x.stride(-1) != 1) or not only_last_stride:
            return x.contiguous()

    return x


def ensure_contiguous(func: Callable, only_last_stride: bool = False) -> Callable:
    def inner(*args, **kwargs):
        args = tree_map(partial(make_contiguous, only_last_stride=only_last_stride), args)
        kwargs = tree_map(partial(make_contiguous, only_last_stride=only_last_stride), kwargs)
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
