from typing import Any, Callable

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.utils._pytree import tree_map


def _make_contiguous(x: Any) -> Any:
    return x.contiguous() if isinstance(x, torch.Tensor) else x


def _wait_for_ACT(x: Any) -> Any:
    return x.wait() if isinstance(x, AsyncCollectiveTensor) else x


def input_guard(func: Callable, ensure_contiguous: bool = True, wait_for_ACT: bool = True) -> Callable:
    def _guard(x: Any) -> Any:
        if ensure_contiguous:
            x = _make_contiguous(x)

        if wait_for_ACT:
            x = _wait_for_ACT(x)

        return x

    def inner(*args, **kwargs):
        args = tree_map(_guard, args)
        kwargs = tree_map(_guard, kwargs)
        return func(*args, **kwargs)

    return inner
