# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.utils._pytree import tree_map


def _wrap(x: torch.Tensor) -> CuteTensor:
    return CuteTensor(x) if isinstance(x, torch.Tensor) else x


def _unwrap(x: CuteTensor) -> torch.Tensor:
    return x.element if isinstance(x, CuteTensor) else x


class CuteTensor(torch.Tensor):
    element: torch.Tensor

    @torch.compiler.disable
    @staticmethod
    def __new__(cls, element: torch.Tensor) -> CuteTensor:
        tensor = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            element.size(),
            strides=element.stride(),
            storage_offset=element.storage_offset(),
            dtype=element.dtype,
            layout=element.layout,
            device=element.device,
            requires_grad=element.requires_grad,
        )
        tensor.element = element
        return tensor

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        args = tree_map(_unwrap, args)
        kwargs = tree_map(_unwrap, kwargs)

        output = func(*args, **kwargs)
        output = tree_map(_wrap, output)

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.element})"
