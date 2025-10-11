# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from .accelerator import Accelerator


class CustomOp(nn.Module):
    def __new__(cls, *args, **kwargs):
        try:
            op_name = cls.__name__
        except AttributeError:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}': its 'name' attribute "
                f"was not set, possibly because it was not decorated with "
                f"@CustomOp.register, or it's the CustomOp base class itself."
            ) from None

        if op_name not in cls.op_registry_oot:
            op_cls_to_instantiate = cls
        else:
            op_cls_to_instantiate = cls.op_registry_oot[op_name]
            logger.debug(
                "Instantiating custom op: %s using %s",
                op_name,
                str(op_cls_to_instantiate),
            )
        return super().__new__(op_cls_to_instantiate)

    def __init__(self):
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        accelerator = self._get_accelerator_from_args_kwargs(*args, **kwargs)

        if accelerator == Accelerator.cpu:
            output = self._forward_cpu(*args, **kwargs)
        elif accelerator == Accelerator.cuda:
            output = self._forward_cuda(*args, **kwargs)
        elif accelerator == Accelerator.rocm:
            output = self._forward_rocm(*args, **kwargs)
        elif accelerator == Accelerator.tpu:
            output = self._forward_tpu(*args, **kwargs)
        elif accelerator == Accelerator.xpu:
            output = self._forward_xpu(*args, **kwargs)
        else:
            output = self._forward_torch(*args, **kwargs)

        return output

    def _forward_cpu(self, *args, **kwargs):
        raise NotImplementedError

    def _forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def _forward_rocm(self, *args, **kwargs):
        raise NotImplementedError

    def _forward_torch(self, *args, **kwargs):
        raise NotImplementedError

    def _forward_tpu(self, *args, **kwargs):
        raise NotImplementedError

    def _forward_xpu(self, *args, **kwargs):
        raise NotImplementedError

    op_registry: dict[str, type["CustomOp"]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(op_cls):
            assert name not in cls.op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            cls.op_registry[name] = op_cls
            return op_cls

        return decorator

    def _get_accelerator_from_args_kwargs(self, *args, **kwargs) -> Accelerator:
        accelerator = None

        for arg in args:
            if isinstance(arg, torch.Tensor):
                accelerator = Accelerator.get_device(arg)
                break

        for key, arg in kwargs.items():
            if isinstance(arg, torch.Tensor):
                accelerator = Accelerator.get_device(arg)
                break

        assert accelerator is not None

        return accelerator
