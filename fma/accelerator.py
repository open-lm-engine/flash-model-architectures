# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum

import torch


_IS_ROCM_AVAILABLE = torch.version.hip is not None
_FORCE_TRITON_BACKEND = False


@contextmanager
def force_triton_backend(enable: bool = True):
    global _FORCE_TRITON_BACKEND

    original_value = _FORCE_TRITON_BACKEND
    _FORCE_TRITON_BACKEND = enable

    yield

    _FORCE_TRITON_BACKEND = original_value


class Accelerator(Enum):
    cpu = "cpu"
    cuda = "cuda"
    rocm = "rocm"
    tpu = "tpu"
    xpu = "xpu"
    # for triton compatible accelerators
    triton = "triton"

    @staticmethod
    def get_device(x: torch.Tensor) -> Accelerator:
        global _FORCE_TRITON_BACKEND
        if _FORCE_TRITON_BACKEND:
            return Accelerator.triton

        device_type = x.device.type

        if device_type == "cuda":
            return Accelerator.rocm if _IS_ROCM_AVAILABLE else Accelerator.cuda
        elif device_type == "cpu":
            return Accelerator.cpu
        elif device_type == "xla":
            return Accelerator.tpu
        elif device_type == "xpu":
            return Accelerator.xpu

        raise ValueError(f"Unsupported device type: {device_type}")
