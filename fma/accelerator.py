# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum

import torch


_IS_ROCM_AVAILABLE = torch.version.hip is not None


class Accelerator(Enum):
    cpu = "cpu"
    cuda = "cuda"
    rocm = "rocm"
    tpu = "tpu"
    xpu = "xpu"

    @staticmethod
    def get_device(x: torch.Tensor) -> Accelerator:
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
