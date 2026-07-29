# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from dataclasses import dataclass

import torch

from ...utils import zeros_like_contiguous


@dataclass
class SGDParamsGroup:
    params: dict[str, torch.Tensor]
    lr: float
    momentum: float
    dampening: float
    weight_decay: float
    nesterov: bool
    maximize: bool
    step: int = 0
    momentum_buffers: dict[str, torch.Tensor | None]
    _lazy_init: bool = False

    def __post_init__(self) -> None:
        self.momentum_buffers = {}

        if self.momentum == 0 or self._lazy_init:
            return

        for name, W in self.params.items():
            self.momentum_buffers[name] = zeros_like_contiguous(W, dtype=torch.float32)

    def get_params_for_optimization(self) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor | None]]:
        params = []
        grads = []
        momentum_buffer_list = []
        for name, W in self.params.items():
            if W.grad is None:
                continue

            params.append(W)
            grads.append(W.grad)

            # lazy initialization for momentum buffers
            if self._lazy_init and self.momentum != 0 and self.momentum_buffers.get(name) is None:
                self.momentum_buffers[name] = zeros_like_contiguous(W, dtype=torch.float32)

            momentum_buffer_list.append(self.momentum_buffers.get(name))

        return params, grads, momentum_buffer_list

    def increment_step(self) -> int:
        self.step += 1
        return self.step
