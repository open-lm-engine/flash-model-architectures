# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import SGD

from ...accelerator import KernelBackend
from ...utils import zeros_like_contiguous
from .op import sgd


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


class SGD:
    def __init__(self, param_groups: list[SGDParamsGroup]) -> SGD:
        self.param_groups = param_groups

    @torch.no_grad()
    def step(self, kernel_backend: KernelBackend | None = None) -> None:
        for group in self.param_groups:
            params, grads, momentum_buffer_list = group.get_params_for_optimization()
            step = group.increment_step()

            sgd(
                params=params,
                grads=grads,
                momentum_buffer_list=momentum_buffer_list,
                lr=group.lr,
                weight_decay=group.weight_decay,
                momentum=group.momentum,
                dampening=group.dampening,
                nesterov=group.nesterov,
                maximize=group.maximize,
                step=step,
                kernel_backend=kernel_backend,
            )
