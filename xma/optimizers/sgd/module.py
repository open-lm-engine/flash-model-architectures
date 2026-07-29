# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, TypeAlias

import torch
from torch.optim import SGD

from ...accelerator import KernelBackend
from .op import sgd


ParamsT: TypeAlias = Iterable[torch.Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, torch.Tensor]]


@dataclass
class SGDParamsGroup:
    params: dict[str, torch.Tensor]
    momentum_buffers: dict[str, torch.Tensor] | None
    lr: float
    momentum: float
    dampening: float
    weight_decay: float
    nesterov: bool
    maximize: bool


class SGD:
    def __init__(self, param_groups: list[SGDParamsGroup]) -> SGD:
        self.param_groups = param_groups

    @torch.no_grad()
    def step(self, closure: Callable | None = None, *, kernel_backend: KernelBackend | None = None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = []
            grads = []
            momentum_buffer_list = []
            for name, W in group.params.items():
                if W.grad is None:
                    continue

                params.append(W)
                grads.append(W.grad)
                momentum_buffer_list.append(group.momentum_buffers[name])

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
                kernel_backend=kernel_backend,
            )

            if group["momentum"] != 0:
                for p, m in zip(params, momentum_buffer_list, strict=True):
                    self.state[p]["momentum_buffer"] = m

        return loss
