# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def _sgd_torch(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    momentum_buffer_list: list[torch.Tensor | None],
    lr: float,
    weight_decay: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    step: int,
) -> None:
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list],  # type: ignore[list-item]
        with_indices=True,
    )

    is_first_step = step == 1

    for (device_params_, device_grads_, device_momentum_buffer_list), indices in grouped_tensors.values():
        device_params: list[torch.Tensor] = device_params_
        device_grads: list[torch.Tensor] = device_grads_

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        if weight_decay != 0:
            # Reuse the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        if momentum != 0:
            torch._foreach_mul_(device_momentum_buffer_list, momentum)
            torch._foreach_add_(device_momentum_buffer_list, device_grads, alpha=1 if is_first_step else 1 - dampening)

            if nesterov:
                torch._foreach_add_(device_grads, device_momentum_buffer_list, alpha=momentum)
            else:
                device_grads = device_momentum_buffer_list

        torch._foreach_add_(device_params, device_grads, alpha=-lr)
