# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch


def _sgd_torch(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    momentum_buffer_list: list[torch.Tensor],
    lr: float,
    weight_decay: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
) -> None:
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list],  # type: ignore[list-item]
        with_indices=True,
    )

    for (
        device_params_,
        device_grads_,
        device_momentum_buffer_list,
    ), indices in grouped_tensors.values():
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
            bufs: list[Tensor] = []

            all_states_with_momentum_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_momentum_buffer = False
                    break
                else:
                    bufs.append(cast(Tensor, device_momentum_buffer_list[i]))

            if all_states_with_momentum_buffer:
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                bufs = []

                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[indices[i]] = (
                            device_grads[i].detach().clone()
                        )
                    else:
                        buf = cast(Tensor, device_momentum_buffer_list[i])
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

                    bufs.append(buf)

            if nesterov:
                torch._foreach_add_(device_grads, bufs, alpha=momentum)
            else:
                device_grads = bufs

        torch._foreach_add_(device_params, device_grads, alpha=-lr)
