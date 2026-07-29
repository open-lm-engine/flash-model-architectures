# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor

from ...accelerator import Accelerator, KernelBackend
from ...utils import is_triton_available


if is_triton_available():
    from .triton_implementation import _single_tensor_sgd_triton


@torch.no_grad()
def sgd(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    momentum_buffer_list: list[torch.Tensor],
    lr: float,
    weight_decay: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    *,
    kernel_backend: KernelBackend | None = None,
) -> None:
    if len(params) == 0:
        return

    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()
    else:
        assert kernel_backend.verify_accelerator()

    if kernel_backend in [KernelBackend.cuda, KernelBackend.triton]:
        is_first_step = False
        if momentum == 0:
            assert len(momentum_buffer_list) == 0
            momentum_buffer_list = [None] * len(params)
        elif momentum_buffer_list[0] is None:
            assert all([m is None for m in momentum_buffer_list])
            is_first_step = True

            for i, p in enumerate(params):
                momentum_buffer_list[i] = torch.empty_like(p, dtype=torch.float32)

        is_dtensor = isinstance(params[0], DTensor)

        if is_dtensor:
            for W, dW, M in zip(params, grads, momentum_buffer_list):
                assert isinstance(dW, DTensor)
                assert W.placements == dW.placements

                if M is not None:
                    assert isinstance(M, DTensor)
                    assert W.placements == M.placements

        for W, dW, M in zip(params, grads, momentum_buffer_list):
            assert W.is_contiguous()
            dW = dW.contiguous()

            if M is not None:
                assert M.is_contiguous()

            if is_dtensor:
                W = W.to_local()
                dW = dW.to_local()

                if M is not None:
                    M = M.to_local()

            _single_tensor_sgd_triton(
                W=W,
                dW=dW,
                M=M,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                dampening=dampening,
                nesterov=nesterov,
                maximize=maximize,
                is_first_step=is_first_step,
            )
    elif kernel_backend == KernelBackend.torch:
        grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
            [params, grads, momentum_buffer_list],  # type: ignore[list-item]
            with_indices=True,
        )

        for (
            device_params_,
            device_grads_,
            device_momentum_buffer_list,
        ), indices in grouped_tensors.values():
            device_params: list[Tensor] = cast(list[Tensor], device_params_)
            device_grads: list[Tensor] = cast(list[Tensor], device_grads_)

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

            # handle internal item() call if lr is a tensor
            if isinstance(lr, torch.Tensor) and torch.compiler.is_compiling():
                grads_x_lr = torch._foreach_mul(device_grads, -lr)
                torch._foreach_add_(device_params, grads_x_lr)
            else:
                torch._foreach_add_(device_params, device_grads, alpha=-lr)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")
