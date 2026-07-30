# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from itertools import product

import pytest


torch = pytest.importorskip("torch")

from xma import KernelBackend
from xma.optimizers import SGD, SGDParamsGroup

from ..utils import (
    assert_equal_tensors,
    get_1d_tensor_sizes,
    get_random_duplicated_tensors,
    skip_if_incompatible_kernel_backend,
)


def _generate_args() -> list:
    args = []
    for nesterov in [False, True]:
        # nesterov requires a non-zero momentum and zero dampening; unrestricted otherwise
        dampening_values = [0] if nesterov else [0, 0.7]
        momentum_values = [0.7] if nesterov else [0, 0.7]

        for dampening in dampening_values:
            for momentum in momentum_values:
                args += list(
                    product(
                        get_1d_tensor_sizes(),  # size
                        [torch.float32, torch.float16, torch.bfloat16],  # dtype
                        [True, False],  # maximize
                        [0, 0.7],  # weight_decay
                        [momentum],  # momentum
                        [dampening],  # dampening
                        [nesterov],  # nesterov
                        [1, 3],  # num_steps
                        [KernelBackend.triton],  # kernel_backend
                    )
                )

    return args


@pytest.mark.parametrize(
    "size,dtype,maximize,weight_decay,momentum,dampening,nesterov,num_steps,kernel_backend", _generate_args()
)
def test_sgd(
    size: int,
    dtype: torch.dtype,
    maximize: bool,
    weight_decay: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    num_steps: int,
    kernel_backend: KernelBackend,
) -> None:
    lr = 1e-3

    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    params_kernel = {}
    params_torch = {}
    for i in range(3):
        param_kernel, param_torch = get_random_duplicated_tensors((size,), device=device, dtype=dtype)

        name = f"param_{i}"
        params_kernel[name] = param_kernel
        params_torch[name] = param_torch

    group_kernel = SGDParamsGroup(
        params=params_kernel,
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        maximize=maximize,
        nesterov=nesterov,
    )

    group_torch = SGDParamsGroup(
        params=params_torch,
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        maximize=maximize,
        nesterov=nesterov,
    )

    sgd_kernel = SGD(param_groups=[group_kernel])
    sgd_torch = SGD(param_groups=[group_torch])

    for expected_step in range(1, num_steps + 1):
        grads = [torch.randint(-8, 8, (size,), device=device, dtype=dtype) for _ in range(3)]

        for (pk, pt), g in zip(zip(params_kernel.values(), params_torch.values()), grads):
            pk.grad = g
            pt.grad = g

        sgd_kernel.step(kernel_backend=kernel_backend)
        sgd_torch.step(kernel_backend=KernelBackend.torch)

        assert group_kernel.step == expected_step
        assert group_torch.step == expected_step

        for name in params_kernel:
            assert_equal_tensors(params_kernel[name], params_torch[name], exact_match=False)

            m_kernel = group_kernel.momentum_buffers.get(name)
            m_torch = group_torch.momentum_buffers.get(name)

            if momentum == 0:
                assert m_kernel is None
                assert m_torch is None
            else:
                assert_equal_tensors(m_kernel, m_torch, exact_match=False)
