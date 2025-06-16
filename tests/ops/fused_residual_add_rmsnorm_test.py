# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import KernelBackend, fused_residual_add_rmsnorm_cute, set_seed

from ..test_commons import TestCommons
from .rmsnorm_test import _get_sizes


_EPSILON = 1e-5
_SEED = 42


class FusedResdidualAddRMSNormTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            _get_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.float32],  # dtype
            [True, False],  # memory_efficient
            [True, False],  # has_weight
            [None, 0.9],  # multiplier
            [
                fused_residual_add_rmsnorm_cute,
                torch.compile(fused_residual_add_rmsnorm_cute, fullgraph=True),
            ],  # function
        )
    )
    def test_fused_residual_add_rmsnorm(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        memory_efficient: bool,
        has_weight: bool,
        multiplier: float | None,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        if isinstance(size, int):
            size = (size,)

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        residual_kernel, residual_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        if has_weight:
            weight_kernel, weight_expected = self.get_random_duplicated_tensors(size[-1], device=device, dtype=dtype)
        else:
            weight_kernel = None
            weight_expected = None

        z_kernel, r_kernel = function(
            x=x_kernel,
            residual=residual_kernel,
            weight=weight_kernel,
            eps=_EPSILON,
            multiplier=multiplier,
            memory_efficient=memory_efficient,
        )
        z_kernel = z_kernel * 2 + r_kernel * 3

        z_expected, r_expected = fused_residual_add_rmsnorm_cute(
            x=x_expected,
            residual=residual_expected,
            weight=weight_expected,
            eps=_EPSILON,
            multiplier=multiplier,
            kernel_backend=KernelBackend.torch,
        )
        z_expected = z_expected * 2 + r_expected * 3

        z_kernel.sum().backward()
        z_expected.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float32=1.4e-4, rtol_float32=0)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float32=1.5e-4, rtol_float32=0)
        self.assert_equal_tensors(
            residual_kernel.grad, residual_expected.grad, False, atol_float32=1.6e-4, rtol_float32=0
        )

        if has_weight:
            self.assert_equal_tensors(
                weight_kernel.grad,
                weight_expected.grad,
                False,
                atol_float32=1.1e-4,
                rtol_float32=0,
            )
