from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import fused_residual_add_rmsnorm_cute, fused_residual_add_rmsnorm_torch, set_seed

from ..test_commons import TestCommons
from .rmsnorm_test import _get_sizes


_EPSILON = 1e-5
_SEED = 42


class FusedResdidualAddRMSNormTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            _get_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.float32, torch.float16],  # dtype
            [True, False],  # memory_efficient
            [True, False],  # has_weight
            [None, 0.9],  # multiplier
            [
                fused_residual_add_rmsnorm_cute,
                # torch.compile(fused_residual_add_rmsnorm_cute, fullgraph=True),
            ],  # function
        )
    )
    def test_rmsnorm(
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

        z_kernel = function(
            x=x_kernel,
            residual=residual_kernel,
            weight=weight_kernel,
            eps=_EPSILON,
            multiplier=multiplier,
            memory_efficient=memory_efficient,
        )
        z_expected = fused_residual_add_rmsnorm_torch(
            x=x_expected, residual=residual_expected, weight=weight_expected, eps=_EPSILON, multiplier=multiplier
        )

        # z_kernel.sum().backward()
        # z_expected.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float16=2e-2, rtol_float16=0)
        # self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float16=9e-2, rtol_float16=0)

        # if has_weight:
        #     self.assert_equal_tensors(
        #         weight_kernel.grad,
        #         weight_expected.grad,
        #         False,
        #         atol_float32=6.5e-5,
        #         rtol_float32=0,
        #         atol_float16=0.1,
        #         rtol_float16=0.01,
        #     )
