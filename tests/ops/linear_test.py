# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import KernelBackend, linear_cute, set_seed

from ..test_commons import TestCommons


_SEED = 42


class LinearTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [False, True],  # has_bias
            [linear_cute, torch.compile(linear_cute, fullgraph=True)],  # function
        )
    )
    def test_linear(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        has_bias: bool,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        M = 417
        input_kernel, input_expected = self.get_random_duplicated_tensors(
            (M, size[-1]), device=device, dtype=dtype, std=0.02
        )
        weight_kernel, weight_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype, std=0.02)

        bias_kernel = None
        bias_expected = None
        if has_bias:
            bias_kernel, bias_expected = self.get_random_duplicated_tensors(
                size[0], device=device, dtype=dtype, std=0.02
            )

        z_kernel = function(
            input=input_kernel,
            weight=weight_kernel,
            bias=bias_kernel,
            kernel_backend_forward=KernelBackend.triton,
            kernel_backend_backward=KernelBackend.triton,
        )

        z_expected = linear_cute(
            input=input_expected,
            weight=weight_expected,
            bias=bias_expected,
            kernel_backend_forward=KernelBackend.torch,
            kernel_backend_backward=KernelBackend.torch,
        )

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(
            z_kernel,
            z_expected,
            False,
            atol_float32=4e-3,
            rtol_float32=1e-4,
            atol_float16=1.5e-4,
            rtol_float16=5e-3,
            atol_bfloat16=2e-3,
            rtol_bfloat16=7e-3,
        )

        self.assert_equal_tensors(input_kernel.grad, input_expected.grad, False)
        self.assert_equal_tensors(weight_kernel.grad, weight_expected.grad, False)

        if has_bias:
            self.assert_equal_tensors(bias_kernel.grad, bias_expected.grad, False)
