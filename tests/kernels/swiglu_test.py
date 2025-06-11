# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import KernelBackend, ceil_divide, swiglu_cute, swiglu_packed_cute

from ..test_commons import TestCommons


class SwiGLUTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [KernelBackend.cuda, KernelBackend.triton],  # kernel_backend
            [swiglu_cute, torch.compile(swiglu_cute, fullgraph=True)],  # function
        )
    )
    def test_swiglu(
        self, size: tuple[int], device: torch.device, dtype: torch.dtype, kernel_backend: str, function: Callable
    ) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        y_kernel, y_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        z_kernel = function(
            x_kernel, y_kernel, kernel_backend_forward=kernel_backend, kernel_backend_backward=kernel_backend
        )
        z_expected = swiglu_cute(
            x_expected,
            y_expected,
            kernel_backend_forward=KernelBackend.torch,
            kernel_backend_backward=KernelBackend.torch,
        )

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float32=5.5e-6, rtol_float32=0)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float32=5e-6, rtol_float32=0)
        self.assert_equal_tensors(y_kernel.grad, y_expected.grad, False, atol_float32=5e-6, rtol_float32=0)

    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [swiglu_packed_cute, torch.compile(swiglu_packed_cute, fullgraph=True)],  # function
        )
    )
    def test_swiglu_packed(
        self, size: tuple[int], device: torch.device, dtype: torch.dtype, function: Callable
    ) -> None:
        size = (size[0], ceil_divide(size[-1], 2) * 2)
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        z_kernel = function(x_kernel)
        z_expected = swiglu_packed_cute(
            x_expected, kernel_backend_forward=KernelBackend.torch, kernel_backend_backward=KernelBackend.torch
        )

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float32=5.5e-6, rtol_float32=0)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float32=5e-6, rtol_float32=0)
