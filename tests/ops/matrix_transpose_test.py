# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import KernelBackend, matrix_transpose_cute

from ..test_commons import TestCommons


class MatrixTransposeTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [matrix_transpose_cute, torch.compile(matrix_transpose_cute, fullgraph=True)],  # function
        )
    )
    def test_matrix_transpose(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        function: Callable,
    ) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        z_kernel = function(x_kernel)
        z_expected = matrix_transpose_cute(x_expected, kernel_backend=KernelBackend.torch)

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
