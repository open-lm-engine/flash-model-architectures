from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import KernelBackend, add_scalar_cute, add_scalar_torch

from ..test_commons import TestCommons


class AddTensorTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [KernelBackend.cuda, KernelBackend.triton],  # kernel_backend
            [add_scalar_cute, torch.compile(add_scalar_cute, fullgraph=True)],  # function
        )
    )
    def test_add_tensor(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        kernel_backend: KernelBackend,
        function: Callable,
    ) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        y = 0.42

        z_kernel = function(x_kernel, y, kernel_backend=kernel_backend)
        z_expected = add_scalar_torch(x_expected, y)

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
