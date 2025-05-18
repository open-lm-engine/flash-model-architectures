from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import (
    CutoTuneParameter,
    KernelBackend,
    add_scalar_cute,
    add_scalar_torch,
    get_counter,
    reset_all_counters,
)
from cute_kernels.kernels.add_scalar import add_scalar_cuda, add_scalar_triton

from ..test_commons import TestCommons


class AddScalarTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [KernelBackend.cuda, KernelBackend.triton, CutoTuneParameter()],  # kernel_backend
            [add_scalar_cute, torch.compile(add_scalar_cute, fullgraph=True)][:1],  # function
        )
    )
    def test_add_scalar(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        kernel_backend: KernelBackend,
        function: Callable,
    ) -> None:
        reset_all_counters()

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        y = 0.42

        z_kernel = function(x_kernel, y, kernel_backend=kernel_backend)
        z_expected = add_scalar_torch(x_expected, y)

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)

        if kernel_backend == KernelBackend.cuda:
            assert get_counter(add_scalar_cuda) > 0
            assert get_counter(add_scalar_triton) == 0
        elif kernel_backend == KernelBackend.triton:
            assert get_counter(add_scalar_cuda) == 0
            assert get_counter(add_scalar_triton) > 1
