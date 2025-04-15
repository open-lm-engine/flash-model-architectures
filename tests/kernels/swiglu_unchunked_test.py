from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import ceil_divide, swiglu_unchunked_cute, swiglu_unchunked_torch

from ..test_commons import TestCommons


class SwigluUnchunkedTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [swiglu_unchunked_cute, torch.compile(swiglu_unchunked_cute, fullgraph=True)],  # function
        )
    )
    def test_swiglu_unchunked(
        self, size: tuple[int], device: torch.device, dtype: torch.dtype, function: Callable
    ) -> None:
        size = (size[0], ceil_divide(size[-1], 2) * 2)
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        z_kernel = function(x_kernel)
        z_expected = swiglu_unchunked_torch(x_expected)

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float32=5.5e-6, rtol_float32=0)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float32=5e-6, rtol_float32=0)
