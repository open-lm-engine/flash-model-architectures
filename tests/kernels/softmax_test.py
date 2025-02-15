from typing import Callable

import torch
from parameterized import parameterized
from transformers import set_seed

from cute_kernels import softmax_cute, softmax_torch

from ..test_commons import TestCommons
from .rmsnorm_test import _get_sizes


_EPSILON = 1e-5
_SEED = 42


class RMSNormTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            _get_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.float32, torch.bfloat16],  # dtype
            [softmax_cute, torch.compile(softmax_cute, fullgraph=True)],  # function
        )
    )
    def test_softmax(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        if isinstance(size, int):
            size = (size,)

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        z_kernel = function(x_kernel)
        z_expected = softmax_torch(x_expected)

        # z_kernel.sum().backward()
        # z_expected.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float16=8e-3, rtol_float16=0)
        # self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float16=9e-2, rtol_float16=0)
