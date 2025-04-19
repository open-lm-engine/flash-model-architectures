from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import set_seed, softmax_cute, softmax_torch

from ..test_commons import TestCommons
from .rmsnorm_test import _get_sizes


_SEED = 42


class SoftmaxTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            _get_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.float32, torch.bfloat16],  # dtype
            [None, 0.7],  # logits_multiplier
            [softmax_cute, torch.compile(softmax_cute, fullgraph=True)],  # function
        )
    )
    def test_softmax(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        logits_multiplier: float | None,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        if isinstance(size, int):
            size = (size,)

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype, std=0.02)

        z_kernel = function(x_kernel, logits_multiplier)
        z_expected = softmax_torch(x_expected, logits_multiplier)

        z_kernel.sum().backward()
        z_expected.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False)
        self.assert_equal_tensors(
            x_kernel.grad,
            x_expected.grad,
            False,
            atol_float32=4e-5,
            rtol_float32=0,
            atol_bfloat16=1.1e-3,
            rtol_bfloat16=0,
        )
