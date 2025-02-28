from typing import Callable

import torch
from parameterized import parameterized
from transformers import set_seed

from cute_kernels import softmax_cute, softmax_torch

from ..test_commons import TestCommons
from .rmsnorm_test import _get_sizes


_SEED = 42


class SoftmaxTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            _get_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.float32, torch.bfloat16],  # dtype
            ["triton"],  # kernel_backend_forward
            ["triton"],  # kernel_backend_backward
            [softmax_cute, torch.compile(softmax_cute, fullgraph=True)],  # function
        )
    )
    def test_softmax(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        kernel_backend_forward: str,
        kernel_backend_backward: str,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        if isinstance(size, int):
            size = (size,)

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype, std=0.02)
        logits_multiplier = 0.7

        z_kernel = function(
            x_kernel,
            logits_multiplier,
            kernel_backend_forward=kernel_backend_forward,
            kernel_backend_backward=kernel_backend_backward,
        )
        z_expected = softmax_torch(x_expected, logits_multiplier)

        z_kernel.sum().backward()
        z_expected.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float32=6e-5, rtol_float32=0)
        self.assert_equal_tensors(
            x_kernel.grad,
            x_expected.grad,
            False,
            atol_float32=4e-5,
            rtol_float32=0,
            atol_bfloat16=4e-5,
            rtol_bfloat16=0,
        )
