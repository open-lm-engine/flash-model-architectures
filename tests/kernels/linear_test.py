from typing import Callable

import torch
from parameterized import parameterized
from transformers import set_seed

from cute_kernels import linear_cute, linear_torch

from ..test_commons import TestCommons


_EPSILON = 1e-5
_SEED = 42


class LinearTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [True, False],  # has_bias
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

        input_kernel, input_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        weight_kernel, weight_expected = self.get_random_duplicated_tensors(size[-1], device=device, dtype=dtype)

        if has_bias:
            bias_kernel = self.get_random_duplicated_tensors(size[-1], device=device, dtype=dtype)
            bias_expected = self.get_random_duplicated_tensors(size[-1], device=device, dtype=dtype)

        z_kernel = function(x=x_kernel, weight=weight_kernel, eps=_EPSILON, memory_efficient=memory_efficient)
        z_expected = linear_torch(input=input, weight=weight, bias=bias)

        z_kernel.sum().backward()
        z_expected.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float16=8e-3, rtol_float16=0)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float16=9e-2, rtol_float16=0)

        if has_bias:
            self.assert_equal_tensors(
                bias_kernel.grad,
                bias_expected.grad,
                False,
                atol_float32=6.5e-5,
                rtol_float32=0,
                atol_float16=0.1,
                rtol_float16=0.01,
            )
