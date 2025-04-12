from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import rnn_cute, rnn_torch, set_seed

from ..test_commons import TestCommons


_SEED = 42


class RNNTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [torch.float32],
            [4],  # batch_size
            [1024],  # sequence_length
            [64],  # state_size
            [4],  # num_heads
            [rnn_cute, torch.compile(rnn_cute, fullgraph=True)],  # function
        )
    )
    def test_rnn(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        sequence_length: int,
        state_size: int,
        num_heads: int,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        x_kernel, x_expected = self.get_random_duplicated_tensors(
            (batch_size, sequence_length, num_heads, state_size), device=device, dtype=dtype
        )
        weight_kernel, weight_expected = self.get_random_duplicated_tensors(
            (num_heads, state_size, state_size), device=device, dtype=dtype
        )

        y_kernel = function(x_kernel, weight_kernel)
        y_expected = rnn_torch(x_expected, weight_expected)

        y_kernel.sum().backward()
        y_expected.sum().backward()

        self.assert_equal_tensors(
            y_kernel,
            y_expected,
            False,
            atol_float16=4e-3,
            rtol_float16=0,
            atol_bfloat16=2e-2,
            rtol_bfloat16=0,
            atol_float32=6e-3,
            rtol_float32=0,
        )

        self.assert_equal_tensors(
            x_kernel.grad,
            x_expected.grad,
            False,
            atol_float16=4e-3,
            rtol_float16=0,
            atol_bfloat16=4e-2,
            rtol_bfloat16=0,
            atol_float32=6e-3,
            rtol_float32=0,
        )

        self.assert_equal_tensors(
            weight_kernel.grad,
            weight_expected.grad,
            False,
            atol_float16=4e-3,
            rtol_float16=0,
            atol_bfloat16=4e-2,
            rtol_bfloat16=0,
            atol_float32=6e-3,
            rtol_float32=0,
        )
