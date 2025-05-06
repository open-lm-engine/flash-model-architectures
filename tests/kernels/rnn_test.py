from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import pack_sequence_cute, rnn_cute, rnn_torch, set_seed, unpack_sequence_cute

from ..test_commons import TestCommons


_SEED = 42


class RNNTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [torch.float32, torch.float16],
            [4],  # batch_size
            [1024],  # sequence_length
            [64],  # head_dim
            [4],  # num_heads
            [False, True],  # has_input_state
            [rnn_cute, torch.compile(rnn_cute, fullgraph=True)],  # function
        )
    )
    def test_rnn(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        sequence_length: int,
        head_dim: int,
        num_heads: int,
        has_input_state: bool,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        x_kernel, x_expected = self.get_random_duplicated_tensors(
            (batch_size, sequence_length, num_heads, head_dim), device=device, dtype=dtype, std=0.01
        )
        weight_kernel, weight_expected = self.get_random_duplicated_tensors(
            (num_heads, head_dim, head_dim), device=device, dtype=dtype, std=0.01
        )

        input_state_kernel = None
        input_state_expected = None
        if has_input_state:
            input_state_kernel, input_state_expected = self.get_random_duplicated_tensors(
                (batch_size, num_heads, head_dim), device=device, dtype=dtype, std=0.01
            )

        y_kernel = function(x_kernel, weight_kernel, input_state_kernel)
        y_expected = rnn_torch(x_expected, weight_expected, input_state_expected)

        y_kernel.sum().backward()
        y_expected.sum().backward()

        self.assert_equal_tensors(
            y_kernel, y_expected, False, atol_float32=4e-6, rtol_float32=0, atol_float16=6.5e-5, rtol_float16=0
        )
        self.assert_equal_tensors(
            x_kernel.grad, x_expected.grad, False, atol_float32=6e-3, rtol_float32=0, atol_float16=2e-3, rtol_float16=0
        )
        self.assert_equal_tensors(
            weight_kernel.grad,
            weight_expected.grad,
            False,
            atol_float32=6e-3,
            rtol_float32=0,
            atol_float16=2.2e-2,
            rtol_float16=0,
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [torch.float32, torch.float16],
            [[0, 7, 19, 27, 93]],  # cu_seqlens
            [64],  # head_dim
            [4],  # num_heads
            [False, True],  # has_input_state
        )
    )
    def test_rnn_varlen_torch(
        self,
        device: torch.device,
        dtype: torch.dtype,
        cu_seqlens: list[int],
        head_dim: int,
        num_heads: int,
        has_input_state: bool,
    ) -> None:
        set_seed(_SEED)

        batch_size = len(cu_seqlens) - 1
        cu_seqlens = torch.tensor(cu_seqlens, device=device)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        x_packed_kernel, x_packed_expected = self.get_random_duplicated_tensors(
            (cu_seqlens[-1], num_heads, head_dim), device=device, dtype=dtype, std=0.01
        )
        x_unpacked_expected = unpack_sequence_cute(
            x_packed_expected, cu_seqlens, (batch_size, max_seqlen, *x_packed_expected.size()[-2:])
        )

        weight_kernel, weight_expected = self.get_random_duplicated_tensors(
            (num_heads, head_dim, head_dim), device=device, dtype=dtype, std=0.01
        )

        input_state_kernel = None
        input_state_expected = None
        if has_input_state:
            input_state_kernel, input_state_expected = self.get_random_duplicated_tensors(
                (batch_size, num_heads, head_dim), device=device, dtype=dtype, std=0.01
            )

        y_kernel = rnn_torch(
            x_packed_kernel, weight_kernel, input_state_kernel, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )

        y_expected = []
        for i in range(batch_size):
            y_expected.append(
                rnn_torch(
                    x_unpacked_expected[i, cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
                    weight_expected,
                    input_state_expected,
                )
            )
        y_expected = torch.cat(y_expected)
        y_expected = pack_sequence_cute(y_expected, cu_seqlens=cu_seqlens)

        y_kernel.sum().backward()
        y_expected.sum().backward()

        self.assert_equal_tensors(y_kernel, y_expected, True)
        self.assert_equal_tensors(x_packed_kernel.grad, x_packed_expected.grad, True)
        self.assert_equal_tensors(
            weight_kernel.grad,
            weight_expected.grad,
            False,
            atol_float32=1.5e-7,
            rtol_float32=0,
            atol_float16=1.5e-3,
            rtol_float16=0,
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [torch.float32],
            [[0, 7, 19, 27, 93]],  # cu_seqlens
            [64],  # head_dim
            [4],  # num_heads
        )
    )
    def test_rnn_varlen_cute(
        self,
        device: torch.device,
        dtype: torch.dtype,
        cu_seqlens: list[int],
        head_dim: int,
        num_heads: int,
    ) -> None:
        set_seed(_SEED)

        cu_seqlens = torch.tensor(cu_seqlens, device=device)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        x_kernel, x_expected = self.get_random_duplicated_tensors(
            (cu_seqlens[-1], num_heads, head_dim), device=device, dtype=dtype, std=0.01
        )

        weight_kernel, weight_expected = self.get_random_duplicated_tensors(
            (num_heads, head_dim, head_dim), device=device, dtype=dtype, std=0.01
        )

        y_kernel = rnn_cute(x_kernel, weight_kernel, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        y_expected = rnn_torch(x_expected, weight_expected, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        y_kernel.sum().backward()
        y_expected.sum().backward()

        self.assert_equal_tensors(y_kernel, y_expected, False, atol_float32=3e-6, rtol_float32=0)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float32=2e-3, rtol_float32=0)
        self.assert_equal_tensors(weight_kernel.grad, weight_expected.grad, False, atol_float32=4e-3, rtol_float32=0)
