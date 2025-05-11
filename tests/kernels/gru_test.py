import torch
from parameterized import parameterized

from cute_kernels import gru_torch, set_seed

from ..test_commons import TestCommons


_SEED = 42


class GRUTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
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

        input_packed_kernel, input_packed_expected = self.get_random_duplicated_tensors(
            (cu_seqlens[-1], num_heads, head_dim), device=device, dtype=dtype, std=0.01
        )

        weight_kernel, weight_expected = self.get_random_duplicated_tensors(
            (num_heads, head_dim, head_dim), device=device, dtype=dtype, std=0.01
        )

        # forget
        forget_input_packed_kernel, forget_input_packed_expected = self.get_random_duplicated_tensors(
            (cu_seqlens[-1], num_heads, head_dim), device=device, dtype=dtype, std=0.01
        )

        forget_weight_kernel, forget_weight_expected = self.get_random_duplicated_tensors(
            (num_heads, head_dim, head_dim), device=device, dtype=dtype, std=0.01
        )

        # reset
        reset_input_packed_kernel, reset_input_packed_expected = self.get_random_duplicated_tensors(
            (cu_seqlens[-1], num_heads, head_dim), device=device, dtype=dtype, std=0.01
        )

        reset_weight_kernel, reset_weight_expected = self.get_random_duplicated_tensors(
            (num_heads, head_dim, head_dim), device=device, dtype=dtype, std=0.01
        )

        input_state_kernel = None
        input_state_expected = None
        if has_input_state:
            input_state_kernel, input_state_expected = self.get_random_duplicated_tensors(
                (batch_size, num_heads, head_dim), device=device, dtype=dtype, std=0.01
            )

        y_kernel = gru_torch(
            input=input_packed_kernel,
            weight=weight_kernel,
            forget_input=forget_input_packed_kernel,
            forget_weight=forget_weight_kernel,
            reset_input=reset_input_packed_kernel,
            reset_weight=reset_weight_kernel,
            input_state=input_state_kernel,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        y_expected = []
        for i in range(batch_size):
            y = gru_torch(
                input=input_packed_expected[cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
                weight=weight_expected,
                forget_input=forget_input_packed_expected[cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
                forget_weight=forget_weight_expected,
                reset_input=reset_input_packed_expected[cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
                reset_weight=reset_weight_expected,
                input_state=input_state_expected[i].unsqueeze(0) if has_input_state else None,
            ).squeeze(0)
            y_expected.append(y)
        y_expected = torch.cat(y_expected)

        y_kernel.sum().backward()
        y_expected.sum().backward()

        self.assert_equal_tensors(y_kernel, y_expected, True)

        self.assert_equal_tensors(input_packed_kernel.grad, input_packed_expected.grad, True)
        self.assert_equal_tensors(forget_input_packed_kernel.grad, forget_input_packed_expected.grad, True)
        self.assert_equal_tensors(reset_input_packed_kernel.grad, reset_input_packed_expected.grad, True)

        self.assert_equal_tensors(
            weight_kernel.grad,
            weight_expected.grad,
            False,
            atol_float32=1.5e-7,
            rtol_float32=0,
            atol_float16=1.5e-3,
            rtol_float16=0,
            atol_bfloat16=6e-3,
            rtol_bfloat16=0,
        )

        self.assert_equal_tensors(
            forget_weight_kernel.grad,
            forget_weight_expected.grad,
            False,
            atol_float32=1.5e-7,
            rtol_float32=0,
            atol_float16=1.5e-3,
            rtol_float16=0,
            atol_bfloat16=6e-3,
            rtol_bfloat16=0,
        )

        self.assert_equal_tensors(
            reset_weight_kernel.grad,
            reset_weight_expected.grad,
            False,
            atol_float32=1.5e-7,
            rtol_float32=0,
            atol_float16=1.5e-3,
            rtol_float16=0,
            atol_bfloat16=6e-3,
            rtol_bfloat16=0,
        )
