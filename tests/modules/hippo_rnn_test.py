# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn
from parameterized import parameterized

from cute_kernels import HiPPO_RNN, KernelBackend, set_seed

from ..test_commons import TestCommons


_SEED = 42


class RNNTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [torch.float32],
            [3],  # batch_size
            [1024],  # sequence_length
            [256],  # state_size
            [7],  # hippo_size
            [64, 256],  # num_heads
            [False, True],  # has_input_state
            [False, True],  # is_compiling
        )
    )
    def test_hippo_rnn(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        sequence_length: int,
        state_size: int,
        hippo_size: int,
        num_heads: int,
        has_input_state: bool,
        is_compiling: bool,
    ) -> None:
        set_seed(_SEED)

        x_kernel, x_torch, input_state_kernel, input_state_torch, hippo_state_kernel, hippo_state_torch = (
            self._get_packed_tensor_inputs(
                batch_size=batch_size,
                sequence_length=sequence_length,
                total_tokens=None,
                state_size=state_size,
                hippo_size=hippo_size,
                num_heads=num_heads,
                has_input_state=has_input_state,
                dtype=dtype,
                device=device,
            )
        )

        with torch.device(device):
            rnn = HiPPO_RNN(
                input_size=state_size,
                state_size=state_size,
                output_size=state_size,
                num_heads=num_heads,
                hippo_size=hippo_size,
                hippo_measure="legs",
                add_bias=False,
                gradient_clipping=None,
            ).to(dtype)

            nn.init.normal_(rnn.state_weight, std=0.01)
            nn.init.normal_(rnn.hippo_weight, std=0.01)
            nn.init.normal_(rnn.compress_weight, std=0.01)

        rnn_torch = rnn
        rnn_kernel = rnn

        if is_compiling:
            rnn_kernel = torch.compile(rnn_kernel, fullgraph=True)

        y_kernel, output_state_kernel = rnn_kernel(
            input=x_kernel,
            input_state=input_state_kernel,
            hippo_state=hippo_state_kernel,
            kernel_backend=KernelBackend.triton,
        )

        y_torch, output_state_torch = rnn_torch(
            input=x_torch,
            input_state=input_state_torch,
            hippo_state=hippo_state_torch,
            kernel_backend=KernelBackend.torch,
        )

        y_kernel.sum().backward()
        weight_kernel_grads = self.collect_gradients_from_module_and_zero_grads(rnn)

        y_torch.sum().backward()
        weight_torch_grads = self.collect_gradients_from_module_and_zero_grads(rnn)

        self.assert_equal_tensors(
            y_kernel, y_torch, False, atol_float32=1e-5, rtol_float32=0, atol_float16=7.4e-4, rtol_float16=0
        )

        self.assert_equal_tensors(
            output_state_kernel,
            output_state_torch,
            False,
            atol_float32=4e-6,
            rtol_float32=0,
            atol_float16=6.5e-5,
            rtol_float16=0,
        )

        self.assert_equal_tensors(x_kernel.grad, x_torch.grad, False, atol_float32=6.3e-4, rtol_float32=0)

        self.assert_equal_tensors(
            weight_kernel_grads["state_weight"],
            weight_torch_grads["state_weight"],
            False,
            atol_float32=5.4e-5,
            rtol_float32=0,
        )

        self.assert_equal_tensors(
            weight_kernel_grads["hippo_weight"],
            weight_torch_grads["hippo_weight"],
            False,
            atol_float32=1.6e-4,
            rtol_float32=0,
        )

        self.assert_equal_tensors(
            weight_kernel_grads["compress_weight"],
            weight_torch_grads["compress_weight"],
            False,
            atol_float32=8.5e-2,
            rtol_float32=0,
        )

    def _get_packed_tensor_inputs(
        self,
        batch_size: int,
        sequence_length: int | None,
        total_tokens: int | None,
        state_size: int,
        hippo_size: int,
        num_heads: int,
        has_input_state: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor | None]:
        x_kernel, x_torch = self.get_random_duplicated_tensors(
            ((batch_size, sequence_length, state_size) if total_tokens is None else (total_tokens, state_size)),
            device=device,
            dtype=dtype,
            std=0.01,
        )

        input_state_kernel = None
        hippo_state_kernel = None
        input_state_torch = None
        hippo_state_torch = None

        if has_input_state:
            input_state_kernel, input_state_torch = self.get_random_duplicated_tensors(
                (batch_size, state_size), device=device, dtype=dtype, std=0.01
            )

            hippo_state_kernel, hippo_state_torch = self.get_random_duplicated_tensors(
                (batch_size, num_heads, hippo_size), device=device, dtype=dtype, std=0.01
            )

        return x_kernel, x_torch, input_state_kernel, input_state_torch, hippo_state_kernel, hippo_state_torch
