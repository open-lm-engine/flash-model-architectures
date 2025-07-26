# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn
from parameterized import parameterized

from cute_kernels import GRU, KernelBackend, set_seed

from ..test_commons import TestCommons


_SEED = 42


class GRUTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [4],  # batch_size
            [1024],  # sequence_length
            [256],  # state_size
            [4, 256],  # num_heads
            [False, True],  # has_input_state
            [False, True],  # is_compiling
        )
    )
    def test_gru(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        sequence_length: int,
        state_size: int,
        num_heads: int,
        has_input_state: bool,
        is_compiling: bool,
    ) -> None:
        set_seed(_SEED)

        x_kernel, x_torch, input_state_kernel, input_state_torch = self._get_packed_tensor_inputs(
            batch_size=batch_size,
            sequence_length=sequence_length,
            total_tokens=None,
            state_size=state_size,
            has_input_state=has_input_state,
            dtype=dtype,
            device=device,
        )

        with torch.device(device):
            gru = GRU(
                input_size=state_size,
                state_size=state_size,
                output_size=state_size,
                num_heads=num_heads,
                add_bias=False,
                gradient_clipping=None,
            ).to(dtype)

            nn.init.normal_(gru.state_weight, std=0.01)

        gru_torch = gru
        gru_kernel = gru

        if is_compiling:
            gru_kernel = torch.compile(gru_kernel, fullgraph=True)

        y_kernel, output_state_kernel = gru_kernel(
            input=x_kernel, input_state=input_state_kernel, kernel_backend=KernelBackend.triton
        )
        y_torch, output_state_torch = gru_torch(
            input=x_torch, input_state=input_state_torch, kernel_backend=KernelBackend.torch
        )

        y_kernel.sum().backward()
        weight_kernel_grads = self.collect_gradients_from_module_and_zero_grads(gru)

        y_torch.sum().backward()
        weight_torch_grads = self.collect_gradients_from_module_and_zero_grads(gru)

        self.assert_equal_tensors(
            y_kernel,
            y_torch,
            False,
            atol_float32=4e-6,
            rtol_float32=0,
            atol_float16=6.5e-5,
            rtol_float16=0,
            atol_bfloat16=2e-4,
            rtol_bfloat16=0,
        )

        self.assert_equal_tensors(
            output_state_kernel,
            output_state_torch,
            False,
            atol_float32=4e-6,
            rtol_float32=0,
            atol_float16=6.5e-5,
            rtol_float16=0,
            atol_bfloat16=2e-4,
            rtol_bfloat16=0,
        )

        for weight_name in weight_kernel_grads:
            self.assert_equal_tensors(
                weight_kernel_grads[weight_name],
                weight_torch_grads[weight_name],
                False,
                atol_float32=6e-3,
                rtol_float32=0,
                atol_float16=2.3e-2,
                rtol_float16=0,
                atol_bfloat16=6.7e-2,
                rtol_bfloat16=0,
            )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [[0, 7, 19, 27, 93]],  # cu_seqlens
            [256],  # state_size
            [4],  # num_heads
            [False, True],  # has_input_state
        )
    )
    def test_gru_varlen_torch(
        self,
        device: torch.device,
        dtype: torch.dtype,
        cu_seqlens: list[int],
        state_size: int,
        num_heads: int,
        has_input_state: bool,
    ) -> None:
        set_seed(_SEED)

        batch_size = len(cu_seqlens) - 1
        cu_seqlens = torch.tensor(cu_seqlens, device=device)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        x_packed_kernel, x_packed_torch, input_state_kernel, input_state_torch = self._get_packed_tensor_inputs(
            batch_size=batch_size,
            sequence_length=None,
            total_tokens=cu_seqlens[-1],
            state_size=state_size,
            has_input_state=has_input_state,
            dtype=dtype,
            device=device,
        )

        with torch.device(device):
            gru = GRU(
                input_size=state_size,
                state_size=state_size,
                output_size=state_size,
                num_heads=num_heads,
                add_bias=False,
                gradient_clipping=None,
            ).to(dtype)

            nn.init.normal_(gru.state_weight, std=0.01)

        y_kernel, _ = gru(
            input=x_packed_kernel,
            input_state=input_state_kernel,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.torch,
        )

        y_torch = []
        for i in range(batch_size):
            y, _ = gru(
                input=x_packed_torch[cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
                input_state=input_state_torch[i].unsqueeze(0) if has_input_state else None,
                kernel_backend=KernelBackend.torch,
            )
            y_torch.append(y.squeeze(0))
        y_torch = torch.cat(y_torch)

        y_kernel.sum().backward()
        weight_kernel_grads = self.collect_gradients_from_module_and_zero_grads(gru)

        y_torch.sum().backward()
        weight_torch_grads = self.collect_gradients_from_module_and_zero_grads(gru)

        self.assert_equal_tensors(y_kernel, y_torch, False)

        for weight_name in weight_kernel_grads:
            self.assert_equal_tensors(
                weight_kernel_grads[weight_name],
                weight_torch_grads[weight_name],
                False,
                atol_float32=2.4e-7,
                rtol_float32=0,
                atol_float16=1.5e-3,
                rtol_float16=0,
                atol_bfloat16=6e-3,
                rtol_bfloat16=0,
            )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [[0, 7, 19, 27, 93]],  # cu_seqlens
            [256],  # state_size
            [4],  # num_heads
            [False, True],  # has_input_state
            [False, True],  # is_compiling
        )
    )
    def test_gru_varlen_cute(
        self,
        device: torch.device,
        dtype: torch.dtype,
        cu_seqlens: list[int],
        state_size: int,
        num_heads: int,
        has_input_state: bool,
        is_compiling: bool,
    ) -> None:
        set_seed(_SEED)

        batch_size = len(cu_seqlens) - 1
        cu_seqlens = torch.tensor(cu_seqlens, device=device)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        x_kernel, x_torch, input_state_kernel, input_state_torch = self._get_packed_tensor_inputs(
            batch_size=batch_size,
            sequence_length=None,
            total_tokens=cu_seqlens[-1],
            state_size=state_size,
            has_input_state=has_input_state,
            dtype=dtype,
            device=device,
        )

        with torch.device(device):
            gru = GRU(
                input_size=state_size,
                state_size=state_size,
                output_size=state_size,
                num_heads=num_heads,
                add_bias=False,
                gradient_clipping=None,
            ).to(dtype)

            nn.init.normal_(gru.state_weight, std=0.01)

        gru_torch = gru
        gru_kernel = gru

        if is_compiling:
            gru_kernel = torch.compile(gru_kernel, fullgraph=True)

        y_kernel, _ = gru_kernel(
            input=x_kernel,
            input_state=input_state_kernel,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.triton,
        )

        y_torch, _ = gru_torch(
            input=x_torch,
            input_state=input_state_torch,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.torch,
        )

        y_kernel.sum().backward()
        weight_kernel_grads = self.collect_gradients_from_module_and_zero_grads(gru)

        y_torch.sum().backward()
        weight_torch_grads = self.collect_gradients_from_module_and_zero_grads(gru)

        self.assert_equal_tensors(
            y_kernel,
            y_torch,
            False,
            atol_float32=3e-6,
            rtol_float32=0,
            atol_float16=6.5e-5,
            rtol_float16=0,
            atol_bfloat16=1.5e-4,
            rtol_bfloat16=0,
        )

        self.assert_equal_tensors(
            x_kernel.grad,
            x_torch.grad,
            False,
            atol_float32=1.3e-4,
            rtol_float32=0,
            atol_float16=3e-3,
            rtol_float16=0,
            atol_bfloat16=8e-3,
            rtol_bfloat16=0,
        )

        for weight_name in weight_kernel_grads:
            self.assert_equal_tensors(
                weight_kernel_grads[weight_name],
                weight_torch_grads[weight_name],
                False,
                atol_float32=1.3e-4,
                rtol_float32=0,
                atol_float16=3e-3,
                rtol_float16=0,
                atol_bfloat16=8e-3,
                rtol_bfloat16=0,
            )

    def _get_packed_tensor_inputs(
        self,
        batch_size: int,
        sequence_length: int | None,
        total_tokens: int | None,
        state_size: int,
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
        input_state_torch = None
        if has_input_state:
            input_state_kernel, input_state_torch = self.get_random_duplicated_tensors(
                (batch_size, state_size), device=device, dtype=dtype, std=0.01
            )

        return x_kernel, x_torch, input_state_kernel, input_state_torch
