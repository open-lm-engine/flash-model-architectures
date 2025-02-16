from typing import Callable

import torch
from parameterized import parameterized
from transformers import set_seed

from cute_kernels import gemm_cute, gemm_torch

from ..test_commons import TestCommons


_SEED = 42


class GEMMTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [False, True],  # is_A_transposed
            [False, True],  # is_B_transposed
            [False, True],  # has_C
            ["triton"],  # kernel_backend
            [None],  # cuda_kernel_algorithm
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [gemm_cute, torch.compile(gemm_cute, fullgraph=True)],  # function
        )
        + TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [False, True],  # is_A_transposed
            [False, True],  # is_B_transposed
            [False, True],  # has_C
            ["cuda"],  # kernel_backend
            ["naive", "cutlass_gemm_cuda"],  # cuda_kernel_algorithm
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [gemm_cute, torch.compile(gemm_cute, fullgraph=True)],  # function
        )
        + TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [False],  # is_A_transposed
            [False],  # is_B_transposed
            [False, True],  # has_C
            ["cuda"],  # kernel_backend
            ["shared_memory"],  # cuda_kernel_algorithm
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [gemm_cute, torch.compile(gemm_cute, fullgraph=True)],  # function
        )
    )
    def test_gemm(
        self,
        size: tuple[int],
        is_A_transposed: bool,
        is_B_transposed: bool,
        has_C: bool,
        kernel_backend: str,
        cuda_kernel_algorithm: str,
        device: torch.device,
        dtype: torch.dtype,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        std = 0.02
        M = 417
        A = (
            torch.randn(
                (size[0], M) if is_A_transposed else (M, size[0]), device=device, dtype=dtype, requires_grad=False
            )
            * std
        )
        B = (
            torch.randn(
                (size[1], size[0]) if is_B_transposed else size, device=device, dtype=dtype, requires_grad=False
            )
            * std
        )
        C = torch.randn(M, size[1], device=device, dtype=dtype, requires_grad=False) * std if has_C else None

        alpha = 0.3
        beta = 0.7 if has_C else 0

        output_kernel = function(
            A=A,
            B=B,
            C=C,
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
            alpha=alpha,
            beta=beta,
            cuda_kernel_algorithm=cuda_kernel_algorithm,
            kernel_backend=kernel_backend,
        )
        output_expected = gemm_torch(
            A=A, B=B, C=C, alpha=alpha, beta=beta, is_A_transposed=is_A_transposed, is_B_transposed=is_B_transposed
        )

        self.assert_equal_tensors(
            output_kernel,
            output_expected,
            False,
            atol_float32=7e-5,
            rtol_float32=1e-4,
            atol_float16=1e-4,
            rtol_float16=5e-3,
            atol_bfloat16=5e-4,
            rtol_bfloat16=7e-3,
        )
