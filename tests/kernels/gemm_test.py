from typing import Callable

import torch
from parameterized import parameterized
from transformers import set_seed

from cute_kernels import KernelBackend, gemm_cute, gemm_torch
from cute_kernels.kernels.gemm import CUDAKernelAlgorithm

from ..test_commons import TestCommons


_SEED = 42


class GEMMTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [(4, 4), (8, 8), (16, 16), (32, 32), (55, 55), (3179, 3179)],
            # TestCommons.get_2d_tensor_sizes(),  # size
            [False],  # is_a_transposed
            [False],  # is_b_transposed
            [False],  # has_c
            [KernelBackend.cuda],  # kernel_backend
            [CUDAKernelAlgorithm.cutlass_gemm_cuda],  # cuda_kernel_algorithm
            [torch.device("cuda")],  # device
            [torch.float32],  # dtype
            [gemm_cute, torch.compile(gemm_cute, fullgraph=True)][:1],  # function
        )
    )
    def test_gemm(
        self,
        size: tuple[int],
        is_a_transposed: bool,
        is_b_transposed: bool,
        has_c: bool,
        kernel_backend: KernelBackend,
        cuda_kernel_algorithm: CUDAKernelAlgorithm,
        device: torch.device,
        dtype: torch.dtype,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        std = 0.02
        M = size[0]
        a = (
            torch.randn(
                (size[0], M) if is_a_transposed else (M, size[0]), device=device, dtype=dtype, requires_grad=False
            )
            * std
        )
        b = (
            torch.randn(
                (size[1], size[0]) if is_b_transposed else size, device=device, dtype=dtype, requires_grad=False
            )
            * std
        )
        c = torch.randn(M, size[1], device=device, dtype=dtype, requires_grad=False) * std if has_c else None

        alpha = 0.3
        beta = 0.7 if has_c else 0

        output_kernel = function(
            a=a,
            b=b,
            c=c,
            is_a_transposed=is_a_transposed,
            is_b_transposed=is_b_transposed,
            alpha=alpha,
            beta=beta,
            cuda_kernel_algorithm=cuda_kernel_algorithm,
            kernel_backend=kernel_backend,
        )
        output_expected = gemm_torch(
            a=a, b=b, c=c, alpha=alpha, beta=beta, is_a_transposed=is_a_transposed, is_b_transposed=is_b_transposed
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
