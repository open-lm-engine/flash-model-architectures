from typing import Callable

import torch
from parameterized import parameterized
from transformers import set_seed

from cute_kernels import KernelBackend, gemm_cute, gemm_torch

from ..test_commons import TestCommons


_SEED = 42


class GEMMTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [False, True],  # is_a_transposed
            [False, True],  # is_b_transposed
            [KernelBackend.cuda, KernelBackend.triton],  # kernel_backend
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [gemm_cute, torch.compile(gemm_cute, fullgraph=True)],  # function
        )
    )
    def test_linear(
        self,
        size: tuple[int],
        is_a_transposed: bool,
        is_b_transposed: bool,
        kernel_backend: KernelBackend,
        device: torch.device,
        dtype: torch.dtype,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        a_kernel, a_expected = self.get_random_duplicated_tensors(
            (size[0], 400) if is_a_transposed else (400, size[0]), device=device, dtype=dtype, std=0.02
        )
        b_kernel, b_expected = self.get_random_duplicated_tensors(
            (size[1], size[0]) if is_b_transposed else size, device=device, dtype=dtype, std=0.02
        )

        c_kernel = function(
            a=a_kernel,
            b=b_kernel,
            is_a_transposed=is_a_transposed,
            is_b_transposed=is_b_transposed,
            kernel_backend=kernel_backend,
        )
        c_expected = gemm_torch(
            a=a_expected, b=b_expected, is_a_transposed=is_a_transposed, is_b_transposed=is_b_transposed
        )

        self.assert_equal_tensors(
            c_kernel,
            c_expected,
            False,
            atol_float32=4e-3,
            rtol_float32=1e-4,
            atol_float16=1e-4,
            rtol_float16=5e-3,
            atol_bfloat16=2e-3,
            rtol_bfloat16=7e-3,
        )
