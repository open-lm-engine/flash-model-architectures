# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import grouped_gemm_cute, set_seed

from ..test_commons import TestCommons


_SEED = 42


class GroupedGEMMTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [False, True],  # is_A_transposed
            [False, True],  # is_B_transposed
            [torch.device("cuda")],  # device
            [torch.bfloat16],  # dtype
            [grouped_gemm_cute],  # function
        )
    )
    def test_grouped_gemm(
        self,
        is_A_transposed: bool,
        is_B_transposed: bool,
        device: torch.device,
        dtype: torch.dtype,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        E = 16
        K = 4096
        M = 4096
        N = 512

        M_array = torch.tensor(
            [2048, 4096, 512, 2048, 4096, 512, 2048, 4096, 512, 2048, 4096, 512, 2048, 4096, 512, 32256],
            device=device,
            dtype=torch.uint32,
        )
        N_array = torch.full_like(M_array, fill_value=N)
        K_array = torch.full_like(M_array, fill_value=K)

        A = torch.randint(-8, 9, (E * K, M) if is_A_transposed else (E * M, K), device=device, dtype=dtype)
        B = torch.randint(-8, 9, (E * N, K) if is_B_transposed else (E * K, N), device=device, dtype=dtype)

        output = function(
            A=A,
            B=B,
            C=None,
            M_array=M_array,
            N_array=N_array,
            K_array=K_array,
            output_shape=(E * M, N),
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
        )

        M_offsets = M_array.cumsum(dim=-1)

        D = []
        for i in range(E):
            if i == 0:
                start = 0
            else:
                start = M_offsets[i - 1]
            end = M_offsets[i]

            if is_A_transposed:
                a = A.view(E, K, M)[i]
                a = a.T
            else:
                a = A[start:end]

            if is_B_transposed:
                b = B.view(E, N, K)[i]
                b = b.T
            else:
                b = B.view(E, K, N)[i]

            D.append(a @ b)

        D = torch.cat(D)

        assert (output - D).abs().max() == 0
