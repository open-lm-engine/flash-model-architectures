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
            [grouped_gemm_cute, torch.compile(grouped_gemm_cute, fullgraph=True)],  # function
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
            [512, 3176, 2048, 512, 3176, 2048, 512, 3176, 2048, 512, 3176, 2048, 512, 3176, 2048, 36856],
            device=device,
            dtype=torch.uint32,
        )
        N_array = torch.full_like(M_array, fill_value=N)
        K_array = torch.full_like(M_array, fill_value=K)

        def get_tensors():
            As = []
            Bs = []

            for M, N, K in zip(M_array, N_array, K_array):
                A = torch.randint(-8, 9, (K, M) if is_A_transposed else (M, K), device=device, dtype=dtype)
                As.append(A)

                B = torch.randint(-8, 9, (N, K) if is_B_transposed else (K, N), device=device, dtype=dtype)
                Bs.append(B)

            return As, Bs

        As, Bs = get_tensors()

        A = torch.cat([i.view(-1) for i in As])
        B = torch.cat([i.view(-1) for i in Bs])

        output = function(
            A=A,
            B=B,
            C=None,
            M_array=M_array,
            N_array=N_array,
            K_array=K_array,
            output_shape=(E * M * N,),
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
        )

        D = []
        for A, B in zip(As, Bs):
            if is_A_transposed:
                A = A.T
            if is_B_transposed:
                B = B.T

            D.append(A @ B)

        D = torch.cat([i.view(-1) for i in D])

        assert (output - D).abs().max() == 0
