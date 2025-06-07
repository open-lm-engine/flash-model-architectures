# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...grouped_gemm import grouped_gemm_cute


class _GroupedGemmExperts_Cute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, expert_frequency: int) -> torch.Tensor:
        # x -> sum(M), K
        # weight -> E, N, K
        _, N, K = weight.size()

        N_array = torch.full_like(expert_frequency, fill_value=N)
        K_array = torch.full_like(expert_frequency, fill_value=K)

        output = grouped_gemm_cute(
            A=x, B=weight, C=None, M_array=expert_frequency, N_array=N_array, K_array=K_array, is_B_transposed=True
        )

        ctx.save_for_backward(x, weight, expert_frequency, K_array, N_array)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        # x -> sum(M), K
        # weight -> E, N, K
        # output_grad -> sum(M), N
        x, weight, expert_frequency, K_array, N_array = ctx.saved_tensors

        # A -> sum(M), N
        # B -> E, N, K
        x_grad = grouped_gemm_cute(
            A=output_grad,
            B=weight,
            C=None,
            M_array=expert_frequency,
            N_array=K_array,
            K_array=N_array,
            is_A_transposed=False,
            is_B_transposed=False,
        )

        # A -> sum(M), N
        # B -> sum(M), K
        weight_grad = grouped_gemm_cute(A=output_grad, B=x, C=None, is_A_transposed=True, is_B_transposed=False)

        return x_grad, weight_grad, None


def grouped_gemm_experts_cute(x: torch.Tensor, weight: torch.Tensor, expert_frequency: torch.Tensor) -> torch.Tensor:
    return _GroupedGemmExperts_Cute(x, weight, expert_frequency)
