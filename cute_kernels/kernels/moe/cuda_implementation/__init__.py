# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ....math import ceil_divide
from ....utils import ensure_contiguous
from ...grouped_gemm import grouped_gemm_cute
from .group_kernel import group_with_padding_triton_kernel
from .padded_expert_frequency_kernel import padded_expert_frequency_triton_kernel
from .ungroup_kernel import ungroup_with_padding_triton_kernel


class _GroupedGemmExperts_Cute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, expert_frequency: int) -> torch.Tensor:
        # x -> sum(M) x K
        # weight -> EN x K
        _, N, K = weight.size()

        N_array = torch.full_like(expert_frequency, fill_value=N)
        K_array = torch.full_like(expert_frequency, fill_value=K)

        assert N % 8 == 0
        assert K % 8 == 0

        output = grouped_gemm_cute(
            A=x,
            B=weight,
            C=None,
            M_array=expert_frequency,
            N_array=N_array,
            K_array=K_array,
            output_shape=(x.size(0), N),
            is_A_transposed=False,
            is_B_transposed=True,
        )

        ctx.save_for_backward(x, weight, expert_frequency, K_array, N_array)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        # x -> sum(M) x K
        # weight -> EN x K
        # output_grad -> sum(M) x N
        x, weight, expert_frequency, K_array, N_array = ctx.saved_tensors

        # A -> sum(M) x N
        # B -> EN x K
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

        # A -> sum(M) x N
        # B -> sum(M) x K
        weight_grad = grouped_gemm_cute(A=output_grad, B=x, C=None, is_A_transposed=True, is_B_transposed=False)

        return x_grad, weight_grad, None


class _GroupWithPadding(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        expert_frequency: torch.Tensor,
        sorted_idxs: torch.Tensor,
        scattered_idxs: torch.Tensor,
        topk: int,
        pad_to_multiple_of: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert x.dim() == 2

        T, H = x.size()
        E = expert_frequency.size(0)
        K = topk

        assert H % 8 == 0

        if pad_to_multiple_of == 1:
            output = torch.empty(T * K, H, device=x.device, dtype=x.dtype)
            expert_padding_offset = None
            padded_expert_frequency = expert_frequency
        else:
            # we pad to max possible shape to make tensor shape independent of data
            output = torch.zeros(
                (ceil_divide(T * K, pad_to_multiple_of) + E) * pad_to_multiple_of, H, device=x.device, dtype=x.dtype
            )

            expert_padding_frequency = torch.empty_like(expert_frequency)

            with torch.cuda.device(expert_frequency.device):
                BLOCK_SIZE = 4096
                NUM_WARPS = 32

                padded_expert_frequency_triton_kernel[ceil_divide(E, BLOCK_SIZE),](
                    x_ptr=expert_frequency,
                    y_ptr=expert_padding_frequency,
                    pad_to_multiple_of=pad_to_multiple_of,
                    N=E,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=NUM_WARPS,
                )

            padded_expert_frequency = expert_frequency.to(torch.int32) + expert_padding_frequency.to(torch.int32)
            padded_expert_frequency = padded_expert_frequency.to(torch.uint32)

            expert_padding_offset = expert_padding_frequency.cumsum(-1)
            expert_padding_offset = torch.cat(
                [
                    torch.tensor([0], device=expert_padding_offset.device, dtype=expert_padding_offset.dtype),
                    expert_padding_offset,
                ]
            )

        with torch.cuda.device(x.device):
            BLOCK_SIZE_B = 1
            BLOCK_SIZE_H = 4096
            NUM_WARPS = 32

            group_with_padding_triton_kernel[ceil_divide(T * K, BLOCK_SIZE_B),](
                x_ptr=x,
                expert_padding_offset_ptr=expert_padding_offset,
                sorted_idxs_ptr=sorted_idxs,
                scattered_idxs_ptr=scattered_idxs,
                y_ptr=output,
                T=T,
                H=H,
                K=topk,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
                num_warps=NUM_WARPS,
            )

        return output, padded_expert_frequency, expert_padding_offset


class _UngroupWithPadding(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        expert_padding_offset: torch.Tensor,
        sorted_idxs: torch.Tensor,
        scattered_idxs: torch.Tensor,
        topk: int,
        num_tokens: int,
        pad_to_multiple_of: int,
    ) -> torch.Tensor:
        assert x.dim() == 2

        T = num_tokens
        H = x.size(-1)
        E = expert_padding_offset.size(0) - 1
        K = topk

        assert H % 8 == 0

        if pad_to_multiple_of == 1:
            output = torch.empty(T * K, H, device=x.device, dtype=x.dtype)
        else:
            # we pad to max possible shape to make tensor shape independent of data
            output = torch.zeros(
                (ceil_divide(T * K, pad_to_multiple_of) + E) * pad_to_multiple_of, H, device=x.device, dtype=x.dtype
            )

        with torch.cuda.device(x.device):
            BLOCK_SIZE_B = 1
            BLOCK_SIZE_H = 4096
            NUM_WARPS = 32

            ungroup_with_padding_triton_kernel[ceil_divide(T * K, BLOCK_SIZE_B),](
                x_ptr=x,
                expert_padding_offset_ptr=expert_padding_offset,
                sorted_idxs_ptr=sorted_idxs,
                scattered_idxs_ptr=scattered_idxs,
                y_ptr=output,
                T=T,
                H=H,
                K=topk,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
                num_warps=NUM_WARPS,
            )

        return output


def grouped_gemm_experts_cute(x: torch.Tensor, weight: torch.Tensor, expert_frequency: torch.Tensor) -> torch.Tensor:
    return _GroupedGemmExperts_Cute.apply(x, weight, expert_frequency)


def group_with_padding(
    x: torch.Tensor,
    expert_frequency: torch.Tensor,
    sorted_idxs: torch.Tensor,
    scattered_idxs: torch.Tensor,
    topk: int,
    pad_to_multiple_of: int = 1,
) -> torch.Tensor:
    return _GroupWithPadding.apply(x, expert_frequency, sorted_idxs, scattered_idxs, topk, pad_to_multiple_of)


def ungroup_with_padding(
    x: torch.Tensor,
    expert_padding_offset: torch.Tensor,
    sorted_idxs: torch.Tensor,
    scattered_idxs: torch.Tensor,
    topk: int,
    num_tokens: int,
    pad_to_multiple_of: int = 1,
) -> torch.Tensor:
    return _UngroupWithPadding.apply(
        x, expert_padding_offset, sorted_idxs, scattered_idxs, topk, num_tokens, pad_to_multiple_of
    )
