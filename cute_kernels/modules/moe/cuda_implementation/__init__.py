# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ....math import ceil_divide
from ....ops import grouped_gemm_cute
from ....utils import ensure_contiguous
from .group_kernel import group_with_padding_triton
from .padded_expert_frequency_kernel import padded_expert_frequency_triton
from .ungroup_kernel import ungroup_with_padding_triton


class _GroupedGemmExperts_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx, x: torch.Tensor, weight: torch.Tensor, M_array: torch.Tensor, N_array: torch.Tensor, K_array: torch.Tensor
    ) -> torch.Tensor:
        # x -> sum(M) x K
        # weight -> EN x K
        _, N, K = weight.size()

        assert N % 8 == 0
        assert K % 8 == 0

        output = grouped_gemm_cute(
            A=x,
            B=weight,
            C=None,
            M_array=M_array,
            N_array=N_array,
            K_array=K_array,
            output_shape=(x.size(0), N),
            is_A_transposed=False,
            is_B_transposed=True,
        )

        ctx.save_for_backward(x, weight, M_array, K_array, N_array)

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        # x -> sum(M) x K
        # weight -> EN x K
        # output_grad -> sum(M) x N
        x, weight, M_array, K_array, N_array = ctx.saved_tensors

        # A -> sum(M) x N
        # B -> EN x K
        x_grad = grouped_gemm_cute(
            A=output_grad,
            B=weight,
            C=None,
            M_array=M_array,
            N_array=K_array,
            K_array=N_array,
            output_shape=x.size(),
            is_A_transposed=False,
            is_B_transposed=False,
        )

        # A -> sum(M) x N
        # B -> sum(M) x K
        weight_grad = grouped_gemm_cute(
            A=output_grad,
            B=x,
            C=None,
            M_array=N_array,
            N_array=K_array,
            K_array=M_array,
            output_shape=weight.size(),
            is_A_transposed=True,
            is_B_transposed=False,
        )

        return x_grad, weight_grad, *[None] * 3


@torch.no_grad()
def _get_expert_padding_offset(
    expert_frequency: torch.Tensor, E: int, pad_to_multiple_of: int
) -> tuple[torch.Tensor, torch.Tensor]:
    expert_padding_frequency = torch.empty_like(expert_frequency)

    padded_expert_frequency_triton(
        expert_frequency=expert_frequency, output=expert_padding_frequency, pad_to_multiple_of=pad_to_multiple_of
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

    return padded_expert_frequency, expert_padding_offset


class _GroupWithPadding(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        expert_frequency: torch.Tensor,
        sorted_idxs: torch.Tensor,
        scattered_idxs: torch.Tensor,
        top_k: int,
        pad_to_multiple_of: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert x.dim() == 2

        T, H = x.size()
        E = expert_frequency.size(0)
        K = top_k

        assert H % 8 == 0

        if pad_to_multiple_of == 1:
            output = torch.empty(T * K, H, device=x.device, dtype=x.dtype)
            padded_expert_frequency = expert_frequency
            expert_padding_offset = None
        else:
            # we pad to max possible shape to make tensor shape independent of data
            output = torch.zeros(
                (ceil_divide(T * K, pad_to_multiple_of) + E) * pad_to_multiple_of, H, device=x.device, dtype=x.dtype
            )

            padded_expert_frequency, expert_padding_offset = _get_expert_padding_offset(
                expert_frequency=expert_frequency, E=E, pad_to_multiple_of=pad_to_multiple_of
            )

        group_with_padding_triton(
            x=x,
            expert_padding_offset=expert_padding_offset,
            sorted_idxs=sorted_idxs,
            scattered_idxs=scattered_idxs,
            output=output,
            T=T,
            H=H,
            K=K,
            NEEDS_DUPLICATION=True,
        )

        ctx.save_for_backward(expert_padding_offset, sorted_idxs, scattered_idxs)
        ctx.T = T
        ctx.K = K

        return output, padded_expert_frequency, expert_padding_offset

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor, _: torch.Tensor, __: torch.Tensor) -> tuple[torch.Tensor | None]:
        expert_padding_offset, sorted_idxs, scattered_idxs = ctx.saved_tensors
        T = ctx.T
        H = output_grad.size(-1)
        K = ctx.K

        x_grad = torch.zeros(T, H, device=output_grad.device, dtype=torch.float32)

        ungroup_with_padding_triton(
            x=output_grad,
            expert_padding_offset=expert_padding_offset,
            sorted_idxs=sorted_idxs,
            scattered_idxs=scattered_idxs,
            output=x_grad,
            T=T,
            H=H,
            K=K,
            ATOMIC_ADD=True,
        )

        x_grad = x_grad.type_as(output_grad)

        return x_grad, *[None] * 5


class _UngroupWithPadding(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        expert_padding_offset: torch.Tensor,
        sorted_idxs: torch.Tensor,
        scattered_idxs: torch.Tensor,
        top_k: int,
        num_tokens: int,
        pad_to_multiple_of: int,
    ) -> torch.Tensor:
        assert x.dim() == 2

        T = num_tokens
        H = x.size(-1)
        K = top_k

        assert H % 8 == 0
        output = torch.empty(T, K, H, device=x.device, dtype=x.dtype)

        ungroup_with_padding_triton(
            x=x,
            expert_padding_offset=expert_padding_offset,
            sorted_idxs=sorted_idxs,
            scattered_idxs=scattered_idxs,
            output=output,
            T=T,
            H=H,
            K=K,
            ATOMIC_ADD=False,
        )

        ctx.save_for_backward(expert_padding_offset, sorted_idxs, scattered_idxs)
        ctx.x_shape = x.size()
        ctx.pad_to_multiple_of = pad_to_multiple_of

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        expert_padding_offset, sorted_idxs, scattered_idxs = ctx.saved_tensors
        pad_to_multiple_of = ctx.pad_to_multiple_of
        T, K, H = output_grad.size()

        x_grad = (torch.empty if pad_to_multiple_of == 1 else torch.zeros)(
            *ctx.x_shape, device=output_grad.device, dtype=output_grad.dtype
        )

        group_with_padding_triton(
            x=output_grad,
            expert_padding_offset=expert_padding_offset,
            sorted_idxs=sorted_idxs,
            scattered_idxs=scattered_idxs,
            output=x_grad,
            T=T,
            H=H,
            K=K,
            NEEDS_DUPLICATION=False,
        )

        return x_grad, *[None] * 6


def grouped_gemm_experts_cute(
    x: torch.Tensor, weight: torch.Tensor, M_array: torch.Tensor, N_array: torch.Tensor, K_array: torch.Tensor
) -> torch.Tensor:
    return _GroupedGemmExperts_Cute.apply(x, weight, M_array, N_array, K_array)


def group_with_padding(
    x: torch.Tensor,
    expert_frequency: torch.Tensor,
    sorted_idxs: torch.Tensor,
    scattered_idxs: torch.Tensor,
    top_k: int,
    pad_to_multiple_of: int = 1,
) -> torch.Tensor:
    return _GroupWithPadding.apply(x, expert_frequency, sorted_idxs, scattered_idxs, top_k, pad_to_multiple_of)


def ungroup_with_padding(
    x: torch.Tensor,
    expert_padding_offset: torch.Tensor,
    sorted_idxs: torch.Tensor,
    scattered_idxs: torch.Tensor,
    top_k: int,
    num_tokens: int,
    pad_to_multiple_of: int,
) -> torch.Tensor:
    return _UngroupWithPadding.apply(
        x, expert_padding_offset, sorted_idxs, scattered_idxs, top_k, num_tokens, pad_to_multiple_of
    )
