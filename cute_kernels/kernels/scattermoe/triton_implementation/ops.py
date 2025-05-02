import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op
from .kernels import group_triton_kernel, groupXtY_triton_kernel, scatter2scatter_triton_kernel


def _fake_bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return torch.empty(minlength, device=x.device, dtype=torch.int)


@cute_op(f"{LIBRARY_NAME}::bincount", mutates_args={}, fake_func=_fake_bincount)
def bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return x.bincount(minlength=minlength).to(torch.int)


def expert_boundaries(sorted_experts_idxs: torch.Tensor, k: int) -> torch.Tensor:
    # there is an overhead of launching a custom op so we only use the custom op when compiling
    expert_counts = bincount(sorted_experts_idxs, k)
    expert_boundaries_end = expert_counts.cumsum(-1)
    return expert_boundaries_end


@cute_op(f"{LIBRARY_NAME}::scatter2scatter", mutates_args={"out"})
def scatter2scatter(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    out: torch.Tensor,
    FAN_OUT: int,
    x_grouped: bool = False,
    y_grouped: bool = False,
) -> None:
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * FAN_OUT
    assert out.size(0) == sorted_expert_idxs.size(0)
    assert out.size(1) == W.size(-1)

    grid = lambda meta: (
        ceil_divide(sorted_expert_idxs.size(0), meta["BLOCK_M"]) * ceil_divide(meta["N"], meta["BLOCK_N"]),
    )

    BLOCK_M = 128

    with torch.device(X.device):
        scatter2scatter_triton_kernel[grid](
            # X_ptr, stride_xm, stride_xk,
            X,
            X.stride(0),
            X.stride(1),
            # W_ptr, stride_we, stride_wk, stride_wn,
            W,
            W.stride(0),
            W.stride(1),
            W.stride(2),
            # Y_ptr, stride_ym, stride_yn,
            out,
            out.stride(0),
            out.stride(1),
            grouped_idx_ptr=sorted_scattered_idxs,
            expert_idxs_ptr=sorted_expert_idxs,
            FAN_OUT=FAN_OUT,
            M=X.size(0),
            K=X.size(1),
            N=out.size(1),
            E=W.size(0),
            BLOCK_M=BLOCK_M,
            ACC_TYPE=tl.float32,
            allow_tf32=torch.backends.cudnn.allow_tf32,
            x_grouped=x_grouped,
            y_grouped=y_grouped,
        )


def group_bwd_W(DY: torch.Tensor, X: torch.Tensor, expert_offsets: torch.Tensor, DW: torch.Tensor, E: int) -> None:
    grid = lambda meta: (E * ceil_divide(meta["K"], meta["BLOCK_K"]), ceil_divide(meta["N"], meta["BLOCK_N"]))

    with torch.cuda.device(X.device):
        groupXtY_triton_kernel[grid](
            # DY_ptr, stride_dym, stride_dyk,
            DY,
            DY.stride(0),
            DY.stride(1),
            # X_ptr, stride_xm, stride_xn,
            X,
            X.stride(0),
            X.stride(1),
            # DW_ptr, stride_dwe, stride_dwk, stride_dwn,
            DW,
            DW.stride(0),
            DW.stride(1),
            DW.stride(2),
            # expert_offsets_ptr,
            expert_offsets,
            # K: tl.constexpr, N: tl.constexpr,
            N=DY.size(-1),
            K=X.size(-1),
            # ACC_TYPE: tl.constexpr,
            ACC_TYPE=tl.float32,
            allow_tf32=torch.backends.cudnn.allow_tf32,
        )


def group(
    A: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    out: torch.Tensor,
    coeff: torch.Tensor | None = None,
    fan_out: int = 1,
) -> None:
    N = sorted_expert_idxs.size(0)
    K = A.size(1)
    assert A.size(0) * fan_out == N

    grid = lambda meta: (triton.cdiv(meta["N"], meta["BLOCK_N"]),)

    with torch.cuda.device(A.device):
        group_triton_kernel[grid](
            # A_ptr, stride_an, stride_ai,
            A,
            A.stride(0),
            A.stride(1),
            coeff is not None,
            coeff,
            fan_out,
            # Y_ptr, stride_yn, stride_yk,
            out,
            out.stride(0),
            out.stride(1),
            # grouped_idx_ptr,
            sorted_expert_idxs,
            # N: tl.constexpr, K: tl.constexpr,
            N,
            K,
        )


class _ScatteredExperts(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        gates=None,
        grouped_in=False,
        grouped_out=False,
    ):
        output = torch.empty(sorted_expert_idxs.size(0), expert_weights.size(-1), device=x.device, dtype=x.dtype)

        scatter2scatter(
            X=x,
            W=expert_weights,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            out=output,
            FAN_OUT=k,
            x_grouped=grouped_in,
            y_grouped=grouped_out,
        )

        if gates is None:
            output_expanded = None
        else:
            output_expanded = output.view(gates.size(0), gates.size(1), output.size(-1))
            output = torch.bmm(gates.unsqueeze(1), output_expanded).squeeze(1)

        ctx.save_for_backward(
            x,
            expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            gates,
            output_expanded,
        )

        ctx.grouped_in = grouped_in
        ctx.grouped_out = grouped_out
        ctx.k = k

        return output

    @staticmethod
    def backward(ctx, grad_out):
        (
            x,
            expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            gates,
            output_expanded,
        ) = ctx.saved_tensors
        k = ctx.k
        grouped_in = ctx.grouped_in
        grouped_out = ctx.grouped_out

        if gates is None:
            d_gates = None
            gates_flat = None
            gate_fan = 1
            grouped_grad_out = None
        else:
            # calculate gates gradient
            d_gates = torch.bmm(output_expanded, grad_out.unsqueeze(2)).squeeze(-1)
            gates_flat = gates.flatten()
            gate_fan = gates.size(1)
            # print("expanded and grouping")
            grouped_grad_out = output_expanded.flatten(0, 1)  # reuse expanded buffer later

        if grouped_out:
            grouped_grad_out = grad_out
        else:
            group(
                A=grad_out,
                sorted_expert_idxs=sorted_scattered_idxs,
                out=grouped_grad_out,
                coeff=gates_flat,
                fan_out=gate_fan,
            )

        if grouped_in:
            grouped_x = x
            d_expanded_input = torch.empty(
                sorted_expert_idxs.size(0), expert_weights.size(1), device=x.device, dtype=x.dtype
            )
        else:
            grouped_x = torch.empty(sorted_scattered_idxs.size(0), x.size(1), dtype=x.dtype, device=x.device)
            group(
                A=x,
                sorted_expert_idxs=sorted_scattered_idxs,
                out=grouped_x,
                fan_out=k,
            )

            d_expanded_input = grouped_x

        d_weights = torch.zeros_like(expert_weights)

        group_bwd_W(
            DY=grouped_grad_out,
            X=grouped_x,
            expert_offsets=expert_offsets,
            DW=d_weights,
            E=expert_weights.size(0),
        )

        scatter2scatter(
            X=grouped_grad_out,
            W=expert_weights.permute(0, 2, 1),
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            out=d_expanded_input,
            FAN_OUT=1,
            x_grouped=True,
            y_grouped=grouped_in,
        )

        if k == 1:
            d_input = d_expanded_input
        else:
            d_input = d_expanded_input.view(x.size(0), k, d_expanded_input.size(-1)).sum(-2)

        return (
            # x, expert_weights, k,
            d_input,
            d_weights,
            None,
            # sorted_expert_idxs, sorted_scattered_idxs,
            None,
            None,
            # expert_offsets,
            None,
            # gates
            d_gates,
            None,
            None,
        )


def scattered_experts(
    inputs,
    expert_weights,
    k,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    expert_offsets,
    gates=None,
    grouped_in=False,
    grouped_out=False,
):
    return _ScatteredExperts.apply(
        inputs,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        gates,
        grouped_in,
        grouped_out,
    )
