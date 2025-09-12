import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_powers_of_2
from ....utils import cute_op


def _get_autotune_configs() -> list[triton.Config]:
    """Get configurations for autotuning the RoPE kernel."""
    configs = []
    for BLOCK_SIZE_H in get_powers_of_2(16, 128):
        for num_warps in get_powers_of_2(2, 8):
            configs.append(triton.Config({"BLOCK_SIZE_H": BLOCK_SIZE_H}, num_warps=num_warps))
    return configs


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["STRIDE_B", "HEAD_DIM", "ROPE_DIM", "x_dtype"],
)
@triton.jit
def rope_forward_triton_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    STRIDE_B: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    x_dtype: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    offsets_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    ROPE_OFFSET = tl.constexpr(HEAD_DIM - ROPE_DIM)
    offsets_rope = offsets_h - ROPE_OFFSET
    use_rope = offsets_rope >= 0

    cos = tl.load(cos_ptr + tl.where(use_rope, offsets_rope, 0), mask=use_rope, other=1.0)
    sin = tl.load(sin_ptr + tl.where(use_rope, offsets_rope, 0), mask=use_rope, other=0.0)

    offsets_bh = pid_b * STRIDE_B + offsets_h
    mask_h = offsets_h < HEAD_DIM

    x = tl.load(x_ptr + offsets_bh, mask=mask_h)

    HALF_ROPE_DIM = tl.constexpr(ROPE_DIM // 2)
    is_first_half = offsets_rope < HALF_ROPE_DIM

    rope_partner = tl.load(x_ptr + offsets_bh + tl.where(is_first_half, HALF_ROPE_DIM, -HALF_ROPE_DIM), mask=use_rope, other=0.0)
    x_rotated = x * cos + tl.where(is_first_half, -1.0, 1.0) * rope_partner * sin

    tl.store(output_ptr + offsets_bh, tl.where(use_rope, x_rotated, x), mask=mask_h)


@cute_op(f"{LIBRARY_NAME}::rope_forward_triton_kernel", mutates_args={"output"})
def rope_forward_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    output: torch.Tensor,
) -> None:
    HEAD_DIM = x.size(-1)
    ROPE_DIM = cos.size(-1)

    assert ROPE_DIM % 2 == 0, "ROPE_DIM must be even"
    assert ROPE_DIM <= HEAD_DIM, "ROPE_DIM must be less than or equal to HEAD_DIM"

    x_flat = x.contiguous().view(-1, HEAD_DIM)
    B = x_flat.size(0)

    GRID = lambda meta: (B, ceil_divide(HEAD_DIM, meta["BLOCK_SIZE_H"]))

    with torch.device(x.device):
        rope_forward_triton_kernel[GRID](
            x_ptr=x_flat,
            cos_ptr=cos,
            sin_ptr=sin,
            output_ptr=output,
            STRIDE_B=HEAD_DIM,
            HEAD_DIM=HEAD_DIM,
            ROPE_DIM=ROPE_DIM,
            x_dtype=x.dtype,
        )