import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME, MAX_TRITON_BLOCK_SIZE
from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ....math import ceil_divide, get_powers_of_2
from ....utils import cute_op, get_num_elements_and_hidden_size


_KERNEL_NAME = "softmax_forward_triton"


@triton.jit
def _exp_with_offset(x, logits_multiplier, offset):
    x += offset
    x = x.to(tl.float32)
    x *= logits_multiplier
    x = tl.exp(x)
    return x


@triton.jit
def _load_x(x_ptr, h, H, BLOCK_SIZE_H, indices_b, mask_b, other=None):
    indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask_h = indices_h < H

    indices = indices_b[:, None] * H + indices_h[None, :]
    mask_bh = mask_b[:, None] & mask_h[None, :]

    x_ptrs = x_ptr + indices
    x = tl.load(x_ptrs, mask=mask_bh, other=other)

    return x, indices, mask_bh


@triton.jit
def _softmax_forward_triton_kernel(
    x_ptr,
    output_ptr,
    logits_multiplier,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    indices_b = pid * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B

    Z = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    M = tl.full((BLOCK_SIZE_B, 1), -float("inf"), dtype=tl.float32)

    num_blocks_h = tl.cdiv(H, BLOCK_SIZE_H)

    for h in range(num_blocks_h):
        x, indices, mask_bh = _load_x(
            x_ptr=x_ptr, h=h, H=H, BLOCK_SIZE_H=BLOCK_SIZE_H, indices_b=indices_b, mask_b=mask_b, other=-float("inf")
        )

        prev_m = M
        m = tl.max(x, axis=1, keep_dims=True)
        M = max(M, m)

        x = _exp_with_offset(x, logits_multiplier, -M)
        Z = Z * tl.exp(prev_m - M) + tl.sum(x, axis=1, keep_dims=True)

    for h in range(num_blocks_h):
        x, indices, mask_bh = _load_x(
            x_ptr=x_ptr, h=h, H=H, BLOCK_SIZE_H=BLOCK_SIZE_H, indices_b=indices_b, mask_b=mask_b
        )

        x = _exp_with_offset(x, logits_multiplier, -M)
        x /= Z

        output_ptrs = output_ptr + indices
        tl.store(output_ptrs, x, mask=mask_bh)


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        BLOCK_SIZE_B=get_powers_of_2(1, MAX_TRITON_BLOCK_SIZE),
        BLOCK_SIZE_H=get_powers_of_2(1, MAX_TRITON_BLOCK_SIZE),
        condition=lambda **kwargs: 1024 <= kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_H"] <= 8192,
    ),
    default_config=CutoTuneConfig({"BLOCK_SIZE_B": 64, "BLOCK_SIZE_H": 64}),
    triggers={"x.dtype"},
)
@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def softmax_forward_triton(
    x: torch.Tensor, output: torch.Tensor, logits_multiplier: float, BLOCK_SIZE_B: int, BLOCK_SIZE_H: int
) -> None:
    num_elements, hidden_size = get_num_elements_and_hidden_size(x)

    with torch.device(x.device):
        _softmax_forward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE_B),)](
            x_ptr=x,
            output_ptr=output,
            logits_multiplier=logits_multiplier,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
