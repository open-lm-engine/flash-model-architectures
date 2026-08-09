# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Callable

import torch

from ....layers_jax.depthwise_causal_convolution.pallas_implementation import (
    _state_passing_core as _state_passing_core_jax,
)
from ....math import ceil_divide


def _make_state_passing_output_shape_dtype_fn(BLOCK_SIZE_S: int, K: int) -> Callable:
    # make_kernel_from_pallas calls this with only the non-static tensor args (x, h0), so BLOCK_SIZE_S/K
    # (needed for the output shape) must be captured via closure instead of taken as parameters.
    def _output_shape_dtype_fn(x: torch.Tensor, h0: torch.Tensor | None) -> list[tuple[tuple[int, ...], torch.dtype]]:
        B, S, H = x.shape
        NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
        PAD = ceil_divide(K - 1, 8) * 8

        return [((B, NUM_BLOCKS_S, PAD, H), torch.float32)]

    return _output_shape_dtype_fn


def _state_passing_core(x: torch.Tensor, h0: torch.Tensor | None, BLOCK_SIZE_S: int, K: int) -> torch.Tensor:
    if not hasattr(_state_passing_core, "cache"):
        _state_passing_core.cache = {}

    cache_key = (h0 is None, BLOCK_SIZE_S, K)
    kernel = _state_passing_core.cache.get(cache_key)

    if kernel is None:
        from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

        kernel = make_kernel_from_pallas(
            _state_passing_core_jax, _make_state_passing_output_shape_dtype_fn(BLOCK_SIZE_S, K)
        )

        _state_passing_core.cache[cache_key] = kernel

    return kernel(x, h0, static_argnames=("BLOCK_SIZE_S", "K"), BLOCK_SIZE_S=BLOCK_SIZE_S, K=K)
