# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .....layers_jax.linear_attention.pallas_implementation.state_passing import (
    _state_passing_core as _state_passing_core_jax,
)
from .....math import ceil_divide


def _checkpoint_output_shape_dtype_fn(
    k: torch.Tensor, v: torch.Tensor, h0: torch.Tensor | None, N: int, BLOCK_SIZE_S: int, BLOCK_SIZE_V: int
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    B, _, S, K = k.shape
    V = v.shape[-1]
    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)

    return [((B, N * NUM_BLOCKS_S, K, V), torch.float32)]


_STATE_PASSING_CACHE = {}


def _state_passing_core(
    k: torch.Tensor, v: torch.Tensor, h0: torch.Tensor | None, N: int, BLOCK_SIZE_S: int, BLOCK_SIZE_V: int
) -> torch.Tensor:
    cache_key = h0 is None

    if cache_key not in _STATE_PASSING_CACHE:
        from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

        _STATE_PASSING_CACHE[cache_key] = make_kernel_from_pallas(
            _state_passing_core_jax, _checkpoint_output_shape_dtype_fn
        )

    h = _STATE_PASSING_CACHE[cache_key](k, v, h0, N, BLOCK_SIZE_S, BLOCK_SIZE_V, static_argnums=(3, 4, 5))

    return h
