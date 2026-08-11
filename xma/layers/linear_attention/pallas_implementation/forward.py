# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....layers_jax.linear_attention.pallas_implementation import _linear_attention_forward_core as _forward_core_jax


def _output_shape_dtype_fn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h0: torch.Tensor | None, output_state: bool
) -> list[tuple[tuple[int, ...], torch.dtype] | None]:
    B, _, S, K = q.shape
    V = v.size(-1)
    N = max(q.size(1), k.size(1), v.size(1))

    return [((B, N, S, V), q.dtype), ((B, N, K, V), torch.float32) if output_state else None]


_CACHE = {}


def _linear_attention_forward_pallas(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int = 128,
    BLOCK_SIZE_V: int = 128,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    cache_key = (h0 is None, output_state)

    if cache_key not in _CACHE:
        from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

        _CACHE[cache_key] = make_kernel_from_pallas(
            _forward_core_jax, lambda q, k, v, h0: _output_shape_dtype_fn(q, k, v, h0, output_state)
        )

    y, ht = _CACHE[cache_key](
        q,
        k,
        v,
        h0,
        static_argnames=("attention_multiplier", "output_state", "BLOCK_SIZE_S", "BLOCK_SIZE_V"),
        attention_multiplier=attention_multiplier,
        output_state=output_state,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )

    y = y.transpose(1, 2)

    return y, ht
