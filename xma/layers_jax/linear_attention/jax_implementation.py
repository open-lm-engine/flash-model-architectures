# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import jax
import jax.numpy as jnp


def _linear_attention_reference(
    q: jax.Array, k: jax.Array, v: jax.Array, h0: jax.Array | None, attention_multiplier: float
) -> tuple[jax.Array, jax.Array]:
    B, S, Nq, K = q.shape
    Nk = k.shape[-2]
    Nv, V = v.shape[-2:]
    N = max(Nq, Nk, Nv)
    dtype = q.dtype

    q = jnp.repeat(q, N // Nq, axis=-2).astype(jnp.float32)
    k = jnp.repeat(k, N // Nk, axis=-2).astype(jnp.float32)
    v = jnp.repeat(v, N // Nv, axis=-2).astype(jnp.float32)

    h = jnp.zeros((B, N, K, V), dtype=jnp.float32) if h0 is None else h0.astype(jnp.float32)

    y = []
    for s in range(S):
        y.append(jnp.einsum("bnk,bnkv->bnv", q[:, s], h))
        h = h + k[:, s][..., :, None] * v[:, s][..., None, :]

    y = jnp.stack(y, axis=1) * attention_multiplier

    return y.astype(dtype), h
