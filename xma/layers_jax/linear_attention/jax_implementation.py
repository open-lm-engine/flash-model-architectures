# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import jax
import jax.numpy as jnp


def _linear_attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    f: jax.Array | None,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
) -> tuple[jax.Array, jax.Array | None]:
    B, S, Nq, K = q.shape
    Nk = k.shape[-2]
    Nv, V = v.shape[-2:]

    Nf = 0
    if f is not None:
        if f.ndim == 3:
            f = f[..., None]

        Nf = f.shape[-2]
        f = f.astype(jnp.float32)

    N = max(Nq, Nk, Nv, Nf)
    dtype = q.dtype

    q = jnp.repeat(q, N // Nq, axis=-2).astype(jnp.float32)
    k = jnp.repeat(k, N // Nk, axis=-2).astype(jnp.float32)
    v = jnp.repeat(v, N // Nv, axis=-2).astype(jnp.float32)
    if f is not None:
        f = jnp.repeat(v, N // Nf, axis=-2).astype(jnp.float32)

    h = jnp.zeros((B, N, K, V), dtype=jnp.float32) if h0 is None else h0.astype(jnp.float32)

    y = []
    for s in range(S):
        if f is not None:
            h *= f[:, s, ..., None]

        h = h + k[:, s, ..., None] * v[:, s, :, None, :]
        y.append(jnp.einsum("bnk,bnkv->bnv", q[:, s], h))

    y = jnp.stack(y, axis=1) * attention_multiplier

    if not output_state:
        h = None

    return y.astype(dtype), h
