# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _linear_attention_backward_kernel(
    q_ref,
    k_ref,
    v_ref,
    h_ref,
    dy_ref,
    dht_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    dh0_ref,
    dh_scratch,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    N: int,
    Gq: int,
    Gk: int,
    Gv: int,
    NUM_BLOCKS_V: int,
    NUM_BLOCKS_S: int,
) -> None:
    S_CELLS_VISITED = pl.program_id(1)

    @pl.when(S_CELLS_VISITED == 0)
    def _():
        if dht_ref is None:
            dh_scratch[...] = jnp.zeros_like(dh_scratch)
        else:
            dh_scratch[...] = dht_ref[...].astype(jnp.float32)

    dtype = q_ref.dtype

    # one register-level transpose per cell: (S, N, .) -> (N, S, .)
    q = q_ref[...].transpose(1, 0, 2)
    k = k_ref[...].transpose(1, 0, 2)
    v = v_ref[...].transpose(1, 0, 2)
    dy = dy_ref[...].transpose(1, 0, 2)

    hc = h_ref[...].astype(dtype)  # (N, K, NUM_BLOCKS_V * BLOCK_SIZE_V)

    row = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 1)
    causal_mask = row >= col

    K = q.shape[-1]
    for n in range(N):
        q_n = q[n // Gq].astype(dtype)
        k_n = k[n // Gk].astype(dtype)
        dy_n_full = (dy[n].astype(jnp.float32) * attention_multiplier).astype(dtype)

        qk = jax.lax.dot_general(q_n, k_n, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
        qk = jnp.where(causal_mask, qk, 0).astype(dtype)

        # static loop over V-tiles, unrolled at trace time; single tile when V <= BLOCK_SIZE_V
        dqk = jnp.zeros((BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32)
        dq_term2 = jnp.zeros((BLOCK_SIZE_S, K), jnp.float32)
        dk_term2 = jnp.zeros((BLOCK_SIZE_S, K), jnp.float32)
        for vb in range(NUM_BLOCKS_V):
            slab = vb * BLOCK_SIZE_V
            v_n = v[n // Gv][:, slab : slab + BLOCK_SIZE_V].astype(dtype)
            dy_n = dy_n_full[:, slab : slab + BLOCK_SIZE_V]
            hc_n = hc[n][:, slab : slab + BLOCK_SIZE_V]  # (K, BLOCK_SIZE_V)
            g = dh_scratch[n][:, slab : slab + BLOCK_SIZE_V]  # (K, BLOCK_SIZE_V) f32

            dv_n = jax.lax.dot_general(qk, dy_n, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
            # precision asymmetry (deliberate): dv mirrors the forward pass, so g is
            # downcast to the input dtype here to keep this contraction fully on the bf16
            # MXU path; dk_term2 below keeps g in f32 (mixed bf16 x f32 dot_general promotes
            # to the wider type, result f32), since that term feeds dk directly and gradient
            # precision dominates there.
            dv_n += jax.lax.dot_general(
                k_n, g.astype(dtype), (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )
            dv_ref[:, n, slab : slab + BLOCK_SIZE_V] = dv_n.astype(dtype)

            dh_scratch[n, :, slab : slab + BLOCK_SIZE_V] = g + jax.lax.dot_general(
                q_n, dy_n, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )

            dqk += jax.lax.dot_general(dy_n, v_n, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            dq_term2 += jax.lax.dot_general(dy_n, hc_n, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            dk_term2 += jax.lax.dot_general(v_n, g, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

        dqk = jnp.where(causal_mask, dqk, 0).astype(dtype)

        dq_n = jax.lax.dot_general(dqk, k_n, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        dq_n += dq_term2
        dq_ref[:, n, :] = dq_n.astype(dtype)

        dk_n = jax.lax.dot_general(dqk, q_n, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        dk_n += dk_term2
        dk_ref[:, n, :] = dk_n.astype(dtype)

    # last visited block (front of sequence): publish dh0
    @pl.when(S_CELLS_VISITED == NUM_BLOCKS_S - 1)
    def _():
        dh0_ref[...] = dh_scratch[...]


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_backward_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h: jax.Array,
    dy: jax.Array,
    dht: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    B, S, Nq, K = q.shape
    Nk = k.shape[2]
    Nv, V = v.shape[-2:]
    N = dy.shape[2]

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    assert S % BLOCK_SIZE_S == 0

    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    NUM_BLOCKS_V = ceil_divide(V, BLOCK_SIZE_V)
    V_WIDTH = NUM_BLOCKS_V * BLOCK_SIZE_V
    assert V == V_WIDTH, "V must be an integer multiple of BLOCK_SIZE_V (host padding guarantees this)"

    kernel = pl.pallas_call(
        partial(
            _linear_attention_backward_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
            N=N,
            Gq=Gq,
            Gk=Gk,
            Gv=Gv,
            NUM_BLOCKS_V=NUM_BLOCKS_V,
            NUM_BLOCKS_S=NUM_BLOCKS_S,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, S, N, K), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, S, N, K), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, S, N, V), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nq, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nk, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nv, V_WIDTH),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, N, K, V_WIDTH),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, V_WIDTH),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            (
                None
                if dht is None
                else pl.BlockSpec(
                    block_shape=(None, N, K, V_WIDTH),
                    index_map=lambda B, S: (B, 0, 0, 0),
                )
            ),
        ),
        out_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, V_WIDTH),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, N, K, V_WIDTH),
                index_map=lambda B, S: (B, 0, 0, 0),
            ),
        ),
        scratch_shapes=[pltpu.VMEM((N, K, V_WIDTH), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, dimension_semantics=("parallel", "arbitrary")
        ),
    )

    return kernel(q, k, v, h, dy, dht)
