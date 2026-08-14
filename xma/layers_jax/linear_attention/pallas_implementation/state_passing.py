# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""Per-block state checkpoints for the backward kernel.

Recomputes the linear-attention state at every S-tile boundary:
`h[cs] = state at the entry of S-tile cs`. Emitted in a flat
(B, NUM_BLOCKS_S * N, K, V) float32 layout — cell cs occupies rows
[cs * N, (cs + 1) * N) — so the backward kernel fetches ALL heads of one
cell in a single BlockSpec fetch; any layout that mixes batches into those
rows would cost one DMA round-trip per head instead.

All V-tiles are visited inside each cell (a static python loop unrolled at
trace time), so the grid is purely (B, NUM_BLOCKS_S) and the running state
in the VMEM scratch ref chains along the S axis only — the same ordering
guarantee the forward and backward kernels rely on.

The checkpoint pass depends only on k/v/h0 — never on dy or dht — so the
XLA scheduler is free to place its memory traffic anywhere ahead of the
backward kernel that consumes it.
"""

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _state_passing_kernel(
    k_ref, v_ref, h0_ref, h_ref, h_scratch, *, N: int, Gk: int, Gv: int, NUM_V_TILES: int, BLOCK_SIZE_V: int
) -> None:
    @pl.when(pl.program_id(1) == 0)
    def _():
        if h0_ref is None:
            h_scratch[...] = jnp.zeros_like(h_scratch)
        else:
            h_scratch[...] = h0_ref[...].astype(jnp.float32)

    dtype = k_ref.dtype

    # one register-level transpose per cell: (S, N, K) -> (N, S, K)
    k = k_ref[...].transpose(1, 0, 2)
    v = v_ref[...].transpose(1, 0, 2)

    for n in range(N):
        kn = k[n // Gk].astype(dtype)
        for vb in range(NUM_V_TILES):
            slab = vb * BLOCK_SIZE_V
            vn = v[n // Gv][:, slab : slab + BLOCK_SIZE_V].astype(dtype)
            h_ref[n, :, slab : slab + BLOCK_SIZE_V] = h_scratch[n][:, slab : slab + BLOCK_SIZE_V]
            h_scratch[n, :, slab : slab + BLOCK_SIZE_V] = h_scratch[n][
                :, slab : slab + BLOCK_SIZE_V
            ] + jax.lax.dot_general(kn, vn, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)


@partial(jax.jit, static_argnames=("N", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _state_passing_core(
    k: jax.Array, v: jax.Array, h0: jax.Array | None, N: int, BLOCK_SIZE_S: int, BLOCK_SIZE_V: int
) -> jax.Array:
    B, S, Nk, K = k.shape
    Nv = v.shape[2]
    V = v.shape[-1]

    Gk = N // Nk
    Gv = N // Nv

    assert S % BLOCK_SIZE_S == 0

    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    NUM_V_TILES = ceil_divide(V, BLOCK_SIZE_V)
    V_WIDTH = NUM_V_TILES * BLOCK_SIZE_V
    assert V == V_WIDTH, "V must be an integer multiple of BLOCK_SIZE_V (host padding guarantees this)"

    kernel = pl.pallas_call(
        partial(_state_passing_kernel, N=N, Gk=Gk, Gv=Gv, NUM_V_TILES=NUM_V_TILES, BLOCK_SIZE_V=BLOCK_SIZE_V),
        out_shape=jax.ShapeDtypeStruct(
            shape=(B, NUM_BLOCKS_S * N, K, V), dtype=jnp.float32
        ),  # flat: cell s at rows s*N..s*N+N-1
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nk, K),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nv, V_WIDTH),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
            ),
            (
                None
                if h0 is None
                else pl.BlockSpec(
                    block_shape=(None, N, K, V_WIDTH),
                    index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, 0, 0, 0),
                )
            ),
        ),
        out_specs=pl.BlockSpec(
            block_shape=(None, N, K, V_WIDTH),
            index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
        ),
        scratch_shapes=[pltpu.VMEM((N, K, V_WIDTH), jnp.float32)],
        # block shapes are built from host-validated dimensions (op.py guarantees
        # S % BLOCK_SIZE_S == 0, a 128-lane-multiple K, and V % BLOCK_SIZE_V == 0), so
        # bounds checks are provably redundant
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, dimension_semantics=("parallel", "arbitrary")
        ),
    )

    return kernel(k, v, h0)
