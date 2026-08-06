# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.numpy as jnp

from .forward import _forward_core


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _depthwise_causal_convolution_pallas(
    x: jax.Array, W: jax.Array, b: jax.Array | None, h0: jax.Array | None, BLOCK_SIZE_S: int
) -> tuple[jax.Array, jax.Array]:
    H = x.shape[-1]

    W = jnp.transpose(W, (1, 0))
    b = jnp.zeros((1, H), dtype=jnp.float32) if b is None else b.astype(jnp.float32)[None, :]

    if h0 is not None:
        h0 = jnp.transpose(h0[:, :, 1:], (0, 2, 1)).astype(x.dtype)

    y, ht = _forward_core(x=x, W=W, b=b, h0=h0, BLOCK_SIZE_S=BLOCK_SIZE_S)

    return y, ht


def _depthwise_causal_convolution_forward(
    x: jax.Array, W: jax.Array, b: jax.Array | None, h0: jax.Array | None, BLOCK_SIZE_S: int
) -> tuple[jax.Array, jax.Array]:
    y, ht = _depthwise_causal_convolution_pallas(x=x, W=W, b=b, h0=h0, BLOCK_SIZE_S=BLOCK_SIZE_S)
    return (y, ht), (x, W, b, h0)


# _BASE_ACTIVATIONS = {
#     "gelu": jax.nn.gelu,
#     "relu": jax.nn.relu,
#     "sigmoid": jax.nn.sigmoid,
#     "silu": jax.nn.silu,
#     "swish": jax.nn.silu,
#     "tanh": jnp.tanh,
# }


# def _get_activation_function(name: str | None) -> Callable[[jax.Array], jax.Array]:
#     if name is None:
#         return lambda x: x

#     if name not in _BASE_ACTIVATIONS:
#         raise ValueError(f"invalid activation function ({name})")

#     return _BASE_ACTIVATIONS[name]


# def _apply_mask_to_padding_states(x: jax.Array, attention_mask: jax.Array | None) -> jax.Array:
#     """
#     Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
#     """
#     if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
#         x = (x * attention_mask[:, :, None]).astype(x.dtype)

#     return x


# def _last_k_columns(xt: jax.Array, K: int) -> jax.Array:
#     """Return the last K columns of `xt` along its trailing axis, left-zero-padded if `xt` is shorter than K."""

#     S = xt.shape[-1]
#     if S < K:
#         return jnp.pad(xt, ((0, 0), (0, 0), (K - S, 0)))

#     return xt[:, :, -K:]


# def _compute_final_state(x: jax.Array, input_state: jax.Array | None, K: int, output_state: bool) -> jax.Array | None:
#     # the trailing K raw input positions to hand back as `input_state` to a subsequent call - pure indexing,
#     # independent of however `y` itself gets computed, so this is shared unchanged by every kernel_backend.
#     if not output_state:
#         return None

#     xt = jnp.transpose(x, (0, 2, 1))  # (B, H, S)

#     if input_state is None:
#         return _last_k_columns(xt, K)

#     full = jnp.concatenate([input_state.astype(x.dtype), xt], axis=-1)  # (B, H, K + S)
#     return full[:, :, -K:]


# @partial(jax.custom_vjp, nondiff_argnums=(4,))
# def _depthwise_causal_convolution_pallas_op(
#     x: jax.Array, weight: jax.Array, bias: jax.Array | None, input_state: jax.Array | None, BLOCK_SIZE_S: int
# ) -> jax.Array:
#     y, _ = _depthwise_causal_convolution_forward_pallas(x, weight, bias, input_state, BLOCK_SIZE_S)
#     return y


# def _depthwise_causal_convolution_pallas_fwd(
#     x: jax.Array, weight: jax.Array, bias: jax.Array | None, input_state: jax.Array | None, BLOCK_SIZE_S: int
# ) -> tuple[jax.Array, tuple]:
#     y, _ = _depthwise_causal_convolution_forward_pallas(x, weight, bias, input_state, BLOCK_SIZE_S)
#     return y, (x, weight, bias, input_state)


# def _depthwise_causal_convolution_pallas_bwd(BLOCK_SIZE_S: int, residuals: tuple, dy: jax.Array) -> tuple:
#     x, weight, bias, input_state = residuals

#     dx, dweight, dbias, dh0 = _depthwise_causal_convolution_backward_pallas(
#         x, weight, input_state, dy, None, BLOCK_SIZE_S
#     )

#     return dx, dweight, (dbias if bias is not None else None), (dh0 if input_state is not None else None)


# _depthwise_causal_convolution_pallas_op.defvjp(
#     _depthwise_causal_convolution_pallas_fwd, _depthwise_causal_convolution_pallas_bwd
# )


# _BLOCK_SIZE_S = 128


# # def _depthwise_causal_convolution_pallas(
# #     x: jax.Array, W: jax.Array, b: jax.Array | None, h0: jax.Array | None, output_state: bool
# # ) -> tuple[jax.Array, jax.Array | None]:
# #     K = W.shape[-1]

# #     y = _depthwise_causal_convolution_pallas_op(x=x, W=W, b=b, h0=h0, x=_BLOCK_SIZE_S)
# #     ht = _compute_final_state(x, input_state, K, output_state)

# #     return y, ht
