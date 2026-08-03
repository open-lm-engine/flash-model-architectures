# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from ..utils import is_haliax_available
from .depthwise_causal_convolution import depthwise_causal_convolution_jax
from .linear_attention import linear_attention_jax


if is_haliax_available():
    from .depthwise_causal_convolution import DepthwiseCausalConvolutionJAX
    from .linear_attention import LinearAttentionJAX
