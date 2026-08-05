# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from ...utils import is_haliax_available
from .op import depthwise_causal_convolution_jax


if is_haliax_available():
    from .module import DepthwiseCausalConvolutionJAX
