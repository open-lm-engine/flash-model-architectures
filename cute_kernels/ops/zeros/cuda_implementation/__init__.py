# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit


@custom_op(f"{LIBRARY_NAME}::zeros_cuda", mutates_args={"x"})
@cpp_jit()
def zeros_cuda(x: torch.Tensor) -> None: ...
