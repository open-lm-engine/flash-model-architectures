# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from torch._dynamo import lookup_backend

from ..utils import get_boolean_env_variable
from .rmsnorm import replace_rmsnorm
from .swiglu_packed import replace_swiglu_unchunked


_DEBUG_CUTEINDUCTOR = get_boolean_env_variable("DEBUG_CUTEINDUCTOR", True)


class CuteInductor:
    def __init__(
        self,
        use_torch_inductor_after_cute_inductor: bool = True,
        replace_functions=[replace_rmsnorm, replace_swiglu_unchunked],
    ) -> None:
        self.use_torch_inductor_after_cute_inductor = use_torch_inductor_after_cute_inductor
        self.replace_functions = replace_functions

    def compiler(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
        # with enable_cute_tracing():
        if _DEBUG_CUTEINDUCTOR:
            print("graph before cute inductor")
            gm.print_readable()

        for replace_function in self.replace_functions:
            for node in gm.graph.nodes:
                replace_function(gm, node)

        if _DEBUG_CUTEINDUCTOR:
            print("graph after cute inductor")
            gm.print_readable()

        if self.use_torch_inductor_after_cute_inductor:
            inductor = lookup_backend("inductor")
            compiled = inductor(gm, example_inputs)
        else:
            compiled = gm.forward

        return compiled
