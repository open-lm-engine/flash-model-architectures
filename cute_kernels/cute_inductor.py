from typing import Callable

import torch
from torch._dynamo import lookup_backend

from .utils import enable_cute_tracing


class CuteInductor:
    def compiler(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
        with enable_cute_tracing():
            inductor = lookup_backend("inductor")
            compiled = inductor(gm, example_inputs)

            return compiled
