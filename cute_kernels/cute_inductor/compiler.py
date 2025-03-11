from typing import Callable

import torch
from torch._dynamo import lookup_backend
from torch._inductor.fx_passes.joint_graph import patterns
from torch._inductor.pattern_matcher import fwd_only, joint_fwd_bwd, register_replacement

from ..utils import enable_cute_tracing, get_boolean_env_variable
from .config import ReplacementConfig


_DEBUG_CUTEINDUCTOR = get_boolean_env_variable("DEBUG_CUTEINDUCTOR", True)


class CuteInductor:
    def __init__(
        self, replacement_configs: list[ReplacementConfig] = [], apply_torch_inductor_after_cute_inductor: bool = True
    ) -> None:
        self.apply_torch_inductor_after_cute_inductor = apply_torch_inductor_after_cute_inductor

        for replacement_config in replacement_configs:
            args = {
                "search_fn": replacement_config.pattern_function,
                "replace_fn": replacement_config.replacement_function,
                "example_inputs": tuple(replacement_config.example_inputs_function()),
                "pass_dicts": patterns,
            }

            register_replacement(**args, trace_fn=joint_fwd_bwd)
            register_replacement(**args, trace_fn=fwd_only, skip_duplicates=True)

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
        with enable_cute_tracing():
            if _DEBUG_CUTEINDUCTOR:
                print("graph before cute inductor")
                gm.print_readable()

            inductor = lookup_backend("inductor" if self.apply_torch_inductor_after_cute_inductor else "eager")
            compiled = inductor(gm, example_inputs)

            if _DEBUG_CUTEINDUCTOR:
                print("graph after cute inductor")
                gm.print_readable()

            return compiled
