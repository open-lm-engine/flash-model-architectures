import torch

from ...kernels import swiglu_cute, swiglu_torch
from ..config import ReplacementConfig
from .add_tensor import _extra_check, _get_example_inputs


def _replacement_function(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return swiglu_cute(gate, up)


swiglu_replacement_config = ReplacementConfig(
    name="swiglu",
    pattern_function=swiglu_torch,
    replacement_function=_replacement_function,
    example_inputs_function=_get_example_inputs,
    extra_check=_extra_check,
)
