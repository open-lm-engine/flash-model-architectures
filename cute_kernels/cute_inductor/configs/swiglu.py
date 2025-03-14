import torch
import torch.nn.functional as F
from torch._inductor.pattern_matcher import Match

from ...kernels import swiglu_cute, swiglu_torch
from ..config import ReplacementConfig
from .add_tensor import _extra_check, _get_example_inputs


def _replacement_function(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return swiglu_cute(gate, up)


def _extra_check(match: Match) -> bool:
    if len(match.kwargs) != 2:
        return False

    gate = match.kwargs["gate"].meta["val"]
    up = match.kwargs["up"].meta["val"]

    return (
        isinstance(gate, torch.Tensor)
        and isinstance(up, torch.Tensor)
        and gate.size() == up.size()
        and gate.dtype == up.dtype
        and gate.device == up.device
    )


swiglu_replacement_config = ReplacementConfig(
    name="swiglu",
    pattern_function=swiglu_torch,
    replacement_function=_replacement_function,
    example_inputs_function=_get_example_inputs,
    extra_check=_extra_check,
)


swiglu1_replacement_config = ReplacementConfig(
    name="swiglu1",
    pattern_function=lambda gate, up: F.silu(gate) * up,
    replacement_function=_replacement_function,
    example_inputs_function=_get_example_inputs,
    extra_check=_extra_check,
)
