import torch
from torch._inductor.pattern_matcher import Match

from ...kernels import add_tensor_cute, add_tensor_torch
from ..config import ReplacementConfig


def _get_example_inputs() -> tuple[torch.Tensor]:
    return [
        torch.empty(8, 8, device=torch.cuda.current_device(), requires_grad=True),
        torch.empty(8, 8, device=torch.cuda.current_device(), requires_grad=True),
    ]


def _extra_check(match: Match) -> bool:
    return True


add_tensor_replacement_config = ReplacementConfig(
    pattern_function=add_tensor_torch,
    replacement_function=add_tensor_cute,
    example_inputs_function=_get_example_inputs,
    extra_check=_extra_check,
)
