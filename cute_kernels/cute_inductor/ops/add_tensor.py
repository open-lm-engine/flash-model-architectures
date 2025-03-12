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
    if len(match.kwargs) != 2:
        return False

    x = match.kwargs["x"].meta["val"]
    y = match.kwargs["y"].meta["val"]

    return isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.size() == y.size() and x.dtype == y.dtype


def _replacement_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return add_tensor_cute(x, y)


add_tensor_replacement_config = ReplacementConfig(
    pattern_function=add_tensor_torch,
    replacement_function=_replacement_function,
    example_inputs_function=_get_example_inputs,
    extra_check=_extra_check,
)
