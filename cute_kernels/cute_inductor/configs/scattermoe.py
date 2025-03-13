import torch
from torch._inductor.pattern_matcher import Match

from ...kernels import MoE_Torch, MoE_Triton
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

    return (
        isinstance(x, torch.Tensor)
        and isinstance(y, torch.Tensor)
        and x.size() == y.size()
        and x.dtype == y.dtype
        and x.device == y.device
    )


scattermoe_replacement_config = ReplacementConfig(
    name="scattermoe",
    pattern_function=MoE_Torch._compute_experts,
    replacement_function=MoE_Triton._compute_experts,
    example_inputs_function=_get_example_inputs,
    extra_check=_extra_check,
)
