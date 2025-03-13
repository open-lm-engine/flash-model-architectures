from dataclasses import dataclass
from typing import Callable

from torch._inductor.pattern_matcher import _return_true


@dataclass
class ReplacementConfig:
    name: str
    pattern_function: Callable
    replacement_function: Callable
    example_inputs_function: Callable
    extra_check: Callable = _return_true
