from dataclasses import dataclass
from typing import Callable


@dataclass
class ReplacementConfig:
    pattern_function: Callable
    replacement_function: Callable
    example_inputs_function: Callable
