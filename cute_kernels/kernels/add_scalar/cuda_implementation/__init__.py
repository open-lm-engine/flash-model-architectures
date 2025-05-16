import torch

from ....constants import LIBRARY_NAME
from ....cutotune import CutoTuneConfig, cutotune
from ....jit import cpp_jit
from ....utils import cute_op


_CUTOTUNE_CONFIG = CutoTuneConfig({"BLOCK_SIZE": 1024})


@cutotune(configs=[_CUTOTUNE_CONFIG], default_config=_CUTOTUNE_CONFIG)
@cute_op(f"{LIBRARY_NAME}::add_scalar_cuda", mutates_args={"output"})
@cpp_jit()
def add_scalar_cuda(x: torch.Tensor, y: float, output: torch.Tensor, BLOCK_SIZE: int) -> None: ...
