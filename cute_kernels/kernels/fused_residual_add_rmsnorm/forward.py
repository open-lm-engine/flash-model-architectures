import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...math import ceil_divide, get_next_power_of_2
from ...utils import get_num_elements_and_hidden_size
from .triton_implementation import _fused_residual_add_rmsnorm_forward_triton_kernel
