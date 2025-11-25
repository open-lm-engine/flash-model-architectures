# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import neuronxcc.nki.language as nl
import torch
from torch.library import custom_op
from torch_neuronx import nki

from ....constants import LIBRARY_NAME
from ....utils import get_num_elements_and_hidden_size


@nki.jit
def swiglu_forward_nki_kernel(g_ptr, u_ptr, y_ptr):
    BLOCK_SIZE_B = 128
    BLOCK_SIZE_H = 512

    g = nl.load(g_ptr[0:BLOCK_SIZE_B, 0:BLOCK_SIZE_H])
    u = nl.load(u_ptr[0:BLOCK_SIZE_B, 0:BLOCK_SIZE_H])

    y = u * g * nl.sigmoid(g)

    nl.store(y_ptr[0:BLOCK_SIZE_B, 0:BLOCK_SIZE_H], y)


@custom_op(f"{LIBRARY_NAME}::swiglu_forward_nki", mutates_args={"y"})
def swiglu_forward_nki(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None:
    B, H = get_num_elements_and_hidden_size(g)

    swiglu_forward_nki_kernel(g_ptr=g, u_ptr=u, y_ptr=y)
