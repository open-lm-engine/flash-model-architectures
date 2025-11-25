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
def swiglu_forward_nki_kernel(g_ptr, u_ptr, y_ptr, B, H):
    BLOCK_SIZE_B = 128
    BLOCK_SIZE_H = 512

    BLOCK_Bs = 0
    BLOCK_Be = 0

    for b in range(B):
        BLOCK_Be += BLOCK_SIZE_B
        BLOCK_Hs = 0
        BLOCK_He = 0

        for h in range(H):
            BLOCK_He += BLOCK_SIZE_H

            g = nl.load(g_ptr[BLOCK_Bs:BLOCK_Be, BLOCK_Hs:BLOCK_He])
            u = nl.load(u_ptr[BLOCK_Bs:BLOCK_Be, BLOCK_Hs:BLOCK_He])

            y = u * g * nl.sigmoid(g)

            nl.store(y_ptr[BLOCK_Bs:BLOCK_Be, BLOCK_Hs:BLOCK_He], y)

        BLOCK_Bs = BLOCK_Be


@custom_op(f"{LIBRARY_NAME}::swiglu_forward_nki", mutates_args={"y"})
def swiglu_forward_nki(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None:
    B, H = get_num_elements_and_hidden_size(g)

    swiglu_forward_nki_kernel(g_ptr=g, u_ptr=u, y_ptr=y, B=B, H=H)
