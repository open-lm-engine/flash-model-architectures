# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import neuronxcc.nki.language as nl
import torch
from torch.library import custom_op
from torch_neuronx import TorchNeuronNKIKernel

from ....constants import LIBRARY_NAME
from ....math import ceil_divide


def swiglu_forward_nki_kernel(g_ptr, u_ptr, y_ptr):
    BLOCK_SIZE_B = 128
    BLOCK_SIZE_H = 512

    BLOCK_ID_B = nl.program_id(0)
    BLOCK_ID_H = nl.program_id(1)

    BLOCK_B = BLOCK_ID_B * nl.arange(BLOCK_SIZE_B)
    BLOCK_H = BLOCK_ID_H * nl.arange(BLOCK_SIZE_H)

    g = nl.load(g_ptr[BLOCK_B, BLOCK_H])
    u = nl.load(u_ptr[BLOCK_B, BLOCK_H])

    y = u * g * nl.sigmoid(g)

    nl.store(y_ptr[BLOCK_B, BLOCK_H], y)


@custom_op(f"{LIBRARY_NAME}::swiglu_forward_nki", mutates_args={"y"})
def swiglu_forward_nki(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None:
    BLOCK_SIZE_B = 128
    BLOCK_SIZE_H = 512

    B, H = g.size()

    kernel = swiglu_forward_nki.cache.get(g.dtype, None)

    if kernel is None:
        kernel = TorchNeuronNKIKernel(
            func=swiglu_forward_nki_kernel,
            grid=(ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)),
            kernel_return=True,
            return_tensor_overrides=(y,),
        )

        swiglu_forward_nki.cache[g.dtype] = kernel

    kernel(g, u, y)


swiglu_forward_nki.cache = {}
