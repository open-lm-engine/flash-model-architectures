# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import neuronxcc.nki.language as nl
import torch
from torch.library import custom_op
from torch_neuronx import TorchNeuronNKIKernel

from ....constants import LIBRARY_NAME


def swiglu_forward_nki_kernel(g_ptr, u_ptr, y_ptr):
    BLOCK_SIZE_B = 128
    BLOCK_SIZE_H = 512

    B, H = g_ptr.shape

    NUM_BLOCKS_B = B // BLOCK_SIZE_B
    NUM_BLOCKS_H = H // BLOCK_SIZE_H

    for b in nl.affine_range(NUM_BLOCKS_B):
        BLOCK_B = nl.arange(b * BLOCK_SIZE_B, b * BLOCK_SIZE_B + BLOCK_SIZE_B)[:, None]

        for h in nl.affine_range(NUM_BLOCKS_H):
            BLOCK_H = nl.arange(h * BLOCK_SIZE_H, h * BLOCK_SIZE_H + BLOCK_SIZE_H)[None, :]

            g = nl.load(g_ptr[BLOCK_B, BLOCK_H])
            u = nl.load(u_ptr[BLOCK_B, BLOCK_H])

            y = u * g * nl.sigmoid(g)

            nl.store(y_ptr[BLOCK_B, BLOCK_H], y)


@custom_op(f"{LIBRARY_NAME}::swiglu_forward_nki", mutates_args={"y"})
def swiglu_forward_nki(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None:
    # swiglu_forward_nki_kernel(g_ptr=g, u_ptr=u, y_ptr=y)

    traced_kernel = TorchNeuronNKIKernel(
        func=swiglu_forward_nki_kernel, grid=(1,), kernel_return=True, return_tensor_overrides=(y,)
    )

    traced_kernel(g, u, y)
