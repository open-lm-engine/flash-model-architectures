<!-- **************************************************
Copyright (c) 2025, Mayank Mishra
************************************************** -->

# Discord Server
Join the [discord server](https://discord.gg/AFDxmjH5RV) if you are interested in LLM architecture or distributed training/inference research.

# Efficient GPU kernels written in both CUDA and Triton

<p align="center">
  <img src="assets/cute-cat.webp" width="300px" height="300px">
</p>

## Modules

| Module   |  Triton   | CUDA |
|----------|-----------|------|
| [GRU](cute_kernels/modules/gru/__init__.py) | ✅ | ❌ |
| [MoE](cute_kernels/modules/moe/__init__.py) | ✅ | ✅ |
| [RNN](cute_kernels/modules/rnn/__init__.py) | ✅ | ❌ |

## Ops

| Module     | Triton | CUDA |
|------------|--------|------|
| [add_scalar](cute_kernels/ops/add_scalar/__init__.py) | ✅     | ✅   |
| [add_tensor](cute_kernels/ops/add_tensor/__init__.py) | ✅     | ✅   |
| [bmm](cute_kernels/ops/bmm/__init__.py)        | ✅     | ❌   |
| [continuous_count](cute_kernels/ops/continuous_count/__init__.py) | ❌ | ✅ |
| [cross_entropy](cute_kernels/ops/cross_entropy/__init__.py) | ✅ | ❌ |
| [fused_linear_cross_entropy](cute_kernels/ops/fused_linear_cross_entropy.py) | ✅ | ❌ |
| [fused_residual_add_rmsnorm](cute_kernels/ops/fused_residual_add_rmsnorm/__init__.py) | ✅ | ❌ |
| [gemm](cute_kernels/ops/gemm/__init__.py) | ✅ | ❌ |
| [grouped_gemm](cute_kernels/ops/grouped_gemm/__init__.py) | ❌ | ✅ |
| [matrix_transpose](cute_kernels/ops/matrix_transpose/__init__.py) | ✅ | ❌ |
| [rmsnorm](cute_kernels/ops/rmsnorm/__init__.py) | ✅ | ❌ |
| [pack_sequence](cute_kernels/ops/sequence_packing/__init__.py) | ✅ | ✅ |
| [softmax](cute_kernels/ops/softmax/__init__.py) | ✅ | ❌ |
| [swiglu](cute_kernels/ops/swiglu/__init__.py) | ✅ | ✅ |
| [swiglu_packed](cute_kernels/ops/swiglu/__init__.py) | ✅ | ❌ |
| [unpack_sequence](cute_kernels/ops/sequence_packing/__init__.py) | ✅ | ✅ |
| [zeros](cute_kernels/ops/zeros/__init__.py) | ✅ | ✅ |

# Cute Inductor
CuteInductor allows easier injection of kernels contained in this repository into any PyTorch module.
<p align="center">
  <img src="assets/cute-inductor.webp" width="300px" height="300px">
</p>
