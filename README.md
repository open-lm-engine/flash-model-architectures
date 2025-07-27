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
| [GRU](fma/modules/gru/__init__.py) | ✅ | ❌ |
| [MoE](fma/modules/moe/__init__.py) | ✅ | ✅ |
| [RNN](fma/modules/rnn/__init__.py) | ✅ | ❌ |

## Ops

| Module     | Triton | CUDA |
|------------|--------|------|
| [add_scalar](fma/ops/add_scalar/__init__.py) | ✅     | ✅   |
| [add_tensor](fma/ops/add_tensor/__init__.py) | ✅     | ✅   |
| [bmm](fma/ops/bmm/__init__.py)        | ✅     | ❌   |
| [continuous_count](fma/ops/continuous_count/__init__.py) | ❌ | ✅ |
| [cross_entropy](fma/ops/cross_entropy/__init__.py) | ✅ | ❌ |
| [fused_linear_cross_entropy](fma/ops/fused_linear_cross_entropy.py) | ✅ | ❌ |
| [fused_residual_add_rmsnorm](fma/ops/fused_residual_add_rmsnorm/__init__.py) | ✅ | ❌ |
| [gemm](fma/ops/gemm/__init__.py) | ✅ | ❌ |
| [grouped_gemm](fma/ops/grouped_gemm/__init__.py) | ❌ | ✅ |
| [matrix_transpose](fma/ops/matrix_transpose/__init__.py) | ✅ | ❌ |
| [rmsnorm](fma/ops/rmsnorm/__init__.py) | ✅ | ❌ |
| [pack_sequence](fma/ops/sequence_packing/__init__.py) | ✅ | ✅ |
| [softmax](fma/ops/softmax/__init__.py) | ✅ | ❌ |
| [swiglu](fma/ops/swiglu/__init__.py) | ✅ | ✅ |
| [swiglu_packed](fma/ops/swiglu/__init__.py) | ✅ | ❌ |
| [unpack_sequence](fma/ops/sequence_packing/__init__.py) | ✅ | ✅ |
| [zeros](fma/ops/zeros/__init__.py) | ✅ | ✅ |

# Cute Inductor
CuteInductor allows easier injection of kernels contained in this repository into any PyTorch module.
<p align="center">
  <img src="assets/cute-inductor.webp" width="300px" height="300px">
</p>
