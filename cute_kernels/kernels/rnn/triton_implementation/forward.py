import triton
import triton.language as tl


@triton.jit
def _rnn_forward_triton_kernel(
    input_ptr,
    starting_state_ptr,
    weight_input_ptr,
    weight_state_ptr,
    bias_ptr,
    B,
    S,
    H,
    I,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
):
    num_programs = tl.num_programs(axis=0)

    num_programs_bs = B * S


def rnn_forward_triton(): ...
