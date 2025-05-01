import triton
import triton.language as tl


@triton.jit
def _copy_array(source_ptr, destination_ptr, b, s, t, S, N, pack: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    unpacked_offset = (b * S + s) * N
    packed_offset = t * N

    for i in range(tl.cdiv(N, BLOCK_SIZE)):
        indices = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        unpacked_ptrs = unpacked_offset + indices
        packed_ptrs = packed_offset + indices

        if pack:
            source = tl.load(source_ptr + unpacked_ptrs)
            tl.store(destination_ptr + packed_ptrs, source)
        else:
            source = tl.load(source_ptr + packed_ptrs)
            tl.store(destination_ptr + unpacked_ptrs, source)


@triton.jit
def pack_unpack_sequence_triton_kernel(
    x_ptr,
    output_ptr,
    cu_seqlens_ptr,
    S,
    N,
    padding_side: tl.constexpr,
    pack: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    s = tl.program_id(axis=0)
    b = tl.program_id(axis=1)

    cu_seqlens_ptrs = cu_seqlens_ptr + b
    start = tl.load(cu_seqlens_ptrs)
    end = tl.load(cu_seqlens_ptrs + 1)
    seqlens = end - start
    pad_tokens = S - seqlens

    if padding_side == "left":
        if s >= pad_tokens:
            _copy_array(x_ptr, output_ptr, b, s, start + s - pad_tokens, S, N, pack, BLOCK_SIZE)
    else:
        if s < seqlens:
            _copy_array(x_ptr, output_ptr, b, s, start + s, S, N, pack, BLOCK_SIZE)
