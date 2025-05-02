import triton
import triton.language as tl


@triton.jit
def gemm_triton_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    output_ptr,
    alpha,
    beta,
    is_A_transposed: tl.constexpr,
    is_B_transposed: tl.constexpr,
    M,
    K,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # A -> K x M if is_A_transposed else M x K
    # B -> N x K if is_B_transposed else K x N
    # C -> M x N

    pid = tl.program_id(axis=0)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_programs_n
    pid_n = pid % num_programs_n

    indices_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    indices_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = indices_m < M
    mask_n = indices_n < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        indices_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = indices_k < K

        if is_A_transposed:
            mask_A = mask_k[:, None] & mask_m[None, :]
            A_ptrs = A_ptr + indices_k[:, None] * M + indices_m[None, :]
        else:
            mask_A = mask_m[:, None] & mask_k[None, :]
            A_ptrs = A_ptr + indices_m[:, None] * K + indices_k[None, :]

        A = tl.load(A_ptrs, mask=mask_A, other=0)

        if is_A_transposed:
            A = A.T

        if is_B_transposed:
            mask_B = mask_n[:, None] & mask_k[None, :]
            B_ptrs = B_ptr + indices_n[:, None] * K + indices_k[None, :]
        else:
            mask_B = mask_k[:, None] & mask_n[None, :]
            B_ptrs = B_ptr + indices_k[:, None] * N + indices_n[None, :]

        B = tl.load(B_ptrs, mask=mask_B, other=0)

        if is_B_transposed:
            B = B.T

        accumulator = tl.dot(A, B, accumulator, allow_tf32=True)

    accumulator = accumulator.to(A_ptr.dtype.element_ty)
    accumulator *= alpha

    indices_mn = indices_m[:, None] * N + indices_n[None, :]
    mask_mn = mask_m[:, None] & mask_n[None, :]

    if C_ptr is not None:
        C = tl.load(C_ptr + indices_mn, mask=mask_mn)
        accumulator += beta * C

    tl.store(output_ptr + indices_mn, accumulator, mask=mask_mn)
