@triton.jit
def triton_tem_fused_mm_2(arg_A, arg_B, out_ptr0):
    GROUP_M: tl.constexpr = 8
    ALLOW_TF32: tl.constexpr = True
    ACC_TYPE: tl.constexpr = tl.float32
    BLOCK_M: tl.constexpr = 128
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 32
    A = arg_A
    B = arg_B

    M = 4096
    N = 4100
    K = 4096
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 4096
    stride_ak = 1
    stride_bk = 4128
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 4096 * idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 4100 * idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_n + 4128 * idx_m, xindex.shape)).broadcast_to(xindex.shape)))

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 4100 * idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
