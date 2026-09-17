#!/usr/bin/env python3
"""Generate a matmul_<N>.cuh wgmma microbenchmark for a given tile width N.

wgmma.mma_async.sync.aligned.m64nNk16.f32.f16.f16 is valid for any N that is a
multiple of 8 in [8, 256]. Each N needs its own inline-PTX wrapper (the shape
is baked into the instruction mnemonic at compile time), so this script
mechanically derives the wrapper + surrounding kernel/launch boilerplate from
the pattern already established by the hand-written matmul_96.cuh,
matmul_104.cuh, matmul_128.cuh, matmul_152.cuh, matmul_160.cuh, matmul_192.cuh,
matmul_64.cuh, and matmul_8.cuh files in this directory:

  - regs (f32 accumulator registers) = N / 2
  - rows (the "d[rows][8]" accumulator shape)  = ceil(N / 16)
  - the last row holds "regs - (rows - 1) * 8" registers (8 when N % 16 == 0,
    otherwise a partial row whose remaining slots are left unused, exactly as
    matmul_104.cuh / matmul_152.cuh already do for their non-16-divisible N)

Usage: python3 generate_matmul_kernel.py <N> [<N> ...]
Writes matmul_<N>.cuh into the same directory as this script for each N.
"""
import sys
import os

VALID_NS = set(range(8, 257, 8))


def build_wgmma_fn(N, regs, rows, last_row_count):
    flat = []
    for r in range(rows):
        cnt = 8 if r < rows - 1 else last_row_count
        for c in range(cnt):
            flat.append((r, c))
    assert len(flat) == regs

    # PTX inline-asm text is whitespace-insensitive between operands (the only
    # place a literal "\n" matters is where the original files put one after
    # each full instruction), so the register list can just be one flat,
    # comma-joined string - no line-wrapping bookkeeping needed for
    # correctness. It's still wrapped here purely for readability.
    desc_a_idx = regs
    desc_b_idx = regs + 1
    s0, s1, s2, s3, s4 = (regs + 2, regs + 3, regs + 4, regs + 5, regs + 6)

    reg_placeholders = ", ".join(f"%{i}" for i in range(regs))
    mnemonic_line = (
        f'"wgmma.mma_async.sync.aligned.m64n{N}k16.f32.f16.f16 '
        f'{{{reg_placeholders}}}, %{desc_a_idx}, %{desc_b_idx}, '
        f'%{s0}, %{s1}, %{s2}, %{s3}, %{s4};\\n"'
    )

    out_ops = []
    for i in range(0, regs, 6):
        chunk = flat[i:i + 6]
        parts = ", ".join(f"\"+f\"(d[{r}][{c}])" for r, c in chunk)
        out_ops.append("                " + parts + ("," if i + 6 < regs else ""))
    out_ops_block = "\n".join(out_ops)

    fn = f'''template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma{N}(float d[{rows}][8], uint16_t* sA, uint16_t* sB, unsigned long long iterations) {{
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    #pragma unroll 10
    for(unsigned long long i=0; i < iterations; i++){{
        asm volatile(
            "{{\\n"
            {mnemonic_line}
            "}}\\n"
            :
{out_ops_block}
            : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
                "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
    }}
}}'''
    return fn


def build_file(N):
    if N not in VALID_NS:
        raise ValueError(f"N={N} is not a multiple of 8 in [8, 256]")

    regs = N // 2
    rows = (N + 15) // 16
    last_row_count = regs - (rows - 1) * 8

    wgmma_fn = build_wgmma_fn(N, regs, rows, last_row_count)

    static_assert_line = (
        f"    static_assert(sizeof(d) * NUM_THREADS == BM * BN * sizeof(float));\n"
        if N % 16 == 0 else
        f"    //static_assert(sizeof(d) * NUM_THREADS == BM * (16*ceil(BN/16.0)) * sizeof(float));\n"
    )

    return f'''
namespace M{N}{{

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {{ return (((x) & 0x3FFFF) >> 0x4); }}

__device__ uint64_t make_smem_desc(uint16_t* ptr) {{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
    desc |= 1llu << 62; // 128B swizzle
    return desc;
    }}


__device__ void warpgroup_arrive() {{
    asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");
}}

__device__ void warpgroup_commit_batch() {{
    asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");
}}

template <int N>
__device__ void warpgroup_wait() {{
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\\n" ::"n"(N) : "memory");
}}

template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map(CUtensorMap *tma_map, uint16_t* gmem_ptr, int blocks_height, int blocks_width) {{
    void* gmem_address = (void*)gmem_ptr;
    uint64_t gmem_prob_shape[5] = {{(uint64_t)BlockMinorSize*blocks_width, (uint64_t)BlockMajorSize*blocks_height, 1, 1, 1}};
    uint64_t gmem_prob_stride[5] = {{sizeof(uint16_t), sizeof(uint16_t) * BlockMinorSize*blocks_width, 0, 0, 0}};
    uint32_t smem_box_shape[5] = {{uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1}};
    uint32_t smem_box_stride[5] = {{1, 1, 1, 1, 1}};

    CUresult result = cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address, gmem_prob_shape,
        gmem_prob_stride + 1, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
}}

CUtensorMap *d_tma_map_A = 0;
CUtensorMap *d_tma_map_B = 0;
int _prev_m=0, _prev_n=0, _prev_k=0;

template<int st_rows, int st_cols>
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(uint16_t* src, int blocks_height, int blocks_width) {{
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map<st_rows, st_cols>(&tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}}

{wgmma_fn}

template <int BM, int BN, int BK>
struct SMem {{
    alignas(128) uint16_t A[BM*BK];
    alignas(128) uint16_t B[BK*BN];
}};

template<int BM, int BN, int BK, int NUM_THREADS, bool DBG>
__global__ void __launch_bounds__(NUM_THREADS) matmulKernel{N}(int M, int N, int K, uint16_t* C, const CUtensorMap* tensorMapA, const CUtensorMap* tensorMapB, int *DB, unsigned long long iterations) {{
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N=BN;
    constexpr int B_WG_M = BM / (NUM_THREADS / 128);
    extern __shared__ SMem<BM, BN, BK> s;
    uint16_t *sA = s.A;
    uint16_t *sB = s.B;
    // Barriers cannot be in the struct and have to be declared this way
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier barA, barB;
    float d[B_WG_M/WGMMA_M][{rows}][8];
{static_assert_line}    memset(d, 0, sizeof(d));

    const int num_blocks_k = K / BK;
    int num_block_n = blockIdx.x % (N / BN);
    int num_block_m = blockIdx.x / (N / BN);

    if (threadIdx.x == 0) {{
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }}
    __syncthreads();
    int wg_idx = threadIdx.x / 128;

    barrier::arrival_token tokenA, tokenB;
    int sumLoad = 0, cntLoad = 0;
    int sumCompute = 0, cntCompute = 0;
    int sumStore = 0, cntStore = 0;
    for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter) {{
        clock_t start = clock();
        // Load
        if (threadIdx.x == 0) {{
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA, block_k_iter*BK, num_block_m*BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, BK*BM*sizeof(uint16_t));
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, block_k_iter*BK, num_block_n*BN, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, BK*BN*sizeof(uint16_t));
        }} else {{
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }}
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();
        if constexpr (DBG) {{
            sumLoad += clock() - start;
            cntLoad++;
            start = clock();
        }}

        // Compute
        warpgroup_arrive();
        #pragma unroll
        for (int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {{
            uint16_t *wgmma_sA = sA + BK*(m_it + wg_idx*B_WG_M/WGMMA_M)*WGMMA_M;
            #pragma unroll
            for (int k_it = 0; k_it < BK/WGMMA_K; ++k_it) {{
                wgmma{N}<1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it*WGMMA_K], &sB[k_it*WGMMA_K], iterations);
            }}
        }}
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        if constexpr (DBG) {{
            sumCompute += clock() - start;
            cntCompute++;
        }}
    }}

    // Store
    {{
        clock_t start = clock();

        uint32_t tid = threadIdx.x % 128;
        uint32_t lane = tid & 31;
        uint32_t warp = tid / 32;
        uint32_t row = warp*16 + lane / 4;

        uint16_t *block_C = C + num_block_n*BN*M + num_block_m*BM;

        #pragma unroll
        for (uint32_t m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {{
            int yo = m_it*WGMMA_M + wg_idx*B_WG_M;
            #pragma unroll
            for (uint32_t w = 0; w < WGMMA_N/16; ++w) {{
                int col = 16*w + 2*(tid % 4);
                #define IDX(i, j) ((j)*M + ((i) + yo))

                block_C[IDX(row, col)] = d[m_it][w][0];
                block_C[IDX(row, col+1)] = d[m_it][w][1];
                block_C[IDX(row+8, col)] = d[m_it][w][2];
                block_C[IDX(row+8, col+1)] = d[m_it][w][3];
                block_C[IDX(row, col+8)] = d[m_it][w][4];
                block_C[IDX(row, col+9)] = d[m_it][w][5];
                block_C[IDX(row+8, col+8)] = d[m_it][w][6];
                block_C[IDX(row+8, col+9)] = d[m_it][w][7];

                #undef IDX
            }}
        }}
        if constexpr (DBG) {{
            sumStore += clock() - start;
            cntStore++;
            if (threadIdx.x == 63) {{
                int i = blockIdx.x*6;
                DB[i] = sumLoad; DB[i + 1] = cntLoad;
                DB[i + 2] = sumCompute; DB[i + 3] = cntCompute;
                DB[i + 4] = sumStore; DB[i + 5] = cntStore;
            }}
        }}
    }}
}}


void runKernel{N}(int M, int N, int K, uint16_t *A, uint16_t *B, uint16_t *C, int *DB, unsigned long long iterations) {{
    constexpr int BM = 128;
    constexpr int BN = {N};
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128;

    if (!d_tma_map_A) {{
        d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(A, M / BM, K / BK);
        d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(B, N / BN, K / BK);
        _prev_m = M;
        _prev_n = N;
        _prev_k = K;
    }}
    // Assert cached values are of same size
    assert (M == _prev_m && N == _prev_n && K == _prev_k);
    auto* kernel = DB ? matmulKernel{N}<BM, BN, BK, NUM_THREADS, true>
            : matmulKernel{N}<BM, BN, BK, NUM_THREADS, false>;
    size_t sMemSize = sizeof(SMem<BM, BN, BK>);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    kernel<<<(M/BM) * (N/BN), NUM_THREADS, sMemSize>>>(M, N, K, C, d_tma_map_A, d_tma_map_B, DB, iterations);
}}

}} // namespace

using M{N}::runKernel{N};
'''


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <N> [<N> ...]  (N must be a multiple of 8, 8..256)")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for arg in sys.argv[1:]:
        N = int(arg)
        content = build_file(N)
        out_path = os.path.join(out_dir, f"matmul_{N}.cuh")
        with open(out_path, "w") as f:
            f.write(content)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
