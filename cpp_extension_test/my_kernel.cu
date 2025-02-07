#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void my_cuda_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = x[idx] * 2.0f;  // Example: Multiply each element by 2
    }
}

void launch_my_cuda_kernel(torch::Tensor x) {
    const int size = x.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    my_cuda_kernel<<<blocks, threads>>>(x.data_ptr<float>(), size);
}


#define A(i,j) A[(i) * N + (j)]
#define B(i,j) B[(i) * N + (j)]
#define C(i,j) C[(i) * N + (j)]
#define sA(pointer, i,j) sA[(pointer)][((i) << 7) + (j)]
#define sB(pointer, i,j) sB[(pointer)][((i) << 7) + (j)]
#define TILE_WIDTH 128
#define BLOCK_WIDTH 8
#define FLOAT_4(pointer) reinterpret_cast<float4*>(&(pointer))[0]


__global__ __launch_bounds__(256)
void mm_new_8_kernel(float* A, float* B, float* C, int N){

    if (threadIdx.x == 0) {
        printf("gridDim.x = %d, gridDim.y = %d, gridDim.z = %d\n", gridDim.x, gridDim.y, gridDim.z);
        printf("blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    }
    int thread_id = threadIdx.x;
    int block_idx = blockIdx.x;
    int block_idy = blockIdx.y;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int warp_row = (warp_id >> 1) << 5;
    int warp_col = (warp_id & 1) << 6;
    int thread_row = (lane_id >> 3) << 2;
    int thread_col = (lane_id & 7) << 2;


    int sA_row = thread_id >> 1;
    int sA_col = (thread_id & 1) << 2;

    int sB_row = thread_id >> 5;
    int sB_col = (thread_id & 31) << 2;


    int C_row = warp_row + thread_row;
    int C_col = warp_col + thread_col;


    A = &A((block_idx << 7), 0);
    B = &B(0, (block_idy << 7));
    C = &C((block_idx << 7), (block_idy << 7));

    __shared__ float sA[2][BLOCK_WIDTH * TILE_WIDTH];
    __shared__ float sB[2][BLOCK_WIDTH * TILE_WIDTH];


    float rA[4];
    float rB[4];

    float fA[8] = {};
    float fB[8] = {};

    float accum[64] = {};

    int shared_pointer = 0;
    // load first block
    FLOAT_4(rA) = FLOAT_4(A(sA_row, sA_col));
    FLOAT_4(rB) = FLOAT_4(B(sB_row, sB_col));
    #pragma unroll
    for (int i=0; i<4;i++){
        sA(shared_pointer, sA_col + i, sA_row) = rA[i];
    }

    FLOAT_4(sB(shared_pointer, sB_row, sB_col)) = FLOAT_4(rB);

    __syncthreads();

    A += BLOCK_WIDTH;
    B += BLOCK_WIDTH * N;

    for (int kBlock=0; kBlock<N/BLOCK_WIDTH; kBlock++){

        // load from gmem A, B for next block
        if (kBlock < N/BLOCK_WIDTH - 1) {
            FLOAT_4(rA) = FLOAT_4(A(sA_row, sA_col));
            FLOAT_4(rB) = FLOAT_4(B(sB_row, sB_col));
        }
        #pragma unroll
        for (int kFragment=0; kFragment<BLOCK_WIDTH; kFragment++) {
            // load from smem A, B
            FLOAT_4(fA[0]) = FLOAT_4(sA(shared_pointer, kFragment, C_row));
            FLOAT_4(fA[4]) = FLOAT_4(sA(shared_pointer, kFragment, C_row + 16));
            FLOAT_4(fB[0]) = FLOAT_4(sB(shared_pointer, kFragment, C_col));
            FLOAT_4(fB[4]) = FLOAT_4(sB(shared_pointer, kFragment, C_col + 32));
            // compute outer product
            #pragma unroll
            for (int i=0; i<8;i++){
                #pragma unroll
                for (int j=0; j<8; j++) {
                    accum[i*8+j] += fA[i] * fB[j];
                }
             }

        }

        // store to smem sA, sB for next block
        if (kBlock < N/BLOCK_WIDTH - 1) {


            //FLOAT_4(sA[sA_sOffset]) = FLOAT_4(rA);
            #pragma unroll
            for (int i=0; i<4;i++){
                sA(shared_pointer^1, sA_col + i, sA_row) = rA[i];
                //sA[shared_pointer^1][sA_sOffset + i*TILE_WIDTH] = rA[i];
            }

            FLOAT_4(sB(shared_pointer^1, sB_row, sB_col)) = FLOAT_4(rB);

            __syncthreads();

            A += BLOCK_WIDTH;
            B += BLOCK_WIDTH * N;

            shared_pointer ^= 1;
        }

    }

//    storeToGmem_5(accum, C, N, C_gOffset);

    // store to gmem C
    #pragma unroll
    for (int i=0;i<4;i++) {

        FLOAT_4(C(C_row + i, C_col)) = FLOAT_4(accum[i * 8]);
        FLOAT_4(C(C_row + i, C_col + 32)) = FLOAT_4(accum[i * 8 + 4]);
        FLOAT_4(C(C_row + i + 16, C_col)) = FLOAT_4(accum[(i+4) * 8]);
        FLOAT_4(C(C_row + i + 16, C_col + 32)) = FLOAT_4(accum[(i+4) * 8 + 4]);

    }
}


torch::Tensor mm_new_8(torch::Tensor a, torch::Tensor b) {

  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  torch::Tensor a_contig = a.contiguous();
  torch::Tensor b_contig = b.contiguous();
  torch::Tensor c = torch::empty(a_contig.sizes(), a_contig.options());
  float* a_ptr = a_contig.data_ptr<float>();
  float* b_ptr = b_contig.data_ptr<float>();
  float* c_ptr = c.data_ptr<float>();

  int N = a.size(0);
  dim3 gridDim_mm_new_8(N / TILE_WIDTH,N / TILE_WIDTH);
  dim3 blockDim_mm_new_8(256);

  mm_new_8_kernel<<<gridDim_mm_new_8, blockDim_mm_new_8>>>(a_ptr, b_ptr, c_ptr, N);

  return c;
}

#undef A
#undef B
#undef C
#undef sA
#undef sB
#undef TILE_WIDTH
#undef BLOCK_WIDTH
#undef FLOAT_4



#include <cute/tensor.hpp>

using namespace cute;

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ __launch_bounds__(256)
void gemm_register_pipelining_256_kernel(
            ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma
)
{


    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    __shared__ half_t smemA[cosize_v<ASmemLayout>];
    __shared__ half_t smemB[cosize_v<BSmemLayout>];

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K)
    Tensor tArA = make_fragment_like(tAsA);


    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K)
    Tensor tBrB = make_fragment_like(tBsB);

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K)
    Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

    // Allocate the accumulators -- same size as the projected data
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);


    auto s2r_tiled_copy_a = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, half_t>{}, mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(threadIdx.x);
    auto s2r_tCsA = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_copy_view = s2r_thr_copy_a.retile_D(tCrA);

    auto s2r_tiled_copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(threadIdx.x);
    auto s2r_tCsB = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_copy_view = s2r_thr_copy_b.retile_D(tCrB);


    //printf("tCrC: %f\n", tCrC[0]);
    clear(tCrC);


    // prologue
    copy(copy_a, tAgA(_,_,_,0), tAsA);
    copy(copy_b, tBgB(_,_,_,0), tBsB);

    __syncthreads();

    copy(s2r_tiled_copy_a, s2r_tCsA(_,_,0), tCrA_copy_view(_,_,0));
    copy(s2r_tiled_copy_b, s2r_tCsB(_,_,0), tCrB_copy_view(_,_,0));

    auto K_TILE_MAX = size<3>(tAgA);
    auto K_BLOCK_MAX = size<2>(tCsA);
    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile++)
    {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++) {

            if (k_block == K_BLOCK_MAX - 1)
            {
            // Copy rmem to smem
                __syncthreads();
                copy(copy_a, tArA, tAsA);
                copy(copy_b, tBrB, tBsB);
                __syncthreads();
            }

            int k_block_next = (k_block + 1) % K_BLOCK_MAX;
            copy(s2r_tiled_copy_a, s2r_tCsA(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
            copy(s2r_tiled_copy_b, s2r_tCsB(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));

            if (k_block == 0)
            {
            // Copy gmem to rmem for k_tile+1
                int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
                copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
                copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);
            }

            gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        }


    }


    //axpby(1.0f, tCrC, 0.0f, tCgC); //vectorized_load
    copy(tCrC, tCgC);


}

// column major matrices
torch::Tensor gemm_register_pipelining_256(torch::Tensor a, torch::Tensor b) {

    int m = a.size(0);
    int n = b.size(1);
    int k = a.size(1);

    auto prob_shape = make_shape(m, n, k);

    auto dA = make_stride(Int<1>{}, m);                      // (dM, dK)
    auto dB = make_stride(k, Int<1>{});                      // (dN, dK)
    auto dC = make_stride(Int<1>{}, m);                      // (dM, dN)
//     printf("%d\n", prob_shape[1]);
//     printf("%d\n", prob_shape[2]);
    auto bM = Int<256>{};
    auto bN = Int<128>{};
    auto bK = Int< 32>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    using SmemLayoutAtomA = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<64>{}, Int<16>{}),
                    make_stride(Int<1>{}, Int<64>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{},
                                               make_shape(Int<256>{}, Int<32>{})));

    SmemLayoutA sA_layout;


    using SmemLayoutAtomB = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<32>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{}))));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{},
                                               make_shape(Int<128>{}, Int<32>{})));
    SmemLayoutB sB_layout;



    auto sC_layout = make_layout(make_shape (Int<256>{}, Int<128>{}),
                        make_stride(Int<1>{}, Int<256>{}));


    TiledCopy copyA = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, half_t>{},
                               Layout<Shape<_16,_16>, Stride<_1,_16>>{},
                               Layout<Shape< _8,_1>>{});
    TiledCopy copyB = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, half_t>{},
                               Layout<Shape<_64,_4>, Stride<_4,_1>>{},
                               Layout<Shape< _1,_8>>{});

    TiledMMA mmaC = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                    Layout<Shape<_2, _4, _1>>{},
                                    Tile<_256,_128,_8>{});

//     torch::Tensor a_contig = a.contiguous();
//     torch::Tensor b_contig = b.contiguous();
    torch::Tensor c = torch::empty(a.sizes(), a.options().dtype(torch::kFloat32));
//     half_t* a_ptr = a_contig.data_ptr<at::Half>();
//     half_t* b_ptr = b_contig.data_ptr<at::Half>();

//     half_t* a_ptr = reinterpret_cast<half_t*>(a_contig.data_ptr<at::Half>());
//     half_t* b_ptr = reinterpret_cast<half_t*>(b_contig.data_ptr<at::Half>());
    half_t* a_ptr = reinterpret_cast<half_t*>(a.data_ptr());
    half_t* b_ptr = reinterpret_cast<half_t*>(b.data_ptr());

    float* c_ptr = c.data_ptr<float>();
//     for (int i=0;i<10;i++) {
//         printf("%f\n", (float)a_ptr[i]);
//     }

    dim3 dimGrid(size(ceil_div(m, bM)), size(ceil_div(n, bN)));
    dim3 dimBlock(256);
    gemm_register_pipelining_256_kernel<<<dimGrid, dimBlock>>>(prob_shape, cta_tiler,
                                                     a_ptr, dA, sA_layout, copyA,
                                                     b_ptr, dB, sB_layout, copyB,
                                                     c_ptr, dC, sC_layout, mmaC);
    return c;
}
