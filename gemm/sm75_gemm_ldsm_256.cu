#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <cute/tensor.hpp>
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"


using namespace cute;

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ void gemm_ldsm_256_kernel(
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

//     copy(copy_a, tAgA(_,_,_,0), tAsA);
//     copy(copy_b, tBgB(_,_,_,0), tBsB);
//
//     __syncthreads();
//
//     gemm(mma, tCsA, tCsB, tCrC);
//

    auto K_TILE_MAX = size<3>(tAgA);
    auto K_BLOCK_MAX = size<2>(tCsA);
    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile++)
    {

        copy(copy_a, tAgA(_,_,_,k_tile), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

        copy(copy_a, tArA, tAsA);
        copy(copy_b, tBrB, tBsB);

        __syncthreads();
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++) {

//             copy(tCsA(_,_,k_block), tCrA(_,_,k_block));
//             copy(tCsB(_,_,k_block), tCrB(_,_,k_block));
            copy(s2r_tiled_copy_a, s2r_tCsA(_,_,k_block), tCrA_copy_view(_,_,k_block));
            copy(s2r_tiled_copy_b, s2r_tCsB(_,_,k_block), tCrB_copy_view(_,_,k_block));
            gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        }
        // Compute gemm on mma-partitioned smem
//         copy(tCsA, tCrA);
//         copy(tCsB, tCrB);
//
//         gemm(mma, tCrA(_,_,0), tCrB(_,_,0), tCrC);

        __syncthreads();

    }

    //axpby(1.0f, tCrC, 0.0f, tCgC); //vectorized_load
    copy(tCrC, tCgC);

    #if 0
        if(thread0()) {

            print("  mA : "); print(  mA); print("\n");
            print("  gA : "); print(  gA); print("\n");
            print("  sA : "); print(  sA); print("\n");
            print("tAgA : "); print(tAgA); print("\n");
            print("tAsA : "); print(tAsA); print("\n");

        }
    #endif

    #if 0
        if(thread0()) {
            print("  mB : "); print(  mB); print("\n");
            print("  gB : "); print(  gB); print("\n");
            print("  sB : "); print(  sB); print("\n");
            print("tBgB : "); print(tBgB); print("\n");
            print("tBsB : "); print(tBsB); print("\n");
        }
    #endif

    #if 0
        if(thread(1)) {
            print("  mC : "); print(  mC); print("\n");
            print("  gC : "); print(  gC); print("\n");
            print("tCsA : "); print(tCsA); print("\n");
            print("tCsB : "); print(tCsB); print("\n");
            print("tCrA : "); print(tCrA); print("\n");
            print("tCrB : "); print(tCrB); print("\n");
            print("tCgC : "); print(tCgC); print("\n");
            print("tCrC : "); print(tCrC); print("\n");
            print("tCrA(0) : "); print(tCrA(_,_,0)); print("\n");
        }
    #endif

    #if 0
        if(thread(1)) {
            printf("tCsA[0], sA[0]: %f %f\n", static_cast<float>(tCsA[0]),static_cast<float>(sA[0]));
            printf("tCsA[1], sA[16]: %f %f\n", static_cast<float>(tCsA[1]),static_cast<float>(sA[16]));
            printf("tCsA[2], sA[8]: %f %f\n", static_cast<float>(tCsA[2]),static_cast<float>(sA[8]));
            printf("tCsA[3], sA[24]: %f %f\n", static_cast<float>(tCsA[3]),static_cast<float>(sA[24]));
            printf("tCsB[0], sB[0]: %f %f\n", static_cast<float>(tCsB[0]),static_cast<float>(sB[0]));
            printf("tCsB[1], sB[1]: %f %f\n", static_cast<float>(tCsB[1]),static_cast<float>(sB[1]));
            //             for (int i=0;i< 16; i++) {
            //                 for (int j=0;j<8;j++) {
            //                     printf("%f ", static_cast<float>(sA[i  + 16 * j]));
            //                 }
            //                 printf("\n");
            //             }
        }
    #endif


    #if 0
        printf("thread = %d, tCsB[0] = %f\n", threadIdx.x, static_cast<float>(tCsB[0]));
        printf("thread = %d, tCsB[1] = %f\n", threadIdx.x, static_cast<float>(tCsB[1]));
    #endif

    #if 0
        if(thread0()) {
            for (int i=0; i<128; i++){
                printf("i = %d, gA = %f, sA = %f,\n", i, static_cast<float>(gA[i]), static_cast<float>(sA[i]));
            }
            for (int i=0; i<64; i++){
                printf("i = %d, gB = %f, sB = %f,\n", i, static_cast<float>(gB[i]), static_cast<float>(sB[i]));
            }
        }
    #endif


}


void gemm_ldsm_256(half_t* A, half_t* B, float* C, int m, int n, int k) {

    auto prob_shape = make_shape(m, n, k);

    auto dA = make_stride(Int<1>{}, m);                      // (dM, dK)
    auto dB = make_stride(k, Int<1>{});                      // (dN, dK)
    auto dC = make_stride(Int<1>{}, m);                      // (dM, dN)
//     printf("%d\n", prob_shape[1]);
//     printf("%d\n", prob_shape[2]);
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int< 32>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    using SmemLayoutAtomA = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<64>{}, Int<16>{}),
                    make_stride(Int<1>{}, Int<64>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{},
                                               make_shape(Int<128>{}, Int<32>{})));
                                               
    SmemLayoutA sA_layout;   
    

    using SmemLayoutAtomB = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<32>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{}))));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{},
                                               make_shape(Int<128>{}, Int<32>{})));
    SmemLayoutB sB_layout;



    auto sC_layout = make_layout(make_shape (Int<128>{}, Int<128>{}),
                        make_stride(Int<1>{}, Int<128>{}));


    TiledCopy copyA = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, half_t>{},
                               Layout<Shape<_16,_16>, Stride<_1,_16>>{},
                               Layout<Shape< _8,_1>>{});
    TiledCopy copyB = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, half_t>{},
                               Layout<Shape<_64,_4>, Stride<_4,_1>>{},
                               Layout<Shape< _1,_8>>{});

    TiledMMA mmaC = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                    Layout<Shape<_2, _4, _1>>{});

    dim3 dimGrid(size(ceil_div(m, bM)), size(ceil_div(n, bN)));
    dim3 dimBlock(256);
    gemm_ldsm_256_kernel<<<dimGrid, dimBlock>>>(prob_shape, cta_tiler,
                                                     A, dA, sA_layout, copyA,
                                                     B, dB, sB_layout, copyB,
                                                     C, dC, sC_layout, mmaC);
}

