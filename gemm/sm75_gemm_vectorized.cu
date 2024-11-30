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
          class ASmemLayout, class TiledCopyA,
          class BSmemLayout, class TiledCopyB,
          class CSmemLayout, class TiledMma>
__global__ void gemm_vectorized_kernel(
           ProblemShape shape_MNK, CtaTiler cta_tiler,
           half_t* A, ASmemLayout sA_layout, TiledCopyA copy_a,
           half_t* B, BSmemLayout sB_layout, TiledCopyB copy_b,
           float*  C, CSmemLayout sC_layout, TiledMma      mma)
{

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), make_stride(Int<1>{}, Int<1024>{})); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), make_stride(Int<1024>{}, Int<1>{})); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), make_stride(Int<1>{}, Int<1024>{})); // (M,N)

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

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K)


    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K)
    Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

    // Allocate the accumulators -- same size as the projected data
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

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

    for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile++)
    {

        copy(copy_a, tAgA(_,_,_,k_tile), tAsA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBsB);

        __syncthreads();

        // Compute gemm on mma-partitioned smem
        gemm(mma, tCsA, tCsB, tCrC);

        __syncthreads();

    }

    axpby(1.0f, tCrC, 0.0f, tCgC); //test


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
            print("tCgC : "); print(tCgC); print("\n");
            print("tCrC : "); print(tCrC); print("\n");
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


void gemm_vectorized(half_t* A, half_t* B, float* C, int M, int N, int K) {

    auto prob_shape = make_shape(M, N, K);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int< 8>{};
    auto cta_tiler = make_shape(bM, bN, bK);


    auto sA_layout = make_layout(make_shape (Int<128>{}, Int<8>{}),
                        make_stride(Int<1>{}, Int<128>{}));
//     auto sB_layout = make_layout(make_shape (Int<8>{}, Int<8>{}),
//                         make_stride(Int<1>{}, Int<8>{}));
    auto sB_layout = make_layout(make_shape (Int<128>{}, Int<8>{}),
                        make_stride(Int<8>{}, Int<1>{}));
    auto sC_layout = make_layout(make_shape (Int<128>{}, Int<128>{}),
                        make_stride(Int<1>{}, Int<128>{}));

    TiledCopy copyA = make_tiled_copy(Copy_Atom<DefaultCopy, half_t>{},
                               Layout<Shape<_32,_8>, Stride<_1,_32>>{},
                               Layout<Shape< _4,_1>>{});
    TiledCopy copyB = make_tiled_copy(Copy_Atom<DefaultCopy, half_t>{},
                               Layout<Shape<_128,_2>, Stride<_2,_1>>{},
                               Layout<Shape< _1,_4>>{});
    TiledMMA mmaC = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                    Layout<Shape<_2, _4, _1>>{},
                                    Tile<_128,_128,_8>{});

    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
    dim3 dimBlock(256);
    gemm_vectorized_kernel<<<dimGrid, dimBlock>>>(prob_shape, cta_tiler,
                                     A, sA_layout, copyA,
                                     B, sB_layout, copyB,
                                     C, sC_layout, mmaC);
}
