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
__global__ void gemm_ldsm_kernel(
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
    Tensor tCrA = thr_mma.partition_fragment_A(sA);                               // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

    // Allocate the accumulators -- same size as the projected data
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);


    auto s2r_tiled_copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(threadIdx.x);
    auto s2r_tAsA = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_copy_view = s2r_thr_copy_a.retile_D(tCrA);

    auto s2r_tiled_copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(threadIdx.x);
    auto s2r_tBsB = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_copy_view = s2r_thr_copy_b.retile_D(tCrB);


    #if 1
        if(thread0()) {
            print("tCrA : "); print(  tCrA); print("\n");
            print("tCrA_copy_view : "); print(  tCrA_copy_view); print("\n");
        }
    #endif


    //printf("tCrC: %f\n", tCrC[0]);
    clear(tCrC);

    auto K_TILE_MAX = size<3>(tAgA);

    for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile++)
    {

        copy(copy_a, tAgA(_,_,_,k_tile), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

        copy(tArA, tAsA);
        copy(tBrB, tBsB);

        __syncthreads();

        // Compute gemm on mma-partitioned smem
        //gemm(mma, tCsA, tCsB, tCrC);

        copy(s2r_tiled_copy_a, s2r_tAsA, tCrA_copy_view);
        copy(s2r_tiled_copy_b, s2r_tBsB, tCrB_copy_view);
    //     copy(s2r_tiled_copy_a, s2r_tAsA, tCrA);
    //     copy(s2r_tiled_copy_b, s2r_tBsB, tCrB);

        gemm(mma, tCrA, tCrB, tCrC);

        __syncthreads();

    }

    //axpby(1.0f, tCrC, 0.0f, tCgC); //ldsm
    copy(tCrC, tCgC);

}


void gemm_ldsm(half_t* A, half_t* B, float* C, int m, int n, int k) {

    auto prob_shape = make_shape(m, n, k);

    auto dA = make_stride(Int<1>{}, m);                      // (dM, dK)
    auto dB = make_stride(k, Int<1>{});                      // (dN, dK)
    auto dC = make_stride(Int<1>{}, m);                      // (dM, dN)
//     printf("%d\n", prob_shape[1]);
//     printf("%d\n", prob_shape[2]);
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<  8>{};
    auto cta_tiler = make_shape(bM, bN, bK);


//     auto sA_layout = make_layout(make_shape (Int<128>{}, Int<8>{}),
//                         make_stride(Int<8>{}, Int<1>{}));
//     auto sB_layout = make_layout(make_shape (Int<128>{}, Int<8>{}),
//                         make_stride(Int<8>{}, Int<1>{}));
    auto sA_layout = composition(Swizzle<2, 3, 3>{},
                                Layout<Shape<_128, _8>,
                                Stride<_8, _1>>{});
    auto sB_layout = composition(Swizzle<2, 3, 3>{},
                                Layout<Shape<_128, _8>,
                                Stride<_8, _1>>{});

    auto sC_layout = make_layout(make_shape (Int<128>{}, Int<128>{}),
                        make_stride(Int<1>{}, Int<128>{}));

    TiledCopy copyA = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, half_t>{},
                               Layout<Shape<_16,_8>, Stride<_1,_16>>{},
                               Layout<Shape< _8,_1>>{});
    TiledCopy copyB = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, half_t>{},
                               Layout<Shape<_128,_1>, Stride<_1,_0>>{},
                               Layout<Shape< _1,_8>>{});

    TiledMMA mmaC = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                    Layout<Shape<_2, _2, _1>>{},
                                    Tile<_128,_128,_8>{});

    dim3 dimGrid(size(ceil_div(m, bM)), size(ceil_div(n, bN)));
    dim3 dimBlock(128);
    gemm_ldsm_kernel<<<dimGrid, dimBlock>>>(prob_shape, cta_tiler,
                                                     A, dA, sA_layout, copyA,
                                                     B, dB, sB_layout, copyB,
                                                     C, dC, sC_layout, mmaC);
}

