
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

using namespace cute;

template <class ASmemLayout, class TiledCopyA,
          class BSmemLayout, class TiledCopyB,
          class CSmemLayout, class TiledMma>
__global__ void mm_kernel(
           half_t* A, ASmemLayout sA_layout, TiledCopyA copy_a,
           half_t* B, BSmemLayout sB_layout, TiledCopyB copy_b,
           float*  C, CSmemLayout sC_layout, TiledMma      mma)
{

    Tensor gA = make_tensor(make_gmem_ptr(A), sA_layout);
    Tensor gB = make_tensor(make_gmem_ptr(B), sB_layout);
    Tensor gC = make_tensor(make_gmem_ptr(C), sC_layout);

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

    copy(copy_a, tAgA, tAsA);
    copy(copy_b, tBgB, tBsB);

    __syncthreads();

    #if 1
        if(thread0()) {
        print("  gA : "); print(  gA); print("\n");
        print("  sA : "); print(  sA); print("\n");
        print("tAgA : "); print(tAgA); print("\n");
        print("tAsA : "); print(tAsA); print("\n");
        }
    #endif

    #if 1
        if(thread0()) {
        print("  gB : "); print(  gB); print("\n");
        print("  sB : "); print(  sB); print("\n");
        print("tBgB : "); print(tBgB); print("\n");
        print("tBsB : "); print(tBsB); print("\n");
        }
    #endif

    #if 1
        if(thread0()) {
        print("  gC : "); print(  gC); print("\n");
        print("tCsA : "); print(tCsA); print("\n");
        print("tCsB : "); print(tCsB); print("\n");
        print("tCgC : "); print(tCgC); print("\n");
        print("tCrC : "); print(tCrC); print("\n");
        }
    #endif

    //gemm(mma, tCsA, tCsB, tCrC);

    axpby(1.0f, tCrC, 0.0f, tCgC);

    C[0] = 0.0f;
}


void mm(half_t* A, half_t* B, float* C) {

    auto sA_layout = make_layout(make_shape (Int<16>{}, Int<8>{}),
                        make_stride(Int<1>{}, Int<16>{}));
    auto sB_layout = make_layout(make_shape (Int<8>{}, Int<8>{}),
                        make_stride(Int<1>{}, Int<8>{}));
    auto sC_layout = make_layout(make_shape (Int<16>{}, Int<8>{}),
                        make_stride(Int<1>{}, Int<16>{}));

    TiledCopy copyA = make_tiled_copy(Copy_Atom<DefaultCopy, half_t>{},
                                     Layout<Shape<_4,_8>, Stride<_1,_4>>{},
                                     Layout<Shape< _4,_1>>{});
    TiledCopy copyB = make_tiled_copy(Copy_Atom<DefaultCopy, half_t>{},
                                     Layout<Shape<_4,_8>, Stride<_1,_4>>{},
                                     Layout<Shape< _2,_1>>{});

    TiledMMA mmaC = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{});

    dim3 dimGrid(1);
    dim3 dimBlock(32);
    mm_kernel<<<dimGrid, dimBlock>>>(A, sA_layout, copyA,
                                     B, sB_layout, copyB,
                                     C, sC_layout, mmaC);
}

int main(int argc, char** argv)
{
    int m = 16;
    int n = 8;
    int k = 8;

    using TA = half_t;
    using TB = half_t;
    using TC = float;

    cute::device_init(0);

    thrust::host_vector<TA> h_A(m*k);
    thrust::host_vector<TB> h_B(n*k);
    thrust::host_vector<TC> h_C(m*n);

    for (int j = 0; j < m*k; ++j) {
        h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
        //printf("%f\n", static_cast<float>(h_A[j]));
    }
    for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;


    mm(d_A.data().get(), d_B.data().get(), d_C.data().get());

    thrust::host_vector<TC> cute_result = d_C;



    return 0;
}