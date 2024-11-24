#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "utils.cuh"

using namespace cute;

template <class AgmemLayout, class ASmemLayout, class TiledCopyA,
          class BgmemLayout, class BSmemLayout, class TiledCopyB,
          class CgmemLayout, class CSmemLayout, class TiledMma>
__global__ void mm_kernel(
           half_t* A, AgmemLayout gA_layout, ASmemLayout sA_layout, TiledCopyA copy_a,
           half_t* B, BgmemLayout gB_layout, BSmemLayout sB_layout, TiledCopyB copy_b,
           float*  C, CgmemLayout gC_layout, CSmemLayout sC_layout, TiledMma      mma)
{

    Tensor gA = make_tensor(make_gmem_ptr(A), gA_layout);
    Tensor gB = make_tensor(make_gmem_ptr(B), gB_layout);
    Tensor gC = make_tensor(make_gmem_ptr(C), gC_layout);

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
    Tensor tCrA = thr_mma.partition_fragment_A(sA);                               // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

    // Allocate the accumulators -- same size as the projected data
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);


    auto s2r_tiled_copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x1_LDSM_N, half_t>{}, mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(threadIdx.x);
    auto s2r_tAsA = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_copy_view = s2r_thr_copy_a.retile_D(tCrA);

    auto s2r_tiled_copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x1_LDSM_N, half_t>{}, mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(threadIdx.x);
    auto s2r_tBsB = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_copy_view = s2r_thr_copy_b.retile_D(tCrB);


    //printf("tCrC: %f\n", tCrC[0]);
    clear(tCrC);

    copy(copy_a, tAgA, tAsA);
    copy(copy_b, tBgB, tBsB);

    __syncthreads();

    if (thread0()) {
        print_tensor(sA);
        print_tensor(sB);
    }


    copy(s2r_tiled_copy_a, s2r_tAsA, tCrA_copy_view);
    copy(s2r_tiled_copy_b, s2r_tBsB, tCrB_copy_view);
//     copy(s2r_tiled_copy_a, s2r_tAsA, tCrA);
//     copy(s2r_tiled_copy_b, s2r_tBsB, tCrB);

    gemm(mma, tCrA, tCrB, tCrC);

    axpby(1.0f, tCrC, 0.0f, tCgC); //test

    #if 0
        if(thread0()) {
        print("  gA : "); print(  gA); print("\n");
        print("  sA : "); print(  sA); print("\n");
        print("tAgA : "); print(tAgA); print("\n");
        print("tAsA : "); print(tAsA); print("\n");

        }
    #endif

    #if 0
        if(thread0()) {
        print("  gB : "); print(  gB); print("\n");
        print("  sB : "); print(  sB); print("\n");
        print("tBgB : "); print(tBgB); print("\n");
        print("tBsB : "); print(tBsB); print("\n");
        }
    #endif

    #if 0
        if(thread(1)) {
            print("  gC : "); print(  gC); print("\n");
            print("tCsA : "); print(tCsA); print("\n");
            print("tCsB : "); print(tCsB); print("\n");
            print("tCgC : "); print(tCgC); print("\n");
            print("tCrC : "); print(tCrC); print("\n");
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


void mm(half_t* A, half_t* B, float* C) {

    auto gA_layout = make_layout(make_shape (Int<16>{}, Int<8>{}),
                        make_stride(Int<1>{}, Int<16>{}));

    auto sA_layout = make_layout(make_shape (Int<16>{}, Int<8>{}),
                        make_stride(Int<8>{}, Int<1>{}));

    auto gB_layout = make_layout(make_shape (Int<8>{}, Int<8>{}),
                        make_stride(Int<8>{}, Int<1>{}));

    auto sB_layout = make_layout(make_shape (Int<8>{}, Int<8>{}),
                        make_stride(Int<8>{}, Int<1>{}));

//     using sB_layout = decltype(composition(Swizzle<1, 1, 1>{},
//                                  make_layout(make_shape (Int<8>{}, Int<8>{}),
//                                  make_stride(Int<1>{}, Int<8>{}))));

    auto sC_layout = make_layout(make_shape (Int<16>{}, Int<8>{}),
                        make_stride(Int<1>{}, Int<16>{}));

    TiledCopy copyA = make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<16>, half_t>{},
                                     Layout<Shape<_4,_8>, Stride<_1,_4>>{},
                                     Layout<Shape< _4,_1>>{});
    TiledCopy copyB = make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<16>, half_t>{},
                                     Layout<Shape<_4,_8>, Stride<_1,_4>>{},
                                     Layout<Shape< _2,_1>>{});

    TiledMMA mmaC = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{});

    dim3 dimGrid(1);
    dim3 dimBlock(32);
    mm_kernel<<<dimGrid, dimBlock>>>(A, gA_layout, sA_layout, copyA,
                                     B, gB_layout, sB_layout, copyB,
                                     C, sC_layout, sC_layout, mmaC);
}



void mm_cublas(half_t* A, half_t* B, float* C) {
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context
    cublasCreate(&handle);
    //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 16, 8, 8, &alpha, A, 16, B, 8, &beta, C, 16);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 16, 8, 8, &alpha,
                            A, CUDA_R_16F, 16,
                            B, CUDA_R_16F, 8, &beta,
                            C, CUDA_R_32F, 16,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cublasDestroy(handle);
}


void mm_cpu(float* A, float* B, float* C) {
    for (int k = 0; k < 8; k ++) {
        for (int i=0; i< 16; i++) {
            for (int j=0; j < 8; j++) {
                C[i + 16 * j] += A[i + 16 * k] * B[k + 8 * j];
            }
        }
    }
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
    thrust::host_vector<TC> h_C_cublas(m*n);

    thrust::host_vector<TC> h_A_cpu(m*k);
    thrust::host_vector<TC> h_B_cpu(n*k);
    thrust::host_vector<TC> h_C_cpu(m*n);

    for (int j = 0; j < m*k; ++j) {
        //h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
        h_A[j] = static_cast<TA>(j);
        h_A_cpu[j] = static_cast<float>(h_A[j]);
    }
    for (int j = 0; j < n*k; ++j) {
        //h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
        h_B[j] = static_cast<TB>(j);
        h_B_cpu[j] = static_cast<float>(h_B[j]);
    }
    for (int j = 0; j < m*n; ++j) {
        h_C[j] = static_cast<TC>(0);
        h_C_cublas[j] = static_cast<TC>(0);
        h_C_cpu[j] = static_cast<TC>(0);
    }

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;
    thrust::device_vector<TC> d_C_cublas = h_C_cublas;

    mm(d_A.data().get(), d_B.data().get(), d_C.data().get());
    mm_cublas(d_A.data().get(), d_B.data().get(), d_C_cublas.data().get());
    mm_cpu(h_A_cpu.data(), h_B_cpu.data(), h_C_cpu.data());

    h_C = d_C;
    h_C_cublas = d_C_cublas;


    if (isSameMatrices(h_C.data(), h_C_cpu.data(), m, n) && isSameMatrices(h_C.data(), h_C_cublas.data(), m, n)) {
        printf("Correct answer\n");
    } else {
        printf("Wrong answer\n");
    }

    printf("cutlass  : \n");
    for (int i = 0; i < 16; i++){
        for (int j=0;j<8;j++){
            printf("%2.4f ", h_C[i*8+j]);
        }
        printf("\n");
    }

    printf("==========\n");
    printf("cublas : \n");
    for (int i = 0; i < 16; i++){
        for (int j=0;j<8;j++){
            printf("%2.4f ", h_C_cublas[i*8+j]);
        }
        printf("\n");
    }


    return 0;
}