// using retile_D

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
    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);

    auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<DefaultCopy, half_t>{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(sA);                  // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)


    auto smem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<DefaultCopy, half_t>{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(sB);                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)





//     Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K)
//     Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K)
//     Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

    // Allocate the accumulators -- same size as the projected data
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

    //printf("tCrC: %f\n", tCrC[0]);
    clear(tCrC);

    copy(copy_a, tAgA, tAsA);
    copy(copy_b, tBgB, tBsB);

    __syncthreads();

    copy(smem_tiled_copy_A, tCsA, tCrA_copy_view);
    copy(smem_tiled_copy_B, tCsB, tCrB_copy_view);

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
        if(thread(0)) {
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

    dim3 dimGrid(1);
    dim3 dimBlock(256);
    mm_kernel<<<dimGrid, dimBlock>>>(A, sA_layout, copyA,
                                     B, sB_layout, copyB,
                                     C, sC_layout, mmaC);
}



void mm_cublas(half_t* A, half_t* B, float* C,
                int M, int N, int K) {
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context
    cublasCreate(&handle);
    //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 16, 8, 8, &alpha, A, 16, B, 8, &beta, C, 16);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                            A, CUDA_R_16F, M,
                            B, CUDA_R_16F, K, &beta,
                            C, CUDA_R_32F, M,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cublasDestroy(handle);
}


void mm_cpu(float* A, float* B, float* C,
            int M, int N, int K) {
    for (int k = 0; k < K; k ++) {
        for (int i=0; i< M; i++) {
            for (int j=0; j < N; j++) {
                C[i + M * j] += A[i + M * k] * B[k + K * j];
            }
        }
    }
}

int main(int argc, char** argv)
{
    int m = 128;
    int n = 128;
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
        //h_A[j] = static_cast<TA>(j);
        h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
//         if (j==16) {
//             h_A[j] = static_cast<TA>(1);
//         } else {
//             h_A[j] = static_cast<TA>(0);
//         }
        h_A_cpu[j] = static_cast<float>(h_A[j]);
    }
    for (int j = 0; j < n*k; ++j) {
        //h_B[j] = static_cast<TB>(j);
//         if (j==1) {
//             h_B[j] = static_cast<TB>(1);
//         } else {
//             h_B[j] = static_cast<TB>(0);
//         }
        h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
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
    mm_cublas(d_A.data().get(), d_B.data().get(), d_C_cublas.data().get(), m, n, k);
    mm_cpu(h_A_cpu.data(), h_B_cpu.data(), h_C_cpu.data(), m, n, k);
//
//     thrust::host_vector<TC> h_C_result = d_C;
//     thrust::host_vector<TC> h_C_cublas_result = d_C_cublas;
    h_C = d_C;
    h_C_cublas = d_C_cublas;


    if (isSameMatrices(h_C.data(), h_C_cpu.data(), m, n) && isSameMatrices(h_C.data(), h_C_cublas.data(), m, n)) {
        printf("Correct answer\n");
    } else {
        printf("Wrong answer\n");
    }

    #if 0
        for (int i=0;i< 16; i++) {
            for (int j=0;j<8;j++) {
                printf("%f ", h_C_result[i + 16 * j]);
            }
            printf("\n");
        }
    #endif

    #if 0
        for (int i=0; i<128; i++){
        //for (int i=0; i<32; i++){
            printf("i = %d, cutlass = %f, cublas = %f, cpu = %f\n", i, h_C_result[i], h_C_cublas_result[i], h_C_cpu[i]);
        }
    #endif


    return 0;
}
