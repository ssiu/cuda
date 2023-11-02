// naive kernel where each thread computes a single value
#include <iostream>
//#include "cutlass/cutlass.h"
#include <cuda_fp16.h>




__global__ void matrix_multiplication(float* A, float* B, float* C) {
    int idx = threadIdx.x;
    printf("thread id %d\n", idx);
    //int col = blockIdx.y * blockDim.y + threadIdx.y;


//struct SM70_8x8x4_F32F16F16F32_TN
//{
//  using DRegisters = float[8];
//  using ARegisters = uint32_t[2];
//  using BRegisters = uint32_t[2];
//  using CRegisters = float[8];
//
//  // Register asm fma
//  CUTE_HOST_DEVICE static void
//  fma(float         & d0, float         & d1, float      & d2, float      & d3,
//      float         & d4, float         & d5, float      & d6, float      & d7,
//      uint32_t const& a0, uint32_t const& a1,
//      uint32_t const& b0, uint32_t const& b1,
//      float    const& c0, float    const& c1, float const& c2, float const& c3,
//      float    const& c4, float    const& c5, float const& c6, float const& c7)
//  {
//#if defined(CUTE_ARCH_MMA_SM70_ENABLED)
//    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
//                 "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
//                 "{%8,  %9},"
//                 "{%10, %11},"
//                 "{%12, %13, %14, %15, %16, %17, %18, %19};\n"
//        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3),
//          "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
//        :  "r"(a0),  "r"(a1),
//           "r"(b0),  "r"(b1),
//           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
//           "f"(c4),  "f"(c5),  "f"(c6),  "f"(c7));
//#else
//    CUTE_RUNTIME_ASSERT("Attempting to use SM70_8x8x4_F32F16F16F32_TN without CUTE_ARCH_MMA_SM70_ENABLED");
//#endif
//  }
//};
}

int main() {

    const int M = 8;
    const int N = 8;
    const int K = 4;


    // Allocate memory on the host
    half *h_A = (half*)malloc(M*K*sizeof(half));
    half *h_B = (half*)malloc(N*K*sizeof(half));
    float *h_C = (float*)malloc(M*N*sizeof(float));

    // Initialize matrices h_A and h_B with data
    for (int i=0; i< M*K; i++){
        h_A[i] = __float2half(1.0f);
    }

    for (int i=0; i< N*K; i++){
        h_A[i] = __float2half(1.0f);
    }

    for (int i=0; i< M*N; i++){
        h_C[i] = 0.0f;
    }

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M*K*sizeof(half));
    cudaMalloc((void**)&d_B, N*K*sizeof(half));
    cudaMalloc((void**)&d_C, M*N*sizeof(float));

    // Copy matrices h_A and h_B from host to device
    cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*K*sizeof(half), cudaMemcpyHostToDevice);

    // Define the grid and block dimensions for the kernel launch
    dim3 dimGrid(1); // You can adjust this based on your GPU's capability
    dim3 dimBlock(32);

    // Launch the matrix multiplication kernel
    matrix_multiplication<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Copy the result matrix d_C from device to host
    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);


    return 0;
}