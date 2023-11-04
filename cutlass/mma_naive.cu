// naive kernel where each thread computes a single value
#include <iostream>
//#include "cutlass/cutlass.h"
#include <cuda_fp16.h>

const int M = 8;
const int N = 8;
const int K = 4;


__global__ void mma_test(float* C) {
    //matrix multiplication of a single quadpair
    //threads 0-3, 16-19
    int idx = threadIdx.x;

    //
    // matrix A fragments
    //
    if (idx < 4 or (idx >= 16 and idx < 20)){
        uint a[2] = { 0 };
        half* a_16 = reinterpret_cast<half*>(a);
        for (int i = 0; i< K; i++){
            a_16[i] = 1.0;
        }
        //
        // matrix B fragments
        //
        uint b[2] = { 0 };
        half* b_16 = reinterpret_cast<half*>(b);
        for (int i = 0; i < K; i++){
            b_16[i] = 1.0;
        }
        //
        // matrix C fragments
        //
        float c[8] = {0.0f};

        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
                     "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
                     "{%8,  %9},"
                     "{%10, %11},"
                     "{%12, %13, %14, %15, %16, %17, %18, %19};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3]),
              "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
            : "r"(a[0]),  "r"(a[1]),
              "r"(b[0]),  "r"(b[1]),
              "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3]),
              "f"(c[4]),  "f"(c[5]),  "f"(c[6]),  "f"(c[7]));

        for (int i = 0; i < 8; i++) {
            int row = (idx & 0b1) + (i & 0b10) + (i & 0b10000);
            int column = (i & 0b100) + (idx & 0b10) + (i & 0b1);
            printf("thread id %d, (%d, %d)\n", idx, row, column);
            C[row*8 + column] = c[i];
        }
    }
}

//https://forums.developer.nvidia.com/t/wrong-answer-with-mma-sync-aligned-m8n8k4/248442
__global__ void matrix_multiplication(half* A, half* B, float* C) {
    //matrix multiplication of a single quadpair
    //threads 0-3, 16-19
    int idx = threadIdx.x;

    //
    // matrix A fragments
    //
    if (idx < 4 or (idx >= 16 and idx < 20)){
        uint a[2] = { 0 };
        half* a_16 = reinterpret_cast<half*>(a);
        for (int i = 0; i< K; i++){
            a_16[i] = A[M*idx + i];
        }
        //
        // matrix B fragments
        //
        uint b[2] = { 0 };
        half* b_16 = reinterpret_cast<half*>(b);
        for (int i = 0; i < K; i++){
            b_16[i] = B[N*i + idx];
        }
        //
        // matrix C fragments
        //
        float c[8] = {0.0f};

        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
                     "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
                     "{%8,  %9},"
                     "{%10, %11},"
                     "{%12, %13, %14, %15, %16, %17, %18, %19};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3]),
              "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
            : "r"(a[0]),  "r"(a[1]),
              "r"(b[0]),  "r"(b[1]),
              "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3]),
              "f"(c[4]),  "f"(c[5]),  "f"(c[6]),  "f"(c[7]));

        for (int i = 0; i < 8; i++){
            int row = (idx & 0b1) + (i & 0b10);
            int column = (i & 0b100) + (idx & 0b10) + (i & 0b1);
            //printf("thread id %d, (%d, %d)\n", idx, row, column);
            C[row*8 + column] = c[i];
        }
    }




//        uint MultiB[2] = { 0 };
//
//        half* test1 = reinterpret_cast<half*>(MultiA);
//        half* test2 = reinterpret_cast<half*>(MultiB);
//        test1[0] = 0.8;
//        test1[1] = 0.8;
//        test1[2] = 0.8;
//        test1[3] = 0.8;
//        test2[0] = 0.7;
//        test2[1] = 0.7;
//        test2[2] = 0.7;
//        test2[3] = 0.7;

    //int col = blockIdx.y * blockDim.y + threadIdx.y;
    // need to figure out how to store two half into a single float

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

//    for (int i=0; i< M*N; i++){
//        h_C[i] = 0.0f;
//    }

    // Allocate memory on the device
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc((void**)&d_A, M*K*sizeof(half));
    cudaMalloc((void**)&d_B, N*K*sizeof(half));
    cudaMalloc((void**)&d_C, M*N*sizeof(float));

    // Copy matrices h_A and h_B from host to device
    cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*K*sizeof(half), cudaMemcpyHostToDevice);

    // Define the grid and block dimensions for the kernel launch
//    dim3 dimGrid(1, 1); // You can adjust this based on your GPU's capability
//    dim3 dimBlock(32, 32);

    // Launch the matrix multiplication kernel
    //matrix_multiplication<<<1, 32>>>(d_A, d_B, d_C);
    mma_test<<<1, 32>>>(d_C);


    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }

    // Copy the result matrix d_C from device to host
    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i< M; i++){
        for (int j=0; j< N; j++){
            std::cout << h_C[i*N+j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);


    return 0;
}