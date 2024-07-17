#include <cublas_v2.h>

void mm_cublas(float* A, float* B, float* C, int N) {
    float alpha = 1.0f;
    float beta = 1.0f;

    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B, N, A, N, &beta, C, N);

    cublasDestroy(handle);
}

