#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

// 2048x2048x2048 gemm is 8.5 x 10^9 FLOPs
// shared memory per block is 48KB, which means 48*1024/4 = 12288 floats
// let each SM perform a 128x128 submatrix in the accumulator => 16x16 = 256 threadblocks in total
// at each SRAM load we do 128x128x32 so total SRAM need is 128x32x2x4 / 1024 = 32KB
// we are using 1024 threads per block so we are using 1024 threads to do 128x128x32 gemm
// so we are using 1024 threads to load 128x32x2 = 8192 numbers -> each threads load 8 numbers = 8*4 = 32
// using vectorized access each thread can load 4*4 = 16B numbers so need to do load twice
// at each load we need 1024*4*4 = 16KB registers

#define WIDTH 2048
#define OUTER_TILE_WIDTH 128
#define INNER_TILE_WIDTH 32

__global__ void basic_mm(float* A, float* B, float* C, int N) {
    const int OUTER_TILE_WIDTH = 128;
    const int INNER_TILE_WIDTH = 32;

    //global memory coalescing
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sA[OUTER_TILE_WIDTH*INNER_TILE_WIDTH];
    __shared__ float sB[OUTER_TILE_WIDTH*INNER_TILE_WIDTH];

    for (int tile = 0; tile < WIDTH/INNER_TILE_WIDTH; tile++) {
        //global -> shared
        //each thread load 4 numbers for each A,B, so 8 numbers in total
        reinterpret_cast<float4*>(sA)[add?] = reinterpret_cast<float4*>(A)[add?];
        __syncthreads();

        //shared -> register

    }

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i=0; i< N; i++){
            sum += A[row * N + i] * B[i*N + col];
        }
        C[row * N + col] = sum;
    }
}



int main() {
    const int N = 2048;
    // Assume column major
    // Allocate memory on the host
    thrust::host_vector<float> hA(N*N);
    thrust::host_vector<float> hB(N*N);
    thrust::host_vector<float> hC(N*N);

    // Initialize matrices h_A and h_B with data
    for (int i=0; i< N*N; i++){
        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }


    thrust::device_vector<float> dA = hA;
    thrust::device_vector<float> dB = hB;
    thrust::device_vector<float> dC = hC;

    //call mma
    //mma_atom<<<1,1>>>(dA.data().get(), dB.data().get(), dC.data().get());

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }


    hC = dC;
    printf("C = %f \n", hC[0]);

    return 0;
}