// matrix multiplication for 8192 x 8192 x 8192

// global memory coalescing
// shared memory blocking
// register blocking
// vectorized memory load
// global + shared memory pipelining

// https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/

//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/11_kernel_double_buffering.cuh


// thread-tiling: each thread loads 8+8 = 16 floats and computes 8x8 = 64 results
// warp-tiling: each warp computes 64x32 = 2048 results
// block-tiling: each thread block has 2x4 = 8 warps = 256 threads computing 128x128 = 16384 results

// shared memory:
// 128(M) * 8(N) * 4(size) * 2(A,B) = 16KB
// registers:
// each thread needs at least 64 * 4 = 256B
// so a threadblock needs at least 256 * 256 = 64 KB

// dim3 dimGrid(16, 16);
// dim3 dimBlock(256, 1);


#include <iostream>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include "mm.cuh"
#include "utils.cuh"

#include <iostream>
#define BLOCK_WIDTH 8
#define TILE_WIDTH 128
#define thread_id threadIdx.x
#define warp_id threadIdx.x / 32
#define lane_id threadIdx.x % 32

// warp tiling
#define warp_row (warp_id / 2) * 32
#define warp_col (warp_id % 2) * 64
#define thread_row lane_id / 8
#define thread_col (lane_id % 8) * 4


#define gC_row TILE_WIDTH * blockIdx.y
#define gC_col TILE_WIDTH * blockIdx.x

// shared memory offsets
#define sA_row thread_id / 2
#define sA_col (thread_id % 2) * 4
#define sB_row threadIdx.x / 32
#define sB_col (threadIdx.x % 32) * 4
//
#define gA_row (gC_row + sA_row)
#define gA_col ((kBlock + 1) * BLOCK_WIDTH + sA_col)
#define gB_row ((kBlock + 1) * BLOCK_WIDTH + sB_row)
#define gB_col (gC_col + sB_col)


__global__ void kernel_test(float* A, float* B, float* C, int N){
    printf("thread id is %d, warp id is %d ", threadIdx.x, threadIdx.x >> 5);

}


#define TILE_WIDTH 128
int main(){
    int N = 128;

    thrust::host_vector<float> hA = generateMatrices(N);
    thrust::host_vector<float> hB = generateMatrices(N);
    thrust::host_vector<float> hC(N*N);

    thrust::device_vector<float> dA = hA;
    thrust::device_vector<float> dB = hB;
    thrust::device_vector<float> dC = hC;



    dim3 dimGrid(1, 1);
    dim3 dimBlock(256, 1);
    kernel_test<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                                   thrust::raw_pointer_cast(dC.data()), N);


    return 0;
}