// matrix multiplication for 2048 x 2048

// global memory coalescing
// shared memory blocking
// register blocking
// avoid shared memory bank conflicts
// vectorized memory load


// https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/

// thread-tiling: each thread loads 8+8 = 16 floats and computes 8x8 = 64 results
// warp-tiling: each warp computes 64x32 = 2048 results
// block-tiling: each thread block has 2x4 = 8 warps = 256 threads computing 128x128 = 16384 results

// shared memory:
// 128 * 8 * 4 * 2 = 8KB
// registers:
// each thread needs at least 64 * 4 = 256B
// so a threadblock needs at least 256 * 256 = 64 KB

// dim3 dimGrid(16, 16);
// dim3 dimBlock(256, 1);

#include <iostream>
#define BLOCK_WIDTH 8
#define TILE_WIDTH 128


__global__ void mm_7(float* A, float* B, float* C, int N){
    int thread_id = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int warp_row = (warp_id / 2) * 32;
    int warp_col = (warp_id % 2) * 64;
    int thread_row = lane_id / 8;
    int thread_col = (lane_id % 8) * 4;

    // offset for output matrix C
    int gC_row =  TILE_WIDTH * blockIdx.y;
    int gC_col =  TILE_WIDTH * blockIdx.x;

    int gA_row;
    int gA_col;
    int gB_row;
    int gB_col;

    int sA_row;
    int sA_col;
    int sB_row;
    int sB_col;

    __shared__ float sA[TILE_WIDTH * BLOCK_WIDTH];
    __shared__ float sB[TILE_WIDTH * BLOCK_WIDTH];

    // fragments
    float fragment_A[8] = {};
    float fragment_B[8] = {};
    float accum[64] = {};

    for (int kBlock = 0; kBlock < N / BLOCK_WIDTH; kBlock++){
        sA_row = thread_id / 2;
        sA_col = (thread_id % 2) * 4;
        sB_row = threadIdx.x / 32;
        sB_col = (threadIdx.x % 32) * 4;

        gA_row = gC_row + sA_row;
        gA_col = kBlock * BLOCK_WIDTH + sA_col;
        gB_row = kBlock * BLOCK_WIDTH + sB_row;
        gB_col = gC_col + sB_col;
        // each thread loads 16 floats for each A, B, broken into 16 loads
        // load A, B tile into shared memory
        // each thread load every 8th row
        #pragma unroll

        for (int i=0; i<4; i+=1) {
            // load shared memory A
            sA[sA_row * BLOCK_WIDTH + sA_col + i] = A[gA_row * N + gA_col + i];
            sB[sB_row * TILE_WIDTH + sB_col + i] = B[gB_row * N + gB_col + i];
        }

        __syncthreads();


        //load a fragment from shared memory to register
        for (int kFragment = 0; kFragment < BLOCK_WIDTH; kFragment++){

            // Load A fragment, 8 floats
            #pragma unroll
            for (int i=0; i<4; i++){
                fragment_A[i] = sA[(warp_row + thread_row + i) * BLOCK_WIDTH + kFragment];
                fragment_A[i+4] = sA[(warp_row + thread_row + 16 + i) * BLOCK_WIDTH + kFragment];
            }

            // Load B fragment, 8 floats
            #pragma unroll
            for (int i=0; i<4; i++){
                fragment_B[i] = sB[kFragment * TILE_WIDTH + warp_col + thread_col + i];
                fragment_B[i+4] = sB[kFragment * TILE_WIDTH + warp_col + thread_col + 32 + i];
            }

            // Compute accumulator, 64 floats
            #pragma unroll
            for (int x=0; x<8; x++){
                for (int y=0; y<8; y++){
                    accum[x * 8 + y] += fragment_A[x] * fragment_B[y];
                }
            }

        }
        __syncthreads();
    }
    // non-vectorized
    for (int x=0; x<4; x+=1){
        for (int y=0; y<4; y+=1){
            C[(gC_row + warp_row + thread_row + x ) * N + gC_col + warp_col + thread_col + y] = accum[x*8 + y];
            C[(gC_row + warp_row + thread_row + x ) * N + gC_col + warp_col + thread_col + y + 32] = accum[x * 8 + y + 4];
            C[(gC_row + warp_row + thread_row + x + 16) * N + gC_col + warp_col + thread_col + y] = accum[(x + 4)* 8 + y];
            C[(gC_row + warp_row + thread_row + x + 16) * N + gC_col + warp_col + thread_col + y + 32] = accum[(x + 4) * 8 + y + 4];
        }
    }

    //vectorized
    // reinterpret_cast<float2*>(d_out)[i]
//    for (int x=0; x<8; x+=1){
//        reinterpret_cast<float4*>(C)[(gC_row + warp_row + thread_row + x * 4) * N + (gC_col + warp_col + thread_col) / 4] = reinterpret_cast<float4*>(accum)[x*8];
//        reinterpret_cast<float4*>(C)[(gC_row + warp_row + thread_row + x * 4) * N + (gC_col + warp_col + thread_col + 32) / 4] = reinterpret_cast<float4*>(accum)[x * 8 + 1];
//    }

}