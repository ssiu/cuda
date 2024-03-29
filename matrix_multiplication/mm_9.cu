// matrix multiplication for 8192 x 8192

// global memory coalescing
// shared memory blocking
// register blocking
// vectorized memory load
// shared memory bank conflict

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
#define thread_id threadIdx.x
#define warp_id (threadIdx.x >> 5)
#define lane_id (threadIdx.x & 31)

// warp tiling
#define warp_row (warp_id >> 1) * 32
#define warp_col (warp_id & 1) * 64
#define thread_row (lane_id >> 3) * 4
#define thread_col (lane_id & 7) * 4


#define gC_row (TILE_WIDTH * blockIdx.y)
#define gC_col (TILE_WIDTH * blockIdx.x)

// shared memory offsets
#define sA_row (thread_id >> 1)
#define sA_col (thread_id & 1) * 4
#define sB_row (threadIdx.x >> 5)
#define sB_col (threadIdx.x & 31) * 4
//
#define gA_row (gC_row + sA_row)
#define gA_col (kBlock * BLOCK_WIDTH + sA_col)
#define gB_row (kBlock * BLOCK_WIDTH + sB_row)
#define gB_col (gC_col + sB_col)


__global__ void mm_8(float* A, float* B, float* C, int N){


    __shared__ float sA[TILE_WIDTH * BLOCK_WIDTH];
    __shared__ float sB[TILE_WIDTH * BLOCK_WIDTH];

    float tmp_original[4];
    float tmp_permuted[4];
    // fragments
    float fragment_A[8] = {};
    float fragment_B[8] = {};
    float accum[64] = {};

    for (int kBlock = 0; kBlock < N / BLOCK_WIDTH; kBlock++){

        //reinterpret_cast<float4*>(sA)[(sA_row * BLOCK_WIDTH + sA_col) >> 2] = reinterpret_cast<float4*>(A)[(gA_row * N + gA_col) >> 2];
        // no bank conflict for B
        reinterpret_cast<float4*>(sB)[(sB_row * TILE_WIDTH + sB_col) >> 2] = reinterpret_cast<float4*>(B)[(gB_row * N + gB_col) >> 2];

        // bank conflict for A, first load it to a tmp register then permute the data
        reinterpret_cast<float4*>(tmp_original)[0] = reinterpret_cast<float4*>(A)[(gA_row * N + gA_col) >> 2];
        #pragma unroll
        for (int i=0;i<4;i++) {
            tmp_permuted[(i + lane_id >> 3) & 3] = tmp_original[i];
        }
        reinterpret_cast<float4*>(sA)[(sA_row * BLOCK_WIDTH + sA_col) >> 2] = reinterpret_cast<float4*>(tmp_permuted)[0];



        __syncthreads();


        //load a fragment from shared memory to register
        for (int kFragment = 0; kFragment < BLOCK_WIDTH; kFragment++){

            #pragma unroll
            for (int i=0; i<4; i++){
                // Load A fragment, 8 floats
//                fragment_A[i] = sA[(warp_row + thread_row + i) * BLOCK_WIDTH + kFragment];
//                fragment_A[i+4] = sA[(warp_row + thread_row + 16 + i) * BLOCK_WIDTH + kFragment];

                // bank conflict free load
                // column shift resets every 16 rows
                // thread_row >> 2 gives us the permutation status
                //
                fragment_A[i] = sA[(warp_row + thread_row + i) * BLOCK_WIDTH + (thread_row >> 2 + kFragment & 3) & 3 + (kFragment >> 2) * 4];
                fragment_A[i+4] = sA[(warp_row + thread_row + 16 + i) * BLOCK_WIDTH + (thread_row >> 2 + kFragment & 3) & 3 + (kFragment >> 2) * 4];
                // Load B fragment, 8 floats
                fragment_B[i] = sB[kFragment * TILE_WIDTH + warp_col + thread_col + i];
                fragment_B[i+4] = sB[kFragment * TILE_WIDTH + warp_col + thread_col + 32 + i];
              }


            // Compute accumulator, 64 floats
            #pragma unroll
            for (int x=0; x<8; x++){
                #pragma unroll
                for (int y=0; y<8; y++){
                    accum[x * 8 + y] += fragment_A[x] * fragment_B[y];
                }
            }

        }
        __syncthreads();
    }
    #pragma unroll
    for (int x=0; x<4; x+=1){
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x ) * N + gC_col + warp_col + thread_col) >> 2] = reinterpret_cast<float4*>(accum)[(x * 8) /4];
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x ) * N + gC_col + warp_col + thread_col + 32) >> 2] = reinterpret_cast<float4*>(accum)[(x * 8 + 4) /4];
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x + 16) * N + gC_col + warp_col + thread_col) >> 2] = reinterpret_cast<float4*>(accum)[((x + 4) * 8) /4];
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x + 16) * N + gC_col + warp_col + thread_col + 32) >> 2] = reinterpret_cast<float4*>(accum)[((x + 4) * 8 + 4) /4];
    }

}