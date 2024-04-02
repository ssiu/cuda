// matrix multiplication for 8192 x 8192

// global memory coalescing
// shared memory blocking
// register blocking
// vectorized memory load
// shared memory bank conflict
// double buffering
//    block 0               block 1             block 0            block 2              block 1          block N / BLOCK_WIDTH - 1   block N / BLOCK_WIDTH - 2       block N / BLOCK_WIDTH - 1
// global -> shared1  | global -> shared2, shared1 -> FFMA | global -> shared 1, shared 2 -> FFMA | ... |     global -> shared     ,      shared -> FFMA         |        shared -> FFMA


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
#define warp_id threadIdx.x / 32
#define lane_id threadIdx.x % 32

// warp tiling
#define warp_row (warp_id / 2) * 32
#define warp_col (warp_id % 2) * 64
#define thread_row (lane_id / 8) * 4
#define thread_col (lane_id % 8) * 4


#define gC_row (TILE_WIDTH * blockIdx.y)
#define gC_col (TILE_WIDTH * blockIdx.x)

// shared memory offsets
#define sA_row (thread_id / 2)
#define sA_col (thread_id % 2) * 4
#define sB_row (threadIdx.x / 32)
#define sB_col (threadIdx.x % 32) * 4
//
#define gA_row (gC_row + sA_row)
#define gA_col ((kBlock + 1) * BLOCK_WIDTH + sA_col)
#define gB_row ((kBlock + 1) * BLOCK_WIDTH + sB_row)
#define gB_col (gC_col + sB_col)


__global__ void //__launch_bounds__(256, 2)
mm_9(float* A, float* B, float* C, int N){


    __shared__ float sA[2][TILE_WIDTH * BLOCK_WIDTH];
    __shared__ float sB[2][TILE_WIDTH * BLOCK_WIDTH];

    float tmp_original[4];
    float tmp_permuted[4];
    // fragments
    float fragment_A[8] = {};
    float fragment_B[8] = {};
    float accum[64] = {};


    // prologue
    //reinterpret_cast<float4*>(sA)[(sA_row * BLOCK_WIDTH + sA_col) / 4] = reinterpret_cast<float4*>(A)[(gA_row * N + gA_col) / 4];
    // no bank conflict for B
    reinterpret_cast<float4*>(sB[0])[(sB_row * TILE_WIDTH + sB_col) / 4] = reinterpret_cast<float4*>(B)[(sB_row * N + gB_col) / 4];

    // bank conflict for A, first load it to a tmp register then permute the data
    reinterpret_cast<float4*>(tmp_original)[0] = reinterpret_cast<float4*>(A)[(gA_row * N + sA_col) / 4];
    #pragma unroll
    for (int i=0;i<4;i++) {
        tmp_permuted[(i + lane_id / 8) % 4] = tmp_original[i];
    }
    reinterpret_cast<float4*>(sA[0])[(sA_row * BLOCK_WIDTH + sA_col) / 4] = reinterpret_cast<float4*>(tmp_permuted)[0];

    __syncthreads();


    for (int kBlock = 0; kBlock < N / BLOCK_WIDTH - 1; kBlock++){
        // load next block
        //reinterpret_cast<float4*>(sA)[(sA_row * BLOCK_WIDTH + sA_col) / 4] = reinterpret_cast<float4*>(A)[(gA_row * N + gA_col) / 4];
        // no bank conflict for B
        reinterpret_cast<float4*>(sB[(kBlock + 1) & 1])[(sB_row * TILE_WIDTH + sB_col) / 4] = reinterpret_cast<float4*>(B)[(gB_row * N + gB_col) / 4];

        // bank conflict for A, first load it to a tmp register then permute the data
        reinterpret_cast<float4*>(tmp_original)[0] = reinterpret_cast<float4*>(A)[(gA_row * N + gA_col) / 4];

        #pragma unroll
        for (int i=0;i<4;i++) {
            tmp_permuted[(i + lane_id / 8) % 4] = tmp_original[i];
        }
        reinterpret_cast<float4*>(sA[(kBlock + 1) & 1])[(sA_row * BLOCK_WIDTH + sA_col) / 4] = reinterpret_cast<float4*>(tmp_permuted)[0];


        //load a fragment from shared memory to register
        for (int kFragment = 0; kFragment < BLOCK_WIDTH; kFragment++){

            #pragma unroll
            for (int i=0; i<4; i++){
                // Load A fragment, 8 floats
//                fragment_A[i] = sA[(warp_row + thread_row + i) * BLOCK_WIDTH + kFragment];
//                fragment_A[i+4] = sA[(warp_row + thread_row + 16 + i) * BLOCK_WIDTH + kFragment];

                // bank conflict free load
                // column shift resets every 16 rows
                // thread_row / 4 gives us the permutation status
                //
                fragment_A[i] = sA[kBlock & 1][(warp_row + thread_row + i) * BLOCK_WIDTH + (thread_row / 4 + kFragment % 4) % 4 + (kFragment / 4) * 4];
                fragment_A[i+4] = sA[kBlock & 1][(warp_row + thread_row + 16 + i) * BLOCK_WIDTH + (thread_row / 4 + kFragment % 4) % 4 + (kFragment / 4) * 4];
                // Load B fragment, 8 floats
                fragment_B[i] = sB[kBlock & 1][kFragment * TILE_WIDTH + warp_col + thread_col + i];
                fragment_B[i+4] = sB[kBlock & 1][kFragment * TILE_WIDTH + warp_col + thread_col + 32 + i];
              }


            // Compute accumulator, 64 floats
            #pragma unroll 1
            for (int x=0; x<8; x++){
                #pragma unroll 1
                for (int y=0; y<8; y++){
                    accum[x * 8 + y] += fragment_A[x] * fragment_B[y];
                }
            }

        }
        __syncthreads();
    }


    // epilogue
    for (int kFragment = 0; kFragment < BLOCK_WIDTH; kFragment++){

            #pragma unroll
            for (int i=0; i<4; i++){
                // Load A fragment, 8 floats
//                fragment_A[i] = sA[(warp_row + thread_row + i) * BLOCK_WIDTH + kFragment];
//                fragment_A[i+4] = sA[(warp_row + thread_row + 16 + i) * BLOCK_WIDTH + kFragment];

                // bank conflict free load
                // column shift resets every 16 rows
                // thread_row / 4 gives us the permutation status
                //
                fragment_A[i] = sA[(N / BLOCK_WIDTH - 1) & 1][(warp_row + thread_row + i) * BLOCK_WIDTH + (thread_row / 4 + kFragment % 4) % 4 + (kFragment / 4) * 4];
                fragment_A[i+4] = sA[(N / BLOCK_WIDTH - 1) & 1][(warp_row + thread_row + 16 + i) * BLOCK_WIDTH + (thread_row / 4 + kFragment % 4) % 4 + (kFragment / 4) * 4];
                // Load B fragment, 8 floats
                fragment_B[i] = sB[(N / BLOCK_WIDTH - 1) & 1][kFragment * TILE_WIDTH + warp_col + thread_col + i];
                fragment_B[i+4] = sB[(N / BLOCK_WIDTH - 1) & 1][kFragment * TILE_WIDTH + warp_col + thread_col + 32 + i];
              }


            // Compute accumulator, 64 floats
            #pragma unroll 1
            for (int x=0; x<8; x++){
                #pragma unroll 1
                for (int y=0; y<8; y++){
                    accum[x * 8 + y] += fragment_A[x] * fragment_B[y];
                }
            }

        }



    #pragma unroll
    for (int x=0; x<4; x+=1){
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x ) * N + gC_col + warp_col + thread_col) / 4] = reinterpret_cast<float4*>(accum)[(x * 8) /4];
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x ) * N + gC_col + warp_col + thread_col + 32) / 4] = reinterpret_cast<float4*>(accum)[(x * 8 + 4) /4];
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x + 16) * N + gC_col + warp_col + thread_col) / 4] = reinterpret_cast<float4*>(accum)[((x + 4) * 8) /4];
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x + 16) * N + gC_col + warp_col + thread_col + 32) / 4] = reinterpret_cast<float4*>(accum)[((x + 4) * 8 + 4) /4];
    }

}