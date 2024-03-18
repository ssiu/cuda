// matrix multiplication for 8192 x 8192 x 8192

// global memory coalescing
// shared memory blocking
// register blocking
// vectorized memory load
// global + shared memory pipelining

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
//#define gA_row gC_row + sA_row
//#define gA_col kBlock * BLOCK_WIDTH + sA_col
//#define gB_row kBlock * BLOCK_WIDTH + sB_row
//#define gB_col gC_col + sB_col


__global__ void mm_8(float* A, float* B, float* C, int N){
//    int thread_id = threadIdx.x;
//    int warp_id = threadIdx.x / 32;
//    int lane_id = threadIdx.x % 32;
//
//    int warp_row = (warp_id / 2) * 32;
//    int warp_col = (warp_id % 2) * 64;
//    int thread_row = lane_id / 8;
//    int thread_col = (lane_id % 8) * 4;

    // offset for output matrix C
//    int gC_row =  TILE_WIDTH * blockIdx.y;
//    int gC_col =  TILE_WIDTH * blockIdx.x;

//    int sA_row;
//    int sA_col;
//    int sB_row;
//    int sB_col;


//    int gA_row;
//    int gA_col;
//    int gB_row;
//    int gB_col;

    __shared__ float sA[TILE_WIDTH * BLOCK_WIDTH];
    __shared__ float sB[TILE_WIDTH * BLOCK_WIDTH];

    // fragments
    float fragment_A[8] = {};
    float fragment_B[8] = {};
    float accum[64] = {};

    float buffer_A[8] = {};
    float buffer_B[8] = {};
    int pointer = 0;
    // prologue
    // global -> reg for kBlock = 0
    // reinterpret_cast<float4*>(buffer_A)[0] = reinterpret_cast<float4*>(A)[(gA_row * N + gA_col) / 4];
    // reinterpret_cast<float4*>(buffer_B)[0] = reinterpret_cast<float4*>(B)[(gB_row * N + gB_col) / 4];
    reinterpret_cast<float4*>(buffer_A)[0] = reinterpret_cast<float4*>(A)[((gC_row + sA_row) * N + sA_col) / 4];
    reinterpret_cast<float4*>(buffer_B)[0] = reinterpret_cast<float4*>(B)[(sB_row * N + gC_col + sB_col) / 4];

    // mainloop
    // for kBlock = 0,..., N/BLOCK_WIDTH - 1
    // reg -> shared for kBlock = k
    // global -> reg for kBlock = k + 1
    // FMA for kBlock = k

    for (int kBlock = 0; kBlock < N / BLOCK_WIDTH - 1; kBlock++){
//        gA_row = gC_row + sA_row;
//        gA_col = kBlock * BLOCK_WIDTH + sA_col;
//        gB_row = kBlock * BLOCK_WIDTH + sB_row;
//        gB_col = gC_col + sB_col;




//        reinterpret_cast<float4*>(sA)[(sA_row * BLOCK_WIDTH + sA_col) / 4] = reinterpret_cast<float4*>(A)[(gA_row * N + gA_col) / 4];
//        reinterpret_cast<float4*>(sB)[(sB_row * TILE_WIDTH + sB_col) / 4] = reinterpret_cast<float4*>(B)[(gB_row * N + gB_col) / 4];

        // reg -> shared for kBlock = k
        reinterpret_cast<float4*>(sA)[(sA_row * BLOCK_WIDTH + sA_col) / 4] = reinterpret_cast<float4*>(buffer_A)[pointer];
        reinterpret_cast<float4*>(sB)[(sB_row * TILE_WIDTH + sB_col) / 4] = reinterpret_cast<float4*>(buffer_B)[pointer];

        pointer = (pointer = 0) ? 1 : 0;
        // global -> reg for kBlock = k + 1
        reinterpret_cast<float4*>(buffer_A)[pointer] = reinterpret_cast<float4*>(A)[((gC_row + sA_row) * N + (kBlock + 1) * BLOCK_WIDTH + sA_col) / 4];
        reinterpret_cast<float4*>(buffer_B)[pointer] = reinterpret_cast<float4*>(B)[(((kBlock + 1) * BLOCK_WIDTH + sB_row) * N + gC_col + sB_col) / 4];

        __syncthreads();


        //load a fragment from shared memory to register
        for (int kFragment = 0; kFragment < BLOCK_WIDTH; kFragment++){


            #pragma unroll
            for (int i=0; i<4; i++){
                // Load A fragment, 8 floats
                fragment_A[i] = sA[(warp_row + thread_row + i) * BLOCK_WIDTH + kFragment];
                fragment_A[i+4] = sA[(warp_row + thread_row + 16 + i) * BLOCK_WIDTH + kFragment];

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


    // epilogue
    // reg -> shared for kBlock = N/BLOCK - 1
    // FMA for kBlock = N/BLOCK - 1

    // reg -> shared for kBlock = N/BLOCK - 1
    reinterpret_cast<float4*>(sA)[(sA_row * BLOCK_WIDTH + sA_col) / 4] = reinterpret_cast<float4*>(buffer_A)[pointer];
    reinterpret_cast<float4*>(sB)[(sB_row * TILE_WIDTH + sB_col) / 4] = reinterpret_cast<float4*>(buffer_B)[pointer];
    // FMA for kBlock = N/BLOCK - 1
    #pragma unroll
    for (int kFragment = 0; kFragment < BLOCK_WIDTH; kFragment++){

        #pragma unroll
        for (int i=0; i<4; i++){
            // Load A fragment, 8 floats
            fragment_A[i] = sA[(warp_row + thread_row + i) * BLOCK_WIDTH + kFragment];
            fragment_A[i+4] = sA[(warp_row + thread_row + 16 + i) * BLOCK_WIDTH + kFragment];

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

    #pragma unroll
    for (int x=0; x<4; x+=1){
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x ) * N + gC_col + warp_col + thread_col) / 4] = reinterpret_cast<float4*>(accum)[(x * 8) /4];
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x ) * N + gC_col + warp_col + thread_col + 32) / 4] = reinterpret_cast<float4*>(accum)[(x * 8 + 4) /4];
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x + 16) * N + gC_col + warp_col + thread_col) / 4] = reinterpret_cast<float4*>(accum)[((x + 4) * 8) /4];
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x + 16) * N + gC_col + warp_col + thread_col + 32) / 4] = reinterpret_cast<float4*>(accum)[((x + 4) * 8 + 4) /4];
    }



}