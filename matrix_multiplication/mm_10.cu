// matrix multiplication for 128 x 128 submatrix
// https://leimao.github.io/blog/CUDA-Shared-Memory-Capacity/
// https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/

// global memory coalescing
// block tiling



// shared memory:
// 128(m) * 32(n) * 4 (float) * 2 (A and B) = 32KB

// dim3 dimGrid(16, 16);
// dim3 dimBlock(256, 1);

#include <iostream>
#define BLOCK_A_WIDTH 32
#define BLOCK_B_WIDTH 8
#define TILE_WIDTH 128
#define thread_id threadIdx.x
#define warp_id (threadIdx.x / 32)
#define lane_id (threadIdx.x % 32)

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
#define gA_col (kBlock * BLOCK_WIDTH + sA_col)
#define gB_row (kBlock * BLOCK_WIDTH + sB_row)
#define gB_col (gC_col + sB_col)


__global__ void mm_7(float* A, float* B, float* C, int N){


    __shared__ float sA[TILE_WIDTH * BLOCK_WIDTH];
    __shared__ float sB[TILE_WIDTH * BLOCK_WIDTH];

    // fragments
    float fragment_A[8] = {};
    float fragment_B[8] = {};
    float accum[64] = {};

    for (int kBlock = 0; kBlock < N / BLOCK_WIDTH; kBlock++){
//        sA_row = thread_id / 2;
//        sA_col = (thread_id % 2) * 4;
//        sB_row = threadIdx.x / 32;
//        sB_col = (threadIdx.x % 32) * 4;

        gA_row = gC_row + sA_row;
        gA_col = kBlock * BLOCK_WIDTH + sA_col;
        gB_row = kBlock * BLOCK_WIDTH + sB_row;
        gB_col = gC_col + sB_col;

        // global memory load -> shared memory store
//        #pragma unroll
//
//        for (int i=0; i<4; i+=1) {
//            // load shared memory A
//            sA[sA_row * BLOCK_WIDTH + sA_col + i] = A[gA_row * N + gA_col + i];
//            sB[sB_row * TILE_WIDTH + sB_col + i] = B[gB_row * N + gB_col + i];
//        }
        reinterpret_cast<float4*>(sA)[(sA_row * BLOCK_WIDTH + sA_col) / 4] = reinterpret_cast<float4*>(A)[(gA_row * N + gA_col) / 4];
        reinterpret_cast<float4*>(sB)[(sB_row * TILE_WIDTH + sB_col) / 4] = reinterpret_cast<float4*>(B)[(gB_row * N + gB_col) / 4];

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
    // non-vectorized
//    for (int x=0; x<4; x+=1){
//        for (int y=0; y<4; y+=1){
//            C[(gC_row + warp_row + thread_row + x ) * N + gC_col + warp_col + thread_col + y] = accum[x*8 + y];
//            C[(gC_row + warp_row + thread_row + x ) * N + gC_col + warp_col + thread_col + y + 32] = accum[x * 8 + y + 4];
//            C[(gC_row + warp_row + thread_row + x + 16) * N + gC_col + warp_col + thread_col + y] = accum[(x + 4)* 8 + y];
//            C[(gC_row + warp_row + thread_row + x + 16) * N + gC_col + warp_col + thread_col + y + 32] = accum[(x + 4) * 8 + y + 4];
//        }
//    }
    #pragma unroll
    for (int x=0; x<4; x+=1){
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x ) * N + gC_col + warp_col + thread_col) / 4] = reinterpret_cast<float4*>(accum)[(x * 8) /4];
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x ) * N + gC_col + warp_col + thread_col + 32) / 4] = reinterpret_cast<float4*>(accum)[(x * 8 + 4) /4];
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x + 16) * N + gC_col + warp_col + thread_col) / 4] = reinterpret_cast<float4*>(accum)[((x + 4) * 8) /4];
        reinterpret_cast<float4*>(C)[((gC_row + warp_row + thread_row + x + 16) * N + gC_col + warp_col + thread_col + 32) / 4] = reinterpret_cast<float4*>(accum)[((x + 4) * 8 + 4) /4];
    }



}