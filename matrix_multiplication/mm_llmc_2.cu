// matrix multiplication for llm.c fused gelu
// A is row major, B is column major, C is column major
// use CUTLASS gemm setup
// becuase C is row major, we need to modify so that we can use 128-byte transactions for global memory store
// we used the configuration from CUTLASS https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
// we launch 256 threads per block, each using 128 registers -> 2 blocks per SM
// we use 128 x 8 shared memory blocks
// each thread computes a 8x8x1 outer product
// total shared memory usage = 128 x 8 x 4 (A tile) + 128 x 8 x 4 (B tile) + 128 * 4 (bias) = 4.5MB. This should be fine for most GPU
//
// assumptions
// BT and OC are divisible by 128
// C is divisible by 8

#include <iostream>
#define A(i,j) A[(i) * N + (j)]
#define B(i,j) B[(i) + N * (j)]
#define C(i,j) C[(i) + N * (j)]
// shared memory tiles are 128 x 8 row major matrices
#define sA(pointer, i,j) sA[(pointer)][((i) << 7) + (j)]
#define sB(pointer, i,j) sB[(pointer)][((i) << 7) + (j)]
#define TILE_WIDTH 128
#define BLOCK_WIDTH 8
#define FLOAT_4(pointer) reinterpret_cast<float4*>(&(pointer))[0]

// we use launch bounds to indicate 256 threads per block at compile time (saves ~x registers)
__global__ __launch_bounds__(256)
void mm_llmc_2(float* A, float* B, float* C, float* bias, int N){

    int block_idx = blockIdx.x;
    int block_idy = blockIdx.y;
    int thread_id = threadIdx.x;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    // global memory offset for this block
    int g_row = block_idx << 7;
    int g_col = block_idy << 7;


    // block tiling 
    int sA_row_sB_col = thread_id >> 1; 
    int sA_col_sB_row = (thread_id & 1) << 2; 
    

    // warp tiling + register tiling 
    int warp_row = (warp_id & 1) << 6;
    int warp_col = (warp_id >> 1) << 5;
    int thread_row = (lane_id & 7) << 2;
    int thread_col = (lane_id >> 3) << 2;
    int c_row = warp_row + thread_row;
    int c_col = warp_col + thread_col;


    A = &A(g_row, 0);
    B = &B(0, g_col);
    C = &C(g_row, g_col);

    // shared memory double buffering
    __shared__ float sA[2][BLOCK_WIDTH * TILE_WIDTH];
    __shared__ float sB[2][BLOCK_WIDTH * TILE_WIDTH];
    __shared__ float shared_bias[TILE_WIDTH];

    int pointer = 0;

    float rA[4];
    float rB[4];

    float fA[8] = {};
    float fB[8] = {};

    float accum[64] = {};

    if (thread_id < 32) {
        FLOAT_4(shared_bias[4 * thread_id]) = FLOAT_4(bias[g_row + 4 * thread_id]);
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++){
            accum[i + 8 * j] = shared_bias[c_row + i];
            accum[i + 4 + 8 * j] = shared_bias[c_row + 32 + i];
        }
    }


    // prologue, load first tile

    FLOAT_4(rA[0]) = FLOAT_4(A(sA_row_sB_col, sA_col_sB_row));
    FLOAT_4(rB[0]) = FLOAT_4(B(sA_col_sB_row, sA_row_sB_col));

    // transpose the tiles so that shared memory load is bank conflict free
    // TODO: make it so that shared memory store is also bank conflict free
    #pragma unroll
    for (int i=0; i<4;i++){
        sA(pointer, sA_col_sB_row + i, sA_row_sB_col) = rA[i];
        sB(pointer, sA_col_sB_row + i, sA_row_sB_col) = rB[i];
    }

    __syncthreads();

    A += BLOCK_WIDTH;
    B += BLOCK_WIDTH;

    // mainloop
    for (int kTile=0; kTile < N/BLOCK_WIDTH; kTile++){

        // load next tile from global memory -> register
        if (kTile < N/BLOCK_WIDTH - 1) {
            FLOAT_4(rA[0]) = FLOAT_4(A(sA_row_sB_col, sA_col_sB_row));
            FLOAT_4(rB[0]) = FLOAT_4(B(sA_col_sB_row, sA_row_sB_col));
        }


        // compute the outer product for the current tile
        #pragma unroll
        for (int kFragment=0; kFragment < BLOCK_WIDTH; kFragment++){

            FLOAT_4(fA[0]) = FLOAT_4(sA(pointer, kFragment, c_row));
            FLOAT_4(fA[4]) = FLOAT_4(sA(pointer, kFragment, c_row+32));
            FLOAT_4(fB[0]) = FLOAT_4(sB(pointer, kFragment, c_col));
            FLOAT_4(fB[4]) = FLOAT_4(sB(pointer, kFragment, c_col+16));

            #pragma unroll
            for (int i=0; i<8;i++){
                #pragma unroll
                for (int j=0; j<8; j++) {
                    accum[i+8*j] += fA[i] * fB[j];
                }
            }
        }

        // store next tile from register -> shared memory
        if (kTile < N/BLOCK_WIDTH - 1) {
            #pragma unroll
            for (int i=0; i<4;i++){
                sA(pointer^1, sA_col_sB_row + i, sA_row_sB_col) = rA[i];
                sB(pointer^1, sA_col_sB_row + i, sA_row_sB_col) = rB[i];
            }

            __syncthreads();

            A += BLOCK_WIDTH;
            B += BLOCK_WIDTH;

            pointer ^= 1;
        }

    }

    // epilogue, apply gelu


    // store to global memory
    #pragma unroll
    for (int i=0;i<4;i++) {
        FLOAT_4(C(c_row, c_col + i)) = FLOAT_4(accum[i * 8]);
        FLOAT_4(C(c_row, c_col + i + 16)) = FLOAT_4(accum[(i + 4) * 8]);
        FLOAT_4(C(c_row + 32, c_col + i)) = FLOAT_4(accum[i * 8 + 4]);
        FLOAT_4(C(c_row + 32, c_col + i + 16)) = FLOAT_4(accum[(i + 4) * 8 + 4]);
    }


}









