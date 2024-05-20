// matrix multiplication for llm.c fused gelu
// A is row major, B is column major, C is column major
// use CUTLASS gemm setup
// becuase C is row major, we need to modify so that we can use 128-byte transactions for global memory store

#include <iostream>
#define A(i,j) A[(i) * N + (j)]
#define B(i,j) B[(i) + N * (j)]
#define C(i,j) C[(i) + N * (j)]
#define sA(i,j) sA[((i) << 7) + (j)]
#define sB(i,j) sB[((i) << 7) + (j)]
//#define sA(pointer, i,j) sA[(pointer)][((i) << 7) + (j)]
//#define sB(pointer, i,j) sB[(pointer)][((i) << 7) + (j)]
#define TILE_WIDTH 128
#define BLOCK_WIDTH 8
#define FLOAT_4(pointer) reinterpret_cast<float4*>(&(pointer))[0]


__global__ __launch_bounds__(256, 2)
void mm_llmc_1(float* A, float* B, float* C, int N){

    int block_idx = blockIdx.x;
    int block_idy = blockIdx.y;
    int thread_id = threadIdx.x;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    // global memory offset that this threadblock is responsible for
    int g_row = block_idx << 7;
    int g_col = block_idy << 7;


    //shared memory address that this thread is responsible for
    int s_row = thread_id >> 1; // 16
    int s_col = (thread_id & 1) << 2; // 0


    //warptiling
    int warp_row = (warp_id & 1) << 6;
    int warp_col = (warp_id >> 1) << 5;
    int thread_row = (lane_id & 7) << 2;
    int thread_col = (lane_id >> 3) << 2;


    int c_row = warp_row + thread_row;
    int c_col = warp_col + thread_col;

    A = &A((block_idx << 7), 0);
    B = &B(0, (block_idy << 7));
    C = &C((block_idx << 7), (block_idy << 7));

    __shared__ float sA[BLOCK_WIDTH * TILE_WIDTH];
    __shared__ float sB[BLOCK_WIDTH * TILE_WIDTH];

    float rA[4];
    float rB[4];

    float fA[8] = {};
    float fB[8] = {};

    float accum[64] = {};

    for (int kTile=0; kTile < N/BLOCK_WIDTH; kTile++){

        FLOAT_4(rA[0]) = FLOAT_4(A(s_row, s_col));
        FLOAT_4(rB[0]) = FLOAT_4(B(s_row, s_col));


        for (int i=0; i<4;i++){
            sA(s_col + i, s_row) = rA[i];
            sB(s_col + i, s_row) = rB[i];
        }


        for (int kFragment=0; kFragment < BLOCK_WIDTH; kFragment++){

            FLOAT_4(fA[0]) = FLOAT_4(sA(kFragment, c_row));
            FLOAT_4(fA[4]) = FLOAT_4(sA(kFragment, c_row+32));
            FLOAT_4(fB[0]) = FLOAT_4(sB(kFragment, c_col));
            FLOAT_4(fB[4]) = FLOAT_4(sB(kFragment, c_col+16));


            for (int i=0; i<8;i++){

                for (int j=0; j<8; j++) {
                    accum[i+8*j] += fA[i] * fB[j];
                }
             }
        }

        __syncthreads();

        A += BLOCK_WIDTH;
        B += BLOCK_WIDTH;
    }

    #pragma unroll
    for (int i=0;i<4;i++) {

        FLOAT_4(C(c_row, c_col + i)) = FLOAT_4(accum[i * 8]);
        FLOAT_4(C(c_row, c_col + i + 16)) = FLOAT_4(accum[(i + 4) * 8]);
        FLOAT_4(C(c_row + 32, c_col + i)) = FLOAT_4(accum[i * 8 + 4]);
        FLOAT_4(C(c_row + 32, c_col + i + 16)) = FLOAT_4(accum[(i + 4) * 8 + 4]);

    }


}









