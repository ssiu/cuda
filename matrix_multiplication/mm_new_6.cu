#include <iostream>
#define TILE_WIDTH 128
#define BLOCK_WIDTH 8
#define FLOAT_4(shared_pointer) reinterpret_cast<float4*>(&(shared_pointer))[0]


__global__ void mm_new_6(float* A, float* B, float* C, int N){
    int block_idx = blockIdx.x;
    int block_idy = blockIdx.y;
    int thread_id = threadIdx.x;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int g_row = block_idx * TILE_WIDTH;
    int g_col = block_idy * TILE_WIDTH;

    int sA_row = thread_id >> 1; // 16
    int sA_col = (thread_id & 1) << 2; // 0

    int sB_row = thread_id >> 5; // 1
    int sB_col = (thread_id & 31) << 2; // 0

    int sA_gOffset = sA_row * N + sA_col;
    int sB_gOffset = sB_row * N + sB_col;
    int sA_sOffset = sA_row * BLOCK_WIDTH + sA_col;
    int sB_sOffset = sB_row * TILE_WIDTH + sB_col;

    int warp_row = (warp_id >> 1) << 5; // 0
    int warp_col = (warp_id & 1) << 6; // 64
    int thread_row = (lane_id >> 3) << 2; // 0
    int thread_col = (lane_id & 7) << 2; // 0


    int sA_rOffset = (warp_row + thread_row) * BLOCK_WIDTH; // 0
    int sB_rOffset = warp_col + thread_col; // 64
    int C_gOffset = (warp_row + thread_row) * N + (warp_col + thread_col); // 64

    A = &A[g_row*N];
    B = &B[g_col];
    C = &C[g_row*N + g_col];

    __shared__ float sA[2][BLOCK_WIDTH * TILE_WIDTH];
    __shared__ float sB[2][BLOCK_WIDTH * TILE_WIDTH];

    float rA[2][4];
    float rB[2][4];

    float fA[8] = {};
    float fB[8] = {};
    float accum[64] = {};

    int shared_pointer = 0;
    // prologue, load kBLock = 0 from global to shared
    //load from gmem A, B
    FLOAT_4(rA[shared_pointer][0]) = FLOAT_4(A[sA_gOffset]);
    FLOAT_4(rB[shared_pointer][0]) = FLOAT_4(B[sB_gOffset]);

    // store to smem sA, sB
    FLOAT_4(sA[shared_pointer][sA_sOffset]) = FLOAT_4(rA[shared_pointer][0]);
    FLOAT_4(sB[shared_pointer][sB_sOffset]) = FLOAT_4(rB[shared_pointer][0]);


    //shift A,B shared_pointers
    A += BLOCK_WIDTH;
    B += BLOCK_WIDTH * N;

    __syncthreads();
//    if (block_idx==0 and block_idy==0 and thread_id ==0) {
//        printf("%f %f %f", rA[shared_pointer][0], sA[shared_pointer][0], sB[shared_pointer][0]);
//    }

    //mainloop
    // compute kblock = 0,..., N/BLOCK_WIDTH - 2
    // load kblock = 1,..., N/BLOCK_WIDTH - 1
    for (int kBlock=0; kBlock<N/BLOCK_WIDTH; kBlock++){

        //load from gmem for next block
        if (kBlock < N/BLOCK_WIDTH - 1) {

            //load from gmem A, B
            FLOAT_4(rA[shared_pointer ^ 1][0]) = FLOAT_4(A[sA_gOffset]);
            FLOAT_4(rB[shared_pointer ^ 1][0]) = FLOAT_4(B[sB_gOffset]);
            
            // store to smem sA, sB
            FLOAT_4(sA[shared_pointer ^ 1][sA_sOffset]) = FLOAT_4(rA[shared_pointer ^ 1][0]);
            FLOAT_4(sB[shared_pointer ^ 1][sA_sOffset]) = FLOAT_4(rB[shared_pointer ^ 1][0]);
            
            A += BLOCK_WIDTH;
            B += BLOCK_WIDTH * N;
        }

        for (int kFragment=0; kFragment<BLOCK_WIDTH; kFragment++) {

            // load from smem A, B

            for (int i=0; i<4; i++) {
                fA[i] = sA[shared_pointer][sA_rOffset + kFragment + i * BLOCK_WIDTH];
                fA[i+4] = sA[shared_pointer][sA_rOffset + kFragment + (i + 16) * BLOCK_WIDTH];
                fB[i] = sB[shared_pointer][sB_rOffset + kFragment * TILE_WIDTH + i];
                fB[i+4] = sB[shared_pointer][sB_rOffset + kFragment * TILE_WIDTH + i + 32];
            }


            // compute outer product
            for (int i = 0; i < 8; i++) { 
                for (int j = 0; j < 8; j++) { 
                    accum[i * 8 + j] += fA[i] * fB[j]; 
                }   
            }

        }

        shared_pointer ^= 1;
        
        __syncthreads();

    }


    // store to gmem C
    for (int i=0;i<4;i++) {
        FLOAT_4(C[C_gOffset + i * N]) = FLOAT_4(accum[i * 8]);
        FLOAT_4(C[C_gOffset + i * N + 32]) = FLOAT_4(accum[i * 8 + 4]);
        FLOAT_4(C[C_gOffset + (i + 16) * N ]) = FLOAT_4(accum[(i+4) * 8]);
        FLOAT_4(C[C_gOffset + (i + 16) * N + 32]) = FLOAT_4(accum[(i+4) * 8 + 4]);
    }



}