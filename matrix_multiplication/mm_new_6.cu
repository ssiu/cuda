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

    float fA[2][8];
    float fB[2][8];
    float accum[8][8];

    for(int i=0; i<8; i++){

        for(int j=0; j<8; j++){
            accum[i][j]=0.0;
        }
    }

    int shared_pointer = 0;
    int reg_pointer = 0;
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

        }


        //
        #pragma unroll
        for (int i=0; i<4; i++) {
                fA[reg_pointer][i] = sA[shared_pointer][sA_rOffset + i * BLOCK_WIDTH];
                fA[reg_pointer][i+4] = sA[shared_pointer][sA_rOffset + (i + 16) * BLOCK_WIDTH];
                fB[reg_pointer][i] = sB[shared_pointer][sB_rOffset + i];
                fB[reg_pointer][i+4] = sB[shared_pointer][sB_rOffset + i + 32];
        }


        for (int kFragment=0; kFragment<BLOCK_WIDTH; kFragment++) {


            if (kFragment < BLOCK_WIDTH -1) {
                // load from smem A,B for next tile
                for (int i=0; i<4; i++) {
                    fA[reg_pointer ^ 1][i] = sA[shared_pointer][sA_rOffset + kFragment + 1 + i * BLOCK_WIDTH];
                    fA[reg_pointer ^ 1][i+4] = sA[shared_pointer][sA_rOffset + kFragment + 1 + (i + 16) * BLOCK_WIDTH];
                    fB[reg_pointer ^ 1][i] = sB[shared_pointer][sB_rOffset + (kFragment + 1) * TILE_WIDTH + i];
                    fB[reg_pointer ^ 1][i+4] = sB[shared_pointer][sB_rOffset + (kFragment + 1) * TILE_WIDTH + i + 32];
                }
            }


            // compute outer product

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) { 
                    accum[i][j] += fA[reg_pointer][i] * fB[reg_pointer][j];
                }   
            }

            reg_pointer ^= 1;

        }

        // store to smem for next block
        if (kBlock < N/BLOCK_WIDTH - 1) {

            // store to smem sA, sB
            FLOAT_4(sA[shared_pointer ^ 1][sA_sOffset]) = FLOAT_4(rA[shared_pointer ^ 1][0]);
            FLOAT_4(sB[shared_pointer ^ 1][sA_sOffset]) = FLOAT_4(rB[shared_pointer ^ 1][0]);

            A += BLOCK_WIDTH;
            B += BLOCK_WIDTH * N;
        }

        shared_pointer ^= 1;
        
        __syncthreads();

    }


    // store to gmem C

    for (int i=0;i<4;i++) {
        FLOAT_4(C[C_gOffset + i * N]) = FLOAT_4(accum[i][0]);
        FLOAT_4(C[C_gOffset + i * N + 32]) = FLOAT_4(accum[i][4]);
        FLOAT_4(C[C_gOffset + (i + 16) * N ]) = FLOAT_4(accum[i+4][0]);
        FLOAT_4(C[C_gOffset + (i + 16) * N + 32]) = FLOAT_4(accum[i+4][4]);
    }



}