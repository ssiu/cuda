#include <iostream>
#define TILE_WIDTH 128
#define BLOCK_WIDTH 8
#define FLOAT_4(pointer) reinterpret_cast<float4*>(&(pointer))[0]


__global__ void mm_new_5(float* A, float* B, float* C, int N){
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
    // need to transpose A tile
    //int sA_sOffset = sA_row * BLOCK_WIDTH + sA_col;
    int sA_sOffset = sA_col * TILE_WIDTH + sA_row;
    int sB_sOffset = sB_row * TILE_WIDTH + sB_col;

    int warp_row = (warp_id >> 1) << 5; // 0
    int warp_col = (warp_id & 1) << 6; // 64
    int thread_row = (lane_id >> 3) << 2; // 0
    int thread_col = (lane_id & 7) << 2; // 0


    int sA_rOffset = warp_row + thread_row; // 0
    int sB_rOffset = warp_col + thread_col; // 64
    int C_gOffset = (warp_row + thread_row) * N + (warp_col + thread_col); // 64

    A = &A[g_row*N];
    B = &B[g_col];
    C = &C[g_row*N + g_col];

    __shared__ float sA[BLOCK_WIDTH * TILE_WIDTH];
    __shared__ float sB[BLOCK_WIDTH * TILE_WIDTH];

    float rA[4];
    float rB[4];

    float fA[8] = {};
    float fB[8] = {};
    float accum[64] = {};
    for (int kBlock=0; kBlock<N/BLOCK_WIDTH; kBlock++){
//        sA[sPos] = A[gPos];
//        sB[sPos] = B[gPos];

        //load from gmem A, B
        FLOAT_4(rA) = FLOAT_4(A[sA_gOffset]);
        FLOAT_4(rB) = FLOAT_4(B[sB_gOffset]);


        // store to smem sA, sB
        //FLOAT_4(sA[sA_sOffset]) = FLOAT_4(rA);
        for (int i=0; i<4;i++){
            sA[sA_sOffset + i*TILE_WIDTH] = rA[i];
        }

        FLOAT_4(sB[sB_sOffset]) = FLOAT_4(rB);


        //shift A,B pointers
        __syncthreads();

        A += BLOCK_WIDTH;
        B += BLOCK_WIDTH * N;

        for (int kFragment=0; kFragment<BLOCK_WIDTH; kFragment++) {

//            loadFromSmemA_5(sA, fA, sA_rOffset + kFragment);
//            loadFromSmemB_5(sB, fB, sB_rOffset + kFragment * TILE_WIDTH);


            // load from smem A, B
            for (int i=0; i<4; i++) {
                fA[i] = sA[sA_rOffset + kFragment + i];
                fA[i+4] = sA[sA_rOffset + kFragment + (i + 16)];
                fB[i] = sB[sB_rOffset + kFragment * TILE_WIDTH + i];
                fB[i+4] = sB[sB_rOffset + kFragment * TILE_WIDTH + i + 32];
            }


            // compute outer product
            for (int i=0; i<8;i++){
                for (int j=0; j<8; j++) {
                    accum[i*8+j] += fA[i] * fB[j];
                }
             }

        }
        __syncthreads();

    }

//    storeToGmem_5(accum, C, N, C_gOffset);

    // store to gmem C
    for (int i=0;i<4;i++) {
        FLOAT_4(C[C_gOffset + i * N]) = FLOAT_4(accum[i * 8]);
        FLOAT_4(C[C_gOffset + i * N + 32]) = FLOAT_4(accum[i * 8 + 4]);
        FLOAT_4(C[C_gOffset + (i + 16) * N ]) = FLOAT_4(accum[(i+4) * 8]);
        FLOAT_4(C[C_gOffset + (i + 16) * N + 32]) = FLOAT_4(accum[(i+4) * 8 + 4]);
//        reinterpret_cast<float4*>(&C[offset + i * N])[0] = reinterpret_cast<float4*>(&accum[i * 8])[0];
//        reinterpret_cast<float4*>(&C[offset + i * N + 32])[0] = reinterpret_cast<float4*>(&accum[i * 8 + 4])[0];
//        reinterpret_cast<float4*>(&C[offset + (i + 16) * N ])[0] = reinterpret_cast<float4*>(&accum[(i+4) * 8])[0];
//        reinterpret_cast<float4*>(&C[offset + (i + 16) * N + 32])[0] = reinterpret_cast<float4*>(&accum[(i+4) * 8 + 4])[0];
    }



}