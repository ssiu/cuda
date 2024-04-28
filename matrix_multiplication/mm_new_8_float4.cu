#include <iostream>
#define TILE_WIDTH 128
#define BLOCK_WIDTH 8
#define FLOAT_4(pointer) reinterpret_cast<float4*>(&(pointer))[0]
#define VS_MUL(vc, sa, vb) \
        vc.x += sa * vb.x; \
        vc.y += sa * vb.y; \
        vc.z += sa * vb.z; \
        vc.w += sa * vb.w;

__global__ __launch_bounds__(256)
void mm_new_8_float4(float* A, float* B, float* C, int N){
    int block_idx = blockIdx.x;
    int block_idy = blockIdx.y;
    int thread_id = threadIdx.x;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int g_row = block_idx << 7;
    int g_col = block_idy << 7;

    int sA_row = thread_id >> 1; // 16
    int sA_col = (thread_id & 1) << 2; // 0

    int sB_row = thread_id >> 5; // 1
    int sB_col = (thread_id & 31) << 2; // 0

    int sA_gOffset = sA_row * N + sA_col;
    int sB_gOffset = sB_row * N + sB_col;
    // need to transpose A tile to give vectorized shared memory load
    //int sA_sOffset = sA_row * BLOCK_WIDTH + sA_col;
    int sA_sOffset = (sA_col << 7) + sA_row;
    int sB_sOffset = (sB_row << 7) + sB_col;

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

    __shared__ float sA[2][BLOCK_WIDTH * TILE_WIDTH];
    __shared__ float sB[2][BLOCK_WIDTH * TILE_WIDTH];


    float4 rA;
//    float4 rB;

//    float rA[4];
    float rB[4];

    float fA[8] = {};
    float fB[8] = {};

    float accum[64] = {};

    int shared_pointer = 0;
    // load first block
    FLOAT_4(rA) = FLOAT_4(A[sA_gOffset]);
    FLOAT_4(rB) = FLOAT_4(B[sB_gOffset]);

//    for (int i=0; i<4;i++){
//        sA[shared_pointer][sA_sOffset + i*TILE_WIDTH] = rA[i];
//    }

    sA[shared_pointer][sA_sOffset + (0 << 7)] = rA.x;
    sA[shared_pointer][sA_sOffset + (1 << 7)] = rA.y;
    sA[shared_pointer][sA_sOffset + (2 << 7)] = rA.z;
    sA[shared_pointer][sA_sOffset + (3 << 7)] = rA.w;


    FLOAT_4(sB[shared_pointer][sB_sOffset]) = FLOAT_4(rB);

    __syncthreads();

    A += BLOCK_WIDTH;
    B += BLOCK_WIDTH * N;


    for (int kBlock=0; kBlock<N/BLOCK_WIDTH; kBlock++){

        // load from gmem A, B for next block
        if (kBlock < N/BLOCK_WIDTH - 1) {

            rA = FLOAT_4(A[sA_gOffset]);
            FLOAT_4(rB) = FLOAT_4(B[sB_gOffset]);
        }

        for (int kFragment=0; kFragment<BLOCK_WIDTH; kFragment++) {

            // load from smem A, B
            FLOAT_4(fA[0]) = FLOAT_4(sA[shared_pointer][sA_rOffset + kFragment * TILE_WIDTH]);
            FLOAT_4(fA[4]) = FLOAT_4(sA[shared_pointer][sA_rOffset + kFragment * TILE_WIDTH + 16]);
            FLOAT_4(fB[0]) = FLOAT_4(sB[shared_pointer][sB_rOffset + kFragment * TILE_WIDTH]);
            FLOAT_4(fB[4]) = FLOAT_4(sB[shared_pointer][sB_rOffset + kFragment * TILE_WIDTH + 32]);

            // compute outer product
            for (int i=0; i<8;i++){
                for (int j=0; j<8; j++) {
                    accum[i*8+j] += fA[i] * fB[j];
                }
             }
//            VS_MUL(FLOAT_4(accum[0]), FLOAT_4(fA[0]).x, FLOAT_4(fB[0]));
//            VS_MUL(FLOAT_4(accum[4]), FLOAT_4(fA[0]).x, FLOAT_4(fB[4]));
//            VS_MUL(FLOAT_4(accum[8]), FLOAT_4(fA[0]).y, FLOAT_4(fB[0]));
//            VS_MUL(FLOAT_4(accum[12]), FLOAT_4(fA[0]).y, FLOAT_4(fB[4]));
//            VS_MUL(FLOAT_4(accum[16]), FLOAT_4(fA[0]).z, FLOAT_4(fB[0]));
//            VS_MUL(FLOAT_4(accum[20]), FLOAT_4(fA[0]).z, FLOAT_4(fB[4]));
//            VS_MUL(FLOAT_4(accum[24]), FLOAT_4(fA[0]).w, FLOAT_4(fB[0]));
//            VS_MUL(FLOAT_4(accum[28]), FLOAT_4(fA[0]).w, FLOAT_4(fB[4]));
//            VS_MUL(FLOAT_4(accum[32]), FLOAT_4(fA[4]).x, FLOAT_4(fB[0]));
//            VS_MUL(FLOAT_4(accum[36]), FLOAT_4(fA[4]).x, FLOAT_4(fB[4]));
//            VS_MUL(FLOAT_4(accum[40]), FLOAT_4(fA[4]).y, FLOAT_4(fB[0]));
//            VS_MUL(FLOAT_4(accum[44]), FLOAT_4(fA[4]).y, FLOAT_4(fB[4]));
//            VS_MUL(FLOAT_4(accum[48]), FLOAT_4(fA[4]).z, FLOAT_4(fB[0]));
//            VS_MUL(FLOAT_4(accum[52]), FLOAT_4(fA[4]).z, FLOAT_4(fB[4]));
//            VS_MUL(FLOAT_4(accum[56]), FLOAT_4(fA[4]).w, FLOAT_4(fB[0]));
//            VS_MUL(FLOAT_4(accum[60]), FLOAT_4(fA[4]).w, FLOAT_4(fB[4]));

        }

        // store to smem sA, sB for next block
        if (kBlock < N/BLOCK_WIDTH - 1) {


            sA[shared_pointer^1][sA_sOffset + (0 << 7)] = rA.x;
            sA[shared_pointer^1][sA_sOffset + (1 << 7)] = rA.y;
            sA[shared_pointer^1][sA_sOffset + (2 << 7)] = rA.z;
            sA[shared_pointer^1][sA_sOffset + (3 << 7)] = rA.w;
//            for (int i=0; i<4;i++){
//                sA[shared_pointer^1][sA_sOffset + i*TILE_WIDTH] = rA[i];
//            }

            FLOAT_4(sB[shared_pointer^1][sB_sOffset]) = FLOAT_4(rB);

            __syncthreads();

            A += BLOCK_WIDTH;
            B += BLOCK_WIDTH * N;

            shared_pointer ^= 1;
        }

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