#include <iostream>
// 1024 threads computing a 128*128 block
#define TILE_WIDTH 128
#define BLOCK_WIDTH 8


__device__ void loadFromGmem3(float* gM, float* r, int offset){
    reinterpret_cast<float4*>(r)[0] = reinterpret_cast<float4*>(&gM[offset])[0];
}

__device__ void storeToSmem3(float* r, float* sM, int offset){
    reinterpret_cast<float4*>(&sM[offset])[0] = reinterpret_cast<float4*>(r)[0];
}

__device__ void loadFromSmemA3(float* sM, float* r, int offset){
    for (int i=0; i<4; i++) {
        r[i] = sM[offset + i*BLOCK_WIDTH];
        r[i+4] = sM[offset + i*BLOCK_WIDTH + 16];
    }
}

__device__ void loadFromSmemB3(float* sM, float* r, int offset){
    for (int i=0; i<4; i++) {
        r[i] = sM[offset + i];
        r[i+4] = sM[offset + i + 32];
    }

}

__device__ void computeOuterProduct3(float* rA, float* rB, float* accum){
    for (int i=0; i<8;i++){
        for (int j=0; j<8; j++) {
            accum[i*8+j] = rA[i] * rB[j];
        }
    }
}

__device__ void storeToGmem3(float* accum, float* C, int N, int offset){
    for (int i=0;i<4;i++) {
        reinterpret_cast<float4*>(&C[offset + i * N])[0] = reinterpret_cast<float4*>(&accum[i])[0];
        reinterpret_cast<float4*>(&C[offset + i * N + 32])[0] = reinterpret_cast<float4*>(&accum[i * BLOCK_WIDTH + 32])[0];
        reinterpret_cast<float4*>(&C[offset + i * N + 16 * N])[0] = reinterpret_cast<float4*>(&accum[i * BLOCK_WIDTH + 16 * BLOCK_WIDTH])[0];
        reinterpret_cast<float4*>(&C[offset + i * N + 16 * N + 32])[0] = reinterpret_cast<float4*>(&accum[i * BLOCK_WIDTH + 16 * BLOCK_WIDTH + 32])[0];
    }
}


__global__ void mm_new_3(float* A, float* B, float* C, int N){
    int block_idx = blockIdx.x;
    int block_idy = blockIdx.y;
    int thread_id = threadIdx.x;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int g_row = block_idx * TILE_WIDTH;
    int g_col = block_idy * TILE_WIDTH;

    int sA_row = thread_id >> 1;
    int sA_col = (thread_id & 1) * 4;

    int sB_row = thread_id >> 5;
    int sB_col = (thread_id & 31) * 4;

    int sA_gOffset = sA_row * BLOCK_WIDTH + sA_col;
    int sB_gOffset = sB_row * TILE_WIDTH + sB_col;

    int warp_row = (warp_id / 2) * 32;
    int warp_col = (warp_id % 2) * 64;
    int thread_row = (lane_id / 8) * 4;
    int thread_col = (lane_id % 8) * 4;


    int sA_rOffset = (warp_row + thread_row) * BLOCK_WIDTH;
    int sB_rOffset = warp_col + thread_col;
    int C_gOffset = (warp_row + thread_row) * N + (warp_col + thread_col);

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

    for (int kBlock=0; kBlock<N/TILE_WIDTH; kBlock++){
//        sA[sPos] = A[gPos];
//        sB[sPos] = B[gPos];

        //load from gmem
        loadFromGmem3(A, rA, sA_gOffset);
        loadFromGmem3(B, rB, sB_gOffset);

        // store to sram
        storeToSmem3(rA, sA, sA_gOffset);
        storeToSmem3(rB, sB, sB_gOffset);

        //shift A,B pointers
        __syncthreads();
        A += BLOCK_WIDTH;
        B += BLOCK_WIDTH * N;
        // sync thread
        if (kBlock==0 and block_idx==0 and thread_id==0) {
            //printf("thread is %d, kBlock is %d, kFragment is %d, frag_A is %f\n", 1, kBlock, kFragment, fragment_A[i]);
            printf("kBlock is %d, thread id is %d, rA[0] is %f, sA[0] is %f\n", 1, kBlock, thread_id, rA[0], sA[0]);
        }

        for (int k=0; k<TILE_WIDTH; k++) {

            loadFromSmemA3(sA, fA, sA_rOffset);
            loadFromSmemB3(sB, fB, sB_rOffset);
            sA_rOffset += 1;
            sB_rOffset += TILE_WIDTH;

            //load from sram
            computeOuterProduct3(fA, fB, accum);

        }
        __syncthreads();

    }
    storeToGmem3(accum, C, N, C_gOffset);

}