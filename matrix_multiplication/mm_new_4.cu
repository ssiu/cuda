#include <iostream>
#define TILE_WIDTH 128
#define BLOCK_WIDTH 8


__device__ void loadFromGmem_4(float* gM, float* r, int offset){
    reinterpret_cast<float4*>(r)[0] = reinterpret_cast<float4*>(&gM[offset])[0];
}

__device__ void storeToSmem_4(float* r, float* sM, int offset){
    reinterpret_cast<float4*>(&sM[offset])[0] = reinterpret_cast<float4*>(r)[0];
}

__device__ void loadFromSmemA_4(float* sM, float* f, int offset){
    for (int i=0; i<4; i++) {
        f[i] = sM[offset + i * BLOCK_WIDTH];
        f[i+4] = sM[offset + (i + 16) * BLOCK_WIDTH];
    }
}

__device__ void loadFromSmemB_4(float* sM, float* f, int offset){
    for (int i=0; i<4; i++) {
        f[i] = sM[offset + i];
        f[i+4] = sM[offset + i + 32];
    }

}

__device__ void computeOuterProduct_4(float* fA, float* fB, float* accum){
    for (int i=0; i<8;i++){
        for (int j=0; j<8; j++) {
            accum[i*8+j] += fA[i] * fB[j];
        }
    }
}

__device__ void storeToGmem_4(float* accum, float* C, int N, int offset){
    for (int i=0;i<4;i++) {
        reinterpret_cast<float4*>(&C[offset + i * N])[0] = reinterpret_cast<float4*>(&accum[i * 8])[0];
        reinterpret_cast<float4*>(&C[offset + i * N + 32])[0] = reinterpret_cast<float4*>(&accum[i * 8 + 4])[0];
        reinterpret_cast<float4*>(&C[offset + (i + 16) * N ])[0] = reinterpret_cast<float4*>(&accum[(i+4) * 8])[0];
        reinterpret_cast<float4*>(&C[offset + (i + 16) * N + 32])[0] = reinterpret_cast<float4*>(&accum[(i+4) * 8 + 4])[0];
    }
}


__global__ void mm_new_4(float* A, float* B, float* C, int N){
    int block_idx = blockIdx.x;
    int block_idy = blockIdx.y;
    int thread_id = threadIdx.x;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    // addresses for block-tiling, gmem -> smem
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

    // addresses for warp-tiling, smem -> outer product
    int warp_row = (warp_id >> 1) << 5; // 0
    int warp_col = (warp_id & 1) << 6; // 64
    int thread_row = (lane_id >> 3) << 2; // 0
    int thread_col = (lane_id & 7) << 2; // 0
    int sA_rOffset = (warp_row + thread_row) * BLOCK_WIDTH; // 0
    int sB_rOffset = warp_col + thread_col; // 64
    int C_gOffset = (warp_row + thread_row) * N + (warp_col + thread_col); // 64

    // move to the current thread block for C
    A = &A[g_row*N];
    B = &B[g_col];
    C = &C[g_row*N + g_col];

    // double buffer
    __shared__ float sA[2][BLOCK_WIDTH * TILE_WIDTH];
    __shared__ float sB[2][BLOCK_WIDTH * TILE_WIDTH];
    float rA[2][4];
    float rB[2][4];

    // registers for computing the outer products
    float fA[8] = {};
    float fB[8] = {};
    float accum[64] = {};

    //buffer index
    int pointer = 0;

    //prologue, preload kblock = 0
    loadFromGmem_4(A, &rA[pointer][0], sA_gOffset);
    loadFromGmem_4(B, &rB[pointer][0], sB_gOffset);


    //
    A += BLOCK_WIDTH;
    B += BLOCK_WIDTH * N;

    //mainloop
    // compute kblock = 0,..., N/BLOCK_WIDTH - 2
    // load kblock = 1,..., N/BLOCK_WIDTH - 1
    for (int kBlock=0; kBlock<N/BLOCK_WIDTH; kBlock++){
//        sA[sPos] = A[gPos];
//        sB[sPos] = B[gPos];

        // store to smem for current block
        storeToSmem_4(&rA[pointer][0], &sA[pointer][0], sA_sOffset);
        storeToSmem_4(&rB[pointer][0], &sB[pointer][0], sB_sOffset);

        if (kBlock < N/BLOCK_WIDTH - 1) {
            //load from gmem for next block
            reinterpret_cast<float4*>(&rA[pointer ^ 1][0])[0] = reinterpret_cast<float4*>(&A[sA_gOffset])[0];
            reinterpret_cast<float4*>(&rB[pointer ^ 1][0])[0] = reinterpret_cast<float4*>(&B[sB_gOffset])[0];

        }

        //shift A,B pointers
        __syncthreads();

        A += BLOCK_WIDTH;
        B += BLOCK_WIDTH * N;

        for (int kFragment=0; kFragment<BLOCK_WIDTH; kFragment++) {

            loadFromSmemA_4(&sA[pointer][0], fA, sA_rOffset + kFragment) ;
            loadFromSmemB_4(&sB[pointer][0], fB, sB_rOffset + kFragment * TILE_WIDTH);

            computeOuterProduct_4(fA, fB, accum);
        }
        pointer ^= 1;

    }

    storeToGmem_4(accum, C, N, C_gOffset);
}