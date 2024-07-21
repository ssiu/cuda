#include <iostream>
#define A(i,j) A[(i) * N + (j)]
#define B(i,j) B[(i) * N + (j)]
#define C(i,j) C[(i) * N + (j)]
#define sA(pointer, i,j) sA[(pointer)][((i) << 7) + (j)]
#define sB(pointer, i,j) sB[(pointer)][((i) << 7) + (j)]
#define TILE_WIDTH 128
#define BLOCK_WIDTH 8
#define FLOAT_4(pointer) reinterpret_cast<float4*>(&(pointer))[0]


__global__ __launch_bounds__(256,2)
void mm_double_buffering_new_kernel(float* A, float* B, float* C, int N){
    int thread_id = threadIdx.x;
    int block_idx = blockIdx.x;
    int block_idy = blockIdx.y;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int warp_row = (warp_id >> 1) << 5;
    int warp_col = (warp_id & 1) << 6;
    int thread_row = (lane_id >> 3) << 2;
    int thread_col = (lane_id & 7) << 2;


    int sA_row = thread_id >> 1;
    int sA_col = (thread_id & 1) << 2;

    int sB_row = thread_id >> 5;
    int sB_col = (thread_id & 31) << 2;


    int permuted_warp_id = (warp_id ) ^ (thread_id & 1);
    int permuted_thread_id = (permuted_warp_id << 5) + lane_id;
    int permuted_sA_row = permuted_thread_id >> 1;

    int C_row = warp_row + thread_row;
    int C_col = warp_col + thread_col;


    A = &A((block_idx << 7), 0);
    B = &B(0, (block_idy << 7));
    C = &C((block_idx << 7), (block_idy << 7));

    __shared__ float sA[2][BLOCK_WIDTH * TILE_WIDTH];
    __shared__ float sB[2][BLOCK_WIDTH * TILE_WIDTH];


    float rA[4];
    float rB[4];

    float fA[8] = {};
    float fB[8] = {};

    float accum[64] = {};

    int shared_pointer = 0;
    // load first block
    FLOAT_4(rA) = FLOAT_4(A(sA_row, sA_col));
    FLOAT_4(rB) = FLOAT_4(B(sB_row, sB_col));
    #pragma unroll
    for (int i=0; i<4;i++){
        sA(shared_pointer, sA_col + i, permuted_sA_row) = rA[i];
    }

    FLOAT_4(sB(shared_pointer, sB_row, sB_col)) = FLOAT_4(rB);

    __syncthreads();

    A += BLOCK_WIDTH;
    B += BLOCK_WIDTH * N;

    for (int kBlock=0; kBlock<N/BLOCK_WIDTH; kBlock++){

        // load from gmem A, B for next block
        if (kBlock < N/BLOCK_WIDTH - 1) {
            FLOAT_4(rA) = FLOAT_4(A(sA_row, sA_col));
            FLOAT_4(rB) = FLOAT_4(B(sB_row, sB_col));
        }
        #pragma unroll
        for (int kFragment=0; kFragment<4; kFragment++) {
            // load from smem A, B
            FLOAT_4(fA[0]) = FLOAT_4(sA(shared_pointer, kFragment, C_row));
            FLOAT_4(fA[4]) = FLOAT_4(sA(shared_pointer, kFragment, C_row + 16));
            FLOAT_4(fB[0]) = FLOAT_4(sB(shared_pointer, kFragment, C_col));
            FLOAT_4(fB[4]) = FLOAT_4(sB(shared_pointer, kFragment, C_col + 32));
            // compute outer product
            #pragma unroll
            for (int i=0; i<8;i++){
                #pragma unroll
                for (int j=0; j<8; j++) {
                    accum[i*8+j] += fA[i] * fB[j];
                }
             }

        }

        #pragma unroll
        for (int kFragment=4; kFragment<BLOCK_WIDTH; kFragment++) {
            // load from smem A, B
            FLOAT_4(fA[0]) = FLOAT_4(sA(shared_pointer, kFragment, C_row + 16));
            FLOAT_4(fA[4]) = FLOAT_4(sA(shared_pointer, kFragment, C_row));
            FLOAT_4(fB[0]) = FLOAT_4(sB(shared_pointer, kFragment, C_col));
            FLOAT_4(fB[4]) = FLOAT_4(sB(shared_pointer, kFragment, C_col + 32));
            // compute outer product
            #pragma unroll
            for (int i=0; i<8;i++){
                #pragma unroll
                for (int j=0; j<8; j++) {
                    accum[i*8+j] += fA[i] * fB[j];
                }
             }

        }
        // store to smem sA, sB for next block
        if (kBlock < N/BLOCK_WIDTH - 1) {


            //FLOAT_4(sA[sA_sOffset]) = FLOAT_4(rA);
            #pragma unroll
            for (int i=0; i<4;i++){
                sA(shared_pointer^1, sA_col + i, permuted_sA_row) = rA[i];
                //sA[shared_pointer^1][sA_sOffset + i*TILE_WIDTH] = rA[i];
            }

            FLOAT_4(sB(shared_pointer^1, sB_row, sB_col)) = FLOAT_4(rB);

            __syncthreads();

            A += BLOCK_WIDTH;
            B += BLOCK_WIDTH * N;

            shared_pointer ^= 1;
        }

    }

//    storeToGmem_5(accum, C, N, C_gOffset);

    // store to gmem C
    #pragma unroll
    for (int i=0;i<4;i++) {

        FLOAT_4(C(C_row + i, C_col)) = FLOAT_4(accum[i * 8]);
        FLOAT_4(C(C_row + i, C_col + 32)) = FLOAT_4(accum[i * 8 + 4]);
        FLOAT_4(C(C_row + i + 16, C_col)) = FLOAT_4(accum[(i+4) * 8]);
        FLOAT_4(C(C_row + i + 16, C_col + 32)) = FLOAT_4(accum[(i+4) * 8 + 4]);

    }
}


void mm_double_buffering_new(float* A, float* B, float* C, int N) {
    dim3 dimGrid(N / TILE_WIDTH, N / TILE_WIDTH);
    dim3 dimBlock(256);
    mm_double_buffering_new_kernel<<<dimGrid, dimBlock>>>(A, B, C, N);
}