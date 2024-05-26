// fused matmul-bias-gelu kernel
//
// we use the same configuration as in matmul_forward2, where
// - weight is (OC, C) row major
// - inp is (C, B * T) column major
// - out is (OC, B * T) column major
//
// we use the configuration from CUTLASS https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
// - 256 threads per block, each using <= 128 registers -> 2 blocks per SM
// - 128 x 8 shared memory tiles
// - each thread computes a 8x8x1 outer product
// total shared memory usage = 128 x 8 x 4 (weight tile) + 128 x 8 x 4 (inp tile) + 128 * 4 (bias) = 4.5MB. This should be fine for most GPU
//
// assumptions
// B * T and OC are divisible by 128
// C is divisible by 8

#define weight(i,j) weight[(i) * C + (j)]
#define inp(i,j) inp[(i) + C * (j)]
#define out(i,j) out[(i) + OC * (j)]
// shared memory tiles are 128 x 8 row major matrices
#define shared_weight(pointer, i,j) shared_weight[(pointer)][((i) << 7) + (j)]
#define shared_inp(pointer, i,j) shared_inp[(pointer)][((i) << 7) + (j)]
#define BLOCK_WIDTH 128
#define TILE_WIDTH 8
#define FLOAT_4(pointer) reinterpret_cast<float4*>(&(pointer))[0]

__global__ __launch_bounds__(256,2)
void fused_matmul_forward_gelu_kernel(float* A, float* B, float* C, float* bias, int N){

    int block_idx = blockIdx.x;
    int block_idy = blockIdx.y;
    int thread_id = threadIdx.x;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    // global memory offset for out
    int out_row = block_idx << 7;
    int out_col = block_idy << 7;

    // block tiling
    int shared_weight_row_shared_inp_col = thread_id >> 1;
    int shared_weight_col_shared_inp_row = (thread_id & 1) << 2;

    // warp tiling + register tiling
    int warp_row = (warp_id & 1) << 6;
    int warp_col = (warp_id >> 1) << 5;
    int thread_row = (lane_id & 7) << 2;
    int thread_col = (lane_id >> 3) << 2;
    int accum_row = warp_row + thread_row;
    int accum_col = warp_col + thread_col;

    // shared memory double buffering
    __shared__ float shared_weight[2][TILE_WIDTH * BLOCK_WIDTH];
    __shared__ float shared_inp[2][TILE_WIDTH * BLOCK_WIDTH];

    // since bias will be reused BLOCK_TILE times, better to load into shared memory
    __shared__ float shared_bias[BLOCK_WIDTH];

    int pointer = 0;
    float reg_weight[4];
    float reg_inp[4];
    float frag_weight[8] = {};
    float frag_inp[8] = {};
    float accum[64] = {};

    // move to first tile
    weight = &weight(out_row, 0);
    inp = &inp(0, out_col);
    out = &out(out_row, out_col);

    // prologue: load bias and the first tile
    // TODO: test to see if we should compute bias in epilogue instead
    if (thread_id < 32) {
        FLOAT_4(shared_bias[4 * thread_id]) = FLOAT_4(bias[out_row + 4 * thread_id]);
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++){
            accum[i + 8 * j] = shared_bias[accum_row + i];
            accum[i + 4 + 8 * j] = shared_bias[accum_row + 32 + i];
        }
    }


    FLOAT_4(reg_weight[0]) = FLOAT_4(weight(shared_weight_row_shared_inp_col, shared_weight_col_shared_inp_row));
    FLOAT_4(reg_inp[0]) = FLOAT_4(inp(shared_weight_col_shared_inp_row, shared_weight_row_shared_inp_col));

    // transpose the tiles so that shared memory load is bank conflict free
    // TODO: make it so that shared memory store is also bank conflict free
    #pragma unroll
    for (int i = 0; i < 4; i++){
        shared_weight(pointer, shared_weight_col_shared_inp_row + i, shared_weight_row_shared_inp_col) = reg_weight[i];
        shared_inp(pointer, shared_weight_col_shared_inp_row + i, shared_weight_row_shared_inp_col) = reg_inp[i];
    }

    __syncthreads();

    weight += TILE_WIDTH;
    inp += TILE_WIDTH;

    // mainloop
    for (int kTile = 0; kTile < C/TILE_WIDTH; kTile++){

        // load next tile from global memory -> register
        if (kTile < C/TILE_WIDTH - 1) {
            FLOAT_4(reg_weight[0]) = FLOAT_4(weight(shared_weight_row_shared_inp_col, shared_weight_col_shared_inp_row));
            FLOAT_4(reg_inp[0]) = FLOAT_4(inp(shared_weight_col_shared_inp_row, shared_weight_row_shared_inp_col));
        }


        // compute the outer product for the current tile
        #pragma unroll
        for (int kFragment=0; kFragment < TILE_WIDTH; kFragment++){

            FLOAT_4(frag_weight[0]) = FLOAT_4(shared_weight(pointer, kFragment, accum_row));
            FLOAT_4(frag_weight[4]) = FLOAT_4(shared_weight(pointer, kFragment, accum_row + 32));
            FLOAT_4(frag_inp[0]) = FLOAT_4(shared_inp(pointer, kFragment, accum_col));
            FLOAT_4(frag_inp[4]) = FLOAT_4(shared_inp(pointer, kFragment, accum_col + 16));

            #pragma unroll
            for (int i=0; i<8;i++){
                #pragma unroll
                for (int j=0; j<8; j++) {
                    accum[i+8*j] += frag_weight[i] * frag_inp[j];
                }
            }
        }

        // store next tile from register -> shared memory
        if (kTile < C/TILE_WIDTH - 1) {
            #pragma unroll
            for (int i=0; i<4;i++){
                shared_weight(pointer ^ 1, shared_weight_col_shared_inp_row + i, shared_weight_row_shared_inp_col) = reg_weight[i];
                shared_inp(pointer ^ 1, shared_weight_col_shared_inp_row + i, shared_weight_row_shared_inp_col) = reg_inp[i];
            }

            __syncthreads();

            weight += TILE_WIDTH;
            inp += TILE_WIDTH;

            pointer ^= 1;
        }

    }

//    // epilogue: apply gelu
//    #pragma unroll
//    for (int i=0; i<8;i++){
//        #pragma unroll
//        for (int j=0; j<8; j++) {
//
//            accum[i+8*j] = 0.5f * accum[i+8*j] * (1.0f + tanhf(GELU_SCALING_FACTOR * (accum[i+8*j] + accum[i+8*j] * accum[i+8*j] * accum[i+8*j])));
//        }
//    }



    // store to global memory
    #pragma unroll
    for (int i=0;i<4;i++) {
        FLOAT_4(out(accum_row, accum_col + i)) = FLOAT_4(accum[i * 8]);
        FLOAT_4(out(accum_row, accum_col + i + 16)) = FLOAT_4(accum[(i + 4) * 8]);
        FLOAT_4(out(accum_row + 32, accum_col + i)) = FLOAT_4(accum[i * 8 + 4]);
        FLOAT_4(out(accum_row + 32, accum_col + i + 16)) = FLOAT_4(accum[(i + 4) * 8 + 4]);
    }


}