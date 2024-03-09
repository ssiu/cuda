// matrix multiplication for 2048 x 2048

// global memory coalescing
// shared memory blocking
// register blocking
// https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
// we will use 256 threads dimBlock(8, 32);

// thread-tiling: each thread loads 8+8 = 16 floats and computes 8x8 = 64 results
// warp-tiling: each warp computes 64x32 = 2048 results
// block-tiling: each thread block has 2x4 = 8 warps = 256 threads computing 128x128 = 16384 results

// shared memory:
// 128 * 32 * 4 * 2 = 32KB
// registers:
// each thread needs at least 64 * 4 = 256B
// so a threadblock needs at least 256 * 256 = 64 KB

// dim3 dimGrid(16, 16);
// dim3 dimBlock(256, 1);

#define TILE_WIDTH 64
#define TILE_LENGTH 128


// df/dx = (f(t+dt) - f(t)) / dt

__global__ void mm_4(float* A, float* B, float* C, int N){

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;


    // offset for output matrix C
    int gRow_C =  TILE_LENGTH * blockIdx.y;
    int gCol_C =  TILE_LENGTH * blockIdx.x;


    int gRow_A;
    int gCol_A;
    int gRow_B;
    int gCol_B;

    int sRow_A;
    int sCol_A;
    int sRow_B;
    int sCol_B;

    __shared__ float sA[TILE_LENGTH * TILE_WIDTH];
    __shared__ float sB[TILE_LENGTH * TILE_WIDTH];

    // fragments
    float fragment_A[8] = {};
    float fragment_B[8] = {};
    float accum[64] = {};

    // offset for A loading into shared memory
    for (int kBlock=0; kBlock<N/TILE_WIDTH; kBlock++){
        sRow_A = threadIdx.x / 32;
        sCol_A = threadIdx.x % 32;
        sRow_B = threadIdx.x / 128;
        sCol_B = threadIdx.x % 128;

        gRow_A = gRow_C + sRow_A;
        gCol_A = kBlock * TILE_WIDTH + sCol_A;
        gRow_B = kBlock * TILE_WIDTH + sRow_B;
        gCol_B = gCol_C + sCol_B;
        // each thread loads 16 floats for each A, B, broken into 16 loads
        // load A, B tile into shared memory
        // each thread load every 8th row
        #pragma unroll
        for (int i=0; i<128; i+=8) {
            //    32
            //  ________
      //128 // | 0 | 1 |
            // | 2 | 3 |
            // | ..... |
            // 128 rows
            sA[(sRow_A + i) * TILE_WIDTH + sCol_A] = A[(gRow_A + i) * N + gCol_A];

        }
        #pragma unroll
        for (int i=0; i<32; i+=2) {
            // ________________________________
            // | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
            // | 8 | 9 | 10| 11| 12| 13| 14| 15|
            // | ............................. |
            // 32 rows
            sB[(sRow_B + i) * TILE_WIDTH + sCol_B] = B[(gRow_B + i) * N + gCol_B];
        }

        __syncthreads();

        #pragma unroll
        //load a fragment from shared memory to register
        for (int kFragment = 0; kFragment < TILE_WIDTH; kFragment++){
            // 256 thread = 8 warps,

            // each warp computes 32 x 64 matrix
            // the 32 x 64 is further split into 4 16 x 32 matrix
            // in the 16 x 32 matrix, each thread computes 4x4 matrix

            // warp 0,1 need rows 0 - 31 of sA
            // warp 2,3 need rows 32 - 63 of sA
            // warp 4,5 need rows 64 - 95 of sA
            // warp 6,7 need rows 96 - 127 of sA

            // warp 0, 2, 4, 6 need cols 0 - 63 of sB
            // warp 1, 3, 5, 7 need cols 64 - 127 of sB


            int warp_offset_row = (warp_id / 2) * 32;
            int warp_offset_col = (warp_id % 2) * 64;


            int thread_offset_row = threadIdx.x / 8;
            int thread_offset_col = (threadIdx.x % 8) * 4;

            // offsets for A and B
            // sA row is (warp_id / 4) * 64 + (threadIdx.x / 4) * 4 + {0,32}
            // sA column is kFragment
            // sB row is kFragment
            // sB column is (warp_id % 4) * 32 + (threadIdx.x % 4) * 4 + {0, 16}

            #pragma unroll
            // todo: make vectorized access
            for (int i=0; i<32; i+=8){
                fragment_A[i] = sA[(warp_offset_row + thread_offset_row + i) * TILE_WIDTH + kFragment];
            }

            #pragma unroll
            for (int i=0; i<4; i++){
                fragment_B[i] = sB[kFragment * TILE_WIDTH + warp_offset_col + thread_offset_col + i];
                fragment_B[i+4] = sB[kFragment * TILE_WIDTH + warp_offset_col + thread_offset_col + 32 + i];
            }



            #pragma unroll

            for (int kTx=0; kTx<8; kTx++){
                for (int kTy=0; kTy<8; kTy++){
                    accum[kTx * 8 + kTy] += fragment_A[kTx] * fragment_B[kTy];
                }
            }

        }
        __syncthreads();
    }
    // non-vectorized
//    for (kTx=0; kTx<8; kTx+=1){
//        for (kTy=0; kTy<4; kTy+=1){
//            C[(gRow_C + warp_offset_row + thread_offset_row + kTx * 4) * N + gCol_C + warp_offset_col + thread_offset_col + kTy] = accum[kThreadx*8 + kThready];
//            C[(gRow_C + warp_offset_row + thread_offset_row + kTx * 4) * N + gCol_C + warp_offset_col + thread_offset_col + kTy + 32] = accum[kThreadx * 8 + kThready + 4];
//        }
//    }

    //vectorized
    // reinterpret_cast<float2*>(d_out)[i]
    for (int kTx=0; kTx<8; kTx+=1){
        reinterpret_cast<float4*>(C)[(gRow_C + warp_offset_row + thread_offset_row + kTx * 4) * N + (gCol_C + warp_offset_col + thread_offset_col) / 4] = reinterpret_cast<float4*>(accum)[kThreadx*8];
        reinterpret_cast<float4*>(C)[(gRow_C + warp_offset_row + thread_offset_row + kTx * 4) * N + (gCol_C + warp_offset_col + thread_offset_col + 32) / 4] = reinterpret_cast<float4*>(accum)[kThreadx * 8 + 1];
    }

}