#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <float.h>
#include <torch/extension.h>
#include <cute/tensor.hpp>
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"


using namespace cute;


#define HEAD_SIZE 128
#define Q_TILE_SIZE 32
#define KV_TILE_SIZE 32



template <class SmemLayoutQ, class TiledCopyQ, class TiledMmaS,
          class SmemLayoutK, class TiledCopyK, class TiledMmaO,
          class SmemLayoutV, class TiledCopyV,
          class SmemLayoutS,
          class SmemLayoutO, class TiledCopyO>
__global__ __launch_bounds__(256)
void flash_fwd_v4_kernel(
    half_t const* q, SmemLayoutQ sQ_layout, TiledCopyQ copy_Q, TiledMmaS mma_S,
    half_t const* k, SmemLayoutK sK_layout, TiledCopyK copy_K, TiledMmaO mma_O,
    half_t const* v, SmemLayoutV sV_layout, TiledCopyV copy_V,
                     SmemLayoutS sS_layout,
    float* o,        SmemLayoutO sO_layout, TiledCopyO copy_O,
    int batch_size, int seq_len, int num_heads, int head_dim
)
{
//  todo:
//  do everything in registers from sS -> sP
//  test tensors are initialized to 0
//  remove bank conflicts

// q : (batch_size, seq_len, num_heads, head_dim)
// k : (batch_size, seq_len, num_heads, head_dim)
// v : (batch_size, seq_len, num_heads, head_dim)
// o : (batch_size, seq_len, num_heads, head_dim)
// m :
// we load a 32 x 64 tile of q, k, v into shared memory and compute 32x64 of o
// each thread loads 8 numbers
// qk^T = s 32 x 32 -> softmax
// define mma

// we launch a 3d grid with batch_size * num_heads * (seq_len / 16)
// blockIdx.x batch_size
// blockIdx.y num_heads
// blockIdx.z (seq_len / 16)
//     if (threadIdx.x == 0) {
//         printf("gridDim.x = %d, gridDim.y = %d, gridDim.z = %d\n", gridDim.x, gridDim.y, gridDim.z);
//         printf("blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
//     }
    Tensor mQ = make_tensor(make_gmem_ptr(q),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gQ = local_tile(mQ(blockIdx.x, _, blockIdx.y, _), Shape<Int<Q_TILE_SIZE>, Int<HEAD_SIZE>>{},
                           make_coord(blockIdx.z, 0));  // (16, 128)

    Tensor mK = make_tensor(make_gmem_ptr(k),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gK = local_tile(mK(blockIdx.x, _, blockIdx.y, _), Shape<Int<KV_TILE_SIZE>, Int<HEAD_SIZE>>{},
                           make_coord(_, 0));  // (16, 128, seq_len / 16)

    // this is a (seq_len, head_dim) column major matrix, so its V^T in row major
    // for gmem -> smem copy, view it as 16 x 128
    // for gemm, view it as 128 x 16
    Tensor mV = make_tensor(make_gmem_ptr(v),
                            make_shape(batch_size, head_dim, num_heads, seq_len),
                            make_stride(seq_len * num_heads * head_dim, Int<1>{}, head_dim, num_heads * head_dim));

    // load V as 16 x 128 matrix, perform matmul as 128 x 16 matrix
    Tensor gV = local_tile(mV(blockIdx.x, _, blockIdx.y, _), Shape<Int<HEAD_SIZE>, Int<KV_TILE_SIZE>>{},
                           make_coord(0, _));  // (128, 16, seq_len / 16)

    Tensor mO = make_tensor(make_gmem_ptr(o),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gO = local_tile(mO(blockIdx.x, _, blockIdx.y, _), Shape<Int<Q_TILE_SIZE>, Int<HEAD_SIZE>>{},
                           make_coord(blockIdx.z, 0));  // (16, 128)


    extern __shared__ char smem_[];

// first load sQ into registers
// load sK into smem, sV into registers
// compute S, store sV into smem
// store S into smem
// compute sP
// downcast sP and store to smem
// store V to smem
// compute O


    // smem total = 50KB
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), sQ_layout); // 8KB
    Tensor sK = make_tensor(sQ.data() + Q_TILE_SIZE*HEAD_SIZE, sK_layout);                   // 8KB
    Tensor sV = make_tensor(sK.data() + KV_TILE_SIZE*HEAD_SIZE, sV_layout);                  // 8KB
    Tensor sP = make_tensor(sV.data() + KV_TILE_SIZE*HEAD_SIZE, sS_layout);                  // 2KB

    int array_offset = (Q_TILE_SIZE * HEAD_SIZE + KV_TILE_SIZE * HEAD_SIZE * 2 + Q_TILE_SIZE * KV_TILE_SIZE) * sizeof(half_t);
    Tensor sS = make_tensor(make_smem_ptr(reinterpret_cast<float*>(&smem_[0] + array_offset)), sS_layout); // 4KB
    Tensor sP_float = make_tensor(sS.data() + Q_TILE_SIZE * KV_TILE_SIZE, sS_layout);   // 4KB
    Tensor sO = make_tensor(sP_float.data() + Q_TILE_SIZE * KV_TILE_SIZE, sO_layout);  // 16KB

    int thread_id = threadIdx.x;
    int lane_id = thread_id % 32;
    int warp_id = thread_id / 32;
    int warp_row = warp_id * 4;
    int thread_row = warp_row + (lane_id / 8);
    int thread_col = (lane_id % 8) * 4;

    float rM_old = -FLT_MAX;
    float rM = 0.0f;
    float rL_old = 0.0f;
    float rL = 0.0f;
    // for storing rowsum(P)
    float rD = 0.0f;

    unsigned mask;
    if (lane_id < 8)       mask = 0x000000FF;  // Lanes 0-7
    else if (lane_id < 16) mask = 0x0000FF00;  // Lanes 8-15
    else if (lane_id < 24) mask = 0x00FF0000;  // Lanes 16-23
    else                   mask = 0xFF000000;  // Lanes 24-31

    int lane_id_to_read_from;
    if (lane_id < 8)       lane_id_to_read_from = 0;  // Lanes 0-7
    else if (lane_id < 16) lane_id_to_read_from = 8;  // Lanes 8-15
    else if (lane_id < 24) lane_id_to_read_from = 16;  // Lanes 16-23
    else                   lane_id_to_read_from = 24;  // Lanes 24-31

    // q should be 16 x 128 tensor
    // k, v should be seq_len x 128 tensor
    // further partition k, v to (seq_len / 16) x 16 x 128 tensor

    // gmem -> smem for Q, K, V
    ThrCopy thr_copy_Q = copy_Q.get_slice(threadIdx.x);
    Tensor tQgQ = thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = thr_copy_Q.partition_D(sQ);


    ThrCopy thr_copy_K = copy_K.get_slice(threadIdx.x);
    Tensor tKgK = thr_copy_K.partition_S(gK);
    Tensor tKsK = thr_copy_K.partition_D(sK);
    Tensor tKrK = make_fragment_like(tKsK);

    ThrCopy thr_copy_V = copy_V.get_slice(threadIdx.x);
    Tensor tVgV = thr_copy_V.partition_S(gV);
    Tensor tVsV = thr_copy_V.partition_D(sV);
    Tensor tVrV = make_fragment_like(tVsV);

    // smem -> gmem for O
    ThrCopy thr_copy_O = copy_O.get_slice(threadIdx.x);
    Tensor tOsO_copy = thr_copy_O.partition_S(sO);
    Tensor tOgO_copy = thr_copy_O.partition_D(gO);

    // mma for S = QK^T
    ThrMMA thr_mma_S = mma_S.get_slice(threadIdx.x);
    Tensor tSsQ = thr_mma_S.partition_A(sQ);
    Tensor tSsK = thr_mma_S.partition_B(sK);
    Tensor tSsS = thr_mma_S.partition_C(sS);

    Tensor tSrQ = thr_mma_S.make_fragment_A(tSsQ);
    Tensor tSrK = thr_mma_S.make_fragment_B(tSsK);
    Tensor tSrS = thr_mma_S.make_fragment_C(tSsS);


    // mma for O = PV
    ThrMMA thr_mma_O = mma_O.get_slice(threadIdx.x);
    Tensor tOsP = thr_mma_O.partition_A(sP);
    Tensor tOsV = thr_mma_O.partition_B(sV);
    Tensor tOgO = thr_mma_O.partition_C(gO);
    Tensor tOsO = thr_mma_O.partition_C(sO);

    Tensor tOrP = thr_mma_O.make_fragment_A(tOsP);
    Tensor tOrV = thr_mma_O.make_fragment_B(tOsV);
    Tensor tOrO = thr_mma_O.make_fragment_C(tOgO);


    auto KV_TILE_MAX = size<3>(tKgK);

    // prologue

    copy(copy_Q, tQgQ, tQsQ);

    // clear sO and rO
    clear(tOrO);
    for (int i = 0; i< 4;i++) {
        for (int j = 0; j< 4;j++) {
            sO(thread_row, thread_col + j + 32 * i) = 0.0f;
        }
    }

    // main loop
    for (int kv_tile = 0; kv_tile < KV_TILE_MAX; ++kv_tile) {
        // load K, V into shared memory
        copy(copy_K, tKgK(_,_,_,kv_tile), tKsK);
        copy(copy_V, tVgV(_,_,_,kv_tile), tVsV);

        __syncthreads();
        // compute S = QK^T
        clear(tSrS);
        copy(tSsQ, tSrQ);
        copy(tSsK, tSrK);
        gemm(mma_S, tSrQ, tSrK, tSrS);


        copy(tSrS, tSsS);
        __syncthreads();


        for (int i = 0; i < 4; i++) {
            sS(thread_row, thread_col + i) *= 1.0f / sqrtf(HEAD_SIZE);
        }
        __syncthreads();



        // compute m = rowmax(S)
        rM = rM_old;

        // intra-thread reduction
        for (int i=0; i< 4; i++) {
            rM = fmaxf(rM, sS(thread_row, thread_col + i));
        }


        // inter-warp reduction
        for (int offset = 4; offset > 0; offset /= 2) {
           rM = fmaxf(rM, __shfl_down_sync(mask, rM, offset));
        }

        // sync rM
        rM = __shfl_sync(mask, rM, lane_id_to_read_from);

        // compute P = softmax(S)

        for (int i=0; i< 4; i++) {
            sP_float(thread_row,thread_col + i) = expf(sS(thread_row, thread_col+ i) - rM);
        }

        __syncthreads();


        // rescale l and also reset rD to 0
        rL = expf(rM_old - rM) * rL_old;
        rD = 0.0f;


        // compute sum(sP)

        // thread reduction
        for (int i=0; i< 4; i++) {
            rD += sP_float(thread_row, thread_col + i);
        }


        // warp reduction
        for (int offset = 4; offset > 0; offset /= 2) {
           rD +=  __shfl_down_sync(mask, rD, offset);
        }


        // can just keep the correct rL to lane 0
        rL += rD;
        // sync rL
        rL = __shfl_sync(mask, rL, lane_id_to_read_from);


        // cast sP from float to half_t
        for (int i=0; i< 4; i++) {
            sP(thread_row, thread_col + i) = __float2half(sP_float(thread_row, thread_col + i));
        }

        // rescale O

        copy(tOrO, tOsO);

        __syncthreads();

        for (int i = 0; i< 4;i++) {
            for (int j = 0; j< 4;j++) {
                sO(thread_row, thread_col + j + 32 * i) = expf(rM_old - rM) * sO(thread_row, thread_col + j + 32 * i);
            }
        }

        __syncthreads();

        copy(tOsO, tOrO);

        __syncthreads();

        gemm(mma_O, tOsP, tOsV, tOrO);

        // update m and l
        rM_old = rM;
        rL_old = rL;
    }
    // end of KV loop

    copy(tOrO, tOsO);
    __syncthreads();
    // rescale rO

    for (int i = 0; i< 4;i++) {
        for (int j = 0; j< 4;j++) {
            sO(thread_row, thread_col + j + 32 * i) /= rL;
        }
    }
//     if (threadIdx.x == 0){
//
//         for (int i=0;i < Q_TILE_SIZE;i++) {
//             for (int j=0; j<HEAD_SIZE; j++) {
//                 sO(i,j) /= rL[i];
//             }
//         }
//     }
    __syncthreads();

    copy(copy_O, tOsO_copy, tOgO_copy);

}


// define mQ, mK, mV, mO
// define gQ, gK, gV, gO
// how to compute softmax
torch::Tensor flash_fwd_v4(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                int batch_size, int seq_len, int num_heads, int head_dim)
{
    //  input : (B, S, NH, HD)
    // output : (B, S, NH, HD)
    //

    // q : 16 x 128
    // k : 16 x 128
    // v : 16 x 128
    // s : 16 x 16
    // p : 16 x 16
    // o : 16 x 128

    auto sQ_layout = make_layout(make_shape (Int<Q_TILE_SIZE>{}, Int<HEAD_SIZE>{}),
                        make_stride(Int<HEAD_SIZE>{}, Int<1>{}));

    auto sK_layout = make_layout(make_shape (Int<KV_TILE_SIZE>{}, Int<HEAD_SIZE>{}),
                        make_stride(Int<HEAD_SIZE>{}, Int<1>{}));

    // we should view sV as tranposed
    auto sV_layout = make_layout(make_shape (Int<HEAD_SIZE>{}, Int<KV_TILE_SIZE>{}),
                        make_stride(Int<1>{}, Int<HEAD_SIZE>{}));


    auto sS_layout = make_layout(make_shape (Int<Q_TILE_SIZE>{}, Int<KV_TILE_SIZE>{}),
                        make_stride(Int<KV_TILE_SIZE>{}, Int<1>{}));


    auto sO_layout = make_layout(make_shape (Int<Q_TILE_SIZE>{}, Int<HEAD_SIZE>{}),
                        make_stride(Int<HEAD_SIZE>{}, Int<1>{}));



    // 64 threads loading a 16 x 128 tile
    TiledCopy copy_Q = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, half_t>{},
                                Layout<Shape<_16,_16>, Stride<_16,_1>>{},
                                Layout<Shape< _1,_8>>{});

    TiledCopy copy_K = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, half_t>{},
                                Layout<Shape<_16,_16>, Stride<_16,_1>>{},
                                Layout<Shape< _1,_8>>{});

    // 64 threads loading a 128 x 16 tile
    TiledCopy copy_V = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, half_t>{},
                                Layout<Shape<_16,_16>, Stride<_1,_16>>{},
                                Layout<Shape< _8,_1>>{});

    TiledCopy copy_O = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, float>{},
                                Layout<Shape<_8,_32>, Stride<_32,_1>>{},
                                Layout<Shape< _1,_4>>{});


    TiledMMA mma_S = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                        Layout<Shape<_2, _4, _1>>{},
                                        Tile<_32,_32,_8>{});

    TiledMMA mma_O = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                        Layout<Shape<_2, _4, _1>>{},
                                        Tile<_32,_128,_8>{});


    torch::Tensor o = torch::empty(q.sizes(), q.options().dtype(torch::kFloat32));

    half_t* q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
    half_t* k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
    half_t* v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
    float* o_ptr = o.data_ptr<float>();


//     dim3 dimGrid(batch_size, num_heads, seq_len / 16);
    dim3 dimGrid(batch_size, num_heads, seq_len / Q_TILE_SIZE);
    dim3 dimBlock(256);
    int maxbytes = 65536;


    auto kernel = flash_fwd_v4_kernel<decltype(sQ_layout), decltype(copy_Q), decltype(mma_S),
                                      decltype(sK_layout), decltype(copy_K), decltype(mma_O),
                                      decltype(sV_layout), decltype(copy_V),
                                      decltype(sS_layout),
                                      decltype(sO_layout), decltype(copy_O)>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    flash_fwd_v4_kernel<<<dimGrid, dimBlock, maxbytes>>>(q_ptr, sQ_layout, copy_Q, mma_S,
                                                         k_ptr, sK_layout, copy_K, mma_O,
                                                         v_ptr, sV_layout, copy_V,
                                                                sS_layout,
                                                         o_ptr, sO_layout, copy_O,
                                                         batch_size, seq_len, num_heads, head_dim);

    return o;

}