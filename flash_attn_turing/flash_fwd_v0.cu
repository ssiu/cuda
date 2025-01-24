#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <cute/tensor.hpp>
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"


using namespace cute;



template <class SmemLayoutQ, class TiledCopyQ, class TiledMmaS,
          class SmemLayoutK, class TiledCopyK, class TiledMmaO,
          class SmemLayoutV, class TiledCopyV,
          class SmemLayoutS,
          class SmemLayoutO>
__global__ __launch_bounds__(64)
void flash_fwd_v0_kernel(
    half_t const* q, SmemLayoutQ sQ_layout, TiledCopyQ copy_Q, TiledMmaS mma_S,
    half_t const* k, SmemLayoutK sK_layout, TiledCopyK copy_K, TiledMmaO mma_O,
    half_t const* v, SmemLayoutV sV_layout, TiledCopyV copy_V,
                     SmemLayoutV sS_layout,
    float* o,        SmemLayoutV sO_layout,
    int batch_size, int seq_len, int num_heads, int head_dim
)
{

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

    Tensor mQ = make_tensor(make_gmem_ptr(q),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gQ = local_tile(mQ(blockIdx.x, _, blockIdx.y, _), Shape<Int<16>, Int<128>>{},
                           make_coord(blockIdx.z, 0));  // (16, 128)

    Tensor mK = make_tensor(make_gmem_ptr(k),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gK = local_tile(mK(blockIdx.x, _, blockIdx.y, _), Shape<Int<16>, Int<128>>{},
                           make_coord(_, 0));  // (16, 128, seq_len / 16)

    // this is a (seq_len, head_dim) column major matrix, so its V^T in row major
    // for gmem -> smem copy, view it as 16 x 128
    // for gemm, view it as 128 x 16
    Tensor mV = make_tensor(make_gmem_ptr(v),
                            make_shape(batch_size, head_dim, num_heads, seq_len),
                            make_stride(seq_len * num_heads * head_dim, Int<1>{}, head_dim, num_heads * head_dim));

    // load V as 16 x 128 matrix, perform matmul as 128 x 16 matrix
    Tensor gV = local_tile(mV(blockIdx.x, _, blockIdx.y, _), Shape<Int<128>, Int<16>>{},
                           make_coord(0, _));  // (128, 16, seq_len / 16)

    Tensor mO = make_tensor(make_gmem_ptr(o),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gO = local_tile(mQ(blockIdx.x, _, blockIdx.y, _), Shape<Int<16>, Int<128>>{},
                           make_coord(blockIdx.z, 0));  // (16, 128)

    __shared__ half_t smemQ[16*128];
    __shared__ half_t smemK[16*128];
    __shared__ half_t smemV[16*128];
    __shared__ float smemS[16*16];
    __shared__ float smemP_float[16*16];
    __shared__ half_t smemP[16*16];
    __shared__ float smemO[16*128];


    float rM_old[16] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX,
                       -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    float rM[16] = {0.0f};
    float rL_old[16] = {0.0f};
    float rL[16] = {0.0f};
    // for storing rowsum(P)
    float rD[16] = {0.0f};

    Tensor sQ = make_tensor(make_smem_ptr(smemQ), sQ_layout);
    Tensor sK = make_tensor(make_smem_ptr(smemK), sK_layout);
    Tensor sV = make_tensor(make_smem_ptr(smemV), sV_layout);
    Tensor sS = make_tensor(make_smem_ptr(smemS), sS_layout);
    Tensor sP = make_tensor(make_smem_ptr(smemP), sS_layout);
    Tensor sP_float = make_tensor(make_smem_ptr(smemP_float), sS_layout);
    Tensor sO = make_tensor(make_smem_ptr(smemO), sO_layout);

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

    ThrCopy thr_copy_V = copy_V.get_slice(threadIdx.x);
    Tensor tVgV = thr_copy_V.partition_S(gV);
    Tensor tVsV = thr_copy_V.partition_D(sV);

    // mma for S = QK^T
    ThrMMA thr_mma_S = mma_S.get_slice(threadIdx.x);
    Tensor tSsQ = thr_mma_S.partition_A(sQ);
    Tensor tSsK = thr_mma_S.partition_B(sK);
    Tensor tSsS = thr_mma_S.partition_C(sS);

//     Tensor tSrQ = thr_mma_S.make_fragment_A(tSsQ);
//     Tensor tSrK = thr_mma_S.make_fragment_B(tSsK);
    Tensor tSrS = thr_mma_S.make_fragment_C(tSsS);


    // mma for O = PV
    ThrMMA thr_mma_O = mma_O.get_slice(threadIdx.x);
    Tensor tOsP = thr_mma_O.partition_A(sP);
    Tensor tOsV = thr_mma_O.partition_B(sV);
    Tensor tOgO = thr_mma_O.partition_C(gO);
    Tensor tOsO = thr_mma_O.partition_C(sO);
//     Tensor tOrP = thr_mma_O.make_fragment_A(tOsP);
//     Tensor tOrV = thr_mma_O.make_fragment_B(tOsV);
    Tensor tOrO = thr_mma_O.make_fragment_C(tOgO);


    auto KV_TILE_MAX = size<3>(tKgK);
    // prologue
    // load Q into smem
    copy(copy_Q, tQgQ, tQsQ);

    // main loop
    for (int kv_tile = 0; kv_tile < KV_TILE_MAX; ++kv_tile) {
        // load K, V into shared memory
        copy(copy_K, tKgK(_,_,_,kv_tile), tKsK);
        copy(copy_V, tVgV(_,_,_,kv_tile), tVsV);

        __syncthreads();
        // compute S = QK^T
        gemm(mma_S, tSsQ, tSsK, tSrS);

        copy(tSrS, tSsS);
        __syncthreads();

        // compute m = rowsum(S)
        for (int i = 0; i < 16; i++) {
            rM[i] = rM_old[i];
            for (int j=0; j < 16; j++) {
                rM[i] = fmaxf(rM[i], sS(i,j));
            }
        }

        // compute P = softmax(S)
        for (int i=0;i<16;i++) {
            for (int j=0;j<16;j++){
                sP_float(i,j) = expf(sS(i,j) - rM[i]);
            }
        }


        // rescale l and also reset rD to 0
        for (int i = 0; i < 16; i++) {
            rL[i] = expf(rM_old[i] - rM[i]) * rL_old[i];
            rD[i] = 0;
        }


        // compute sum(sP)
        for (int i = 0; i < 16; i++) {
            for (int j=0; j< 16; j++) {
                rD[i] += sP_float(i, j);
            }
        }

        // compute l
        for (int i=0; i<16; i++) {
            rL[i] += rD[i];
        }


        // cast sP from float to half_t
        for (int i=0;i<16;i++) {
            for (int j=0;j<16;j++){
                sP(i, j) = __float2half(sP_float(i, j));
            }
        }

        __syncthreads();
        // compute O = PV
        gemm(mma_O, tOsP, tOsV, tOrO);


        __syncthreads();

        // update m and l
        for (int i = 0; i < 16; i++) {
            rM_old[i] = rM[i];
            rL_old[i] = rL[i];
        }

    }
    // rescale rO

    copy(tOrO, tOsO);
    __syncthreads();

    for (int i=0;i<16;i++) {
        for (int j=0; j<16; j++) {
            sO(i,j) /= rL[i];

        }

    }

    copy(tOsO, tOgO);

}


// define mQ, mK, mV, mO
// define gQ, gK, gV, gO
// how to compute softmax
void flash_fwd_v0(half_t const* q, half_t const* k, half_t const* v, float* o,
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

    auto sQ_layout = make_layout(make_shape (Int<16>{}, Int<128>{}),
                        make_stride(Int<128>{}, Int<1>{}));

    auto sK_layout = make_layout(make_shape (Int<16>{}, Int<128>{}),
                        make_stride(Int<128>{}, Int<1>{}));

    // we should view sV as tranposed
    auto sV_layout = make_layout(make_shape (Int<128>{}, Int<16>{}),
                        make_stride(Int<1>{}, Int<128>{}));


    auto sS_layout = make_layout(make_shape (Int<16>{}, Int<16>{}),
                        make_stride(Int<16>{}, Int<1>{}));


    auto sO_layout = make_layout(make_shape (Int<16>{}, Int<128>{}),
                        make_stride(Int<128>{}, Int<1>{}));



    // 64 threads loading a 16 * 128 tile
    TiledCopy copy_Q = make_tiled_copy(Copy_Atom<DefaultCopy, half_t>{},
                                Layout<Shape<_4,_16>, Stride<_16,_1>>{},
                                Layout<Shape< _1,_8>>{});
    TiledCopy copy_K = make_tiled_copy(Copy_Atom<DefaultCopy, half_t>{},
                                Layout<Shape<_4,_16>, Stride<_16,_1>>{},
                                Layout<Shape< _1,_8>>{});


    TiledCopy copy_V = make_tiled_copy(Copy_Atom<DefaultCopy, half_t>{},
                                Layout<Shape<_16,_4>, Stride<_1,_16>>{},
                                Layout<Shape< _8,_1>>{});

    TiledMMA mma_S = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                        Layout<Shape<_1, _2, _1>>{},
                                        Tile<_16,_16,_8>{});

    TiledMMA mma_O = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                        Layout<Shape<_1, _2, _1>>{},
                                        Tile<_16,_128,_8>{});


    dim3 dimGrid(batch_size, num_heads, seq_len / 16);
    dim3 dimBlock(64);
    flash_fwd_v0_kernel<<<dimGrid, dimBlock>>>(q, sQ_layout, copy_Q, mma_S,
                                               k, sK_layout, copy_K, mma_O,
                                               v, sV_layout, copy_V,
                                                  sS_layout,
                                               o, sO_layout);

}