#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "utils.cuh"

using namespace cute;

template <class LayoutIn, class LayoutOut, class TiledCopy>
__global__ void mm_kernel(
           half_t* in, LayoutIn layout_in,
           half_t* out, LayoutOut layout_out,
           LayoutSmem layout_smem, TiledCopy tiled_copy
{

    Tensor g_in = make_tensor(make_gmem_ptr(in), layout_in);
    Tensor g_out = make_tensor(make_gmem_ptr(out), layout_out);

    __shared__ half_t smem[cosize_v<LayoutOut>];


    Tensor s_mid = make_tensor(make_smem_ptr(smem), layout_smem);


    ThrCopy thr_copy = tiled_copy.get_slice(threadIdx.x);
    Tensor tg_in = thr_copy.partition_S(g_in);                            // (CPY,CPY_M,CPY_K,k)
    Tensor ts_out = thr_copy.partition_D(g_out);                            // (CPY,CPY_M,CPY_K)


    copy(tiled_copy, tg_in, ts_out);


}


void mm(half_t* in, half_t* out) {

    auto in_layout = make_layout(make_shape (Int<8>{}, Int<8>{}),
                        make_stride(Int<8>{}, Int<1>{}));

    auto out_layout = make_layout(make_shape (Int<8>{}, Int<8>{}),
                        make_stride(Int<1>{}, Int<8>{}));

    auto smem_layout = make_layout(make_shape (Int<8>{}, Int<8>{}),
                                               make_stride(Int<1>{}, Int<8>{}));
//     using sB_layout = decltype(composition(Swizzle<1, 1, 1>{},
//                                  make_layout(make_shape (Int<8>{}, Int<8>{}),
//                                  make_stride(Int<1>{}, Int<8>{}))));

    TiledCopy tiled_copy = make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<16>, half_t>{},
                                     Layout<Shape<_4,_8>, Stride<_1,_4>>{},
                                     Layout<Shape< _2,_1>>{});

    dim3 dimGrid(1);
    dim3 dimBlock(32);
    mm_kernel<<<dimGrid, dimBlock>>>(in, in_layout,
                                     out, out_layout,
                                     smem_layout, tiled_copy);
}



int main(int argc, char** argv)
{
    int m = 8;
    int n = 8;

    using T = half_t;

    cute::device_init(0);

    thrust::host_vector<T> h_in(m*n);
    thrust::host_vector<T> h_out(m*n);


    for (int j = 0; j < m*n; ++j) {
        h_in[j] = static_cast<T>( j );
    }


    thrust::device_vector<T> d_in = h_in;
    thrust::device_vector<T> d_out;

    mm(d_A.data().get(), d_B.data().get(), d_C.data().get());

    h_out = d_out;




    return 0;
}