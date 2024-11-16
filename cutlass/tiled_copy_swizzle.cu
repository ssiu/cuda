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

using namespace cute;

template <class T, class LayoutIn, class LayoutOut, class TiledCopy>
__global__ void mm_kernel(
           T* in, LayoutIn layout_in,
           T* out, LayoutOut layout_out,
           TiledCopy tiled_copy)
{

    Tensor g_in = make_tensor(make_gmem_ptr(in), layout_in);
    Tensor g_out = make_tensor(make_gmem_ptr(out), layout_out);

    ThrCopy thr_copy = tiled_copy.get_slice(threadIdx.x);
    Tensor tg_in = thr_copy.partition_S(g_in);                            // (CPY,CPY_M,CPY_K,k)
    Tensor ts_out = thr_copy.partition_D(g_out);                            // (CPY,CPY_M,CPY_K)

    #if 1
        if(thread0()) {
        print("  g_in : "); print(  g_in); print("\n");
        print(" g_out : "); print( g_out); print("\n");
        print(" tg_in : "); print( tg_in); print("\n");
        print("ts_out : "); print(ts_out); print("\n");

        }
    #endif

    copy(tiled_copy, tg_in, ts_out);


}

template <class T>
void mm(T* in, T* out) {

    auto in_layout = make_layout(make_shape (Int<8>{}, Int<8>{}),
                        make_stride(Int<8>{}, Int<1>{}));

//     auto out_layout = make_layout(make_shape (Int<8>{}, Int<8>{}),
//                         make_stride(Int<1>{}, Int<8>{}));

    auto out_layout = composition(Swizzle<1, 1, 1>{},
                                 make_layout(make_shape (Int<8>{}, Int<8>{}),
                                 make_stride(Int<1>{}, Int<8>{})));

    print_layout(out_layout);

    TiledCopy tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, T>{},
                                     Layout<Shape<_4,_8>, Stride<_1,_4>>{},
                                     Layout<Shape< _2,_1>>{});

    //print_latex(tiled_copy);
    dim3 dimGrid(1);
    dim3 dimBlock(32);
    mm_kernel<<<dimGrid, dimBlock>>>(in, in_layout,
                                     out, out_layout,
                                     tiled_copy);
}



int main(int argc, char** argv)
{
    int m = 8;
    int n = 8;

    //using T = half_t;
    using T = float;
    cute::device_init(0);

    thrust::host_vector<T> h_in(m*n);
    thrust::host_vector<T> h_out(m*n);


    for (int j = 0; j < m*n; ++j) {
        h_in[j] = static_cast<T>( j );
        h_out[j] = 0.0f;
    }


    thrust::device_vector<T> d_in = h_in;
    thrust::device_vector<T> d_out = h_out;

    mm(d_in.data().get(), d_out.data().get());

    h_out = d_out;

    for (int i = 0; i < 8; i++){
        for (int j=0;j<8;j++){
            printf("%2.0f ", h_in[i*8+j]);
        }
        printf("\n");
    }

    printf("==========\n");

    for (int i = 0; i < 8; i++){
        for (int j=0;j<8;j++){
            printf("%2.0f ", h_out[i*8+j]);
        }
        printf("\n");
    }



    return 0;
}

// nvcc -lineinfo -std=c++17 -arch=sm_70 -O3 -lcublas -I./cutlass/include -I./cutlass/tools/util/include -o tiled_copy_swizzle.o tiled_copy_swizzle.cu