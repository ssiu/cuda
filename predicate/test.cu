/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout) {
    using namespace cute;
    Tensor tile_S = S(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)
    Tensor tile_D = D(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)

    Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{},
                                        threadIdx.x);  // (ThrValM, ThrValN)
    Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{},
                                        threadIdx.x);  // (ThrValM, ThrValN)

    Tensor fragment = make_tensor_like(thr_tile_S);  // (ThrValM, ThrValN)

    // Copy from GMEM to RMEM and from RMEM to GMEM
    copy(thr_tile_S, fragment);
    copy(fragment, thr_tile_D);
}

/// Main function
int main(int argc, char** argv) {
  //
  // Given a 2D shape, perform an efficient copy
  //

  using namespace cute;
  using Element = float;

  int M = 32768;
  int N = 16384;

  auto tensor_shape = make_shape(M, N);

  thrust::host_vector<Element> h_S(size(tensor_shape));
  thrust::host_vector<Element> h_D(size(tensor_shape));

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
    h_D[i] = Element{};
  }

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  Tensor tensor_S =
      make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())),
                  make_layout(tensor_shape));
  Tensor tensor_D =
      make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())),
                  make_layout(tensor_shape));

  auto block_shape = make_shape(Int<256>{}, Int<128>{});

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape);  // ((M, N), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape);  // ((M, N), m', n')

  // Thread arrangement
  Layout thr_layout =
      make_layout(make_shape(Int<32>{}, Int<8>{}));  // (32,8) -> thr_idx

  dim3 gridDim(
      size<1>(tiled_tensor_D),
      size<2>(tiled_tensor_D));  // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(thr_layout));

  copy_kernel<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_D,
                                     thr_layout);

  cudaError result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
              << std::endl;
    return -1;
  }

  h_D = d_D;

  int32_t errors = 0;
  int32_t const kErrorLimit = 10;

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_S[i] != h_D[i]) {
      std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i
                << "]: " << h_D[i] << std::endl;

      if (++errors >= kErrorLimit) {
        std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
        return -1;
      }
    }
  }

  std::cout << "Success." << std::endl;

  return 0;
}