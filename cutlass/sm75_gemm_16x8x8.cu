
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"




int main(int argc, char** argv)
{
  int m = 16;
  int n = 8;
  int k = 8;

  using TA = half_t;
  using TB = half_t;
  using TC = half_t;

  cute::device_init(0);

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 ); print("%f\n", __half2float(h_A[j]));
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

//   int ldA = m, ldB = n, ldC = m;
//
//   // Run once
//   d_C = h_C;
//   matmul(
//        d_A.data().get(), ldA,
//        d_B.data().get(), ldB,
//        d_C.data().get(), ldC);
//   CUTE_CHECK_LAST();
//   thrust::host_vector<TC> cute_result = d_C;



  return 0;
}