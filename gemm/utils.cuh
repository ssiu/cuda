#include <iostream>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <cmath> // For std::fabs

// todo: investigate the correct epsilon
bool areFloatsEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}


int isSameMatrices(float* A_1, float* A_2, int M, int N){

    for (int i = 0; i < M*N; i++){
        if (!(areFloatsEqual(A_1[i], A_2[i]))) {
            //std::cout << "Wrong answer:" << A_1[i] << " " << A_2[i] << std::endl;
            return 0;
        }
    }
    return 1;
}

template <typename T>
thrust::host_vector<T> generateRandomMatrix(int M, int N) {
    thrust::host_vector<T> A(M * N);

    for (int i = 0; i < M * N; i++) {
        h_A[i] = static_cast<T>( 2*(rand() / double(RAND_MAX)) - 1 );
    }

    // Return both matrices
    return A;
}

