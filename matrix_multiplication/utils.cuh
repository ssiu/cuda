#include <iostream>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>

int isSameMatrices(float* A_1, float* A_2, int N){
    for (int i = 0; i < N*N; i++){
        if (A_1[i] != A_2[i]) {
            return 0;
        }
    }
    return 1;
}

thrust::host_vector<float> generateMatrices(int N) {
    thrust::host_vector<float> A(N * N);

    // Create random engine
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define distribution range
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    // Generate random matrix
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            float randomFloat = dis(gen);
            A[i * N + j] = randomFloat;
        }
    }

    // Return both matrices
    return A;
}


