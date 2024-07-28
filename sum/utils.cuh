#include <iostream>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <cmath> // For std::fabs

// todo: investigate the correct epsilon
bool areFloatsEqual(float a, float b, float epsilon = 1e-2f) {
    return std::fabs(a - b) < epsilon;
}


int areSameArrays(float* A_1, float* A_2, int N){
    for (int i = 0; i < N; i++){
        if (!(areFloatsEqual(A_1[i], A_2[i]))) {
            //std::cout << "Wrong answer:" << A_1[i] << " " << A_2[i] << std::endl;
            return 0;
        }
    }
    return 1;
}


thrust::host_vector<float> generateRandomArray(int N) {
    thrust::host_vector<float> A(N);

    // Create random engine
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define distribution range
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    // Generate random matrix
    for (int i=0; i<N; i++) {
        float randomFloat = dis(gen);
        A[i] = randomFloat;
    }

    // Return both matrices
    return A;
}
