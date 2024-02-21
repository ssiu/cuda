#include <iostream>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "mm.h"


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


int main(){
    int N = 2048;

    thrust::host_vector<float> hA = generateMatrices(N);
    thrust::host_vector<float> hB = generateMatrices(N);
    thrust::host_vector<float> hC;

    thrust::device_vector<float> dA = hA;
    thrust::device_vector<float> dB = hB;
    thrust::device_vector<float> dC = hC;


    mm_cublas(dA, dB, dC, N);

}