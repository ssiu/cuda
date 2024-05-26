#include <iostream>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <cmath> // For std::fabs

// todo: investigate the correct epsilon
bool areFloatsEqual(float a, float b, float epsilon = 1e-2) {
    return std::fabs(a - b) < epsilon;
}


int isSameMatrices(float* A_1, float* A_2, int N){
    for (int i = 0; i < N*N; i++){
        if (!(areFloatsEqual(A_1[i], A_2[i]))) {
            //std::cout << "Wrong answer:" << A_1[i] << " " << A_2[i] << std::endl;
            return 0;
        }
    }
    return 1;
}


//int elementwise_difference()


int countZeros(float* A, int N) {
    int num = 0;
//    for (int i = 0; i < N*N; i++) {
//        if (A[i] == 0.0f) {
//            num += 1;
//        }
//    }

    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            if (A[i*N + j] == 0.0f) {
                num += 1;
            }
        }
    }
    return num;

}

thrust::host_vector<float> generateRandomBias(int N) {
    thrust::host_vector<float> A(N);

    // Create random engine
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define distribution range
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    // Generate random matrix
    for (int i=0; i<N; i++) {
        A[i] = randomFloat;
    }

    // Return matrix
    return A;
}



thrust::host_vector<float> generateRandomMatrices(int N) {
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


thrust::host_vector<float> generateTestMatrices(int N) {
    thrust::host_vector<float> A(N * N);

    // Generate random matrix
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            if (i<32 and j<32) {
                A[i * N + j] = static_cast<float>(i);
            } else {
                A[i * N + j] = 0.0f;
            }
        }
    }

    // Return both matrices
    return A;
}