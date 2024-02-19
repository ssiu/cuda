#include <iostream>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>

// This kernel
// variables: number of threads
//            number of ILP

#define N 1000000

__global__ void test_1(float a, float c) {
        float a1 = a;
        float c1 = c;

        #pragma unroll 1
        for (int i = 0; i < N; i++)
        {
                c1 += a1;
        }
        if (c1 == 0) printf("!");
}



__global__ void test_5(float a, float c) {
        float a1 = a;
        float c1 = c;
        float a2 = a+1;
        float c2 = c+1;
        float a3 = a+2;
        float c3 = c+2;
        float a4 = a+3;
        float c4 = c+3;
        float a5 = a+4;
        float c5 = c+4;

        #pragma unroll
        for (int i = 0; i < N; i++)
        {
                c1 += a1;
                c2 += a2;
                c3 += a3;
                c4 += a4;
                c5 += a5;
        }
        if (c1 == 0) printf("?");
        if (c2 == 0) printf("!");
        if (c3 == 0) printf(".");
        if (c4 == 0) printf("*");
        if (c5 == 0) printf("/");
}



int main(){

    test_1<<<256, 128>>>(0.1, 0.2);
    test_5<<<256, 128>>>(0.1, 0.2);


    return 0;

}