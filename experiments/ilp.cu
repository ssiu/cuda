
// This kernel
// variables: number of threads
//            number of ILP

__global__ void arithmetic_kernel(int num_instructions, int N) {
    int a = 1;

    #pragma unroll 1
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
    }
}



int main(){

    arithmetic_kernel<<<1024, 128>>>(1, 1024);
    return 0;

}