
// This kernel
// variables: number of threads
//            number of ILP

__global__ void arithmetic_kernel(float num_instructions, int N) {
    #pragma unroll num_instructions
    int a = 1;
    for (int i = idx; i < N; i ++) {
        a = a * 1 + 1;
    }
}



int main(){

    arithmetic_kernel<<<1024, 128>>>(1, 1024);
    return 0;

}