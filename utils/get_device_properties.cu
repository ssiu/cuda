// System includes
#include <stdio.h>
#include <assert.h>
#include <iostream>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>


// testing upload
int main() {
    cudaProfilerStart();

    int device = 0; // Assuming the device ID is 0

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    
    std::cout << "--- High-Level ---" << std::endl;
    std::cout << "Device:  " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Number of SMs: " << deviceProp.multiProcessorCount << std::endl;
    
    std::cout << std::endl;
    
    std::cout << "--- Global Memory ---" << std::endl;
    std::cout << "Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << "MB" << std::endl;
    
    std::cout << std::endl;
    
    std::cout << "--- Shared Memory ---" << std::endl;
    std::cout << "Shared memory per SM: " << deviceProp.sharedMemPerMultiprocessor / (1024) << "KB" << std::endl;
    std::cout << "Shared memory per block: " << deviceProp.sharedMemPerBlock / (1024) << "KB" << std::endl;
    
    std::cout << std::endl;    
    
    std::cout << "--- Registers ---" << std::endl;
    std::cout << "Registers per SM: " << deviceProp.regsPerMultiprocessor * 4 / (1024) << "KB" << std::endl;
    std::cout << "Registers per block: " << deviceProp.regsPerBlock * 4 / (1024) << "KB" << std::endl;
    
    std::cout << std::endl;
    
    std::cout << "--- Grid ---" << std::endl;
    std::cout << "Max grid dimension: (" << deviceProp.maxGridSize[0] << ", "
              << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
              
    std::cout << std::endl;
    
    std::cout << "--- Block ---" << std::endl;
    std::cout << "Max block per SM: " << deviceProp.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Max block dimension: (" << deviceProp.maxThreadsDim[0] << ", "
              << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
              
    std::cout << std::endl;          
    
    std::cout << "--- Warp ---" << std::endl;
    std::cout << "threads in a warp: " << deviceProp.warpSize << std::endl;
    std::cout << "Max warp per SM: " << deviceProp.maxThreadsPerMultiProcessor << " / " 
              << deviceProp.warpSize << " = " << deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize << std::endl;           
    std::cout << "Max warp per block: " << deviceProp.maxThreadsPerBlock << " / " 
              << deviceProp.warpSize << " = " << deviceProp.maxThreadsPerBlock / deviceProp.warpSize << std::endl;
    
    std::cout << std::endl;
    
    std::cout << "--- Thread ---" << std::endl;    
    std::cout << "Max threads per SM: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    
    std::cout << std::endl;
    
    cudaProfilerStop();
    

    return 0;
}