NVCC_FLAGS = -lineinfo -std=c++17 --expt-relaxed-constexpr -arch=sm_75 -O3
# NVCC_FLAGS = -lineinfo -std=c++17 -arch=sm_70 -O3
NVCC_LDFLAGS = -lcublas
NVCC_INCLUDES = -I./cutlass/include -I./cutlass/tools/util/include
#NVCC_GEMM_FILES = #*.cu

INPUT = $<
OUTPUT = -o $@

clean:
	rm -rf *.o

%.o: %.cu
	nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(OUTPUT) $(INPUT)


# nvcc -lineinfo -std=c++17 --expt-relaxed-constexpr -arch=sm_70 -O3 -I/mnt/shared/swsiu/cuda/cutlass/cutlass/include -I/mnt/shared/swsiu/cuda/cutlass/cutlass/tools/util/include -o sgemm_sm70.o sgemm_sm70.cu