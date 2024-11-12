NVCC_FLAGS = -lineinfo -std=c++17 -arch=sm_75 -O3
NVCC_LDFLAGS = -lcublas
NVCC_INCLUDES = -I/cutlass/include -I/cutlass/tools/util/include

INPUT = $<
OUTPUT = -o $@

clean:
	rm -rf *.o

%.o: %.cu
	nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(OUTPUT) $(INPUT)