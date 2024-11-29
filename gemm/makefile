NVCC_FLAGS = -lineinfo -std=c++17 --expt-relaxed-constexpr -arch=sm_75 -O3
# NVCC_FLAGS = -lineinfo -std=c++17 -arch=sm_70 -O3
NVCC_LDFLAGS = -lcublas
NVCC_INCLUDES = -I./cutlass/include -I./cutlass/tools/util/include

INPUT = $<
OUTPUT = -o $@

clean:
	rm -rf *.o

%.o: %.cu
	nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(OUTPUT) $(INPUT)


#/usr/local/cuda/bin/nvcc -ccbin=g++ -O3 -std=c++17 -ftemplate-backtrace-limit=0 -arch sm_70 -uumn --expt-extended-lambda --expt-relaxed-constexpr --use_fast_math -Xptxas -v --compiler-options "-O3 -std=c++17 -ftemplate-backtrace-limit=0 -Wall -Wno-unused-local-typedefs -Wno-strict-aliasing -Wno-unused-function -Wno-format-security -Wno-unknown-pragmas -Wno-psabi" -I. -I/usr/local/cuda/include -I/home/ccecka/Desktop/mnt/ccecka_nvresearch/kernel_store/cutlass/include -I/home/ccecka/Desktop/mnt/ccecka_nvresearch/kernel_store/cutlass/tools/util/include -I/home/ccecka/Desktop/mnt/ccecka_nvresearch/kernel_store/cutlass/test -I/home/ccecka/Desktop/mnt/ccecka_nvresearch/kernel_store/cutlass/examples -o scratch_bug scratch_bug.cu -L/usr/local/cuda/lib64
