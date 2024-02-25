say_hello:
	echo "Hello World!"

%.o: %.cu
	nvcc -o $@ -std=c++17 -arch=sm_70 -I/mnt/shared/swsiu/cutlass/sm70/cutlass/include $<

matrix_multiplication/launch_mm.o: matrix_multiplication/launch_mm.cu
	rm matrix_multiplication/launch_mm.o && nvcc -o $@ -std=c++17 -arch=sm_70 -I/mnt/shared/swsiu/cutlass/sm70/cutlass/include $< \
	matrix_multiplication/mm_0.cu matrix_multiplication/mm_1.cu matrix_multiplication/utils.cuh -lcublas

clean:
	rm -f *.o


