say_hello:
	echo "Hello World!"

%.o: %.cu
	nvcc -o $@ -std=c++17 -arch=sm_70 -I/mnt/shared/swsiu/cutlass/sm70/cutlass/include $<

matrix_multiplication/launch_mm.o: matrix_multiplication/launch_mm.cu
	nvcc -o $@ -std=c++17 -arch=sm_70 -I/mnt/shared/swsiu/cutlass/sm70/cutlass/include $< matrix_multiplication/mm_0.cu matrix_multiplication/mm_cublas.cu -lcublas

clean:
	rm -f *.o


