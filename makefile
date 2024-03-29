say_hello:
	echo "Hello World!"

%.o: %.cu
	nvcc -o $@ -std=c++17 -arch=sm_70 -I/mnt/shared/swsiu/cutlass/sm70/cutlass/include $<


mm: clean_mm matrix_multiplication/launch_mm.o

matrix_multiplication/launch_mm.o: matrix_multiplication/launch_mm.cu
	nvcc -o $@ -std=c++17 -arch=sm_70 -I/mnt/shared/swsiu/cutlass/sm70/cutlass/include $< \
	matrix_multiplication/mm_*.cu -lcublas


clean_mm:
	rm -f matrix_multiplication/*.o


test: clean_test matrix_multiplication/test.o


matrix_multiplication/test.o: matrix_multiplication/test.cu
	nvcc -o $@ -std=c++17 -arch=sm_70  $<

clean_test:
	rm -f matrix_multiplication/test.o
