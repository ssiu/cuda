say_hello:
	echo "Hello World!"

.PHONY: matmul_talk matmul_talk/launch_mm.o

mm: clean_mm matrix_multiplication/launch_mm.o

sum: sum/launch_sum.o

sum/launch_sum.o: sum/launch_sum.cu
	nvcc -O3 -lineinfo -o $@ -std=c++17 -arch=sm_70 $< \
	sum/sum_*.cu -lcudart


matmul_talk: matmul_talk/launch_mm.o

# matrix_multiplication/launch_mm.o: matrix_multiplication/launch_mm.cu
# 	nvcc -O3 -lineinfo -o $@ -std=c++17 -arch=sm_70 -I/mnt/shared/swsiu/cutlass/sm70/cutlass/include $< \
# 	matrix_multiplication/mm_*.cu matrix_multiplication/yz_mm_*.cu -lcublas

matmul_talk/launch_mm.o: matmul_talk/launch_mm.cu
	nvcc -O3 -lineinfo -o $@ -std=c++17 -arch=sm_70 $< \
	matmul_talk/mm_*.cu -lcublas

profile: sum/launch_sum.o #matmul_talk/launch_mm.o
	sudo ncu -f --target-processes all --set full -o $@ ./$<

matrix_multiplication/launch_mm.o: matrix_multiplication/launch_mm.cu
	nvcc -O3 -lineinfo -o $@ -std=c++17 -arch=sm_70 $< \
	matrix_multiplication/mm_*.cu matrix_multiplication/yz_mm_*.cu -lcublas

clean_mm:
	rm -f matrix_multiplication/*.o


test: clean_test matrix_multiplication/test.o


matrix_multiplication/test.o: matrix_multiplication/test.cu
	nvcc -o $@ -std=c++17 -arch=sm_70  $<


%.o: %.cu
	nvcc -lineinfo -o $@ -std=c++17 -arch=sm_70 -I/mnt/shared/swsiu/cutlass/sm70/cutlass/include $< -lcublas