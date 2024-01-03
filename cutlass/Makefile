say_hello:
	echo "Hello World!"

%.o: %.cu
	nvcc -o *.o -std=c++17 -arch=sm_70 -I/mnt/shared/swsiu/cutlass/sm70/cutlass/include *.cu

clean:
	rm -f *.o
