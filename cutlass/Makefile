say_hello:
	echo "Hello World!"

%.o: %.cu
	#nvcc -o $@ -std=c++17 -arch=sm_70 -I/mnt/shared/swsiu/cutlass/sm70/cutlass/include $<
	nvcc -std=c++17 -arch=sm_70 -I/mnt/shared/swsiu/cutlass/sm70/cutlass/include -o $@ -c $<

clean:
	rm -f *.o
