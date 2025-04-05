import torch
import my_cuda_extension

x = torch.ones(10, device="cuda")
print("Before:", x)

my_cuda_extension.cuda_kernel(x)

print("After:", x)  # Each element should be multiplied by 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = torch.randn(1024, 1024, dtype=torch.float32, device=device)
b = torch.randn(1024, 1024, dtype=torch.float32, device=device)
c = my_cuda_extension.mm_new_8(a, b)
c_cublas = torch.matmul(a,b)


sum_error = torch.sum(torch.abs(c - c_cublas))

print(sum_error)

###


a = torch.randn(1024, 1024, dtype=torch.float16, device=device)
b = torch.randn(1024, 1024, dtype=torch.float16, device=device)

c = my_cuda_extension.gemm_register_pipelining_256(b.T, a.T)


c_cublas = torch.matmul(a,b)
c_cublas_fp32 = torch.matmul(a.float(),b.float())


sum_error = torch.sum(torch.abs(c - c_cublas))
sum_error_fp32 = torch.sum(torch.abs(c - c_cublas_fp32))

print(sum_error)
print(sum_error_fp32)




