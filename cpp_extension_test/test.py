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

lse = 0
for i in range(1024):
    for j in range(1024):
        lse += (c[i,j] - c_cublas[i,j])**2

print(lse)

a = torch.randn(1024, 1024, dtype=torch.float16, device=device)
b = torch.randn(1024, 1024, dtype=torch.float16, device=device)

c = my_cuda_extension.gemm_register_pipelining_256(b.T, a.T)


c_cublas = torch.matmul(a,b)
c_cublas_fp32 = torch.matmul(a.float(),b.float())

lse = 0
lse_fp32 = 0
for i in range(1024):
    for j in range(1024):
        lse += (c[i,j] - c_cublas[i,j])**2
        lse_fp32 += (c[i,j] - c_cublas_fp32[i,j])**2

print(f"lse is {lse}")
print(f"lse_fp32 is {lse_fp32}")

for i in range(100):
    print(c[777,865+i], c_cublas[777, 865+i])

