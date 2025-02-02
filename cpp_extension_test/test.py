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

