import torch
import my_cuda_extension

x = torch.ones(10, device="cuda")
print("Before:", x)

my_cuda_extension.cuda_kernel(x)

print("After:", x)  # Each element should be multiplied by 2