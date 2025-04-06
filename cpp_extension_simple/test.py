import torch
import simple
# outputs = simple.Dummy(3)
# print(outputs)
t = torch.randn(2,3, dtype=torch.float16).to("cuda")
a, b = simple.Dummy(2)
print(a)
print(b)
