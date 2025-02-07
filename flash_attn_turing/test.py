import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import flash_attn_turing

def get_lse(batch_size=1, seqlen=16, nheads=1, headdim=128):
    # for custom flash attention
    # (batch_size, seqlen, nheads, headdim)
    query = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")
    key = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")
    value = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")
    output = flash_attn_turing.flash_fwd_v0(query, key, value,
                                           batch_size, seqlen, nheads, headdim)

    # for pytorch function
    # (batch_size, nheads, seqlen, headdim)
    query_torch = query.permute(0, 2, 1, 3).contiguous().clone()
    key_torch = key.permute(0, 2, 1, 3).contiguous().clone()
    value_torch = value.permute(0, 2, 1, 3).contiguous().clone()

    # (batch_size, nheads, seqlen, headdim)
    output_torch = F.scaled_dot_product_attention(query_torch, key_torch, value_torch)
    output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()

    lse = 0
    count = 0
    for i in range(batch_size):
        for j in range(seqlen):
            for k in range(nheads):
                for l in range(headdim):
                    if count < 1:
                        print(i,j,k,l,output[i,j,k,l], output_torch[i,j,k,l])
                        count += 1
                    #print(i, j, k, l, output[i,j,k,l].item(), output_torch[i,j,k,l].item())
                    lse += (output[i,j,k,l] - output_torch[i,j,k,l])**2
    print("=====")
    return lse

lse_16=0
#lse_16 = get_lse(batch_size=1, seqlen=16, nheads=1, headdim=128)
lse_32 = get_lse(batch_size=4, seqlen=32, nheads=4, headdim=128)

print(f"lse_16 = {lse_16}, lse_32 = {lse_32}")


# batch_size = 1
# seqlen = 32
# nheads = 1
# headdim = 128
#
# #(batch_size, seqlen, nheads, headdim)
#
# # for custom flash attention
# # (batch_size, seqlen, nheads, headdim)
# query = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")
# key = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")
# value = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")
# #output = flash_attn_turing.flash_fwd_v0(query, key, value,
# #                                        batch_size, seqlen, nheads, headdim)
#
# # for pytorch function
# # (batch_size, nheads, seqlen, headdim)
# query_torch = query.permute(0, 2, 1, 3).contiguous().clone()
# key_torch = key.permute(0, 2, 1, 3).contiguous().clone()
# value_torch = value.permute(0, 2, 1, 3).contiguous().clone()
#
# # (batch_size, nheads, seqlen, headdim)
# #output_torch = F.scaled_dot_product_attention(query_torch, key_torch, value_torch)
#
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     # with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
#     #     output =  F.scaled_dot_product_attention(query, key, value)
#     output = flash_attn_turing.flash_fwd_v0(query, key, value,
#                                             batch_size, seqlen, nheads, headdim)
#     output_torch = F.scaled_dot_product_attention(query_torch, key_torch, value_torch)
#
#
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#
#
# # (batch_size, seqlen, nheads, headdim)
# output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()



