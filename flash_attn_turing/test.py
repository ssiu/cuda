import torch
import torch.nn.functional as F
import flash_attn_turing


batch_size = 32
seqlen = 1024
nheads = 8
headdim = 128

(batch_size, seqlen, nheads, headdim)
query = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")
key = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")
value = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")
output = flash_attn_turing.flash_fwd_v0(query, key, value,
                                        batch_size, seqlen, nheads, headdim)


# (batch_size, nheads, seqlen, headdim)
query_torch = query.permute(0, 2, 1, 3).contiguous().clone()
key_torch = key.permute(0, 2, 1, 3).contiguous().clone()
value_torch = value.permute(0, 2, 1, 3).contiguous().clone()

# (batch_size, nheads, seqlen, headdim)
output_torch = F.scaled_dot_product_attention(query_torch, key_torch, value_torch)

# (batch_size, seqlen, nheads, headdim)
output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()

print(output[0,0,0,0])
print(output_torch[0,0,0,0])



# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     # with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
#     #     output =  F.scaled_dot_product_attention(query, key, value)
#     output = F.scaled_dot_product_attention(query, key, value)