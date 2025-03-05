import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import flash_attn_turing

torch.set_printoptions(precision=8)


def get_error(batch_size=1, seqlen=16, nheads=1, headdim=128):

    query = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")
    key = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")
    value = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")

    # for pytorch function
    # (batch_size, nheads, seqlen, headdim)
    query_torch = query.permute(0, 2, 1, 3).contiguous().clone()
    key_torch = key.permute(0, 2, 1, 3).contiguous().clone()
    value_torch = value.permute(0, 2, 1, 3).contiguous().clone()


    output = flash_attn_turing.flash_fwd_v15(query, key, value,
                                           batch_size, seqlen, nheads, headdim)

    torch.cuda.synchronize()
    # (batch_size, nheads, seqlen, headdim)
    output_torch = F.scaled_dot_product_attention(query_torch, key_torch, value_torch)
    torch.cuda.synchronize()

    output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()

    sum_error = torch.sum(torch.abs(output - output_torch))
    avg_error = sum_error / (batch_size * seqlen * nheads * headdim)
    max_error = torch.max(torch.abs(output - output_torch))

    max_error_index = torch.argmax(torch.abs(output - output_torch))

    # Convert the flat index to multi-dimensional indices (if needed)
    max_error_indices = torch.unravel_index(max_error_index, output.shape)

    # Extract the values at the maximum error location
    output_value = output[max_error_indices]
    output_torch_value = output_torch[max_error_indices]


    return sum_error, avg_error, max_error, output_value, output_torch_value


with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:

    sum_error, avg_error, max_error, output_value, output_torch_value = get_error(batch_size=4, seqlen=4096, nheads=32, headdim=128)


print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(f"sum_error = {sum_error}, avg_error = {avg_error}, max_error = {max_error},\nmax_error output = {output_value}, max_error output torch = {output_torch_value}")






