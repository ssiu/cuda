from __future__ import annotations

import argparse
from contextlib import contextmanager
from typing import Iterator

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

from flash_attention_v2 import FlashAttentionForwardAmpere


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local CUTE FlashAttention v2 kernel and/or PyTorch cuDNN SDPA "
            "on identical inputs."
        )
    )
    parser.add_argument("--impl", choices=["compare", "fa2", "cudnn"], default="compare")
    parser.add_argument("--dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seqlen_q", type=int, default=8192)
    parser.add_argument("--seqlen_k", type=int, default=8192)
    parser.add_argument("--num_head", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--softmax_scale", type=float, default=0.5)
    parser.add_argument("--m_block_size", type=int, default=128)
    parser.add_argument("--n_block_size", type=int, default=64)
    parser.add_argument("--num_threads", type=int, default=128)
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--warmup_iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def cutlass_to_torch_dtype(dtype: cutlass.Numeric) -> torch.dtype:
    if dtype == cutlass.Float16:
        return torch.float16
    if dtype == cutlass.BFloat16:
        return torch.bfloat16
    raise ValueError(f"Unsupported attention dtype: {dtype}")


def create_input(
    shape: tuple[int, ...],
    dtype: cutlass.Numeric,
) -> tuple[cute.Tensor, torch.Tensor]:
    torch_tensor = (
        torch.empty(*shape, dtype=torch.int32, device="cuda")
        .random_(-2, 2)
        .to(dtype=cutlass_to_torch_dtype(dtype))
        .contiguous()
    )
    cute_tensor = (
        from_dlpack(torch_tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(
            mode=3,
            stride_order=torch_tensor.dim_order(),
            divisibility=(128 // dtype.width),
        )
    )
    return cute_tensor, torch_tensor


def create_output(shape: tuple[int, ...], dtype: cutlass.Numeric) -> tuple[cute.Tensor, torch.Tensor]:
    torch_tensor = torch.empty(*shape, dtype=cutlass_to_torch_dtype(dtype), device="cuda")
    cute_tensor = (
        from_dlpack(torch_tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(
            mode=3,
            stride_order=torch_tensor.dim_order(),
            divisibility=(128 // dtype.width),
        )
    )
    return cute_tensor, torch_tensor


@contextmanager
def cudnn_sdp_only() -> Iterator[None]:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:
        SDPBackend = None
        sdpa_kernel = None

    if SDPBackend is not None and hasattr(SDPBackend, "CUDNN_ATTENTION"):
        try:
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                yield
        except TypeError:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                yield
        return

    if not hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        raise RuntimeError(
            "This PyTorch build does not expose cuDNN SDPA. Install a PyTorch build "
            "with torch.backends.cuda.enable_cudnn_sdp or SDPBackend.CUDNN_ATTENTION."
        )

    toggles = (
        ("flash", "flash_sdp_enabled", "enable_flash_sdp"),
        ("math", "math_sdp_enabled", "enable_math_sdp"),
        ("mem_efficient", "mem_efficient_sdp_enabled", "enable_mem_efficient_sdp"),
        ("cudnn", "cudnn_sdp_enabled", "enable_cudnn_sdp"),
    )
    saved: dict[str, bool] = {}
    try:
        for name, getter, _ in toggles:
            if hasattr(torch.backends.cuda, getter):
                saved[name] = getattr(torch.backends.cuda, getter)()
        for _, _, setter in toggles:
            if hasattr(torch.backends.cuda, setter):
                getattr(torch.backends.cuda, setter)(setter == "enable_cudnn_sdp")
        yield
    finally:
        for name, _, setter in toggles:
            if name in saved:
                getattr(torch.backends.cuda, setter)(saved[name])


def run_cudnn_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
) -> torch.Tensor:
    q_ref = q.permute(0, 2, 1, 3).contiguous()
    k_ref = k.permute(0, 2, 1, 3).contiguous()
    v_ref = v.permute(0, 2, 1, 3).contiguous()
    with cudnn_sdp_only():
        out = torch.nn.functional.scaled_dot_product_attention(
            q_ref,
            k_ref,
            v_ref,
            scale=softmax_scale,
            is_causal=is_causal,
        )
    return out.permute(0, 2, 1, 3).contiguous()


def compile_fa2(
    args: argparse.Namespace,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    stream: cuda.CUstream,
):
    if not FlashAttentionForwardAmpere.can_implement(
        args.dtype,
        args.head_dim,
        args.m_block_size,
        args.n_block_size,
        args.num_threads,
        args.is_causal,
    ):
        raise TypeError(
            "Unsupported testcase: "
            f"{args.dtype}, head_dim={args.head_dim}, "
            f"m_block_size={args.m_block_size}, n_block_size={args.n_block_size}, "
            f"num_threads={args.num_threads}, is_causal={args.is_causal}"
        )

    fa2 = FlashAttentionForwardAmpere(
        args.head_dim,
        args.m_block_size,
        args.n_block_size,
        args.num_threads,
        args.is_causal,
    )
    return cute.compile(fa2, q, k, v, o, args.softmax_scale, stream, options="")


def cuda_time_us(fn, warmup_iterations: int, iterations: int):
    for _ in range(warmup_iterations):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    result = None
    start.record()
    for _ in range(iterations):
        result = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / max(iterations, 1), result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    shape_q = (args.batch_size, args.seqlen_q, args.num_head, args.head_dim)
    shape_kv = (args.batch_size, args.seqlen_k, args.num_head, args.head_dim)
    q, q_torch = create_input(shape_q, args.dtype)
    k, k_torch = create_input(shape_kv, args.dtype)
    v, v_torch = create_input(shape_kv, args.dtype)
    o, o_torch = create_output(shape_q, args.dtype)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    print("FlashAttention problem:")
    print(f"  impl: {args.impl}")
    print(f"  dtype: {args.dtype}")
    print(f"  q: {shape_q}")
    print(f"  k/v: {shape_kv}")
    print(f"  softmax_scale: {args.softmax_scale}")
    print(f"  is_causal: {args.is_causal}")

    compiled_fa2 = None
    if args.impl in ("compare", "fa2"):
        compiled_fa2 = compile_fa2(args, q, k, v, o, stream)

    if args.impl == "compare":
        compiled_fa2(q, k, v, o, args.softmax_scale, stream)
        cudnn_out = run_cudnn_sdpa(
            q_torch, k_torch, v_torch, args.softmax_scale, args.is_causal
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(o_torch, cudnn_out, atol=args.atol, rtol=args.rtol)
        diff = (o_torch.float() - cudnn_out.float()).abs()
        print(f"max_abs_diff={diff.max().item():.6f}")
        print(f"mean_abs_diff={diff.mean().item():.6f}")
        print("PASS")
        return

    if args.impl == "fa2":
        avg_us, _ = cuda_time_us(
            lambda: compiled_fa2(q, k, v, o, args.softmax_scale, stream),
            args.warmup_iterations,
            args.iterations,
        )
        print(f"fa2_avg_us={avg_us:.3f}")
        print("PASS")
        return

    avg_us, _ = cuda_time_us(
        lambda: run_cudnn_sdpa(
            q_torch, k_torch, v_torch, args.softmax_scale, args.is_causal
        ),
        args.warmup_iterations,
        args.iterations,
    )
    print(f"cudnn_avg_us={avg_us:.3f}")
    print("PASS")


if __name__ == "__main__":
    main()
