from __future__ import annotations

import argparse

import torch


def configure_reference_math() -> None:
    # Disable TF32 so the reference path is as strict and repeatable as possible.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def _validate_output(out: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    if out.device.type != "cuda":
        raise ValueError("out must be a CUDA tensor")
    if tuple(out.shape) != shape:
        raise ValueError(f"out has shape {tuple(out.shape)}, expected {shape}")
    if out.dtype != dtype:
        raise ValueError(f"out has dtype {out.dtype}, expected {dtype}")


def cublas_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    accumulate_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if a.ndim not in (2, 3) or b.ndim != a.ndim:
        raise ValueError("cublas_matmul expects matching 2D or 3D tensors")
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise ValueError("cublas_matmul expects CUDA tensors")
    if a.dtype != b.dtype:
        raise ValueError(f"a and b must have the same dtype, got {a.dtype} and {b.dtype}")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported dtype: {a.dtype}")

    compute_dtype = accumulate_dtype or a.dtype
    if compute_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported accumulate dtype: {compute_dtype}")

    if a.ndim == 2:
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"incompatible shapes: {a.shape} x {b.shape}")
        result = torch.matmul(a.to(compute_dtype), b.to(compute_dtype))
    else:
        if a.shape[1] != b.shape[1] or a.shape[2] != b.shape[2]:
            raise ValueError(f"incompatible dense_gemm shapes: {a.shape} x {b.shape}")

        # dense_gemm stores tensors as A[m,k,l], B[n,k,l], C[m,n,l].
        a_batch = a.permute(2, 0, 1).to(compute_dtype)
        b_batch = b.permute(2, 1, 0).to(compute_dtype)
        result = torch.matmul(a_batch, b_batch).permute(1, 2, 0)

    if out is None:
        return result.to(a.dtype)

    _validate_output(out, tuple(result.shape), out.dtype)
    out.copy_(result.to(out.dtype))
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cuBLAS-backed matmul through PyTorch.")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_reference_math()
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)

    if args.l == 1:
        a = torch.randn((args.m, args.k), device="cuda", dtype=dtype)
        b = torch.randn((args.k, args.n), device="cuda", dtype=dtype)
    else:
        a = torch.randn((args.m, args.k, args.l), device="cuda", dtype=dtype)
        b = torch.randn((args.n, args.k, args.l), device="cuda", dtype=dtype)

    c = cublas_matmul(a, b, accumulate_dtype=torch.float32 if dtype != torch.float32 else None)
    torch.cuda.synchronize()

    print(f"A: {tuple(a.shape)}")
    print(f"B: {tuple(b.shape)}")
    print(f"C: {tuple(c.shape)}")
    print(f"mean={c.mean().item():.6f} max={c.max().item():.6f} min={c.min().item():.6f}")


if __name__ == "__main__":
    main()
