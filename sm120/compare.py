from __future__ import annotations

import argparse
from typing import Tuple

import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as cutlass_utils

from cublas_matmul import cublas_matmul, configure_reference_math
from dense_gemm import Sm120GemmKernel


def parse_comma_separated_ints(s: str) -> tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dense_gemm and cuBLAS matmul on the same tensors and compare results."
    )
    parser.add_argument("--mnkl", type=parse_comma_separated_ints, default=(128, 128, 128, 1))
    parser.add_argument(
        "--tile_shape_mnk",
        type=parse_comma_separated_ints,
        choices=[
            (64, 64, 64),
            (64, 128, 64),
            (128, 64, 64),
            (128, 128, 64),
            (128, 256, 64),
            (128, 128, 128),
        ],
        default=(128, 128, 64),
    )
    parser.add_argument("--a_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--b_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument("--a_major", choices=["k", "m"], default="k")
    parser.add_argument("--b_major", choices=["k", "n"], default="k")
    parser.add_argument("--c_major", choices=["n", "m"], default="n")
    parser.add_argument("--tolerance", type=float, default=1e-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--print-first",
        type=int,
        default=32,
        help="Print the first N output values from dense_gemm and cuBLAS.",
    )
    parser.add_argument(
        "--print-fp32-ref",
        action="store_true",
        help="Also print the first N pure-FP32 cuBLAS reference values.",
    )
    return parser.parse_args()


def create_cute_tensor(data_ref: torch.Tensor, cutlass_dtype) -> tuple[cute.Tensor, torch.Tensor]:
    cute_tensor, torch_tensor = cutlass_torch.cute_tensor_like(data_ref, cutlass_dtype, True, 16)

    if cutlass_dtype.is_float and cutlass_dtype.width == 8:
        f32_torch_tensor = data_ref.to(dtype=torch.float32)
        cute_tensor = cutlass_torch.convert_cute_tensor(
            f32_torch_tensor,
            cute_tensor,
            cutlass_dtype,
            is_dynamic_layout=True,
        )

    return cute_tensor, torch_tensor


def cutlass_to_torch_dtype(dtype) -> torch.dtype:
    if dtype == cutlass.Float16:
        return torch.float16
    if dtype == cutlass.BFloat16:
        return torch.bfloat16
    if dtype == cutlass.Float32:
        return torch.float32
    raise ValueError(f"Unsupported dtype for torch conversion: {dtype}")


def build_inputs(
    mnkl: Tuple[int, int, int, int],
    a_dtype,
    b_dtype,
    c_dtype,
    a_major: str,
    b_major: str,
    c_major: str,
) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n, k, l = mnkl

    a_torch_cpu = cutlass_torch.matrix(l, m, k, a_major, a_dtype)
    b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major, b_dtype)
    c_torch_cpu = cutlass_torch.matrix(l, m, n, c_major, c_dtype)

    a_tensor, a_torch_gpu = create_cute_tensor(a_torch_cpu, a_dtype)
    b_tensor, b_torch_gpu = create_cute_tensor(b_torch_cpu, b_dtype)
    c_tensor, c_torch_gpu = create_cute_tensor(c_torch_cpu, c_dtype)

    return a_tensor, b_tensor, c_tensor, a_torch_gpu, b_torch_gpu, c_torch_gpu


def run_dense_gemm(
    a_tensor: cute.Tensor,
    b_tensor: cute.Tensor,
    c_tensor: cute.Tensor,
    acc_dtype,
    tile_shape_mnk: Tuple[int, int, int],
):
    gemm = Sm120GemmKernel(acc_dtype, tile_shape_mnk)
    hardware_info = cutlass_utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(1)
    stream = cutlass_torch.default_stream()
    compiled_gemm = cute.compile(
        gemm, a_tensor, b_tensor, c_tensor, max_active_clusters, stream
    )
    compiled_gemm(a_tensor, b_tensor, c_tensor, stream)
    return stream


def print_first_values(
    dense_out: torch.Tensor,
    cublas_out: torch.Tensor,
    count: int = 32,
) -> None:
    if count <= 0:
        return

    dense_flat = dense_out.detach().reshape(-1).cpu()
    cublas_flat = cublas_out.detach().reshape(-1).cpu()
    count = min(count, dense_flat.numel(), cublas_flat.numel())

    print(f"First {count} values:")
    for idx in range(count):
        dense_value = format(float(dense_flat[idx].item()), ".17g")
        cublas_value = format(float(cublas_flat[idx].item()), ".17g")
        print(f"[{idx:02d}] dense_gemm={dense_value} cublas={cublas_value}")


def print_first_fp32_reference(a: torch.Tensor, b: torch.Tensor, count: int = 32) -> None:
    if count <= 0:
        return

    fp32_ref = cublas_matmul(a, b, accumulate_dtype=torch.float32).to(torch.float32)
    fp32_flat = fp32_ref.detach().reshape(-1).cpu()
    count = min(count, fp32_flat.numel())

    print(f"First {count} FP32 reference values:")
    for idx in range(count):
        print(f"[{idx:02d}] fp32_ref={fp32_flat[idx].item():.9f}")


def main() -> None:
    args = parse_args()
    if len(args.mnkl) != 4:
        raise ValueError("--mnkl must contain exactly 4 values")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this comparison")

    configure_reference_math()
    torch.manual_seed(args.seed)

    a_tensor, b_tensor, c_tensor, a_torch_gpu, b_torch_gpu, c_torch_gpu = build_inputs(
        args.mnkl,
        args.a_dtype,
        args.b_dtype,
        args.c_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
    )

    run_dense_gemm(a_tensor, b_tensor, c_tensor, args.acc_dtype, args.tile_shape_mnk)

    cublas_out = torch.empty_like(c_torch_gpu)
    cublas_matmul(
        a_torch_gpu,
        b_torch_gpu,
        out=cublas_out,
        accumulate_dtype=cutlass_to_torch_dtype(args.acc_dtype),
    )
    torch.cuda.synchronize()

    dense_out = c_torch_gpu
    torch.testing.assert_close(dense_out, cublas_out, atol=args.tolerance, rtol=1e-3)
    print_first_values(dense_out, cublas_out, args.print_first)
    if args.print_fp32_ref:
        print_first_fp32_reference(a_torch_gpu, b_torch_gpu, args.print_first)

    diff = (dense_out.float() - cublas_out.float()).abs()
    print(f"A shape: {tuple(a_torch_gpu.shape)} dtype={a_torch_gpu.dtype}")
    print(f"B shape: {tuple(b_torch_gpu.shape)} dtype={b_torch_gpu.dtype}")
    print(f"C shape: {tuple(dense_out.shape)} dtype={dense_out.dtype}")
    print(f"max_abs_diff={diff.max().item():.6f}")
    print(f"mean_abs_diff={diff.mean().item():.6f}")
    print("PASS")


if __name__ == "__main__":
    main()
