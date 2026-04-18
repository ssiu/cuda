from __future__ import annotations

import argparse
import sys

import compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper around sm120/compare.py."
    )
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.argv = [
        "compare.py",
        "--mnkl",
        f"{args.m},{args.n},{args.k},{args.l}",
    ]
    compare.main()


if __name__ == "__main__":
    main()
