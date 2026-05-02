from pathlib import Path


def map_coord(x: int, y: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Map (x, y) in a 32x32 tile to ((m, singleton), (n, k))."""
    if not (0 <= x < 32 and 0 <= y < 32):
        raise ValueError(f"expected 0 <= x,y < 32, got ({x}, {y})")

    return ((x % 32, 0), (y % 8, (y // 8) % 4))


def layout_offset(coord: tuple[tuple[int, int], tuple[int, int]]) -> int:
    """Apply Layout(((32, 1), (8, 4)), ((1, 0), (32, 256)))."""
    (m, singleton), (n, k) = coord
    return m * 1 + singleton * 0 + n * 32 + k * 256


def main() -> None:
    output_path = Path(__file__).with_name("test_layout.txt")
    lines = []

    for y in range(32):
        for x in range(32):
            mapped = map_coord(x, y)
            offset = layout_offset(mapped)
            lines.append(
                f"({x:2d},{y:2d}) -> "
                f"(({mapped[0][0]:2d},{mapped[0][1]:1d}), "
                f"({mapped[1][0]:1d},{mapped[1][1]:1d})) -> "
                f"{offset:4d}"
            )

    output_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
