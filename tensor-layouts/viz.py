import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from tensor_layouts import Layout, Swizzle
from tensor_layouts.viz import draw_layout, draw_swizzle


def main():
    out_dir = Path(__file__).resolve().parent
    # layout_path = out_dir / "row_major_8x8.png"
    # swizzle_path = out_dir / "swizzle_8x8.png"

    # draw_layout(
    #     Layout((8, 8), (8, 1)),
    #     filename=layout_path,
    #     title="Row-Major 8x8",
    #     colorize=True,
    # )
    # draw_swizzle(
    #     Layout((8, 8), (8, 1)),
    #     Swizzle(3, 0, 3),
    #     filename=swizzle_path,
    #     colorize=True,
    # )

    # print(f"Saved {layout_path}")
    # print(f"Saved {swizzle_path}")

    test_path = out_dir / "test.png"
    draw_layout(
        Layout(((32,1),(8,4),(1,2)),((1,0),(32,256),(0,1024))),
        filename=test_path,
        title="Row-Major 8x8",
        colorize=True,
    )
    print(f"Saved {test_path}")



if __name__ == "__main__":
    main()
