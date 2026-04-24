from tensor_layouts import Layout, compose, complement, logical_divide

# A 4x8 column-major layout: offset(i,j) = i + j*4
layout = Layout((4, 8), (1, 4))
print(layout)       # (4, 8) : (1, 4)
print(layout(2, 3)) # 14

# Compose two layouts
a = Layout((4, 2), (1, 4))
b = Layout((2, 4), (4, 1))
print(compose(a, b))

# Tile a layout into 2x4 blocks
tiler = Layout((2, 4))
print(logical_divide(layout, tiler))