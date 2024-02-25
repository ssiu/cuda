# from glob import glob
# # input:
# # list of csv in a folder
# # kernel names to be profiled
#
#
# def get_df('folder_path'):
#

import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
folder = 'kernel_csv'

file_order = [2**i for i in range(10, 21)]
#print(file_order)
copy_32 = []
copy_64 = []
copy_128 = []
for num in file_order:
    file = f"{folder}/profile_{num}.csv"
    df = pd.read_csv(file, skiprows=3)
    #print(df['Time'].iloc[0])
    times = []

    if df['Time'].iloc[0] == 's':
        for i in range(3):
            times.append(float(df['Time'].iloc[i+3])*1000)
    else:
        for i in range(3):
            times.append(float(df['Time'].iloc[i+3]))


    copy_32.append(times[0])
    copy_64.append(times[1])
    copy_128.append(times[2])

nums = [2**i for i in range(10,21)]
# print(copy_32)
# print(copy_64)
# print(copy_128)

# Plotting the lines with dots on the points
plt.plot(nums, copy_32, marker='o', label='copy_32')
plt.plot(nums, copy_64, marker='o', label='copy_64')
plt.plot(nums, copy_128, marker='o', label='copy_128')


# Adding a title
plt.title('Multiple Line Graphs')

# Adding X and Y axis labels
plt.xlabel('X axis')
plt.ylabel('Y axis')

# Showing the legend
plt.legend()

# Display the plot
plt.show()


# # Sample data for three lines
# x1 = [1, 2, 3, 4, 5]
# y1 = [1, 2, 3, 4, 5]  # Line 1 - y = x
#
# x2 = [1, 2, 3, 4, 5]
# y2 = [2, 3, 4, 5, 6]  # Line 2 - y = x + 1
#
# x3 = [1, 2, 3, 4, 5]
# y3 = [3, 4, 5, 6, 7]  # Line 3 - y = x + 2
#
# # Plotting the lines with dots on the points
# plt.plot(x1, y1, marker='o', label='Line 1')
# plt.plot(x2, y2, marker='o', label='Line 2')
# plt.plot(x3, y3, marker='o', label='Line 3')
#
#
# # Adding a title
# plt.title('Multiple Line Graphs')
#
# # Adding X and Y axis labels
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
#
# # Showing the legend
# plt.legend()
#
# # Display the plot
# plt.show()