import pandas as pd
import matplotlib.pyplot as plt

NUMS  = [512, 1024, 2048, 4096, 8192]

def clean_kernel_names(kernel_name):
    if 'void' in kernel_name:
        return kernel_name.split('<')[0].replace('void ', '')
    else:
        return "cublas_turing"


def count_skip_rows(file_path):
    marker_line = "==PROF== Disconnected from process"
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if marker_line in line:
                return i + 1  # Skip the marker line as well
    return 0  # Default to 0 if the marker is not found


def plot_kernels(df):
    kernel_names = df["Kernel Name"].unique()
    # Display the list of unique kernels
    print(kernel_names)

    x_ticks = sorted(df["size"].unique())

    plt.figure(figsize=(10, 6))
    for kernel in kernel_names:
        kernel_df = df[df["Kernel Name"] == kernel]  # Filter rows for the kernel
        plt.plot(kernel_df["size"], kernel_df["Metric Value"], marker='o', label=kernel)

    #plt.xscale("log")
    plt.xticks(x_ticks)
    # Add labels, title, legend, and grid
    plt.xlabel("Size", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.title("Metric Value vs. Size for Different Kernels", fontsize=14)
    plt.legend(title="Kernels", fontsize=10)
    plt.grid(True)

    # Show the plot
    # plt.show()

    plt.savefig("kernel_plots.png")
    plt.close()


list_of_dfs = []
for num in NUMS:
    # Path to your CSV file
    csv_file_path = f"{num}.csv"
    skip_rows = count_skip_rows(csv_file_path)
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, skiprows=skip_rows)

    # Display the DataFrame

    columns_to_keep = ["Kernel Name", "Metric Value"]
    subset_df = df[columns_to_keep].copy()
    subset_df["Kernel Name"] = subset_df["Kernel Name"].apply(clean_kernel_names)
    subset_df["size"] = num
    # print(subset_df)
    # print(subset_df.dtypes)
    # Display the first few rows of the DataFrame
    list_of_dfs.append(subset_df)

df = pd.concat(list_of_dfs)

plot_kernels(df)