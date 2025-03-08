# two plots
# one for time, for each seq len
# one for throughput, for each seq len
# assume 512, 1024, 2048, 4096, 16384

import pandas as pd
import matplotlib.pyplot as plt

# remove pytorch reduction kernels
# remove unwanted columns
def filter_df(df):
    # remove unwanted kernels
    keywords = ['void at::']

    pattern = '|'.join(keywords)

    df_filtered = df[~df['Kernel Name'].str.contains(pattern, case=False)]

    # remove unwanted columns
    columns_to_keep = ["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"]
    df_filtered = df_filtered[columns_to_keep].copy()


    return df_filtered


def clean_kernel_names(kernel_name):
    if 'flash' in kernel_name:
        return "flash attention"
    elif "fmha" in kernel_name:
        return "memory efficient attention"
    else:
        return kernel_name

def clean_metric_names(metric_name):
    if metric_name == "gpu__time_duration.sum":
        return "duration"
    elif metric_name == "sm__throughput.avg.pct_of_peak_sustained_elapsed":
        return "compute throughput"
    else:
        return metric_name

def count_skip_rows(file_path):
    marker_line = "==PROF== Disconnected from process"
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if marker_line in line:
                return i + 1  # Skip the marker line as well
    return 0  # Default to 0 if the marker is not found



df_list = []
NUMS  = [512, 1024, 2048, 4096, 8192]

for num in NUMS:
    # Path to your CSV file
    csv_file_path = f"{num}.csv"
    skip_rows = count_skip_rows(csv_file_path)
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, skiprows=skip_rows)

    #columns_to_keep = ["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"]
    # = df[columns_to_keep].copy()
    filtered_df = filter_df(df)

    filtered_df["Kernel Name"] = filtered_df["Kernel Name"].apply(clean_kernel_names)
    filtered_df["Metric Name"] = filtered_df["Metric Name"].apply(clean_metric_names)
    filtered_df["seq_len"] = num
    #print(filtered_df.head()
    df_list.append(filtered_df)

df_combined = pd.concat(df_list, axis=0, ignore_index=True)

print(df_combined)