import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib import font_manager  # added

# Define the main folder path
main_folder = r"D:\Mphil\Dataset"
output_excel = r"D:\Mphil\Dataset\statistics_summary.xlsx"

# Define custom font properties for ticks
x_tick_font = font_manager.FontProperties(family='cursive', style='italic', size=12)
# y will use family/size/color directly

# List to store statistics for all folders
stats_list = []

# Iterate through all subdirectories (P1, P10, etc.)
for subfolder in Path(main_folder).iterdir():
    if subfolder.is_dir():
        npy_file = subfolder / "data.npy"

        if npy_file.exists():
            # Load data
            data = np.load(npy_file)
            data_flatten = data.flatten()

            # Calculate statistics
            stats = {
                'Folder': subfolder.name,
                'Shape': str(data.shape),
                'Flatten_Shape': str(data_flatten.shape),
                'Min': np.min(data_flatten),
                'Max': np.max(data_flatten),
                'PTP': np.ptp(data_flatten),
                'Mean': np.mean(data_flatten),
                'Std': np.std(data_flatten),
                'Q1': np.quantile(data_flatten, 0.25),
                'Median': np.median(data_flatten),
                'Q3': np.quantile(data_flatten, 0.75)
            }
            stats_list.append(stats)

            # Create histogram plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data_flatten, bins=1000, log_scale=False,
                         color='cornflowerblue', ax=ax)

            # Title and axis labels
            ax.set_title(f'Histogram - {subfolder.name}',
                         fontname="Times New Roman", fontsize=20)
            ax.set_xlabel('Value', fontname="Times New Roman", fontsize=16)
            ax.set_ylabel('Frequency', fontname="Times New Roman", fontsize=16)

            # Apply custom font to x-axis tick labels
            for tick in ax.get_xticklabels():
                tick.set_fontfamily('Times New Roman')
                tick.set_fontsize(16)
                # tick.set_color('cornflowerblue')

            # Apply different font/style to y-axis tick labels
            for tick in ax.get_yticklabels():
                tick.set_fontfamily('Times New Roman')
                tick.set_fontsize(16)
                # tick.set_color('cornflowerblue')

            # Style the offset text (e.g., ×10⁶)
            offset_text = ax.yaxis.get_offset_text()
            offset_text.set_fontfamily('Times New Roman')  # or any family
            offset_text.set_fontsize(16)  # match y-ticks if you like
            # offset_text.set_color('green')

            # Save plot
            plot_path = subfolder / f"{subfolder.name}_histogram.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Processed {subfolder.name}")

# Convert to DataFrame and save to Excel
df_stats = pd.DataFrame(stats_list)
df_stats.to_excel(output_excel, index=False)

print(f"\nStatistics saved to {output_excel}")
print("\nSummary:")
print(df_stats)

