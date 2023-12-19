import matplotlib.pyplot as plt
import pandas as pd


def save_feature_important_plots(df, save_path):
    # Iterate over columns and create bar plots
    for i, (column, values) in enumerate(df.items()):
        # Create individual subplot for each column
        fig, ax = plt.subplots(figsize=(6, 4))  # Adjust figsize as needed
        bar_width = 0.6
        bar_positions = range(len(df.index))  # Adjust the bar positions

        ax.bar(bar_positions, values, width=bar_width, align='center')  # Adjust the width, color, and alignment of the bars
        ax.set_title(column, fontsize=12)  # Adjust title font size
        ax.set_ylabel('Values', fontsize=10)  # Adjust y-axis label font size
        ax.set_xticks(bar_positions)  # Set the x-tick positions
        ax.set_xticklabels(df.index, rotation=45, fontsize=8, ha='right')  # Rotate x-axis labels and adjust font size

        # Save the individual subplot as a separate PNG file
        plt.savefig(f"{save_path}_{column}.png", bbox_inches='tight')

        # Close the current plot to prevent overlapping when creating the next subplot
        plt.close()