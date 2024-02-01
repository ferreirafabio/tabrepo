import matplotlib.pyplot as plt
import pandas as pd
from tabrepo.loaders import Paths
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from tabrepo.loaders import Paths
import os


def visualize_config_selected(exp_name, title, save_name, top_n_configs=10, portfolio_size_to_plot=4):

    print(os.getcwd())
    df = pd.read_csv(f'../../data/simulation/{exp_name}/results.csv')

    # Define method names
    # method1 = 'Portfolio-N200 metalearning (ensemble) (4h)'
    # method2 = 'Portfolio-N200 (ensemble) (4h)'
    method_base_name = df['method'].str.extract(fr'(Portfolio-N{portfolio_size_to_plot})').dropna()[0].unique()[0]
    method1 = f'{method_base_name} metalearning (ensemble)'
    method2 = f'{method_base_name} (ensemble) (4h)'

    # Extract relevant columns
    df = df[df["method"].str.contains(r'^Portfolio', regex=True, na=False)]
    # df['config_selected'] = df['config_selected'].apply(literal_eval)  # Convert string representation of list to actual list

    # Function to get the top N configurations for each method
    def get_top_n_configs(df_method, n=5):
        config_freq_method = {}

        # Count the frequency of each config for the given method
        # for configs in df_method['config_selected']:
        #     for config in configs:
        #         if config in config_freq_method:
        #             config_freq_method[config] += 1
        #         else:
        #             config_freq_method[config] = 1

        # to also show ensembles (e.g. when using synthetically generated portfolios), simply take the string
        for config_list in df_method['config_selected']:
            if config_list in config_freq_method:
                config_freq_method[config_list] += 1
            else:
                config_freq_method[config_list] = 1

        # Sort configurations by frequency and get the top N
        sorted_configs = sorted(config_freq_method.items(), key=lambda x: x[1], reverse=True)
        top_n_configs = dict(sorted_configs[:n])

        return top_n_configs

    # Get the top N configurations for each method
    top_n_configs_method1 = get_top_n_configs(df[df['method'] == method1], top_n_configs)
    top_n_configs_method2 = get_top_n_configs(df[df['method'] == method2], top_n_configs)

    # Create dataframes for the top N configurations
    df_method1 = pd.DataFrame(list(top_n_configs_method1.items()), columns=['Config', method1])
    df_method2 = pd.DataFrame(list(top_n_configs_method2.items()), columns=['Config', method2])

    # Plot individual subplots for each method
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 12))

    # Plot for method1
    df_method1.plot(x='Config', kind='bar', ax=axes[0], width=0.8, color='skyblue')
    axes[0].set_title(method1)
    # axes[0].set_xlabel('Config')
    axes[0].set_ylabel('Freq. across all folds / datasets')

    # Plot for method2
    df_method2.plot(x='Config', kind='bar', ax=axes[1], width=0.8, color='salmon')
    axes[1].set_title(method2)
    # axes[1].set_xlabel('Config')
    axes[1].set_ylabel('Freq. across all folds / datasets')

    axes[0].tick_params(axis='x', rotation=45, labelsize=12)
    axes[1].tick_params(axis='x', rotation=45, labelsize=12)

    fig.suptitle(f"{exp_name}, {title}")

    plt.tight_layout()
    plt.savefig(str(Paths.data_root / "results-baseline-comparison" / exp_name / save_name / f"compare_selected_metalearning.png"))
    plt.show()


def save_feature_importance_plots(df, exp_name, save_name):
    # Iterate over columns and create bar plots
    for i, (column, values) in enumerate(df.items()):
        # Create individual subplot for each column
        fig, ax = plt.subplots(figsize=(27, 18))
        bar_width = 0.6
        bar_positions = range(len(df.index))

        ax.bar(bar_positions, values, width=bar_width, align='center')  # Adjust the width, color, and alignment of the bars
        ax.set_title(column, fontsize=12)  # Adjust title font size
        ax.set_ylabel('Values (log scale)', fontsize=10)  # Adjust y-axis label font size
        ax.set_yscale('log')  # Set the y-axis to log scale
        ax.set_xticks(bar_positions)  # Set the x-tick positions
        ax.set_xticklabels(df.index, rotation=45, fontsize=6, ha='right')  # Rotate x-axis labels and adjust font size

        plt.title(f"{column}")
        # Save the individual subplot as a separate PNG file
        plt.savefig(str(Paths.data_root / "results-baseline-comparison" / exp_name / save_name / f"feature_importance_{column}.png"), bbox_inches='tight')
        # plt.savefig(f"{save_name}_{column}.png", bbox_inches='tight')

        # Close the current plot to prevent overlapping when creating the next subplot
        plt.close()


def plot_portfolio_selection(exp_name, save_name, n_portfolios):
    df = pd.read_csv(str(Paths.data_root / "simulation" / exp_name / "results.csv"))

    fig, axes = plt.subplots(nrows=1, ncols=len(n_portfolios), figsize=(15, 5), sharey=True)
    # filter_method = "metalearning with zeroshot and best synthetic portfolios (ensemble)"
    filter_method = "metalearning with zeroshot and zeroshot synthetic portfolios (ensemble)"

    def categorize_portfolio(name):
        if name.startswith('Portfolio-ZS'):
            return 'Zeroshot portfolios'
        else:
            return 'Synthetic portfolios'

    for i, N in enumerate(n_portfolios):
        # Filter for each N-value
        # filtered_df = df[df['method'].str.endswith(f"Portfolio-N{N} metalearning with zeroshot portfolios (ensemble)")]
        filtered_df = df[df['method'].str.endswith(f"Portfolio-N{N} {filter_method}")]

        # Drop NaNs in 'portfolio_name'
        filtered_df = filtered_df.dropna(subset=['portfolio_name'])

        # Categorize portfolio names
        filtered_df['portfolio_category'] = filtered_df['portfolio_name'].apply(categorize_portfolio)

        # Count the occurrences
        category_counts = filtered_df['portfolio_category'].value_counts()
        category_counts = category_counts.reindex(['Zeroshot portfolios', 'Synthetic portfolios'])

        # Plot for each N-value
        axes[i].bar(category_counts.index, category_counts.values)
        axes[i].set_title(f'N={N}')
        axes[0].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].yaxis.set_tick_params(labelbottom=True)

    plt.suptitle(f'What portfolios does {filter_method} pick?', fontsize=13)
    plt.tight_layout()
    plt.savefig(str(Paths.data_root / "results-baseline-comparison" / exp_name / save_name / f"portfolio_zs_selected_comparison.png"))
    plt.show()
