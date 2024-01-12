import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from tabrepo.loaders import Paths

import os


def visualize_config_selected(exp_name, title, save_name, top_n_configs=10):

    print(os.getcwd())
    df = pd.read_csv(f'../../data/simulation/{exp_name}/results.csv')

    # Define method names
    # method1 = 'Portfolio-N200 metalearning (ensemble) (4h)'
    # method2 = 'Portfolio-N200 (ensemble) (4h)'
    method1 = 'Portfolio-N1 metalearning (ensemble) (4h)'
    method2 = 'Portfolio-N1 (ensemble) (4h)'

    # Extract relevant columns
    df = df[df["method"].str.contains(r'^Portfolio', regex=True, na=False)]
    df['config_selected'] = df['config_selected'].apply(literal_eval)  # Convert string representation of list to actual list

    # Function to get the top N configurations for each method
    def get_top_n_configs(df_method, n=5):
        config_freq_method = {}

        # Count the frequency of each config for the given method
        for configs in df_method['config_selected']:
            for config in configs:
                if config in config_freq_method:
                    config_freq_method[config] += 1
                else:
                    config_freq_method[config] = 1

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
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

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
    fig.suptitle(f"{exp_name}, {title}")

    plt.tight_layout()
    plt.savefig(str(Paths.data_root / "results-baseline-comparison" / exp_name / save_name / f"compare_selected_metalearning.png"))
    plt.show()
