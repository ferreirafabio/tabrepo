import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


def convert_df_ranges_dtypes_fillna(df: pd.DataFrame) -> pd.DataFrame:
    max_limit = np.finfo(np.float32).max
    min_limit = np.finfo(np.float32).min
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = df[numerical_columns].clip(lower=min_limit, upper=max_limit)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(value=-100, inplace=True)

    return df


def minmax_normalize_tasks(df):
    tasks = list(df.columns.str.split('_').str[0].unique())
    # Create a new dataframe to store normalized values
    normalized_df = pd.DataFrame(index=df.index, columns=df.columns)

    # Normalize each task separately
    for task in tasks:
        task_columns = [col for col in df.columns if col.startswith(task)]
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform(df[task_columns])
        normalized_df[task_columns] = normalized_values

    return normalized_df


def print_arguments(**kwargs):
    print("Arguments:")
    for key, value in kwargs.items():
        print(f"{key}={value}")


def get_test_dataset_folds(dataset_names, n_splits):
    # total_size = len(dataset_names)
    # n_splits = total_size // int(total_size * test_ratio)
    kf = KFold(n_splits=n_splits)
    test_dataset_folds = []
    for i, (_, test_index) in enumerate(kf.split(X=dataset_names)):
        test_datasets = [dataset_names[i] for i in test_index]
        test_dataset_folds.append(test_datasets)

    return test_dataset_folds


def sparsify_rank(df_rank, sparsify_ratio=0.5, sparsify_on="best"):
    assert sparsify_on in ["best", "worst", "random"]
    row_means = df_rank.mean(axis=1)

    if sparsify_on == "best":
        threshold = row_means.quantile(1 - sparsify_ratio)
        rows_to_keep = row_means[row_means <= threshold].index
    elif sparsify_on == "worst":
        threshold = row_means.quantile(sparsify_ratio)
        rows_to_keep = row_means[row_means >= threshold].index
    else:
        # Random sparsification
        remove_n = len(df_rank) - int(sparsify_ratio * len(df_rank))
        drop_indices = pd.Index(np.random.choice(df_rank.index, remove_n, replace=False))
        rows_to_keep = df_rank.index.difference(drop_indices)

    return df_rank.loc[rows_to_keep]