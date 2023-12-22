from sklearn.preprocessing import MinMaxScaler
import pandas as pd


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