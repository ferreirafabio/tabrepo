import numpy as np


def get_meta_features(repo, n_eval_folds, meta_features_selected, selected_tids):
    df_meta_features = repo._df_metadata[meta_features_selected]

    # add problem_type column as a categorical
    df_meta_features.loc[:, "problem_type"] = df_meta_features.loc[:, "dataset"].map(
        repo._zeroshot_context.dataset_to_problem_type_dict)
    df_meta_features.loc[:, "problem_type"] = df_meta_features.loc[:, "problem_type"].factorize()[0]

    # create the tid with n_eval_folds identifier
    # df_meta_features = df_meta_features.loc[df_meta_features["dataset"].index.repeat(n_eval_folds)].reset_index(
    #     drop=True)

    # fold is used as a helper column and dropped later
    # df_meta_features['fold'] = np.tile(range(n_eval_folds), len(df_meta_features) // n_eval_folds)
    # df_meta_features['dataset'] = df_meta_features.apply(
    #     lambda row: repo.task_name(dataset=row['dataset'], fold=row['fold']), axis=1)
    # df_meta_features.drop(['fold', 'tid'], axis=1, inplace=True)
    # df_meta_features.rename(columns={"dataset": "tid"}, inplace=True)
    # df_meta_features.set_index("tid", inplace=True)

    # keep_ids_str = {str(id) for id in selected_tids}
    # keep_mask = df_meta_features.index.str.startswith(tuple(keep_ids_str))

    # df_meta_features_train = df_meta_features[keep_mask]
    # df_meta_features_test = df_meta_features[~keep_mask]

    keep_datasets = {repo.tid_to_dataset(id) for id in selected_tids}
    keep_mask = df_meta_features["dataset"].isin(keep_datasets)

    df_meta_features_train = df_meta_features[keep_mask]
    df_meta_features_test = df_meta_features[~keep_mask]

    return df_meta_features_train, df_meta_features_test
