
def get_meta_features(repo, meta_features_selected, selected_tids):
    df_meta_features = repo._df_metadata[meta_features_selected]

    # to avoid "A value is trying to be set on a copy of a slice from a DataFrame" warning
    df_meta_features_new = df_meta_features.copy()
    df_meta_features_new.loc[:, "problem_type"] = df_meta_features.loc[:, "dataset"].map(
        repo._zeroshot_context.dataset_to_problem_type_dict)
    # df_meta_features.loc[:, "problem_type"] = df_meta_features.loc[:, "problem_type"].factorize()[0]
    keep_datasets = {repo.tid_to_dataset(id) for id in selected_tids}
    keep_mask = df_meta_features_new["dataset"].isin(keep_datasets)

    df_meta_features_train = df_meta_features_new[keep_mask]
    df_meta_features_test = df_meta_features_new[~keep_mask]

    return df_meta_features_train, df_meta_features_test
