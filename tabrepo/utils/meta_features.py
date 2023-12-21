
def get_meta_features(repo, n_eval_folds, meta_features_selected, selected_tids):
    df_meta_features = repo._df_metadata[meta_features_selected]

    # add problem_type column as a categorical
    df_meta_features.loc[:, "problem_type"] = df_meta_features.loc[:, "dataset"].map(
        repo._zeroshot_context.dataset_to_problem_type_dict)
    df_meta_features.loc[:, "problem_type"] = df_meta_features.loc[:, "problem_type"].factorize()[0]
    keep_datasets = {repo.tid_to_dataset(id) for id in selected_tids}
    keep_mask = df_meta_features["dataset"].isin(keep_datasets)

    df_meta_features_train = df_meta_features[keep_mask]
    df_meta_features_test = df_meta_features[~keep_mask]

    return df_meta_features_train, df_meta_features_test
