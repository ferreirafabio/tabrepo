# for advanced meta-features
from autogluon_benchmark import OpenMLTaskWrapper
from pymfe.mfe import MFE
import pandas as pd
import os
from pathlib import Path
from functools import reduce
from autogluon.tabular import TabularDataset
from autogluon.features.generators import AutoMLPipelineFeatureGenerator, CategoryFeatureGenerator
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_extended_meta_features(repo, args, rewrite_csvs=False):
    if not (args.extended_mf_general or
            args.extended_mf_statistical or
            args.extended_mf_info_theory or
            args.extended_mf_model_based or
            args.extended_mf_landmarking):
        return repo

    def get_mfe_df(group, X, y, dataset_name):
        # we use AutoGluon's feature generator since one-hot-encoding of mfe leads to issues
        feature_generator = AutoMLPipelineFeatureGenerator()
        feature_generator_y = AutoMLPipelineFeatureGenerator()
        # X, y = TabularDataset(X), y.to_numpy()
        X, y = TabularDataset(X), TabularDataset(y)
        X_transf = feature_generator.fit_transform(X=X).to_numpy()
        y_transf = feature_generator_y.fit_transform(X=y).to_numpy()
        mfe = MFE(groups=[group])
        ft = mfe.fit(X_transf, y_transf).extract()
        df_ft = pd.DataFrame([ft[1]], columns=ft[0])
        df_ft["dataset"] = dataset_name
        return df_ft

    def compute_meta_features(mf_group):
        df_ft_list = []
        for tid in task_ids:
            task = OpenMLTaskWrapper.from_task_id(task_id=tid)
            # X, y = task.X.to_numpy(), task.y.to_numpy()
            X, y = task.X, task.y
            dataset_name = repo._tid_to_dataset_dict[int(tid)]
            df_ft = get_mfe_df(mf_group, X, y, dataset_name)
            df_ft_list.append(df_ft)

        df_group = pd.concat(df_ft_list, ignore_index=True)

        return df_group

    flags = [
        ("extended_mf_general", args.extended_mf_general, "general"),
        ("extended_mf_statistical", args.extended_mf_statistical, "statistical"),
        ("extended_mf_info_theory", args.extended_mf_info_theory, "info-theory"),
        ("extended_mf_model_based", args.extended_mf_model_based, "model-based"),
        ("extended_mf_landmarking", args.extended_mf_landmarking, "landmarking"),
        ("extended_mf_concept", args.extended_mf_concept, "concept"),
        ("extended_mf_clustering", args.extended_mf_clustering, "clustering"),
        ("extended_mf_complexity", args.extended_mf_complexity, "complexity"),
        ("extended_mf_itemset", args.extended_mf_itemset, "itemset"),
        ("extended_mf_relative", args.extended_mf_relative, "relative"),
    ]

    out_dir = Path(__file__).parent.parent.parent / "data" / "results-baseline-comparison" / args.repo
    df_meta_features_dict = {}
    task_ids = [repo.dataset_to_tid(dataset) for dataset in repo.datasets()]
    for flag, flag_value, group in flags:
        print(f"processing {group=}")
        if flag_value:
            file_name = f"{flag}.csv"
            file_path = out_dir / file_name
            if os.path.exists(file_path) and not rewrite_csvs:
                print(f"{file_path.name} already exists. Loading dataframe.")
                df = pd.read_csv(file_path, index_col=0)
            else:
                print(f"{file_path.name} does not exist or rewrite_csvs is set to True. "
                      f"Computing meta features for group '{group}' and saving dataframe.")
                df = compute_meta_features(mf_group=group)
                df.to_csv(file_path)
            df_meta_features_dict[group] = df

    all_dataframes = list(df_meta_features_dict.values())
    repo._extended_df_metadata = reduce(lambda x, y: pd.merge(x, y, on='dataset'), all_dataframes)
    print("-> done generating / loading extended meta features")
    return repo


def get_meta_features(repo, meta_features_selected, selected_tids, use_extended_meta_features=False):
    df_meta_features = repo._df_metadata[meta_features_selected]

    # to avoid "A value is trying to be set on a copy of a slice from a DataFrame" warning
    df_meta_features_new = df_meta_features.copy()
    df_meta_features_new.loc[:, "problem_type"] = df_meta_features.loc[:, "dataset"].map(
        repo._zeroshot_context.dataset_to_problem_type_dict)
    # df_meta_features.loc[:, "problem_type"] = df_meta_features.loc[:, "problem_type"].factorize()[0]

    if use_extended_meta_features:
        df_meta_features_new = pd.merge(df_meta_features_new, repo._extended_df_metadata, on='dataset')

    assert len(df_meta_features_new) == len(repo._df_metadata), "generated meta features df differs in length compared to repo._df_metadata"

    keep_datasets = {repo.tid_to_dataset(id) for id in selected_tids}
    keep_mask = df_meta_features_new["dataset"].isin(keep_datasets)

    df_meta_features_train = df_meta_features_new[keep_mask]
    df_meta_features_test = df_meta_features_new[~keep_mask]

    return df_meta_features_train, df_meta_features_test
