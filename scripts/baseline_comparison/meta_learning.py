import ast
import copy
import itertools
import os
import shutil
import pandas as pd
from typing import List, Optional, Tuple

import numpy as np
from dataclasses import dataclass
import random
from sklearn.model_selection import KFold
from scripts.baseline_comparison.meta_learning_utils import minmax_normalize_tasks, print_arguments

from sklearn.metrics import mean_squared_error
from tabrepo.portfolio.zeroshot_selection import zeroshot_configs
from tabrepo.repository import EvaluationRepository
from tabrepo.repository.time_utils import (
    filter_configs_by_runtime,
    sort_by_runtime,
    get_runtime,
)
from tabrepo.utils.meta_features import get_meta_features
from tabrepo.utils.parallel_for import parallel_for
from autogluon.tabular import TabularPredictor
from scripts.baseline_comparison.vis_utils import save_feature_important_plots
from tabrepo.loaders import Paths

from scripts.baseline_comparison.baselines import (
    evaluate_configs,
    ResultRow,
    zeroshot_name,
    filter_configurations_above_budget,
)

default_ensemble_size = 40
n_portfolios_default = 200
default_runtime = 3600 * 4

backup_fast_config = "ExtraTrees_c1_BAG_L1"


def zeroshot_results_metalearning(
        repo: EvaluationRepository,
        dataset_names: List[str],
        framework_types: List[str],
        rank_scorer,
        normalized_scorer,
        n_eval_folds: int,
        n_ensembles: List[int] = [None],
        n_portfolios: List[int] = [n_portfolios_default],
        n_training_datasets: List[int] = [None],
        n_training_folds: List[int] = [None],
        n_training_configs: List[int] = [None],
        max_runtimes: List[float] = [default_runtime],
        engine: str = "ray",
        use_meta_features: bool = True,
        name: str = "",
        expname: str = "",
        loss: str = "metric_error",
        use_extended_mf: bool = False,
        seed: int = 0,
) -> List[ResultRow]:
    """
    :param dataset_names: list of dataset to use when fitting zeroshot
    :param n_eval_folds: number of folds to consider for evaluation
    :param n_ensembles: number of caruana sizes to consider
    :param n_portfolios: number of folds to use when fitting zeroshot
    :param n_training_datasets: number of dataset to use when fitting zeroshot
    :param n_training_folds: number of folds to use when fitting zeroshot
    :param n_training_configs: number of configurations available when fitting zeroshot TODO per framework
    :param max_runtimes: max runtime available when evaluating zeroshot configuration at test time
    :param engine: engine to use, must be "sequential", "joblib" or "ray"
    :param seed: the seed for the random number generator used for shuffling the configs
    :return: evaluation obtained on all combinations
    """
    print_arguments(**locals())

    def evaluate_dataset(test_datasets, n_portfolio, n_ensemble, n_training_dataset, n_training_fold, n_training_config,
                         max_runtime, repo: EvaluationRepository, df_rank, rank_scorer, normalized_scorer,
                         model_frameworks, use_meta_features, seed):
        method_name = zeroshot_name(
            n_portfolio=n_portfolio,
            n_ensemble=n_ensemble,
            n_training_dataset=n_training_dataset,
            n_training_fold=n_training_fold,
            max_runtime=max_runtime,
            n_training_config=n_training_config,
            name_suffix=" metalearning",
        )
        _n_training_dataset = n_training_dataset
        # restrict number of evaluation fold
        if n_training_fold is None:
            n_training_fold = n_eval_folds

        # gets all tids that are possible available
        # test_tid = repo.dataset_to_tid(test_dataset)
        test_tids = [repo.dataset_to_tid(t_d) for t_d in test_datasets] # TODO: use datasets instead of tid
        # available_tids = [repo.dataset_to_tid(dataset) for dataset in dataset_names if dataset != test_dataset]
        # TODO: call this training_tids
        available_tids = [repo.dataset_to_tid(dataset) for dataset in dataset_names if dataset not in test_datasets]
        np.random.shuffle(available_tids)
        if n_training_dataset is None:
            n_training_dataset = len(available_tids)

        # restrict number of training tid availables to fit
        selected_tids = set(available_tids[:n_training_dataset])

        # restrict number of configurations available to fit
        configs = []
        for models_framework in model_frameworks.values():
            if not (n_training_config) or len(models_framework) <= n_training_config:
                configs += models_framework
            else:
                configs += list(np.random.choice(models_framework, n_training_config, replace=False))

        # Randomly shuffle the config order for the passed seed
        rng = np.random.default_rng(seed=seed)
        configs = list(rng.choice(configs, len(configs), replace=False))

        # # exclude configurations from zeroshot selection whose runtime exceeds runtime budget by large amount
        if max_runtime:
            # configs = filter_configurations_above_budget(repo, test_tid, configs, max_runtime)
            configs = filter_configurations_above_budget(repo, test_tids, configs, max_runtime)

        df_rank = df_rank.copy().loc[configs]

        # collects all tasks that are available
        train_tasks = []
        for task in df_rank.columns:
            tid, fold = task.split("_")
            if int(tid) in selected_tids and int(fold) < n_training_fold:
                train_tasks.append(task)

        # get meta features
        # all_featueres = repo._df_metadata.columns.tolist()
        meta_features_selected = [
            "dataset",  # maps to problem_type
            # "tid",
            "MajorityClassSize",
            "MaxNominalAttDistinctValues",
            "MinorityClassSize",
            "NumberOfClasses",
            "NumberOfFeatures",
            "NumberOfInstances",
            "NumberOfInstancesWithMissingValues",
            "NumberOfMissingValues",
            "NumberOfSymbolicFeatures",
            "NumberOfNumericFeatures",
            # "number_samples",  # is nan
            ]

        df_meta_features_train, df_meta_features_test = get_meta_features(repo,
                                                                          meta_features_selected,
                                                                          selected_tids,
                                                                          use_extended_mf=use_extended_mf
                                                                          )

        df_rank_all = df_rank.stack().reset_index(name='rank')
        df_rank_all["dataset"] = df_rank_all["task"].apply(repo.task_to_dataset)

        # create meta train / test splits
        train_mask = df_rank_all["task"].isin(train_tasks)
        df_rank_train = df_rank_all[train_mask].drop(columns=["task"])

        # n_training_datasets is not None when we do dataset size analysis in which case we
        # do not take the negated mask as test data but the tid given
        if _n_training_dataset is None:
            df_rank_test = df_rank_all[~train_mask].drop(columns=["task"])
        else:
            test_tasks = []
            for task in df_rank.columns:
                tid, fold = task.split("_")
                if int(tid) in test_tids and int(fold) < n_training_fold:
                    test_tasks.append(task)
            df_rank_test = df_rank_all[df_rank_all["task"].isin(test_tasks)].drop(columns=["task"])

        df_rank_train = df_rank_train.groupby(["framework", "dataset"])["rank"].mean()
        df_rank_train = df_rank_train.reset_index(drop=False)

        df_rank_test = df_rank_test.groupby(["framework", "dataset"])["rank"].mean()
        df_rank_test = df_rank_test.reset_index(drop=False)

        if use_meta_features:
            # merge meta features into the performance data
            train_meta = df_rank_train.merge(df_meta_features_train, on=["dataset"])
            test_meta = df_rank_test.merge(df_meta_features_test, on=["dataset"])
        else:
            # deactivated meta features (-> in an AG-learned version of zero-shot)
            train_meta = df_rank_train
            test_meta = df_rank_test

        train_meta.drop(['dataset'], axis=1, inplace=True)
        test_meta_new = test_meta.drop(['dataset'], axis=1)

        # print("-------------meta-feature ablation study-----------------")
        # # 1 feature
        # # meta_features_to_consider = ["rank", "NumberOfMissingValues"]
        # # 3 features
        # meta_features_to_consider = ["rank", "NumberOfMissingValues", "NumberOfSymbolicFeatures", "MajorityClassSize"]
        # 5 features
        # meta_features_to_consider = ["rank", "NumberOfMissingValues", "NumberOfSymbolicFeatures", "MajorityClassSize", "NumberOfInstancesWithMissingValues", "MaxNominalAttDistinctValues"]
        # 8 features
        # meta_features_to_consider = ["rank", "NumberOfMissingValues", "NumberOfSymbolicFeatures", "MajorityClassSize", "NumberOfInstancesWithMissingValues", "MaxNominalAttDistinctValues", "NumberOfNumericFeatures", "NumberOfInstances", "NumberOfFeatures"]
        # 9 features
        # meta_features_to_consider = ["rank", "NumberOfMissingValues", "NumberOfSymbolicFeatures", "MajorityClassSize", "NumberOfInstancesWithMissingValues", "MaxNominalAttDistinctValues", "NumberOfNumericFeatures", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]
        # 10 features
        # meta_features_to_consider = ["rank", "NumberOfMissingValues", "NumberOfSymbolicFeatures", "MajorityClassSize", "NumberOfInstancesWithMissingValues", "MaxNominalAttDistinctValues", "NumberOfNumericFeatures", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses", "problem_type"]
        # 11 features
        # meta_features_to_consider = ["rank", "NumberOfMissingValues", "NumberOfSymbolicFeatures", "MajorityClassSize", "NumberOfInstancesWithMissingValues", "MaxNominalAttDistinctValues", "NumberOfNumericFeatures", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses", "problem_type", "MinorityClassSize"]
        # 12 features --> done
        # meta_features_to_consider = ["rank", "NumberOfMissingValues", "NumberOfSymbolicFeatures", "MajorityClassSize", "NumberOfInstancesWithMissingValues", "MaxNominalAttDistinctValues", "NumberOfNumericFeatures", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses", "problem_type", "MinorityClassSize", "framework"]
        # train_meta.drop(columns=train_meta.columns.difference(meta_features_to_consider), inplace=True)
        # test_meta_new.drop(columns=test_meta_new.columns.difference(meta_features_to_consider), inplace=True)
        # print(f"remaining columns train: {list(train_meta.columns)}")
        # print(f"remaining columns test: {list(test_meta_new.columns)}")

        # meta_features_to_consider = ["rank", "framework"]
        # # meta_features_to_consider = ["rank", "problem_type"]
        # # meta_features_to_consider = ["rank", "MinorityClassSize"]
        # # meta_features_to_consider = ["rank", "framework", "MinorityClassSize"]
        # meta_features_to_consider = ["rank", "framework", "problem_type"]
        # # meta_features_to_consider = ["rank", "framework", "MinorityClassSize", "problem_type"]
        # meta_features_to_consider = ["rank", "framework", "NumberOfMissingValues", "NumberOfSymbolicFeatures", "MajorityClassSize"]
        # meta_features_to_consider = [
        #             "rank",
        #             "problem_type",
        #             "MajorityClassSize",
        #             "MaxNominalAttDistinctValues",
        #             "MinorityClassSize",
        #             "NumberOfClasses",
        #             "NumberOfFeatures",
        #             "NumberOfInstances",
        #             "NumberOfInstancesWithMissingValues",
        #             "NumberOfMissingValues",
        #             "NumberOfSymbolicFeatures",
        #             "NumberOfNumericFeatures",
        #             ]

        # train_meta.drop(columns=train_meta.columns.difference(meta_features_to_consider), inplace=True)
        # test_meta_new.drop(columns=test_meta_new.columns.difference(meta_features_to_consider), inplace=True)
        # print(f"remaining columns train: {list(train_meta.columns)}")
        # print(f"remaining columns test: {list(test_meta_new.columns)}")

        # test_meta.drop(['std_dev_rank'], axis=1, inplace=True)

        if use_extended_mf:
            max_limit = np.finfo(np.float32).max
            min_limit = np.finfo(np.float32).min

            numerical_columns = train_meta.select_dtypes(include=[np.number]).columns
            train_meta[numerical_columns] = train_meta[numerical_columns].clip(lower=min_limit, upper=max_limit)
            numerical_columns = test_meta_new.select_dtypes(include=[np.number]).columns
            test_meta_new[numerical_columns] = test_meta_new[numerical_columns].clip(lower=min_limit, upper=max_limit)
            numerical_columns = test_meta.select_dtypes(include=[np.number]).columns
            test_meta[numerical_columns] = test_meta[numerical_columns].clip(lower=min_limit, upper=max_limit)

            train_meta.replace([np.inf, -np.inf], np.nan, inplace=True)
            test_meta.replace([np.inf, -np.inf], np.nan, inplace=True)
            test_meta_new.replace([np.inf, -np.inf], np.nan, inplace=True)

            train_meta.fillna(value=-100, inplace=True)
            test_meta.fillna(value=-100, inplace=True)
            test_meta_new.fillna(value=-100, inplace=True)

        predictor = TabularPredictor(label="rank").fit(
            train_meta,
            # hyperparameters={
            # "RF": {},
            # "DUMMY": {},
            # "GBM": {},
            # },
            # time_limit=7200,
            # time_limit=1200,
            # time_limit=600,
            time_limit=300,
            # time_limit=10,
            # time_limit=30,
            # verbosity=3,
        )
        predictor.leaderboard(display=True)

        # run predict to get the portfolio
        ranks = predictor.predict(test_meta_new)
        if ranks.isna().any():
            print("ranks has NaN")
            print(f"ranks: {ranks}")
            print(f"train_meta: {train_meta}")
            print(f"test id {test_datasets}")
        rmse_test = np.sqrt(mean_squared_error(test_meta["rank"], ranks))
        # feature_importance_df = predictor.feature_importance(test_meta_new)
        feature_importance_df = None

        all_config_results_per_ds, all_portfolios_per_ds = {}, {}
        all_predicted_ranks_all_datasets = pd.concat([ranks, test_meta["dataset"], test_meta["framework"]], axis=1)

        for test_ds in test_datasets:
            ranks_per_ds = all_predicted_ranks_all_datasets[all_predicted_ranks_all_datasets['dataset'] == test_ds]
            ranks_per_ds = ranks_per_ds.sort_values(by="rank", ascending=True)
            portfolio_configs_per_ds = ranks_per_ds[:n_portfolio]["framework"].tolist()

            # TODO: Technically we should exclude data from the fold when computing the average runtime and also pass the
            #  current fold when filtering by runtime.
            # portfolio_configs = sort_by_runtime(repo=repo, config_names=portfolio_configs)

            portfolio_configs_per_ds = filter_configs_by_runtime(
                repo=repo,
                tid=repo.dataset_to_tid(test_ds),
                fold=0,
                config_names=portfolio_configs_per_ds,
                max_cumruntime=max_runtime if max_runtime else default_runtime, # TODO
            )
            if len(portfolio_configs_per_ds) == 0:
                # in case all configurations selected were above the budget, we evaluate a quick backup, we pick a
                # configuration that takes <1s to be evaluated
                portfolio_configs_per_ds = [backup_fast_config]

            all_portfolios_per_ds[test_ds] = portfolio_configs_per_ds

            evaluate_configs_result = evaluate_configs(
                repo=repo,
                rank_scorer=rank_scorer,
                normalized_scorer=normalized_scorer,
                config_selected=portfolio_configs_per_ds,
                ensemble_size=n_ensemble,
                tid=repo.dataset_to_tid(test_ds),
                method=method_name,
                folds=range(n_eval_folds),
            )
            all_config_results_per_ds[test_ds] = evaluate_configs_result

        shutil.rmtree(predictor.path)

        return_dct = {
            "evaluate_configs_result": all_config_results_per_ds,
            "feature_importance_df": feature_importance_df,
            "rmse_test": rmse_test
        }
        return return_dct

    dd = repo._zeroshot_context.df_configs_ranked
    # df_rank = dd.pivot_table(index="framework", columns="dataset", values="score_val").rank()
    # TODO use normalized scores
    # df_rank = dd.pivot_table(index="framework", columns="task", values="metric_error").rank(ascending=False)

    # instead of metric_error, let's use the actual task here; also rank them in ascending order
    assert loss in ["metric_error", "metric_error_val", "rank"]
    if loss == "rank":
        df_rank = dd.pivot_table(index="framework", columns="task", values="rank").rank(ascending=True)
        print("using unnormalized rank as objective")
    elif loss == "metric_error":
        df_rank = dd.pivot_table(index="framework", columns="task", values="metric_error")
        df_rank = minmax_normalize_tasks(df_rank)
        df_rank = df_rank.rank(ascending=True)
        print("using task-normalized metric_error")
    elif loss == "metric_error_val":
        df_rank = dd.pivot_table(index="framework", columns="task", values="metric_error_val")
        df_rank = minmax_normalize_tasks(df_rank)
        df_rank = df_rank.rank(ascending=True)
        print("using task-normalized metric_error_val")
    else:
        import sys
        print("loss not supported")
        sys.exit()

    # df_rank = dd.pivot_table(index="framework", columns="task", values="metric_error").rank(ascending=False)
    df_rank.fillna(value=np.nanmax(df_rank.values), inplace=True)
    assert not any(df_rank.isna().values.reshape(-1))

    model_frameworks = {
        framework: sorted([x for x in repo.configs() if framework in x])
        for framework in framework_types
    }

    # total_size = len(dataset_names)
    # test_ratio = 0.3
    # n_splits = total_size // int(total_size * test_ratio)
    # kf = KFold(n_splits=n_splits)
    # test_dataset_folds = []
    # for i, (_, test_index) in enumerate(kf.split(X=dataset_names)):
    #     test_datasets = [dataset_names[i] for i in test_index]
    #     test_dataset_folds.append(test_datasets)
    #
    # lengths = {fold: len(dataset) for fold, dataset in enumerate(test_dataset_folds)}
    # print(f"test datasets per fold and repetition {lengths}")

    dataset_names_input = [[ds] for ds in dataset_names]
    result_list = parallel_for(
        evaluate_dataset,
        inputs=list(itertools.product(dataset_names_input, n_portfolios, n_ensembles, n_training_datasets, n_training_folds,
                                     n_training_configs, max_runtimes)),
        # inputs=list(itertools.product(test_dataset_folds, n_portfolios, n_ensembles, n_training_datasets, n_training_folds,
        #              n_training_configs, max_runtimes)),
        context=dict(repo=repo, df_rank=df_rank, rank_scorer=rank_scorer, normalized_scorer=normalized_scorer,
                     model_frameworks=model_frameworks, use_meta_features=use_meta_features, seed=seed),
        engine=engine,
    )

    mean_test_error = np.mean([result_dct["rmse_test"] for result_dct in result_list])
    print(f"mean rmse on test for {name}: {mean_test_error:.3f}")

    # if n_training_datasets is not None:
    #     # n_training_datasets is not None when we do dataset size analysis in which case we deactivate feature importance
    #     feature_importances = [result_dct["feature_importance_df"] for result_dct in result_list]
    #     feature_importance_averages = pd.concat(feature_importances).groupby(level=0).mean()
    #     save_feature_important_plots(df=feature_importance_averages[["importance", "stddev", "p_value", "n"]],
    #                                  save_path=str(Paths.data_root / "simulation" / expname / f"{name}_feat_imp"),
    #                                  )

    return [row for result_dct in result_list for result_rows_per_dataset in result_dct["evaluate_configs_result"].values() for row in result_rows_per_dataset]
