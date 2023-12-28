import ast
import copy
import itertools
import os
import shutil
import pandas as pd
import hashlib
from typing import List, Optional, Tuple
import random
import contextlib
from scripts.baseline_comparison.meta_learning_utils import minmax_normalize_tasks, print_arguments

from AutoFolio.autofolio.facade.af_csv_facade import AFCsvFacade

import numpy as np
from dataclasses import dataclass

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


def generate_worker_hash():
    # Get the process ID of the current worker
    worker_id = os.getpid()

    # Create a unique hash using the process ID
    hash_object = hashlib.sha256(str(worker_id).encode())
    worker_hash = hash_object.hexdigest()

    return worker_hash

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
        loss: str = "metric_error",
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
    :return: evaluation obtained on all combinations
    """
    print_arguments(**locals())

    def evaluate_dataset(test_dataset, n_portfolio, n_ensemble, n_training_dataset, n_training_fold, n_training_config,
                         max_runtime, repo: EvaluationRepository, df_rank, rank_scorer, normalized_scorer,
                         model_frameworks, use_meta_features):
        method_name = zeroshot_name(
            n_portfolio=n_portfolio,
            n_ensemble=n_ensemble,
            n_training_dataset=n_training_dataset,
            n_training_fold=n_training_fold,
            max_runtime=max_runtime,
            n_training_config=n_training_config,
            name_suffix=" metalearning",
        )

        # restrict number of evaluation fold
        if n_training_fold is None:
            n_training_fold = n_eval_folds

        # gets all tids that are possible available
        test_tid = repo.dataset_to_tid(test_dataset)
        available_tids = [repo.dataset_to_tid(dataset) for dataset in dataset_names if dataset != test_dataset]
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

        # Randomly shuffle the config order with seed 0
        rng = np.random.default_rng(seed=0)
        configs = list(rng.choice(configs, len(configs), replace=False))

        # # exclude configurations from zeroshot selection whose runtime exceeds runtime budget by large amount
        if max_runtime:
            configs = filter_configurations_above_budget(repo, test_tid, configs, max_runtime)

        df_rank = df_rank.copy().loc[configs]

        # collects all tasks that are available
        train_tasks = []
        for task in df_rank.columns:
            tid, fold = task.split("_")
            if int(tid) in selected_tids and int(fold) < n_training_fold:
                train_tasks.append(task)

        # get meta features
        # all_featueres = repo._df_metadata.columns.tolist()
        meta_features_selected = ["dataset",  # maps to problem_type
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
                                                                          n_eval_folds,
                                                                          meta_features_selected,
                                                                          selected_tids
                                                                          )

        df_rank_all = df_rank.stack().reset_index(name='rank')
        df_rank_all["dataset"] = df_rank_all["task"].apply(repo.task_to_dataset)

        # create meta train / test splits
        train_mask = df_rank_all["task"].isin(train_tasks)
        df_rank_train = df_rank_all[train_mask].drop(columns=["task"])
        df_rank_test = df_rank_all[~train_mask].drop(columns=["task"])

        df_rank_train = df_rank_train.groupby(["framework", "dataset"])["rank"].mean()
        df_rank_train = df_rank_train.reset_index(drop=False)

        df_rank_test = df_rank_test.groupby(["framework", "dataset"])["rank"].mean()
        df_rank_test = df_rank_test.reset_index(drop=False)

        try:
            hsh = generate_worker_hash()
            current_directory = os.getcwd()
            new_directory_path = os.path.join(current_directory, "as_files")

            # Check if the directory already exists before creating it
            if not os.path.exists(new_directory_path):
                os.makedirs(new_directory_path)

            # performance matrix: (column: algorithm, row: instance, delimeter: ,)
            df_rank_train = df_rank_train.pivot(index="dataset", columns="framework", values="rank")

            # df_rank_train.to_csv("perf.csv")
            df_rank_train.to_csv(f"as_files/perf_{hsh}.csv", index=True, index_label="")  # to match AF example 1:1

            # df_rank_test = df_rank_test.pivot(index="dataset", columns="framework", values="rank")

            # feature matrix: (column: features, row: instance, delimeter: ,)
            # df_meta_features_train.drop(["dataset"], axis=1, inplace=True)
            # df_meta_features_train.to_csv("feats.csv")
            df_meta_features_train.set_index("dataset", inplace=True)
            df_meta_features_train.to_csv(f"as_files/feats_{hsh}.csv", index=True, index_label="")

            # df_meta_features_test.drop(["dataset"], axis=1, inplace=True)
            # df_meta_features_test.to_csv("feats_test.csv")
            df_meta_features_test.set_index("dataset", inplace=True)
            # df_meta_features_test.to_csv(f"as_files/feats_test_{hsh}.csv", index=True, index_label="")

            perf_fn = f"as_files/perf_{hsh}.csv"
            feat_fn = f"as_files/feats_{hsh}.csv"

            # will be created (or overwritten) by AutoFolio
            model_fn = f"as_files/af_model_{hsh}.pkl"

            # try:
            af = AFCsvFacade(perf_fn=perf_fn, feat_fn=feat_fn, maximize=False)

            # config = {'StandardScaler': False, 'fgroup_all': True, 'imputer_strategy': 'mean', 'pca': False,
            #           'selector': 'IndRegressor', 'classifier': 'RandomForest', 'imputer_strategy': 'mean', 'pca': False,
            #            'rfreg:bootstrap': False, 'rfreg:criterion': 'gini', 'rfreg:max_depth': 132, 'rfreg:max_features': 'log2',
            #           'rfreg:min_samples_leaf': 3, 'rfreg:min_samples_split': 3, 'rfreg:n_estimators': 68}
            # config = {'StandardScaler': False, 'fgroup_all': True, 'imputer_strategy': 'mean', 'pca': False,
            #           'selector': 'PairwiseClassifier', 'classifier': 'RandomForest', 'rf:bootstrap': False,
            #           'rf:criterion': 'gini', 'rf:max_depth': 132, 'rf:max_features': 'log2', 'rf:min_samples_leaf': 3,
            #           'rf:min_samples_split': 3, 'rf:n_estimators': 68}

            # fit AutoFolio; will use default hyperparameters of AutoFolio
            af.fit(save_fn=model_fn)
            portfolio_configs = AFCsvFacade.load_and_predict(vec=df_meta_features_test.values.reshape(-1), load_fn=model_fn)
            portfolio_configs = portfolio_configs[:n_portfolio]

        except Exception as e:
            with contextlib.suppress(FileNotFoundError):
                os.remove(perf_fn)
                os.remove(feat_fn)
                os.remove(model_fn)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(perf_fn)
                os.remove(feat_fn)
                os.remove(model_fn)

        # fit AF using a loaded configuration on all data!
        # af.fit(config=config)

        # tune AutoFolio's hyperparameter configuration for 4 seconds
        # import errors with SMAC --> don't do tuning for now
        # config = af.tune(wallclock_limit=4)

        # evaluate configuration using a 10-fold cross validation
        # score = af.cross_validation(config=config)

        # re-fit AutoFolio using the (hopefully) better configuration
        # and save model to disk
        # af.fit(config=config, save_fn=model_fn)

        # predictor = TabularPredictor(label="rank").fit(
        #     train_meta,
        #     # hyperparameters={
        #     # "RF": {},
        #     # "DUMMY": {},
        #     # "GBM": {},
        #     # },
        #     # time_limit=1200,
        #     # time_limit=600,
        #     time_limit=300,
        #     # time_limit=10,
        #     # time_limit=30,
        # )
        # predictor.leaderboard(display=True)

        # run predict to get the portfolio
        # ranks = predictor.predict(test_meta)
        # portfolio_configs = pd.concat([ranks, test_meta["framework"]], axis=1).sort_values(by="rank", ascending=True)
        # portfolio_configs = portfolio_configs[:n_portfolio]["framework"].tolist()
        # shutil.rmtree(predictor.path)

        # TODO: Technically we should exclude data from the fold when computing the average runtime and also pass the
        #  current fold when filtering by runtime.
        # portfolio_configs = sort_by_runtime(repo=repo, config_names=portfolio_configs)

        portfolio_configs = filter_configs_by_runtime(
            repo=repo,
            tid=test_tid,
            fold=0,
            config_names=portfolio_configs,
            max_cumruntime=max_runtime if max_runtime else default_runtime, # TODO
        )
        if len(portfolio_configs) == 0:
            # in case all configurations selected were above the budget, we evaluate a quick backup, we pick a
            # configuration that takes <1s to be evaluated
            portfolio_configs = [backup_fast_config]

        return evaluate_configs(
            repo=repo,
            rank_scorer=rank_scorer,
            normalized_scorer=normalized_scorer,
            config_selected=portfolio_configs,
            ensemble_size=n_ensemble,
            tid=test_tid,
            method=method_name,
            folds=range(n_eval_folds),
        )

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

    df_rank.fillna(value=np.nanmax(df_rank.values), inplace=True)
    assert not any(df_rank.isna().values.reshape(-1))

    model_frameworks = {
        framework: sorted([x for x in repo.configs() if framework in x])
        for framework in framework_types
    }

    list_rows = parallel_for(
        evaluate_dataset,
        inputs=list(itertools.product(dataset_names, n_portfolios, n_ensembles, n_training_datasets, n_training_folds,
                                      n_training_configs, max_runtimes)),
        context=dict(repo=repo, df_rank=df_rank, rank_scorer=rank_scorer, normalized_scorer=normalized_scorer,
                     model_frameworks=model_frameworks, use_meta_features=use_meta_features),
        engine=engine,
    )
    return [x for l in list_rows for x in l]
