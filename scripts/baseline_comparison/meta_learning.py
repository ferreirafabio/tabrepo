import ast
import copy
import itertools
import os
import shutil
import pandas as pd
import math
from typing import List, Optional, Tuple
from tabrepo.utils.cache import cache_function, cache_function_dataframe


import numpy as np
from dataclasses import dataclass
import random
from scripts.baseline_comparison.meta_learning_utils import (
    minmax_normalize_tasks,
    print_arguments,
    convert_df_ranges_dtypes_fillna,
    get_test_dataset_folds,
    get_train_val_split,
    transform_ranks,
)
from tabrepo.portfolio.portfolio_generator import RandomPortfolioGenerator, AbstractPortfolioGenerator

from sklearn.metrics import mean_squared_error
from tabrepo.repository import EvaluationRepository
from tabrepo.repository.time_utils import (
    filter_configs_by_runtime,
    sort_by_runtime,
    get_runtime,
)
from tabrepo.utils.meta_features import get_meta_features
from tabrepo.utils.parallel_for import parallel_for
from autogluon.tabular import TabularPredictor
from scripts.baseline_comparison.vis_utils import save_feature_importance_plots
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
        method_name_suffix: str = " metalearning",
        loss: str = "metric_error",
        use_extended_meta_features: bool = False,
        use_synthetic_portfolios: bool = False,
        n_synthetic_portfolios: int = 1000,
        use_metalearning_kfold_training: bool = False,
        n_splits_kfold: int = 5,
        generate_feature_importance: bool = False,
        seed: int = 0,
        save_name: int = "experiment",
        results_dir: str = "",
        ray_process_ratio: float = 1.,
        add_zeroshot_portfolios: bool = False,
        ignore_cache: bool = False,
        metalearning_with_only_zeroshot: bool = False
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
    :param use_synthetic_portfolios: indicates whether we should add synthetic portfolios to the metalearning train data
    :return: evaluation obtained on all combinations
    """
    print_arguments(**locals())


    def evaluate_dataset(test_datasets, n_portfolio_model_frameworks_df_rank, n_ensemble, n_training_dataset, n_training_fold, n_training_config,
                         max_runtime, repo: EvaluationRepository, rank_scorer, normalized_scorer,
                         use_meta_features, method_name_suffix, seed):

        print(f"running now method {method_name_suffix} with {seed}")
        df_rank, model_frameworks, n_portfolio = n_portfolio_model_frameworks_df_rank

        method_name = zeroshot_name(
            n_portfolio=n_portfolio,
            n_ensemble=n_ensemble,
            n_training_dataset=n_training_dataset,
            n_training_fold=n_training_fold,
            max_runtime=max_runtime,
            n_training_config=n_training_config,
            name_suffix=method_name_suffix,
        )

        _n_training_dataset = n_training_dataset
        # restrict number of evaluation fold
        if n_training_fold is None:
            n_training_fold = n_eval_folds

        # gets all tids that are possible available
        test_tids = [repo.dataset_to_tid(t_d) for t_d in test_datasets] # TODO: use datasets instead of tid
        # TODO: rename to training_tids
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
            configs = filter_configurations_above_budget(repo, test_tids, configs, max_runtime)

        df_rank = df_rank.copy().loc[configs]

        # collects all tasks that are available
        train_tasks = []
        for task in df_rank.columns:
            tid, fold = task.split("_")
            if int(tid) in selected_tids and int(fold) < n_training_fold:
                train_tasks.append(task)

        # get meta features
        # all_features = repo._df_metadata.columns.tolist()
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
                                                                          use_extended_meta_features=use_extended_meta_features
                                                                          )

        df_rank_all = df_rank.stack().reset_index(name='rank')
        df_rank_all["dataset"] = df_rank_all["task"].apply(repo.task_to_dataset)

        print(f"unique frameworks in data: {len(df_rank_all['framework'].unique())}")

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

        # merge meta features into the performance data
        if use_meta_features:
            train_meta = df_rank_train.merge(df_meta_features_train, on=["dataset"])
            test_meta = df_rank_test.merge(df_meta_features_test, on=["dataset"])
        else:
            # deactivated meta features (-> in an AG-learned version of zero-shot)
            train_meta = df_rank_train
            test_meta = df_rank_test

        # split train into dataset-disjoint 90% train and 10% val splits
        train_meta, val_meta = get_train_val_split(df=train_meta, seed=seed)

        train_meta.drop(['dataset'], axis=1, inplace=True)
        val_meta.drop(['dataset'], axis=1, inplace=True)
        test_meta_wo_dataset = test_meta.drop(['dataset'], axis=1)

        # test_meta.drop(['std_dev_rank'], axis=1, inplace=True)

        if use_extended_meta_features:
            train_meta = convert_df_ranges_dtypes_fillna(train_meta)
            val_meta = convert_df_ranges_dtypes_fillna(val_meta)
            test_meta_wo_dataset = convert_df_ranges_dtypes_fillna(test_meta_wo_dataset)
            test_meta = convert_df_ranges_dtypes_fillna(test_meta)

        assert (len(train_meta["framework"].unique()) == len(list(df_rank.index)))
        predictor = TabularPredictor(label="rank").fit(
            train_data=train_meta,
            tuning_data=val_meta,
            # hyperparameters={
            # "RF": {},
            # "DUMMY": {},
            # "GBM": {},
            # },
            # time_limit=7200,
            time_limit=1200,
            # time_limit=600,
            # time_limit=300,
            # time_limit=10,
            # time_limit=30,
            # verbosity=3,
            excluded_model_types=["CAT"],
            hyperparameters={"GBM": [
                # {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
                {},
                # "GBMLarge",
            ],
            },
            num_cpus=1,
            # presets="best_quality",
        )

        predictor.leaderboard(data=test_meta, display=True)

        # run predict to get the portfolio
        ranks = predictor.predict(test_meta_wo_dataset)
        rmse_test = np.sqrt(mean_squared_error(test_meta["rank"], ranks))
        feature_importance_df = predictor.feature_importance(test_meta_wo_dataset) if generate_feature_importance else None

        all_config_results_per_ds, all_portfolios_per_ds = {}, {}
        all_predicted_ranks_all_datasets = pd.concat([ranks, test_meta["dataset"], test_meta["framework"]], axis=1)

        for test_ds in test_datasets:
            ranks_per_ds = all_predicted_ranks_all_datasets[all_predicted_ranks_all_datasets['dataset'] == test_ds]
            ranks_per_ds = ranks_per_ds.sort_values(by="rank", ascending=True)

            portfolio_name = None
            if use_synthetic_portfolios:
                # the portfolio size is in this case defined by the synthetic_portfolio_size
                framework_name = ranks_per_ds.iloc[0]["framework"]
                if framework_name.startswith("Portfolio"):
                    portfolio_configs_per_ds = repo.random_portfolio_generator.portfolio_name_to_config[n_portfolio][framework_name]
                    portfolio_name = framework_name
                else:
                    portfolio_configs_per_ds = [framework_name]
            else:
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
                portfolio_name=portfolio_name,
                ensemble_size=n_ensemble,
                tid=repo.dataset_to_tid(test_ds),
                method=method_name,
                folds=range(n_eval_folds),
                seed=seed,
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

    model_frameworks_original = {
        framework: sorted([x for x in repo.configs() if framework in x])
        for framework in framework_types
    }

    assert loss in ["metric_error", "metric_error_val", "rank"]

    dd = dd[[loss, "framework", "task"]]
    random_portfolio_generator = None

    # TODO: impute other columns like train time when generating metric_errors
    if use_synthetic_portfolios:
        assert loss == "metric_error", "synthetic portfolios currently only supported for metric_error loss"

        df_rank_n_portfolios = []
        model_frameworks_n_portfolios = []
        generator_file_path = results_dir / f"random_portfolio_generator_repo_{expname}_num_portfolios_{n_synthetic_portfolios}_seed_{seed}.pkl"

        if os.path.exists(generator_file_path):
            random_portfolio_generator = AbstractPortfolioGenerator.load_generator(generator_file_path)
            print(f"Loaded random portfolio generator: {generator_file_path}.")
        else:
            print(f"No previous random portfolio generator found for settings {n_synthetic_portfolios=} and {seed=}. Generating anew.")
            random_portfolio_generator = RandomPortfolioGenerator(repo=repo, n_portfolios=n_portfolios)

            metric_errors, ensemble_weights, portfolio_info = random_portfolio_generator.generate_evaluate_bulk(
                n_portfolios=n_synthetic_portfolios,
                portfolio_size=n_portfolios,
                ensemble_size=100,
                seed=seed,
                backend="ray"
            )

            random_portfolio_generator.save_generator(generator_file_path)

        repo.random_portfolio_generator = random_portfolio_generator

        for n_portfolio in n_portfolios:
            m_e = random_portfolio_generator.metric_errors[n_portfolio]
            portfolio_names = list(random_portfolio_generator.portfolio_name_to_config[n_portfolio].keys())
            dd_with_syn_portfolio = dd.copy()
            dd_with_syn_portfolio = random_portfolio_generator.concatenate_bulk(base_df=dd_with_syn_portfolio,
                                                                                to_add_series_list=m_e,
                                                                                portfolio_names=portfolio_names
                                                                                )

            if add_zeroshot_portfolios:
                zeroshot_metric_errors, _, zeroshot_config_name, portfolio_configs_zs, zeroshot_config_name = cache_function(fun=lambda: random_portfolio_generator.generate_evaluate_zeroshot(n_portfolio=n_portfolio, dd=dd.copy(), loss=loss),
                                                                                 cache_name=f"random_portfolio_generator_zeroshot_n_portfolio_{n_portfolio}",
                                                                                 cache_path=results_dir,
                                                                                 # ignore_cache=True,
                                                                                 )
                random_portfolio_generator.portfolio_name_to_config[n_portfolio][zeroshot_config_name] = portfolio_configs_zs

                dd_with_syn_portfolio = random_portfolio_generator.concatenate(base_df=dd_with_syn_portfolio,
                                                                               to_add_series=zeroshot_metric_errors,
                                                                               portfolio_name=zeroshot_config_name
                                                                               )

                portfolio_names = list(random_portfolio_generator.portfolio_name_to_config[n_portfolio].keys())

            df_r = transform_ranks(loss, dd_with_syn_portfolio)
            df_r.fillna(value=np.nanmax(df_r.values), inplace=True)
            assert not any(df_r.isna().values.reshape(-1))
            df_rank_n_portfolios.append(df_r)

            model_frameworks_copy = model_frameworks_original.copy()
            model_frameworks_copy["ensemble"] = portfolio_names
            model_frameworks_n_portfolios.append(model_frameworks_copy)

   # no synthetic portfolios
    else:
        df_rank_n_portfolios = transform_ranks(loss, dd)

        # df_rank = dd.pivot_table(index="framework", columns="task", values="metric_error").rank(ascending=False)
        df_rank_n_portfolios.fillna(value=np.nanmax(df_rank_n_portfolios.values), inplace=True)
        assert not any(df_rank_n_portfolios.isna().values.reshape(-1))
        if n_portfolios and n_portfolios[0] is not None:
            df_rank_n_portfolios = [df_rank_n_portfolios for i in range(len(n_portfolios))]
            model_frameworks_n_portfolios = [model_frameworks_original for i in range(len(n_portfolios))]
        else:
            df_rank_n_portfolios = [df_rank_n_portfolios]
            model_frameworks_n_portfolios = [model_frameworks_original]

    if use_metalearning_kfold_training:
        print(f"Using kfold training for metalearning")
        dataset_names_input = get_test_dataset_folds(dataset_names, n_splits=n_splits_kfold)
        lengths = {fold: len(dataset) for fold, dataset in enumerate(dataset_names_input)}
        print(f"number of test datasets per fold {lengths}")
    else:
        dataset_names_input = [[ds] for ds in dataset_names]

    assert len(model_frameworks_n_portfolios) == len(df_rank_n_portfolios) and len(n_portfolios) == len(df_rank_n_portfolios)

    n_portfolio_model_frameworks_df_rank = list(zip(df_rank_n_portfolios, model_frameworks_n_portfolios, n_portfolios))

    if metalearning_with_only_zeroshot:
        filtered_dfs = []
        filtered_model_frameworks = []
        for df in df_rank_n_portfolios:
            # Filter rows where the framework starts with 'Portfolio-ZS-N'
            filtered_df = df[df.index.to_series().str.startswith('Portfolio-ZS-N')]
            filtered_dfs.append(filtered_df)
            portfolio_filtered_name = filtered_df.index.tolist()
            filtered_model_frameworks.append({"ensemble": portfolio_filtered_name})
        n_portfolio_model_frameworks_df_rank = list(zip(filtered_dfs, filtered_model_frameworks, n_portfolios))

    result_list = parallel_for(
        evaluate_dataset,
        inputs=list(itertools.product(dataset_names_input,
                                      n_portfolio_model_frameworks_df_rank,
                                      n_ensembles,
                                      n_training_datasets,
                                      n_training_folds,
                                      n_training_configs,
                                      max_runtimes,
                                      )),
        context=dict(repo=repo,
                     rank_scorer=rank_scorer,
                     normalized_scorer=normalized_scorer,
                     # model_frameworks=model_frameworks,
                     use_meta_features=use_meta_features,
                     seed=seed,
                     method_name_suffix=method_name_suffix,
                     ),
        engine=engine,
    )

    mean_test_error = np.mean([result_dct["rmse_test"] for result_dct in result_list])
    print(f"mean rmse on test for {name}: {mean_test_error:.3f}")

    if len(n_training_datasets) == 1 and n_training_datasets[0] is None and generate_feature_importance:
        # n_training_datasets is not None when we run the sensitivity experiments in which case we do not want to run below code
        feature_importances = [result_dct["feature_importance_df"] for result_dct in result_list]
        feature_importance_averages = pd.concat(feature_importances).groupby(level=0).mean()
        save_feature_importance_plots(df=feature_importance_averages[["importance", "stddev", "p_value"]],
                                      exp_name=expname,
                                      save_name=save_name,
                                      )

    return [row for result_dct in result_list for result_rows_per_dataset in result_dct["evaluate_configs_result"].values() for row in result_rows_per_dataset]
