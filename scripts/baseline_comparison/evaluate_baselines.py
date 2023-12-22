import os
import math
from typing import List, Callable, Dict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from pathlib import Path

from autogluon.common.savers import save_pd
from dataclasses import dataclass

from tabrepo import EvaluationRepository
from tabrepo.loaders import Paths
from tabrepo.utils import catchtime
from tabrepo.utils.cache import cache_function, cache_function_dataframe
from tabrepo.utils.normalized_scorer import NormalizedScorer
from tabrepo.utils.rank_utils import RankScorer

from scripts import output_path, load_context
from scripts.baseline_comparison.baselines import (
    automl_results,
    framework_default_results,
    framework_best_results,
    zeroshot_results,
    zeroshot_name,
    ResultRow,
    framework_name,
    time_suffix,
    default_ensemble_size,
    n_portfolios_default,
)

from scripts.baseline_comparison.meta_learning import zeroshot_results_metalearning

from scripts.baseline_comparison.compare_results import winrate_comparison
from scripts.baseline_comparison.plot_utils import (
    MethodStyle,
    show_latex_table,
    show_cdf,
    show_scatter_performance_vs_time, iqm, show_scatter_performance_vs_time_lower_budgets, figure_path,
    plot_critical_diagrams,
)


@dataclass
class Experiment:
    expname: str  # name of the parent experiment used to store the file
    name: str  # name of the specific experiment, e.g. "localsearch"
    run_fun: Callable[[], List[ResultRow]]  # function to execute to obtain results

    def data(self, ignore_cache: bool = False):
        return cache_function_dataframe(
            lambda: pd.DataFrame(self.run_fun()),
            cache_name=self.name,
            ignore_cache=ignore_cache,
            cache_path=output_path.parent / "data" / "results-baseline-comparison" / self.expname,
        )


def make_scorers(repo: EvaluationRepository, only_baselines=False):
    if only_baselines:
        df_results_baselines = repo._zeroshot_context.df_baselines
    else:
        df_results_baselines = pd.concat([
            repo._zeroshot_context.df_configs_ranked,
            repo._zeroshot_context.df_baselines,
        ], ignore_index=True)

    unique_dataset_folds = [
        f"{repo.dataset_to_tid(dataset)}_{fold}"
        for dataset in repo.datasets()
        for fold in range(repo.n_folds())
    ]
    rank_scorer = RankScorer(df_results_baselines, tasks=unique_dataset_folds, pct=False)
    normalized_scorer = NormalizedScorer(df_results_baselines, tasks=unique_dataset_folds, baseline=None)
    return rank_scorer, normalized_scorer


def impute_missing(repo: EvaluationRepository):
    # impute random forest data missing folds by picking data from another fold
    # TODO remove once we have complete dataset
    df = repo._zeroshot_context.df_configs_ranked
    df["framework_type"] = df.apply(lambda row: row["framework"].split("_")[0], axis=1)

    missing_tasks = [(3583, 0), (58, 9), (3483, 0)]
    for tid, fold in missing_tasks:
        impute_fold = (fold + 1) % 10
        df_impute = df[(df.framework_type == 'RandomForest') & (df.dataset == f"{tid}_{impute_fold}")].copy()
        df_impute['dataset'] = f"{tid}_{fold}"
        df_impute['fold'] = fold
        df = pd.concat([df, df_impute], ignore_index=True)
    repo._zeroshot_context.df_configs_ranked = df


def plot_figure(df, method_styles: List[MethodStyle], title: str = None, figname: str = None, show: bool = False):
    fig, _ = show_cdf(df[df.method.isin([m.name for m in method_styles])], method_styles=method_styles)
    if title:
        fig.suptitle(title)
    if figname:
        fig_save_path = figure_path() / f"{figname}.pdf"
        plt.tight_layout()
        plt.savefig(fig_save_path)
    if show:
        plt.show()


def make_rename_dict(suffix: str) -> Dict[str, str]:
    # return renaming of methods
    rename_dict = {}
    for hour in [1, 4, 24]:
        for automl_framework in ["autosklearn", "autosklearn2", "flaml", "lightautoml", "H2OAutoML"]:
            rename_dict[f"{automl_framework}_{hour}h{suffix}"] = f"{automl_framework} ({hour}h)".capitalize()
        for preset in ["best", "high", "medium"]:
            rename_dict[f"AutoGluon_{preset[0]}q_{hour}h{suffix}"] = f"AutoGluon {preset} ({hour}h)"
    for minute in [5, 10, 30]:
        for preset in ["best"]:
            rename_dict[f"AutoGluon_{preset[0]}q_{minute}m{suffix}"] = f"AutoGluon {preset} ({minute}m)"
    return rename_dict


def time_cutoff_baseline(df: pd.DataFrame, rel_tol: float = 0.1) -> pd.DataFrame:
    df = df.copy()
    # TODO Portfolio excess are due to just using one fold to simulate runtimes, fix it
    mask = (df["time fit (s)"] > df["fit budget"] * (1 + rel_tol)) & (~df.method.str.contains("Portfolio"))

    # gets performance of Extra-trees baseline on all tasks
    dd = repo._zeroshot_context.df_configs_ranked
    dd = dd[dd.framework == "ExtraTrees_c1_BAG_L1"]
    dd["tid"] = dd.dataset.apply(lambda s: int(s.split("_")[0]))
    dd["fold"] = dd.dataset.apply(lambda s: int(s.split("_")[1]))
    dd["rank"] = dd.apply(lambda row: rank_scorer.rank(task=row["dataset"], error=row["metric_error"]), axis=1)
    dd["normalized-score"] = dd.apply(
        lambda row: normalized_scorer.rank(dataset=row["dataset"], error=row["metric_error"]), axis=1)
    df_baseline = dd[["tid", "fold", "rank", "normalized-score"]]

    df.loc[mask, ["normalized_score", "rank"]] = df.loc[mask, ["tid", "fold"]].merge(df_baseline, on=["tid", "fold"])[
        ["normalized-score", "rank"]].values

    return df


def rename_dataframe(df):
    rename_dict = make_rename_dict(suffix="8c_2023_08_21")
    df["method"] = df["method"].replace(rename_dict)
    df.rename({
        "normalized_error": "normalized-error",
        "time_train_s": "time fit (s)",
        "time_infer_s": "time infer (s)",
    },
        inplace=True, axis=1
    )

    def convert_timestamp(s):
        if "h)" in s:
            return float(s.split("(")[-1].replace("h)", "")) * 3600
        elif "m)" in s:
            return float(s.split("(")[-1].replace("m)", "")) * 60
        else:
            return None

    df["fit budget"] = df.method.apply(convert_timestamp)
    df.method = df.method.str.replace("NeuralNetTorch", "MLP")
    return df


def generate_sentitivity_plots(df, show: bool = False):
    # show stds

    # show stds
    fig, axes = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(9, 4))

    dimensions = [
        ("M", "Number of configuration per family"),
        ("D", "Number of training datasets to fit portfolios"),
    ]
    for i, (dimension, legend) in enumerate(dimensions):

        for j, metric in enumerate(["normalized-error", "rank"]):
            df_portfolio = df.loc[df.method.str.contains(f"Portfolio-N.*-{dimension}.*4h"), :].copy()
            df_ag = df.loc[df.method.str.contains("AutoGluon best \(4h\)"), metric].copy()

            df_portfolio.loc[:, dimension] = df_portfolio.loc[:, "method"].apply(
                lambda s: int(s.replace(" (ensemble) (4h)", "").split("-")[-1][1:]))
            df_portfolio = df_portfolio[df_portfolio[dimension] > 1]

            dim, mean, sem = df_portfolio.loc[:, [dimension, metric]].groupby(dimension).agg(
                ["mean", "sem"]).reset_index().values.T
            ax = axes[j][i]
            ax.errorbar(
                dim, mean, sem,
                label="Portfolio",
                linestyle="",
                marker="o",
            )
            ax.set_xlim([0, None])
            if j == 1:
                ax.set_xlabel(legend)
            if i == 0:
                ax.set_ylabel(f"{metric}")
            ax.grid()
            ax.hlines(df_ag.mean(), xmin=0, xmax=max(dim), color="black", label="AutoGluon", ls="--")
            if i == 1 and j == 0:
                ax.legend()
    fig_save_path = figure_path() / f"sensitivity.pdf"
    plt.tight_layout()
    plt.savefig(fig_save_path)
    if show:
        plt.show()


def save_total_runtime_to_file(total_time_h):
    # Save total runtime so that "show_repository_stats.py" can show the ratio of saved time
    with open(output_path / "tables" / "runtime.txt", "w") as f:
        f.write(str(total_time_h))


def generate_selected_config_hist_plots(df):
    # create histograms for selected configs
    from collections import Counter
    import numpy as np
    best_n_configs = 10
    n_datasets = None  # only use a subset of the datasets (unique folds), set to None if all should be used
    methods_in_df = df["method"].unique()
    # [method for method in methods_in_df if "N1" in method]
    for method in methods_in_df:
        if n_datasets:
            df = df.groupby('dataset', group_keys=False).apply(lambda x: x.sample(1))
            # shuffle dataframe rows
            df = df.sample(frac=1.0)[:n_datasets].reset_index(drop=True)

        configs_selected = list(df[df["method"] == method]["config_selected"])
        # convert string to list of strings
        configs_selected = [eval(cfg) for cfg in configs_selected]
        # flatten across all datasets and folds and count occurrences
        config_counts = Counter(np.concatenate(configs_selected))
        config_counts_dict = dict(config_counts)

        # Select the top 'n' configs based on occurrences
        top_n_configs = dict(sorted(config_counts.items(), key=lambda x: x[1], reverse=True)[:best_n_configs])

        # Print the method and corresponding counts
        print(f"Method: {method}")
        print("Config Counts:", config_counts_dict)
        plt.figure(figsize=(30, 25))
        plt.bar(top_n_configs.keys(), top_n_configs.values())
        plt.xticks(rotation=45, ha='right', fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(str(Paths.data_root / "simulation" / expname / ("selected_cfg_histogram_" + method + ".png")))
        plt.show()



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--repo", type=str, help="Name of the repo to load", default="D244_F3_C1416")
    parser.add_argument("--n_folds", type=int, default=-1, required=False,
                        help="Number of folds to consider when evaluating all baselines. Uses all if set to -1.")
    parser.add_argument("--n_datasets", type=int, required=False, help="Number of datasets to consider when evaluating all baselines.")
    parser.add_argument("--ignore_cache", action="store_true", help="Ignore previously generated results and recompute them from scratch.")
    parser.add_argument("--all_configs", action="store_true", help="If True, will use all configs rather than filtering out NeuralNetFastAI. If True, results will differ from the paper.")
    parser.add_argument("--expname", type=str, help="Name of the experiment. If None, defaults to the value specified in `repo`.", required=False, default=None)
    parser.add_argument("--engine", type=str, required=False, default="ray", choices=["sequential", "ray", "joblib"],
                        help="Engine used for embarrassingly parallel loop.")
    parser.add_argument("--ray_process_ratio", type=float,
                        help="The ratio of ray processes to logical cpu cores. Use lower values to reduce memory usage. Only used if engine == 'ray'",)
    parser.add_argument("--deactivate_meta_features", action="store_true",
                        help="Ignore all meta features")
    args = parser.parse_args()
    print(args.__dict__)

    repo_version = args.repo
    ignore_cache = args.ignore_cache
    ray_process_ratio = args.ray_process_ratio
    engine = args.engine
    expname = repo_version if args.expname is None else args.expname
    n_datasets = args.n_datasets
    as_paper = not args.all_configs
    use_meta_features = not args.deactivate_meta_features

    if n_datasets:
        expname += f"-{n_datasets}"

    if engine == "ray" and args.ray_process_ratio is not None:
        assert (ray_process_ratio <= 1) and (ray_process_ratio > 0)
        num_cpus = os.cpu_count()
        num_ray_processes = math.ceil(num_cpus*ray_process_ratio)

        print(f'NOTE: To avoid OOM, we are limiting ray processes to {num_ray_processes} (Total Logical Cores: {num_cpus})\n'
              f'\tThis is based on ray_process_ratio={ray_process_ratio}')

        # FIXME: The large-scale 3-fold 244-dataset 1416-config runs OOM on m6i.32x without this limit
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=num_ray_processes)

    n_eval_folds = args.n_folds
    n_portfolios = [5, 10, 50, 100, n_portfolios_default]
    max_runtimes = [300, 600, 1800, 3600, 3600 * 4, 24 * 3600]
    # n_training_datasets = list(range(10, 210, 10))
    # n_training_configs = list(range(10, 210, 10))
    n_training_datasets = [1, 5, 10, 25, 50, 75, 100, 125, 150, 175, 199]
    n_training_configs = [1, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200]
    n_seeds = 1
    n_training_folds = [1, 2, 5, 10]
    n_ensembles = [10, 20, 40, 80]
    linestyle_ensemble = "--"
    linestyle_default = "-"
    linestyle_tune = "dotted"

    # Number of digits to show in table
    n_digits = {
        "normalized-error": 3,
        "rank": 1,
        "time fit (s)": 1,
        "time infer (s)": 3,
    }

    if not as_paper:
        expname += "_ALL"

    repo: EvaluationRepository = load_context(version=repo_version, ignore_cache=False, as_paper=as_paper)
    repo.print_info()

    if n_eval_folds == -1:
        n_eval_folds = repo.n_folds()

    rank_scorer, normalized_scorer = make_scorers(repo)
    dataset_names = repo.datasets()
    if n_datasets:
        dataset_names = dataset_names[:n_datasets]

    # TODO: filter configs for AutoFolio (put this in AF branch)
    # repo.configs()
    # repo = repo.subset(configs=)

    # TODO: This is a hack, in future repo should know the framework_types via the configs.json input
    configs_default = [c for c in repo.configs() if "_c1_" in c]
    framework_types = [c.rsplit('_c1_', 1)[0] for c in configs_default]

    experiment_common_kwargs = dict(
        repo=repo,
        dataset_names=dataset_names,
        framework_types=framework_types,
        rank_scorer=rank_scorer,
        normalized_scorer=normalized_scorer,
        n_eval_folds=n_eval_folds,
        engine=engine,
        use_meta_features=use_meta_features,
    )

    experiments = [
        Experiment(
            expname=expname, name=f"framework-default-{expname}",
            run_fun=lambda: framework_default_results(**experiment_common_kwargs)
        ),
        # Experiment(
        #     expname=expname, name=f"framework-best-{expname}",
        #     run_fun=lambda: framework_best_results(max_runtimes=[3600, 3600 * 4, 3600 * 24], **experiment_common_kwargs),
        # ),
        # Experiment(
        #     expname=expname, name=f"framework-all-best-{expname}",
        #     run_fun=lambda: framework_best_results(framework_types=[None], max_runtimes=[3600, 3600 * 4, 3600 * 24], **experiment_common_kwargs),
        # ),
        # Automl baselines such as Autogluon best, high, medium quality
        # Experiment(
        #     expname=expname, name=f"automl-baselines-{expname}",
        #     run_fun=lambda: automl_results(**experiment_common_kwargs),
        # ),
        Experiment(
            expname=expname, name=f"zeroshot-metalearning-{expname}",
            run_fun=lambda: zeroshot_results_metalearning(**experiment_common_kwargs,
                                                          name=f"zeroshot-metalearning-{expname}",
                                                          expname=expname
                                                          )
        ),
        Experiment(
            expname=expname, name=f"zeroshot-{expname}",
            run_fun=lambda: zeroshot_results(**experiment_common_kwargs)
        ),
        Experiment(
            expname=expname, name=f"zeroshot-metalearning-singlebest-{expname}",
            run_fun=lambda: zeroshot_results_metalearning(**experiment_common_kwargs,
                                                          n_portfolios=[1],
                                                          name=f"zeroshot-metalearning-singlebest-{expname}",
                                                          expname=expname,
                                                          )
        ),
        Experiment(
            expname=expname, name=f"zeroshot-singlebest-{expname}",
            run_fun=lambda: zeroshot_results(**experiment_common_kwargs, n_portfolios=[1])
        ),
        # Experiment(
        #     expname=expname, name=f"zeroshot-{expname}-maxruntimes",
        #     run_fun=lambda: zeroshot_results(max_runtimes=max_runtimes, **experiment_common_kwargs)
        # ),
        # Experiment(
        #     expname=expname, name=f"zeroshot-{expname}-num-folds",
        #     run_fun=lambda: zeroshot_results(n_training_folds=n_training_folds, ** experiment_common_kwargs)
        # ),
        # Experiment(
        #     expname=expname, name=f"zeroshot-{expname}-num-portfolios",
        #     run_fun=lambda: zeroshot_results(n_portfolios=n_portfolios, n_ensembles=[1, default_ensemble_size], **experiment_common_kwargs)
        # ),
    ]

    # Use more seeds
    # for seed in range(n_seeds):
    #     experiments.append(Experiment(
    #         expname=expname, name=f"zeroshot-{expname}-num-configs-{seed}",
    #         run_fun=lambda: zeroshot_results(n_training_configs=n_training_configs, **experiment_common_kwargs)
    #     ))
    #
    #     experiments.append(Experiment(
    #         expname=expname, name=f"zeroshot-{expname}-num-training-datasets-{seed}",
    #         run_fun=lambda: zeroshot_results(n_training_datasets=n_training_datasets, **experiment_common_kwargs)
    #     ))


    with catchtime("total time to generate evaluations"):
        df = pd.concat([
            experiment.data(ignore_cache=ignore_cache) for experiment in experiments
        ])
    # De-duplicate in case we ran a config multiple times
    df = df.drop_duplicates(subset=["method", "dataset", "fold"])
    df = rename_dataframe(df)

    # Save results
    save_pd.save(path=str(Paths.data_root / "simulation" / expname / "results.csv"), df=df)

    # df = time_cutoff_baseline(df)

    # generate_selected_config_hist_plots(df)

    print(f"Obtained {len(df)} evaluations on {len(df.dataset.unique())} datasets for {len(df.method.unique())} methods.")
    print(f"Methods available:" + "\n".join(sorted(df.method.unique())))
    total_time_h = df.loc[:, "time fit (s)"].sum() / 3600
    print(f"Total time of experiments: {total_time_h} hours")
    save_total_runtime_to_file(total_time_h)


    #generate_sentitivity_plots(df, show=False)

    show_latex_table(df, "all", show_table=True, n_digits=n_digits)
    ag_styles = [
        # MethodStyle("AutoGluon best (1h)", color="black", linestyle="--", label_str="AG best (1h)"),
        MethodStyle("AutoGluon best (4h)", color="black", linestyle="-.", label_str="AG best (4h)", linewidth=2.5),
        # MethodStyle("AutoGluon high quality (ensemble)", color="black", linestyle=":", label_str="AG-high"),
        # MethodStyle("localsearch (ensemble) (ST)", color="red", linestyle="-")
    ]

    method_styles = ag_styles.copy()
    framework_types.remove("NeuralNetTorch")
    framework_types.append("MLP")

    for i, framework_type in enumerate(framework_types):
        method_styles.append(
            MethodStyle(
                framework_name(framework_type, tuned=False),
                color=sns.color_palette('bright')[i],
                linestyle=linestyle_default,
                label=True,
                label_str=framework_type,
            )
        )
        method_styles.append(
            MethodStyle(
                framework_name(framework_type, max_runtime=4 * 3600, ensemble_size=1, tuned=True),
                color=sns.color_palette('bright')[i],
                linestyle=linestyle_tune,
                label=False,
            )
        )
        method_styles.append(
            MethodStyle(
                framework_name(framework_type, max_runtime=4 * 3600, tuned=True),
                color=sns.color_palette('bright')[i],
                linestyle=linestyle_ensemble,
                label=False,
                label_str=framework_type
            )
        )
    show_latex_table(df[df.method.isin([m.name for m in method_styles])], "frameworks", n_digits=n_digits)#, ["rank", "normalized_score", ])

    plot_figure(df, method_styles, figname="cdf-frameworks")

    plot_figure(
        df, [x for x in method_styles if "ensemble" not in x.name], figname="cdf-frameworks-tuned",
        title="Effect of tuning configurations",
    )

    plot_figure(
        df,
        [x for x in method_styles if any(pattern in x.name for pattern in ["tuned", "AutoGluon"])],
        figname="cdf-frameworks-ensemble",
        title="Effect of tuning & ensembling",
        # title="Comparison of frameworks",
    )

    cmap = matplotlib.colormaps["viridis"]
    # Plot effect number of training datasets
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_training_dataset=size),
            color=cmap(i / (len(n_training_datasets) - 1)), linestyle="-", label_str=r"$\mathcal{D}'~=~" + f"{size}$",
        )
        for i, size in enumerate(n_training_datasets)
    ]
    plot_figure(df, method_styles, title="Effect of number of training tasks", figname="cdf-n-training-datasets")

    # # Plot effect number of training fold
    # method_styles = ag_styles + [
    #     MethodStyle(
    #         zeroshot_name(n_training_fold=size),
    #         color=cmap(i / (len(n_training_folds) - 1)), linestyle="-", label_str=f"S{size}",
    #     )
    #     for i, size in enumerate(n_training_folds)
    # ]
    # plot_figure(df, method_styles, title="Effect of number of training folds", figname="cdf-n-training-folds")

    # Plot effect number of portfolio size
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_portfolio=size),
            color=cmap(i / (len(n_portfolios) - 1)), linestyle="-", label_str=r"$\mathcal{N}~=~" + f"{size}$",
        )
        for i, size in enumerate(n_portfolios)
    ]
    plot_figure(df, method_styles, title="Effect of number of portfolio configurations", figname="cdf-n-configs")

    # Plot effect of number of training configurations
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_training_config=size),
            color=cmap(i / (len(n_training_configs) - 1)), linestyle="-", label_str=r"$\mathcal{M}'~=~" + f"{size}$",
        )
        for i, size in enumerate(n_training_configs)
    ]
    plot_figure(df, method_styles, title="Effect of number of offline configurations", figname="cdf-n-training-configs")

    # Plot effect of training time limit
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(max_runtime=size),
            color=cmap(i / (len(max_runtimes) - 1)), linestyle="-",
            label_str=time_suffix(size).replace("(", "").replace(")", "").strip(),
        )
        for i, size in enumerate(max_runtimes)
    ]
    plot_figure(df, method_styles, title="Effect of training time limit", figname="cdf-max-runtime")

    automl_frameworks = ["Autosklearn2", "Flaml", "Lightautoml", "H2oautoml"]
    for budget in ["1h", "4h"]:
        budget_suffix = f"\({budget}\)"
        # df = df[~df.method.str.contains("All")]
        df_selected = df[
            (df.method.str.contains(f"AutoGluon .*{budget_suffix}")) |
            (df.method.str.contains(".*(" + "|".join(automl_frameworks) + f").*{budget_suffix}")) |
            (df.method.str.contains(f"Portfolio-N{n_portfolios_default} .*{budget_suffix}")) |
            (df.method.str.contains(".*(" + "|".join(framework_types) + ")" + f".*{budget_suffix}")) |
            (df.method.str.contains(".*default.*"))
        ].copy()
        df_selected.method = df_selected.method.str.replace(f" {budget_suffix}", "").str.replace(f"\-N{n_portfolios_default}", "")
        show_latex_table(
            df_selected,
            f"selected-methods-{budget}",
            show_table=True,
            n_digits=n_digits,
        )

    show_latex_table(df[(df.method.str.contains("Portfolio") | (df.method.str.contains("AutoGluon ")))], "zeroshot", n_digits=n_digits)

    fig, _, bbox_extra_artists = show_scatter_performance_vs_time(df, metric_cols=["normalized-error", "rank"])
    fig_save_path = figure_path() / f"scatter-perf-vs-time.pdf"
    fig.savefig(fig_save_path, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
    fig.show()

    fig, _, bbox_extra_artists = show_scatter_performance_vs_time_lower_budgets(df, metric_cols=["normalized-error", "rank"])
    fig_save_path = figure_path() / f"scatter-perf-vs-time-lower-budget.pdf"
    fig.savefig(fig_save_path, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
    fig.show()

    # plot_critical_diagrams(df)

    winrate_comparison(df=df, repo=repo)
