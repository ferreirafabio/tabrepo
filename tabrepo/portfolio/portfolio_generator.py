from tabrepo.portfolio.zeroshot_selection import zeroshot_configs
from tabrepo.utils.normalized_scorer import NormalizedScorer
from tabrepo.utils.rank_utils import RankScorer
from typing import List, Tuple, Dict
from tabrepo import EvaluationRepository
import pandas as pd
import numpy as np
import pickle
import os
from tabrepo.portfolio.zeroshot_selection import zeroshot_configs
from scripts.baseline_comparison.meta_learning_utils import transform_ranks
from copy import deepcopy
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from pathlib import Path
from tabrepo.utils.cache import cache_function, cache_function_dataframe


class AbstractPortfolioGenerator:
    def __init__(self, repo: EvaluationRepository):
        self.repo = repo

    def generate(self, *args, **kwargs) -> List[str]:
        raise NotImplementedError()

    def evaluate(self, portfolio: List[str], datasets: List[str] = None, folds: List[int] = None, ensemble_size: int = 100, backend: str = "ray") -> Tuple[pd.Series, pd.DataFrame]:
        """
        evaluates a portfolio on datasets.
        :param portfolio: a list of configurations
        :param datasets: a list of datasets
        :param folds: a list of fold int identifiers
        :param ensemble_size: specifies the size of an ensemble for the Caruana approach
        :param backend: specifies whether the ensemble evaluation should be run in parallel mode (ray) or sequentially
        :return: the metric errors a pd.series of shape (len(datasets) * number of folds, 1) of the ensemble evaluated on the datasets along
        with the ensemble weights, a pd.DataFrame of shape (len(datasets) * number of folds *, len(portfolio)
        """
        if datasets is None:
            datasets = self.repo.datasets()

        if folds is None:
            folds = self.repo.folds

        metric_errors, ensemble_weights = self.repo.evaluate_ensemble(
            datasets=datasets,
            configs=portfolio,
            ensemble_size=ensemble_size,
            backend=backend,
            folds=folds,
            rank=False,
        )
        return metric_errors, ensemble_weights

    @staticmethod
    def _portfolio_name(n_portfolio_iter: int, portfolio_size: int, ensemble_size: int, seed: int, name_suffix: str = ""):
        suffix = [
            f"-{letter}{x}" if x is not None else ""
            for letter, x in
            [("", n_portfolio_iter), ("PS", portfolio_size), ("ES", ensemble_size), ("S", seed)]
        ]
        suffix += name_suffix
        suffix = "".join(suffix)
        return f"Portfolio{suffix}"

    @staticmethod
    def zeroshot_name(
            n_portfolio: int, ensemble_size: int = None, n_training_dataset: int = None,
            n_training_fold: int = None, n_training_config: int = None,
            name_suffix: str = "",
            base_name: str = "Portfolio-ZS",
    ):
        """
        :return: name of the zeroshot method such as Zeroshot-N20-C40 if n_training_dataset or n_training_folds are not
        None, suffixes "-D{n_training_dataset}" and "-S{n_training_folds}" are added, for instance "Zeroshot-N20-C40-D30-S5"
        would be the name if n_training_dataset=30 and n_training_fold=5
        """
        suffix = [
            f"-{letter}{x}" if x is not None else ""
            for letter, x in
            [("N", n_portfolio), ("ES", ensemble_size)]
        ]
        suffix += name_suffix

        # if n_ensemble:
        #     suffix += f"-C{n_ensemble}"
        suffix = "".join(suffix)
        if ensemble_size is None or ensemble_size > 1:
            suffix += " (ensemble)"
        return f"{base_name}{suffix}"

    def concatenate(self, base_df: pd.DataFrame, to_add_series: pd.Series, portfolio_name: str) -> pd.DataFrame:
        """
        concatenate a pd.Series (typically metric errors from an evaluated ensemble) to a base pd.DataFrame (existing
        dataframe containing the ranks that needs additional results) by first preparing the series to have a
        framework column with the portfolio_name and then concatenating both by  according to the framework name.
        :param base_df: the pd.DataFrame containing some metric errors or performance data.
        :param to_add_series:
        :param portfolio_name:
        :return:
        """
        metric_errors_prepared = self._prepare_merge(metric_errors=to_add_series, portfolio_name=portfolio_name)
        return pd.concat([base_df, metric_errors_prepared], axis=0, ignore_index=True)

    def concatenate_bulk(self, base_df: pd.DataFrame, to_add_series_list: List[pd.Series], portfolio_names: List[str]) -> pd.DataFrame:
        """
        concatenates a list of pd.Series to an existing base pd.DataFrame.
        :param base_df: the base dataframe with columns metric_error, framework, task
        :param to_add_series_list: a list of pd.Series of size len(portfolio_names), the list elements (pd.Series) are of shape (n_folds* number of datasets, )
        :param portfolio_names: a list of strings containing the unique portfolio names of size len(to_add_series_list)
        :return: the base dataframe with all series added to it along the row-axis
        """
        assert all(len(series) == len(to_add_series_list[0]) for series in to_add_series_list), "All Series must have the same length"
        assert len(to_add_series_list) == len(portfolio_names)

        for to_add_series, portfolio_name_i in zip(to_add_series_list, portfolio_names):
            base_df = self.concatenate(base_df=base_df, to_add_series=to_add_series, portfolio_name=portfolio_name_i)
        return base_df

    def _prepare_merge(self, metric_errors: pd.Series, portfolio_name: str):
        """

        :param metric_errors: a pd.Series of shape (n_folds * number of datasets, ) where the index has information on metric_error, framework, and task
        :param portfolio_name: a unique portfolio name string
        :return: the transformed metric_errors pd.Series with shape (n_folds * number of datasets, 3) where the columns are (metric_error, framework, task)
        """
        metric_errors = metric_errors.reset_index(drop=False)
        metric_errors["task"] = metric_errors.apply(lambda x: self.repo.task_name(x["dataset"], x["fold"]), axis=1)
        metric_errors["metric_error"] = metric_errors["error"]
        metric_errors["framework"] = portfolio_name
        metric_errors = metric_errors[["metric_error", "framework", "task"]]
        return metric_errors

    def generate_evaluate_zeroshot(self,
                                   n_portfolio: int,
                                   dd: pd.DataFrame,
                                   datasets: List[str] = None,
                                   folds: List[int] = None,
                                   ensemble_size: int = 100,
                                   loss: str = "metric_error",
                                   backend: str = "ray",
                                   config_name_map: Dict = None,
                                   base_name: str = None,
                                   actual_n_portfolio: int = None) -> Tuple[pd.Series, pd.DataFrame, str, List[str]]:
        """
        given datasets and metric errors, generated and evaluates a zeroshot portfolio
        :param n_portfolio: the number of configurations of the zeroshot portfolio
        :param dd: the metric errors of shape (task_id, 3) where 3=(metric_error, framework, task columns)
        :param datasets: a list of str containing the dataset names for which the zeroshot portfolio is evaluated
        :param folds: number of dataset folds
        :param ensemble_size: number of members to select with Caruana
        :param loss: a str indicating which loss to be used, needs to be in ["rank", "metric_error", "metric_error_val"]
        :param backend: whether to use ray for parallel processing or sequential processing
        :param config_name_map: a dictionary mapping from portfolio name to config name, if provided, portfolio configs get mapped, else the portfolio_name is used subsequently
        :param base_name: a str used for the prefix for the zeroshot portfolio name (default 'portfolio-ZS')
        :param actual_n_portfolio: when applying zeroshot to configs that are ensembles (e.g. of size 2), this int parameter indicates what the 'actual' portfolio size is,
        e.g., when a portfolio of size 20 is to be created with zeroshot based on ensembles of size 2, we can use actual_n_portfolio=20 and portfolio_size=10 to achieve this
        :return: a tuple consisting of zeroshot evaluation metrics and parameter (metric_errors, ensemble_weights, zeroshot_config_name, portfolio_configs), even if actual_n_portfolio is provided
        the result is still an ensemble consisting of single config strings
        """
        if datasets is None:
            datasets = self.repo.datasets()

        if folds is None:
            folds = self.repo.folds

        df_rank = transform_ranks(loss, deepcopy(dd))

        indices = zeroshot_configs(df_rank.values.T, n_portfolio)
        portfolio_configs = [df_rank.index[i] for i in indices]
        if config_name_map:
            portfolio_configs = [single_config for portfolio_name in portfolio_configs for single_config in config_name_map[portfolio_name]]

        metric_errors, ensemble_weights = self.repo.evaluate_ensemble(
            datasets=datasets,
            configs=portfolio_configs,
            ensemble_size=ensemble_size,
            backend=backend,
            folds=folds,
            rank=False,
        )

        zeroshot_config_name = self.zeroshot_name(n_portfolio=actual_n_portfolio if actual_n_portfolio else n_portfolio,
                                                  ensemble_size=ensemble_size,
                                                  base_name=base_name
                                                  )

        return metric_errors, ensemble_weights, zeroshot_config_name, portfolio_configs

    def filter_synthetic_portfolios(self, perc_best: float = 0.1) -> (Dict[int, List[pd.Series]], Dict[int, List[pd.Series]], Dict[int, Dict]):
        filtered_metric_errors, filtered_ensemble_weights, filtered_portfolio_name_to_cfg = {}, {}, {}
        for ((key_metric, series_list_metric), (key_ensembles, series_list_ensembles),
             (key_portfolio_name_to_cfg, portfolio_name_to_cfg_dict)) in \
                (zip(self.metric_errors.items(), self.ensemble_weights.items(), self.portfolio_name_to_config.items())):

            mean_values = [s.mean() for s in series_list_metric]
            threshold = pd.Series(mean_values).quantile(perc_best)

            filtered_metric_errors[key_metric] = [s for s, mean in zip(series_list_metric, mean_values)
                                                  if mean <= threshold]
            filtered_ensemble_weights[key_ensembles] = [s for s, mean in zip(series_list_ensembles, mean_values)
                                                        if mean <= threshold]

            filtered_cfg_dict = {k: v for (k, v), mean in zip(portfolio_name_to_cfg_dict.items(), mean_values)
                                                        if mean <= threshold}
            filtered_portfolio_name_to_cfg[key_portfolio_name_to_cfg] = filtered_cfg_dict

        self.metric_errors = filtered_metric_errors
        self.ensemble_weights = filtered_ensemble_weights
        self.portfolio_name_to_config = filtered_portfolio_name_to_cfg

        return filtered_metric_errors, filtered_ensemble_weights, filtered_portfolio_name_to_cfg

    def generate_evaluate(self, portfolio_size: int, datasets: List[str] = None, folds: List[int] = None,
                          ensemble_size: int = 100, n_portfolio_iter: int = 0, seed: int = 0, backend: str = "ray"):
        # ensure to get deterministic variance in portfolios generated
        seed_generator = seed + n_portfolio_iter

        portfolio = self.generate(portfolio_size=portfolio_size, seed=seed_generator)
        metric_errors, ensemble_weights = self.evaluate(portfolio=portfolio, datasets=datasets, folds=folds,
                                                        ensemble_size=ensemble_size, backend=backend)

        portfolio_name = AbstractPortfolioGenerator._portfolio_name(n_portfolio_iter=n_portfolio_iter,
                                                                    portfolio_size=len(portfolio),
                                                                    ensemble_size=ensemble_size, seed=seed)

        self.portfolio_name_to_config[portfolio_size][portfolio_name] = portfolio

        return metric_errors, ensemble_weights, portfolio_name

    def generate_evaluate_bulk(self, n_portfolios: int, portfolio_size: List[int], datasets: List[str] = None,
                               folds: List[int] = None, ensemble_size: int = 100, seed: int = 0, backend: str = "ray"):
        total_iterations = n_portfolios * len(portfolio_size)
        pbar = tqdm(total=total_iterations, desc="Synthetic Portfolio Generation Progress")

        for p_s in portfolio_size:
            metric_errors_ps, ensemble_weights_ps, portfolio_name_ps = [], [], []
            for i in range(n_portfolios):
                pbar.set_description(f"Synthetic Portfolio Generation Progress (size {p_s})")
                metric_errors, ensemble_weights, portfolio_name = self.generate_evaluate(portfolio_size=p_s,
                                                                                         datasets=datasets,
                                                                                         folds=folds,
                                                                                         ensemble_size=ensemble_size,
                                                                                         n_portfolio_iter=i,
                                                                                         seed=seed,
                                                                                         backend=backend)
                metric_errors_ps.append(metric_errors)
                ensemble_weights_ps.append(ensemble_weights)
                pbar.update(1)

            self.metric_errors[p_s] = metric_errors_ps
            self.ensemble_weights[p_s] = ensemble_weights_ps

        pbar.close()
        return self.metric_errors, self.ensemble_weights, self.portfolio_name_to_config

    def add_n_synthetic_portfolio(self, n_portfolio: int, dd: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        adds synthetic portfolios that were previously generated to an existing pd.DataFrame consisting of metric errors or performance data (performance matrix)
        :param n_portfolio: identifier to determine which of the performance matrices is used to augment the performance matrix
        :param dd: the base pd.DataFrame (performance matrix) containing the performance data
        :return: the updated base pd.DataFrame augmented by synthetic portfolios of the respective n_portfolio pool
        """
        m_e = self.metric_errors[n_portfolio]
        portfolio_names = list(self.portfolio_name_to_config[n_portfolio].keys())
        dd_with_syn_portfolio = dd.copy()
        dd_with_syn_portfolio = self.concatenate_bulk(base_df=dd_with_syn_portfolio,
                                                                            to_add_series_list=m_e,
                                                                            portfolio_names=portfolio_names
                                                                            )
        return dd_with_syn_portfolio, portfolio_names

    def add_zeroshot_portfolio(self, n_portfolio: int, dd: pd.DataFrame, dd_with_syn_portfolio: pd.DataFrame,
                               loss: str, results_dir: str,
                               seed: int = 0) -> Tuple[pd.DataFrame, List[str]]:
        """
        used to add a zeroshot portfolio to an existing pd.DataFrame performance matrix
        :param n_portfolio: indicates the size of the zeroshot portfolio that is supposed to be added to the performance matrix
        :param dd: the base performance matrix with the config evaluations without synthetic ensembles, pd. DataFrame of shape (task_id, 3) where 3=("metric_error", "framework", "task")
        :param dd_with_syn_portfolio: the base pd.DataFrame consisting of config evaluations as well as synthetic ensemble evaluations, same shape as dd but with more task_ids
        :param loss: a str indicating which loss to be used, needs to be in ["rank", "metric_error", "metric_error_val"]
        :param results_dir: a path as str that identifies where to store the random portfolio generator file
        :param seed: used to identify the correct random portfolio generator file
        :return: the original performance matrix augmented by the synthetic portfolios as well as a list of strings containing the names of the synthetic portfolios
        """
        zeroshot_metric_errors, _, zeroshot_config_name, portfolio_configs_zs = cache_function(
            fun=lambda: self.generate_evaluate_zeroshot(n_portfolio=n_portfolio,
                                                        dd=dd.copy(),
                                                        loss=loss,
                                                        base_name="Portfolio-ZS"),
            cache_name=f"random_portfolio_generator_zeroshot_n_portfolio_{n_portfolio}_{seed}",
            cache_path=results_dir,
            # ignore_cache=True,
            )
        self.portfolio_name_to_config[n_portfolio][zeroshot_config_name] = portfolio_configs_zs

        dd_with_syn_portfolio = self.concatenate(base_df=dd_with_syn_portfolio,
                                                 to_add_series=zeroshot_metric_errors,
                                                 portfolio_name=zeroshot_config_name
                                                 )

        portfolio_names = list(self.portfolio_name_to_config[n_portfolio].keys())
        return dd_with_syn_portfolio, portfolio_names

    def add_synthetic_zeroshot_portfolio(self, n_portfolio: int, dd_with_syn_portfolio: pd.DataFrame,
                                         dd_with_synthetic_ps_2: pd.DataFrame, portfolio_names: List[str],
                                         results_dir: str, loss: str = "metric_error",
                                         seed: int = 0) -> Tuple[pd.DataFrame, List[str]]:
        """
        adds a zeroshot portfolio based on synthetic portfolios of n_portfolio size 2 to the current performance matrix.
        :param n_portfolio: the size of the resulting portfolio
        :param dd_with_syn_portfolio: the 'running' base performance matrix containing single configurations as well as the synthetic portfolios that keeps getting augmented
        :param dd_with_synthetic_ps_2: the 'base' performance matrix that contains the random synthetic portfolios of size 2 of which the zeroshot algorithm produces larger portfolios
        :param portfolio_names: a list of synthetic portfolio names
        :param results_dir: a path as str that identifies where to store the random synthetic portfolio generator file
        :param loss: a str indicating which loss to be used, needs to be in ["rank", "metric_error", "metric_error_val"]
        :param seed: used to identify the correct random portfolio generator file
        :return: the running performance matrix augmented by the synthetic zeroshot portfolios as well as a list of strings containing the names of the synthetic zeroshot portfolios
        """
        # apply zeroshot only to ensembles of size 2; Portfolio-ZS also included
        if n_portfolio <= 2 or n_portfolio % 2 != 0 or dd_with_synthetic_ps_2 is None:
            return dd_with_syn_portfolio, portfolio_names

        dd_with_synthetic_ps_2_copy = deepcopy(dd_with_synthetic_ps_2)
        dd_syn_portfolios_ps_2_wo_single_configs = dd_with_synthetic_ps_2_copy[dd_with_synthetic_ps_2_copy["framework"].str.startswith('Portfolio')]


        zeroshot_syn_metric_errors, _, zeroshot_syn_config_name, portfolio_syn_configs_zs = cache_function(
            fun=lambda: self.generate_evaluate_zeroshot(n_portfolio=n_portfolio//2,
                                                        dd=dd_syn_portfolios_ps_2_wo_single_configs,
                                                        loss=loss,
                                                        config_name_map=self.portfolio_name_to_config[2],
                                                        actual_n_portfolio=n_portfolio,
                                                        base_name="Portfolio-SynZS"),
            cache_name=f"random_portfolio_generator_synthetic_zeroshot_n_portfolio_{n_portfolio}_{seed}",
            cache_path=results_dir,
        )

        self.portfolio_name_to_config[n_portfolio][zeroshot_syn_config_name] = portfolio_syn_configs_zs

        dd_with_syn_portfolio = self.concatenate(base_df=dd_with_syn_portfolio,
                                                 to_add_series=zeroshot_syn_metric_errors,
                                                 portfolio_name=zeroshot_syn_config_name
                                                 )

        portfolio_names = list(self.portfolio_name_to_config[n_portfolio].keys())
        return dd_with_syn_portfolio, portfolio_names

    def save_generator(self, file_path: str):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_generator(cls, file_path: str):
        with open(file_path, 'rb') as file:
            return pickle.load(file)


class RandomPortfolioGenerator(AbstractPortfolioGenerator):
    """
    a class to randomly sample single configs for ensemble generation
    """
    def __init__(self, repo: EvaluationRepository, n_portfolios: List[int]):
        super().__init__(repo=repo)
        self.n_portfolios = n_portfolios
        self.generated_portfolios = {}
        self.metric_errors = {}
        self.ensemble_weights = {}
        self.portfolio_name_to_config = {portfolio_size: {} for portfolio_size in n_portfolios}

    def generate(self, portfolio_size: int, seed: int = 0) -> List[str]:
        """
        samples configurations at random without replacement
        :param portfolio_size: specifies the number of randomly sampled configs
        :param seed: set to seed the RNG
        :return: a list of sampled config strings
        """
        rng = np.random.default_rng(seed=seed)
        return list(rng.choice(self.repo.configs(), portfolio_size, replace=False))


class ZeroshotEnsembleGenerator(AbstractPortfolioGenerator):
    """

    """
    def __init__(self, repo: EvaluationRepository):
        super().__init__(repo=repo)

    def generate_evaluate_zeroshot(self, n_portfolio: int, synthetic_portofolios_df: pd.DataFrame, datasets: List[str] = None, folds: List[int] = None, ensemble_size: int = 100, loss: str = "metric_error", backend: str = "ray") -> pd.DataFrame:
        if datasets is None:
            datasets = self.repo.datasets()

        if folds is None:
            folds = self.repo.folds

        df_rank = transform_ranks(loss, deepcopy(synthetic_portofolios_df))

        indices = zeroshot_configs(df_rank.values.T, n_portfolio)
        portfolio_configs = [df_rank.index[i] for i in indices]

        metric_errors, ensemble_weights = self.repo.evaluate_ensemble(
            datasets=datasets,
            configs=portfolio_configs,
            ensemble_size=ensemble_size,
            backend=backend,
            folds=folds,
            rank=False,
        )

        zeroshot_config_name = self.zeroshot_name(n_portfolio=n_portfolio, ensemble_size=ensemble_size)
        return metric_errors, ensemble_weights, zeroshot_config_name, portfolio_configs, zeroshot_config_name


class ClusterPortfolioGenerator(AbstractPortfolioGenerator):
    def __init__(self, repo: EvaluationRepository, n_portfolios: List[int]):
        super().__init__(repo=repo)
        self.n_portfolios = n_portfolios
        self.generated_portfolios = {}
        self.metric_errors = {}
        self.ensemble_weights = {}
        self.portfolio_name_to_config = {portfolio_size: {} for portfolio_size in n_portfolios}

    def generate(self, df: pd.DataFrame, portfolio_size: int, seed: int = 0) -> List[str]:
        def get_clusters(df, n_clusters):
            df = df.groupby(["framework", "dataset"])["rank"].mean().reset_index(drop=False)

            encoder_fw = LabelEncoder()
            encoder_ds = LabelEncoder()
            df.fillna(-100, inplace=True)
            df['framework_encoded'] = encoder_fw.fit_transform(df['framework'])
            df['dataset_encoded'] = encoder_ds.fit_transform(df['dataset'])

            features = df.drop(['framework', 'dataset'], axis=1)
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
            df['cluster'] = kmeans.fit_predict(features)

            return df

        df = get_clusters(df, n_clusters=10)

        sampled_frameworks = []
        for cluster in df['cluster'].unique():
            cluster_df = df[df['cluster'] == cluster]
            sampled_frameworks += cluster_df.sample(n=1)['framework'].tolist()

        return sampled_frameworks




