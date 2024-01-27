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
        metric_errors_prepared = self._prepare_merge(metric_errors=to_add_series, portfolio_name=portfolio_name)
        return pd.concat([base_df, metric_errors_prepared], axis=0, ignore_index=True)

    def concatenate_bulk(self, base_df: pd.DataFrame, to_add_series_list: List[pd.Series], portfolio_names: List[str]) -> pd.DataFrame:
        assert all(len(series) == len(to_add_series_list[0]) for series in to_add_series_list), "All Series must have the same length"
        assert len(to_add_series_list) == len(portfolio_names)

        for to_add_series, portfolio_name_i in zip(to_add_series_list, portfolio_names):
            base_df = self.concatenate(base_df=base_df, to_add_series=to_add_series, portfolio_name=portfolio_name_i)
        return base_df

    def _prepare_merge(self, metric_errors: pd.Series, portfolio_name: str):
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

        if n_portfolio <= 2 or n_portfolio % 2 != 0 or dd_with_synthetic_ps_2 is None:
            return dd_with_syn_portfolio, portfolio_names

        # apply zeroshot only to ensembles of size 2; Portfolio-ZS also included
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
    def __init__(self, repo: EvaluationRepository, n_portfolios: List[int]):
        super().__init__(repo=repo)
        self.n_portfolios = n_portfolios
        self.generated_portfolios = {}
        self.metric_errors = {}
        self.ensemble_weights = {}
        self.portfolio_name_to_config = {portfolio_size: {} for portfolio_size in n_portfolios}

    def generate(self, portfolio_size: int, seed: int = 0) -> List[str]:
        rng = np.random.default_rng(seed=seed)
        return list(rng.choice(self.repo.configs(), portfolio_size, replace=False))


class ZeroshotEnsembleGenerator(AbstractPortfolioGenerator):
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




