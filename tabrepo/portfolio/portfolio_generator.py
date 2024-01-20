from tabrepo.portfolio.zeroshot_selection import zeroshot_configs
from tabrepo.utils.normalized_scorer import NormalizedScorer
from tabrepo.utils.rank_utils import RankScorer
from typing import List, Tuple
from tabrepo import EvaluationRepository
import pandas as pd
import numpy as np
import pickle
import os
from tabrepo.portfolio.zeroshot_selection import zeroshot_configs
from scripts.baseline_comparison.meta_learning_utils import transform_ranks
from copy import deepcopy


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
        return f"Portfolio-ZS{suffix}"

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

    def generate_evaluate(self, portfolio_size: int, datasets: List[str] = None, folds: List[int] = None, ensemble_size: int = 100, n_portfolio_iter: int = 0, seed: int = 0, backend: str = "ray"):
        # ensure to get deterministic variance in portfolios generated
        seed_generator = seed+n_portfolio_iter

        portfolio = self.generate(portfolio_size=portfolio_size, seed=seed_generator)
        metric_errors, ensemble_weights = self.evaluate(portfolio=portfolio, datasets=datasets, folds=folds, ensemble_size=ensemble_size, backend=backend)
        portfolio_name = AbstractPortfolioGenerator._portfolio_name(n_portfolio_iter=n_portfolio_iter, portfolio_size=len(portfolio), ensemble_size=ensemble_size, seed=seed)

        self.portfolio_name_to_config[portfolio_size][portfolio_name] = portfolio

        return metric_errors, ensemble_weights, portfolio_name

    def generate_evaluate_bulk(self, n_portfolios: int, portfolio_size: List[int], datasets: List[str] = None, folds: List[int] = None, ensemble_size: int = 100, seed: int = 0, backend: str = "ray"):
        for p_s in portfolio_size:
            print(f"generating synthetic portfolio of size {p_s}")
            metric_errors_ps, ensemble_weights_ps, portfolio_name_ps = [], [], []
            for i in range(n_portfolios):
                print(f"generating portfolio {i}/{n_portfolios}")
                metric_errors, ensemble_weights, portfolio_name = self.generate_evaluate(portfolio_size=p_s, datasets=datasets, folds=folds, ensemble_size=ensemble_size, n_portfolio_iter=i, seed=seed, backend=backend)
                metric_errors_ps.append(metric_errors)
                ensemble_weights_ps.append(ensemble_weights)
                # portfolio_name_ps.append(portfolio_name)

            self.metric_errors[p_s] = metric_errors_ps
            self.ensemble_weights[p_s] = ensemble_weights_ps
            # self.portfolio_name[p_s] = portfolio_name_ps

        return self.metric_errors, self.ensemble_weights, self.portfolio_name_to_config

    def generate_evaluate_zeroshot(self, n_portfolio: int, dd: pd.DataFrame, datasets: List[str] = None, folds: List[int] = None, ensemble_size: int = 100, loss: str = "metric_error", backend: str = "ray") -> pd.DataFrame:
        if datasets is None:
            datasets = self.repo.datasets()

        if folds is None:
            folds = self.repo.folds

        df_rank = transform_ranks(loss, deepcopy(dd))

        indices = zeroshot_configs(-df_rank.values.T, n_portfolio)
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
        # self.portfolio_name_to_config[n_portfolio][zeroshot_config_name] = portfolio_configs
        return metric_errors, ensemble_weights, zeroshot_config_name, portfolio_configs, zeroshot_config_name
