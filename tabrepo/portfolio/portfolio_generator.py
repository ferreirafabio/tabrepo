from tabrepo.portfolio.zeroshot_selection import zeroshot_configs
from tabrepo.utils.normalized_scorer import NormalizedScorer
from tabrepo.utils.rank_utils import RankScorer
from typing import List, Tuple
from tabrepo import EvaluationRepository
import pandas as pd
import numpy as np
import pickle
import os


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

    def concatenate(self, real_errors: pd.DataFrame, synthetic_errors: pd.Series, portfolio_name: str) -> pd.DataFrame:
        metric_errors_prepared = self._prepare_merge(metric_errors=synthetic_errors, portfolio_name=portfolio_name)
        return pd.concat([real_errors, metric_errors_prepared], axis=0, ignore_index=True)

    def concatenate_bulk(self, real_errors: pd.DataFrame, synthetic_errors: List[pd.Series], portfolio_names: List[str]) -> pd.DataFrame:
        assert all(len(series) == len(synthetic_errors[0]) for series in synthetic_errors), "All Series must have the same length"
        assert len(synthetic_errors) == len(portfolio_names)

        for synthetic_errors_i, portfolio_name_i in zip(synthetic_errors, portfolio_names):
            real_errors = self.concatenate(real_errors=real_errors, synthetic_errors=synthetic_errors_i, portfolio_name=portfolio_name_i)
        return real_errors

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


# class BestPortfolioGenerator(AbstractPortfolioGenerator):
#     def __init__(self, repo: EvaluationRepository):
#         super().__init__(repo=repo)
#
#     def generate(self, portfolio_size: int, random_config_fraction: float = 0.0, seed: int = 0) -> List[str]:
#         n_random_ensembles = int(sample_size * random_config_fraction)
#
#         rng = np.random.default_rng(seed=seed)
#         return list(rng.choice(self.repo.configs(), portfolio_size, replace=False))