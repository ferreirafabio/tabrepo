from tabrepo.portfolio.zeroshot_selection import zeroshot_configs
from tabrepo.utils.normalized_scorer import NormalizedScorer
from tabrepo.utils.rank_utils import RankScorer
from typing import List, Tuple
from tabrepo import EvaluationRepository
import pandas as pd
import numpy as np


class AbstractPortfolioGenerator:
    def __init__(self, repo: EvaluationRepository):
        self.repo = repo

    def generate(self, *args, **kwargs) -> List[str]:
        raise NotImplementedError()

    # TODO: implement later
    # def generate_bulk(self, n_portfolios: int, generate_kwargs: dict): -> List[List[str]]

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
    def _portfolio_name(portfolio_size: int, ensemble_size: int, seed: int, name_suffix: str = ""):
        suffix = [
            f"-{letter}{x}" if x is not None else ""
            for letter, x in
            [("N", portfolio_size), ("E", ensemble_size), ("S", seed)]
        ]
        suffix += name_suffix
        suffix = "".join(suffix)
        return f"Portfolio{suffix}"

    def concatenate(self, real_errors: pd.DataFrame, synthetic_errors: pd.Series, portfolio_name: str):
        metric_errors_prepared = self._prepare_merge(metric_errors=synthetic_errors, portfolio_name=portfolio_name)
        return pd.concat([real_errors, metric_errors_prepared], axis=0, ignore_index=True)

    def _prepare_merge(self, metric_errors: pd.Series, portfolio_name: str):
        metric_errors = metric_errors.reset_index(drop=False)
        metric_errors["task"] = metric_errors.apply(lambda x: self.repo.task_name(x["dataset"], x["fold"]), axis=1)
        metric_errors["metric_error"] = metric_errors["error"]
        metric_errors["framework"] = portfolio_name
        metric_errors = metric_errors[["metric_error", "framework", "task"]]
        return metric_errors


class RandomPortfolioGenerator(AbstractPortfolioGenerator):
    def __init__(self, repo: EvaluationRepository):
        super().__init__(repo=repo)

    def generate(self, portfolio_size: int, seed: int = 0) -> List[str]:
        rng = np.random.default_rng(seed=seed)
        return list(rng.choice(self.repo.configs(), portfolio_size, replace=False))

    def generate_evaluate(self, portfolio_size: int, seed: int = 0, datasets: List[str] = None, folds: List[int] = None, ensemble_size: int = 100, backend: str="ray"):
        portfolio = self.generate(portfolio_size=portfolio_size, seed=seed)
        metric_errors, ensemble_weights = self.evaluate(portfolio=portfolio, datasets=datasets, folds=folds, ensemble_size=ensemble_size, backend=backend)
        portfolio_name = AbstractPortfolioGenerator._portfolio_name(portfolio_size=len(portfolio), ensemble_size=ensemble_size, seed=seed)

        return metric_errors, ensemble_weights, portfolio_name