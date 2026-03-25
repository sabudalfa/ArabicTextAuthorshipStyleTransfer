from .metric import EvaluationMetric
from evaluate import load
from typing import *


class CometMetric(EvaluationMetric):

    def __init__(self):
        self._comet_metric = load('comet')

    def __call__(
            self,
            ground_truth: List[str],
            predictions: List[str],
            sources: List[str],
    ) -> Dict[str, float]:
        compute_result = self._comet_metric.compute(
            predictions=predictions,
            references=ground_truth,
            sources=sources,
            progress_bar=True,
        )
        return {'comet': compute_result['mean_score']}
