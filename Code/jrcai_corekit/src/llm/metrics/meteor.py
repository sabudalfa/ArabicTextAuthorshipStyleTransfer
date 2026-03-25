from .metric import EvaluationMetric
from evaluate import load
from typing import *


class MeteorMetric(EvaluationMetric):

    def __init__(self):
        self._meteor_metric = load('meteor')

    def __call__(
            self,
            ground_truth: List[str],
            predictions: List[str],
            sources: List[str],
    ) -> Dict[str, float]:
        compute_result = self._meteor_metric.compute(
            predictions=predictions,
            references=ground_truth,
        )
        return {'meteor': compute_result['meteor']}
