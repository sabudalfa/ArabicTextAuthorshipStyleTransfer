from .metric import EvaluationMetric
from evaluate import load
from typing import *


class RougeMetric(EvaluationMetric):

    def __init__(self):
        self._rouge_metric = load('rouge')

    def __call__(
            self,
            ground_truth: List[str],
            predictions: List[str],
            sources: List[str],
    ) -> Dict[str, float]:
        compute_result = self._rouge_metric.compute(
            predictions=predictions,
            references=ground_truth,
        )
        return {
            'rouge1': compute_result['rouge1'],
            'rouge2': compute_result['rouge2'],
            'rougeL': compute_result['rougeL'],
        }
