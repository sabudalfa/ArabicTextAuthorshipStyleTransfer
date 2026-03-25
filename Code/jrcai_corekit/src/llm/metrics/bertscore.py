from .metric import EvaluationMetric
from typing import *
from evaluate import load
from statistics import mean


class BertScoreMetric(EvaluationMetric):

    def __init__(self, lang):
        self._bertscore = load("bertscore")
        self._lang = lang

    def __call__(self, ground_truth: List[str], predictions: List[str], sources: List[str],) -> Dict[str, float]:
        assert len(ground_truth) == len(predictions)
        compute_result = self._bertscore.compute(
            predictions=predictions,
            references=ground_truth,
            lang=self._lang,
            verbose=True,
        )
        return {
            'bertscore_precision': mean(compute_result['precision']),
            'bertscore_recall': mean(compute_result['recall']),
            'bertscore_f1': mean(compute_result['f1']),
        }
