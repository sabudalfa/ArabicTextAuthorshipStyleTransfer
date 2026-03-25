from .metric import EvaluationMetric
import sacrebleu
from typing import *


class ChrfMetric(EvaluationMetric):

    def __call__(self, ground_truth: List[str], predictions: List[str], sources: List[str], word_order=2):
        chrf = sacrebleu.CHRF(word_order=word_order)
        chrf_score = chrf.corpus_score(predictions, [ground_truth]).score
        return {
            'chrf': chrf_score,
        }
