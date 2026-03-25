from .metric import EvaluationMetric
import sacrebleu
from typing import *


class BleuMetric(EvaluationMetric):

    def __init__(self):
        self.bleu = sacrebleu.BLEU()

    def __call__(self, ground_truth: List[str], predictions: List[str], sources: List[str]):
        for prediction in predictions:
            if not isinstance(prediction, str):
                print(prediction)
        bleu_score = self.bleu.corpus_score(predictions, [ground_truth]).score
        return {
            'bleu': bleu_score,
        }
