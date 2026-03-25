from typing import *


class LLMEvaluator:

    def __init__(self, evaluation_metrics_getter):
        self.evaluation_metrics_getter = evaluation_metrics_getter

    def __call__(
            self,
            ground_truth: List[str],
            predictions: List[str],
            sources: List[str],
    ) -> Dict:
        return {
            k: v
            for metric in self.evaluation_metrics_getter()
            for k, v in metric(
                ground_truth=ground_truth,
                predictions=predictions,
                sources=sources,
            ).items()
        }
